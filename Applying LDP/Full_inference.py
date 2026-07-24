#!/usr/bin/env python3
"""Character-level local differential privacy restoration experiment."""

import json
import logging
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from openai import APIError, OpenAI, RateLimitError
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


json_file_path = "health_fixed_dataset.json"
detailed_csv_path = "Restored_summarized_output.csv"
summary_csv_path = "Restored_summarized_output_summary.csv"
plot_path = "Restored_summarized_output_summary.png"

openai_model = "gpt-5.4-2026-03-05"
embedding_model_name = "all-distilroberta-v1"
number_of_prompts = None   # Set to None to use all prompts in the JSON file
max_completion_tokens = 800
retries = 3
seed = 42

epsilons = np.arange(1.0, 10.5, 0.5)

SYSTEM_PROMPT = (
    "You are a text restoration and summarization assistant. "
    "First, correct only the errors in the given text that were introduced by distortion or noise. "
    "Do not make any changes that are not necessary to restore the original text. "
    "No matter how distorted the text is, please restore it with an understanding of the context. "
    "Preserve the original wording, punctuation, capitalization, and formatting as much as possible. "
    "Second, create a concise and accurate summary of the restored text. "
    "Focus on the main ideas and key details, avoid unnecessary details, and do not add opinions or any prefacing."
)

DOMAIN_CHARS = tuple(chr(code) for code in range(33, 127))
DOMAIN_SET = frozenset(DOMAIN_CHARS)
KAPPA = len(DOMAIN_CHARS)
CHOICES_MAP = {
    char: tuple(candidate for candidate in DOMAIN_CHARS if candidate != char)
    for char in DOMAIN_CHARS
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def set_seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def load_prompts(file_path: str, limit: int | None) -> list[str]:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    prompts = data.get("text")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("The JSON file must contain a non-empty 'text' list.")

    prompts = [text for text in prompts if isinstance(text, str) and text.strip()]
    if limit is not None:
        prompts = prompts[:limit]

    if not prompts:
        raise ValueError("No valid prompts were found in the JSON file.")

    return prompts


def calculate_gamma(kappa: int, epsilon: float) -> float:
    return (kappa - 1) / (kappa - 1 + np.exp(epsilon))


def k_ary_randomized_response(char: str, gamma: float) -> str:
    if char not in DOMAIN_SET:
        return char

    if np.random.random() < gamma:
        return random.choice(CHOICES_MAP[char])

    return char


def apply_ldp_to_text(text: str, epsilon: float) -> str:
    gamma = calculate_gamma(KAPPA, epsilon)
    return "".join(k_ary_randomized_response(char, gamma) for char in text)


def restore_and_summarize(
    client: OpenAI,
    input_text: str,
) -> str:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": input_text},
                ],
                max_completion_tokens=max_completion_tokens,
            )
            content = response.choices[0].message.content
            return content.strip() if content else input_text

        except (RateLimitError, APIError) as error:
            if attempt == retries - 1:
                logging.error(
                    "OpenAI request failed after %d attempts: %s",
                    retries,
                    error,
                )
                break

            delay = 2**attempt
            logging.warning(
                "OpenAI request failed. Retrying in %d second(s): %s",
                delay,
                error,
            )
            time.sleep(delay)

        except Exception as error:
            logging.exception("Unexpected OpenAI error: %s", error)
            break

    return input_text


def compute_semantic_similarity(
    model: SentenceTransformer,
    original_text: str,
    generated_text: str,
) -> float:
    embeddings = model.encode(
        [original_text, generated_text],
        convert_to_tensor=True,
    )
    return float(util.cos_sim(embeddings[0], embeddings[1]).item())


def save_results(
    detailed_results: list[dict[str, object]],
    summary_results: list[dict[str, float]],
) -> None:
    for file_path in (detailed_csv_path, summary_csv_path, plot_path):
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    detailed_df = pd.DataFrame(detailed_results)
    summary_df = pd.DataFrame(summary_results)

    detailed_df.to_csv(detailed_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        summary_df["epsilon"],
        summary_df["average_semantic_similarity"],
        marker="o",
    )
    axis.set_xlabel("Epsilon")
    axis.set_ylabel("Average Semantic Similarity")
    axis.set_title("Privacy-Utility Trade-Off Curve")
    axis.grid(True)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=300)
    plt.close(figure)


def main() -> None:
    configure_logging()
    set_seed(seed)

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set."
        )

    questions = load_prompts(json_file_path, number_of_prompts)
    client = OpenAI(api_key=OPENAI_API_KEY)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(
        embedding_model_name,
        device=device,
    )

    detailed_results = []
    summary_results = []
    total_iterations = len(epsilons) * len(questions)

    with tqdm(
        total=total_iterations,
        desc="Processing",
        unit="step",
    ) as progress:
        for epsilon in epsilons:
            epsilon_scores = []

            for prompt_id, original_text in enumerate(questions):
                ldp_text = apply_ldp_to_text(
                    original_text,
                    float(epsilon),
                )
                enhanced_text = restore_and_summarize(
                    client,
                    ldp_text,
                )
                semantic_score = compute_semantic_similarity(
                    embedding_model,
                    original_text,
                    enhanced_text,
                )

                detailed_results.append(
                    {
                        "prompt_id": prompt_id,
                        "epsilon": float(epsilon),
                        "original_text": original_text,
                        "ldp_modified_text": ldp_text,
                        "enhanced_text": enhanced_text,
                        "semantic_score": semantic_score,
                    }
                )
                epsilon_scores.append(semantic_score)
                progress.update(1)

            summary_results.append(
                {
                    "epsilon": float(epsilon),
                    "average_semantic_similarity": float(
                        np.mean(epsilon_scores)
                    ),
                }
            )

    save_results(detailed_results, summary_results)
    logging.info("Detailed results saved to %s", detailed_csv_path)
    logging.info("Summary results saved to %s", summary_csv_path)
    logging.info("Plot saved to %s", plot_path)


if __name__ == "__main__":
    main()