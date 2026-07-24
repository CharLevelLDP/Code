import json
import logging
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


np.random.seed(42)
random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    logging.error(
        "OpenAI API key not set. Please set the OPENAI_API_KEY environment variable."
    )
    exit(1)



# 1. Load Prompts

json_file_path = "health_fixed_dataset.json"
#json_file_path = "email_dataset.json"

if not os.path.exists(json_file_path):
    logging.error(f"JSON file not found at path: {json_file_path}")
    exit(1)

try:
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        questions_data = json.load(json_file)
except json.JSONDecodeError as e:
    logging.error(f"Error decoding JSON: {e}")
    exit(1)
except OSError as e:
    logging.error(f"Error reading JSON file: {e}")
    exit(1)

questions = questions_data.get("text", [])[]  # Limit to first n prompts for testing; set to None for all

if not questions:
    logging.error("No 'text' key found in JSON data or insufficient prompts.")
    exit(1)



# 2. Local Differential Privacy


# Domain C: printable ASCII without spaces (33–126), so |C| = 94.
DOMAIN_CHARS = [chr(i) for i in range(33, 127)]
KAPPA = len(DOMAIN_CHARS)

# Precompute C \ {c} for each character.
CHOICES_MAP = {
    char: [candidate for candidate in DOMAIN_CHARS if candidate != char]
    for char in DOMAIN_CHARS
}


def calculate_gamma(kappa: int, epsilon: float) -> float:
   
    return (kappa - 1) / (kappa - 1 + np.exp(epsilon))


def k_ary_randomized_response(char: str, gamma: float) -> str:
    
    if char not in DOMAIN_CHARS:
        return char

    if np.random.rand() < gamma:
        return random.choice(CHOICES_MAP[char])

    return char


def apply_ldp_to_text(text: str, epsilon: float) -> str:
    
    gamma = calculate_gamma(KAPPA, epsilon)

    return "".join(
        k_ary_randomized_response(char, gamma)
        for char in text
    )


# 3. Restore Text Using OpenAI


def enhance_text_with_llm(input_text: str) -> str:
    retries = 3
    backoff = 2

    for attempt in range(retries):
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-5.4-2026-03-05",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a text restoration assistant. "
                            "Your task is to correct any errors in the given text "
                            "that were introduced by distortion. "
                            "Please do not make any changes that are not necessary "
                            "to restore the original text. "
                            "No matter how distorted the text is, please restore it. "
                            "When restoring the words, use the surrounding context. "
                            "Preserve the original wording, punctuation, "
                            "capitalization, and formatting."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Please correct the following text:\n\n"
                            f"{input_text}"
                        ),
                    },
                ],
                max_completion_tokens=460,
                temperature=0.5,
            )

            return resp["choices"][0]["message"]["content"].strip()

        except openai.error.RateLimitError:
            wait_time = backoff ** attempt
            logging.warning(
                f"Rate limit reached. Retrying in {wait_time} seconds..."
            )
            time.sleep(wait_time)

        except openai.error.APIError as e:
            wait_time = backoff ** attempt
            logging.warning(
                f"API error: {e}. Retrying in {wait_time} seconds..."
            )
            time.sleep(wait_time)

        except Exception as e:
            logging.error(f"Unexpected error enhancing text: {e}")
            break

    # Return the privatized text if restoration fails.
    return input_text



# 4. Load SentenceTransformer


try:
    model = SentenceTransformer(
        "all-distilroberta-v1",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
except Exception as e:
    logging.error(f"Error loading SentenceTransformer model: {e}")
    exit(1)


# 5. Epsilon Range


epsilons = np.arange(1.0, 10.5, 0.5)




# 6. Output Paths


detailed_csv_path = "Resotred_output.csv"
summary_csv_path = "Resotred_output_summary.csv"
plot_path = "Resotred_output_summary.png"



os.makedirs(
    os.path.dirname(summary_csv_path),
    exist_ok=True,
)

# 7. Run Restoration Experiment


all_results = []
summary_results = []

total_iterations = len(epsilons) * len(questions)

with tqdm(
    total=total_iterations,
    desc="Processing",
    unit="step",
) as pbar:

    for epsilon in epsilons:
        epsilon_scores = []

        for original_text in questions:
            ldp_text = apply_ldp_to_text(
                original_text,
                epsilon=epsilon,
            )

            enhanced_text = enhance_text_with_llm(ldp_text)

            try:
                emb_orig = model.encode(
                    original_text,
                    convert_to_tensor=True,
                )
                emb_enh = model.encode(
                    enhanced_text,
                    convert_to_tensor=True,
                )

                sim = util.pytorch_cos_sim(
                    emb_orig,
                    emb_enh,
                ).item()

            except Exception as e:
                logging.error(
                    f"Error computing semantic similarity: {e}"
                )
                sim = None

            all_results.append(
                {
                    "epsilon": epsilon,
                    "original_text": original_text,
                    "ldp_modified_text": ldp_text,
                    "enhanced_text": enhanced_text,
                    "semantic_score": sim,
                }
            )

            if sim is not None:
                epsilon_scores.append(sim)

            pbar.update(1)

        if epsilon_scores:
            avg_sim = sum(epsilon_scores) / len(epsilon_scores)
        else:
            avg_sim = None

        summary_results.append(
            {
                "epsilon": epsilon,
                "average_semantic_similarity": avg_sim,
            }
        )


# 8. Save Results


detailed_df = pd.DataFrame(all_results)
detailed_df.to_csv(detailed_csv_path, index=False)

summary_df = pd.DataFrame(summary_results)
summary_df.to_csv(summary_csv_path, index=False)

logging.info(f"Detailed results saved to: {detailed_csv_path}")
logging.info(f"Summary results saved to: {summary_csv_path}")



# 9. Plot Results


plt.figure(figsize=(10, 6))

plt.plot(
    summary_df["epsilon"],
    summary_df["average_semantic_similarity"],
    marker="o",
)

plt.xlabel("Epsilon")
plt.ylabel("Average Semantic Similarity")
plt.title("Privacy-Utility Trade-Off Curve")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight",
)

logging.info(f"Plot saved to: {plot_path}")

plt.show()