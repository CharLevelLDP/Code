#!/usr/bin/env python3
"""Summarize original and restored texts and measure summary similarity."""

import logging
import os
import time

import pandas as pd
import torch
from openai import APIError, OpenAI, RateLimitError
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

input_restoration_results_csv = "Resotred_output.csv"
summary_evaluation_output_csv = "Summary_output.csv"

openai_model = "gpt-5.4-2026-03-05"
embedding_model_name = "all-distilroberta-v1"
max_output_tokens = 300
retries = 3

SUMMARIZATION_PROMPT = (
    "You are a summarization assistant. "
    "Create a concise and accurate summary of the given text. "
    "Focus on the main ideas and key details, avoid unnecessary details, and do not add opinions. "
    "Return only the summary, without explanations or prefacing."
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def summarize_text(client: OpenAI, input_text: str) -> str:
    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=openai_model,
                instructions=SUMMARIZATION_PROMPT,
                input=input_text,
                max_output_tokens=max_output_tokens,
            )
            summary = response.output_text.strip()
            return summary or input_text

        except (RateLimitError, APIError) as error:
            if attempt == retries - 1:
                logging.error("OpenAI request failed after %d attempts: %s", retries, error)
                break

            delay = 2**attempt
            logging.warning("OpenAI request failed. Retrying in %d second(s).", delay)
            time.sleep(delay)

        except Exception as error:
            logging.exception("Unexpected OpenAI error: %s", error)
            break

    return input_text


def compute_similarity(
    model: SentenceTransformer,
    first_text: str,
    second_text: str,
) -> float:
    embeddings = model.encode(
        [first_text, second_text],
        convert_to_tensor=True,
    )
    return float(util.cos_sim(embeddings[0], embeddings[1]).item())


def main() -> None:
    configure_logging()

    if not OPENAI_API_KEY:
        raise ValueError("Set OPENAI_API_KEY at the top of the script.")

    if not os.path.isfile(input_restoration_results_csv):
        raise FileNotFoundError(
            f"Input restoration results not found: {input_restoration_results_csv}"
        )

    dataframe = pd.read_csv(input_restoration_results_csv)
    required_columns = {"epsilon", "original_text", "enhanced_text"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    client = OpenAI(api_key=OPENAI_API_KEY)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(embedding_model_name, device=device)

    results = []

    for row_index, row in tqdm(
        dataframe.iterrows(),
        total=len(dataframe),
        desc="Summarizing",
        unit="row",
    ):
        original_text = str(row["original_text"])
        restored_text = str(row["enhanced_text"])

        original_summary = summarize_text(client, original_text)
        restored_summary = summarize_text(client, restored_text)
        similarity = compute_similarity(
            embedding_model,
            original_summary,
            restored_summary,
        )

        results.append(
            {
                "row_id": row_index,
                "prompt_id": row.get("prompt_id", row_index),
                "epsilon": row["epsilon"],
                "original_text": original_text,
                "enhanced_text": restored_text,
                "summary_original": original_summary,
                "summary_enhanced": restored_summary,
                "summary_semantic_score": similarity,
            }
        )

    output_parent = os.path.dirname(summary_evaluation_output_csv)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    pd.DataFrame(results).to_csv(summary_evaluation_output_csv, index=False)
    logging.info("Summary evaluation saved to %s", summary_evaluation_output_csv)


if __name__ == "__main__":
    main()