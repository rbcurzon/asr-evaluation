import logging
import argparse
import sys
from multiprocessing import Pool
from multiprocessing import cpu_count

import torch
import matplotlib.pyplot as plt
import numpy as np

from transformers import pipeline
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from evaluate import load

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def evaluate_model(pipe, test_data):
    """
    Evaluate the model on the provided test data.

    Args:
        model: The trained model to evaluate.
        test_data: The data to evaluate the model on.

    Returns:
        A dictionary containing evaluation metrics.
    """
    logging.info("Starting model evaluation...")

    all_predictions = [
        pred["text"]
        for pred in tqdm(
            pipe(
                KeyDataset(test_data, "audio"),
                return_timestamps=True,
                generate_kwargs={
                    "task": "transcribe",
                    "language": "tagalog",
                    "max_new_tokens": 128,
                },
            ),
            total=len(test_data),
        )
    ]

    wer_metric = load("wer")

    wer = 100 * wer_metric.compute(
        references=test_data["transcription"], predictions=all_predictions
    )

    logging.info("Model evaluation completed.")
    
    return wer


def plot_results(evaluation_results, model_name):

    fig, ax = plt.subplots()

    y_pos = np.arange(len(evaluation_results))

    results_list = list(evaluation_results.values())

    # medium_model = [16.030283080974325, 21.57816005983545, 15.73165947430365, 17.93032786885246]

    for model, scores in zip([model_name], [results_list]):
        ax.barh(y_pos, scores, height=0.4, color='steelblue', label=model)

    for i, v in enumerate(scores):
        ax.text(v + 0.3, i, f"{v:.2f}", va='center', fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(evaluation_results.keys(), fontsize=10)
    ax.invert_yaxis()
    
    ax.set_xlabel('Word Error Rate (%)', fontsize=12)
    ax.set_title(f'WER by Dialect – {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{model_name}_wer.png", dpi=300)
    plt.show()


def main():
    if len(sys.argv) != 2:
        logging.error("Usage: python EvaluateModel.py <model>")
        sys.exit(1)


    if torch.cuda.is_available():
        device = 0                 # GPU‑0
        torch_dtype = torch.float16
    else:
        device = -1                # CPU
        torch_dtype = torch.float32

    model = sys.argv[1]
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        device=device,
        torch_dtype=torch_dtype,
    )

    subsets = ["bik", "ceb", "hil", "ilo", "mrw", "pag", "tgl", "war", "pam"]
    
    ds = {}
    with Pool(processes=cpu_count()) as pool:
        results = {}
        for subset in subsets:
            logging.info(f"Loading dataset for {subset} subset...")
            results[subset] = pool.apply_async(
                load_dataset,
                args=("rbcurzon/ph_dialect_asr", subset),
                kwds={"split": "test"},
            )
        pool.close()
        pool.join()
        ds = {subset: results[subset].get() for subset in subsets}
    
    evaluation_results = {}
    for dataset, test_data in ds.items():
        logging.info(f"Evaluating model on {dataset} dataset...")
        evaluation_results[dataset] = evaluate_model(pipe, test_data)

    logging.info("Plotting results...")
    plot_results(evaluation_results, model.split("/")[-1])

if __name__ == "__main__":
    main()