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
logger = logging.getLogger(__name__)

def evaluate_model(pipe, test_data):
    """
    Evaluate the model on the provided test data.

    Args:
        model: The trained model to evaluate.
        test_data: The data to evaluate the model on.

    Returns:
        A dictionary containing evaluation metrics.
    """
    logger.info("Starting model evaluation...")

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

    logger.info("Model evaluation completed.")
    
    return wer


if __name__ == "__main__":

    if len(sys.argv) != 2:
        logger.error("Usage: python EvaluateModel.py <model_path>")
        sys.exit(1)


    if torch.cuda.is_available():
        device = 0                 # GPU‑0
        torch_dtype = torch.float16
    else:
        device = -1                # CPU
        torch_dtype = torch.float32

    model_path = sys.argv[1]
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device,
        torch_dtype=torch_dtype,
    )

    subsets = ["bik", "ceb", "hil", "ilo", "mrw", "pag", "tgl", "war", "pam", "bisaya"]
    
    evaluation_results = {}
    with Pool(processes=cpu_count()) as pool:
        for subset in subsets:
            logger.info(f"Evaluating model on subset: {subset}")
            test_data = load_dataset("tagalog-speech", subset, split="test")
            result = pool.apply_async(evaluate_model, (pipe, test_data))
            evaluation_results[subset] = result.get()

    logger.info("Plotting results...")

    fig, ax = plt.subplots()

    y_pos = np.arange(len(subsets))

    small_model = list(evaluation_results.values())

    # medium_model = [16.030283080974325, 21.57816005983545, 15.73165947430365, 17.93032786885246]

    for model, scores in zip(['small'], [small_model]):
        ax.barh(y_pos, scores, height=0.4, color='steelblue', label=model)

    for i, v in enumerate(scores):
        ax.text(v + 0.3, i, f"{v:.2f}", va='center', fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(evaluation_results.keys(), fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel('Word Error Rate (%)', fontsize=12)
    ax.set_title('WER by Dialect – Small Model', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("small_model_wer.png", dpi=300)
    plt.show()
