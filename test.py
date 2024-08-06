"""Evaluation script."""

import argparse
import json
import os

import torch

import models
from datasets.kg_dataset import KGDataset
from ensemble import Constants
from utils.train import avg_both, format_metrics

parser = argparse.ArgumentParser(description="Test")
parser.add_argument('--model_dir', help="Model path")


def test(model_dir, mode="test"):
    # load config
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    args = argparse.Namespace(**config)

    # create dataset
    dataset_path = os.path.join("data", args.dataset)
    dataset = KGDataset(dataset_path, False)
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    # load pretrained model weights
    model = getattr(models, args.model)(args)
    device = 'cuda'
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt')))

    # eval
    if mode == "test":
        metrics = avg_both(*model.compute_metrics(test_examples, filters, args.sizes))
    elif mode == "valid":
        metrics = avg_both(*model.compute_metrics(test_examples, filters, args.sizes))
    else:
        raise ValueError(f"The given mode {mode} does not exist!")
    return metrics


if __name__ == "__main__":
    args = parser.parse_args()

    for aggregation in Constants.AGGREGATION_METHODS:
        args.aggregation_method = aggregation

        valid_metrics = test(args.model_dir, mode="valid")
        print(format_metrics(valid_metrics, split='valid'))

    test_metrics = test(args.model_dir, mode="test")
    print(format_metrics(test_metrics, split='test'))
