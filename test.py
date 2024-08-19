"""Evaluation script."""

import argparse
import json
import os

import torch

import models
from datasets.kg_dataset import KGDataset
from ensemble import util
from utils.train import avg_both, format_metrics

parser = argparse.ArgumentParser(description="Test")
parser.add_argument('--model_dir', help="Model path")


def test(model_dir, mode="test", paths=None):
    if paths is None:
        paths = argparse.Namespace(
            result_directory="",
            model_name="model.pt",
            config_name="config.json"
        )

    # load config
    with open(os.path.join(model_dir, paths.result_directory, paths.config_name), "r") as f:
        config = json.load(f)
    args = argparse.Namespace(**config)

    # create dataset
    if not hasattr(paths, "dataset_path"):
        paths.dataset_path = os.path.join("data", args.dataset)

    dataset = KGDataset(paths.dataset_path, False)
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    # load pretrained model weights
    if args.model_name == "Unified":
        args.test_mode = True
        with open(os.path.join(model_dir, paths.result_directory, paths.init_config_name), "r") as init_config_file:
            init_args = argparse.Namespace(**json.load(init_config_file))

        model = getattr(models, args.model)(args, init_args, args)
    else:
        model = getattr(models, args.model)(args)

    device = 'cuda'
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, paths.result_directory, paths.model_name)))

    # eval
    if mode == "test":
        metrics = avg_both(*model.compute_metrics(test_examples, filters, args.sizes))
    elif mode == "valid":
        metrics = avg_both(*model.compute_metrics(test_examples, filters, args.sizes))
    else:
        raise ValueError(f"The given mode {mode} does not exist!")
    return metrics


if __name__ == "__main__":
    args = argparse.Namespace()
    args.model_dir = "FB15K-237_Entity_sampling_N30_min0.6_max0.7"

    # args = parser.parse_args()

    result_directories = os.listdir(os.path.join("data", args.model_dir))

    for result_directory in result_directories:
        if "result" in result_directory:
            paths = argparse.Namespace(
                result_directory=result_directory,
                config_name=os.path.join("model_files", "config_unified.json"),
                init_config_name=os.path.join("model_files", "config_init.json"),
                model_name=os.path.join("model_files", "unified_model.pt"),
                dataset_path=os.path.join("data", util.get_dataset_name(args.model_dir))
            )
            print(f"Testing from result directory \"{result_directory}\"")
            test_metrics = test(os.path.join("data", args.model_dir), mode="test", paths=paths)
            print(format_metrics(test_metrics, split='test'))
