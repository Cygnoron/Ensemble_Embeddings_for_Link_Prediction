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
    filters = dataset.get_filters()
    if mode == "test":
        test_examples = dataset.get_examples("test")
        metrics = avg_both(*model.compute_metrics(test_examples, filters, args.sizes))
    elif mode == "valid":
        valid_examples = dataset.get_examples("valid")
        metrics = avg_both(*model.compute_metrics(valid_examples, filters, args.sizes))
    else:
        raise ValueError(f"The given mode {mode} does not exist!")
    return metrics, args


def format_metrics_latex(metrics, args):
    """Format metrics for logging."""
    if (hasattr(args, "subgraph_amount") and hasattr(args, "subgraph_size_range") and
            not isinstance(args.subgraph_size_range, str) and hasattr(args, "rho")):
        result = (f"{args.subgraph_amount}\t& "
                  f"${util.format_set(args.subgraph_size_range, first_char='(', last_char=')')}$\t& "
                  f"${args.rho if args.rho > 1 else '-'}$\t\t\t\t& ")
    else:
        result = "\\multicolumn{3}{c}{Baseline}\t\t\t& "

    for mode in ['average']:
        result += f"{metrics['MRR'][mode]:.3f}\t\t\t\t& "
        result += f"{metrics['MR'][mode]:.3f}\t\t\t\t& "
        result += f"{metrics['AMRI'][mode]:.3f}\t\t\t\t& "
        result += f"{metrics['hits@[1,3,10]'][mode][0]:.3f}\t\t\t\t& "
        result += f"{metrics['hits@[1,3,10]'][mode][1]:.3f}\t\t\t\t& "
        result += f"{metrics['hits@[1,3,10]'][mode][2]:.3f}\t\t\t\t& "
        result += f"{metrics['rank_deviation'][mode]:.5f}\t\t\t\\\\"

    return result


if __name__ == "__main__":
    args = argparse.Namespace()
    split = "test"
    dataset = "WN18RR"
    model = "TransE"
    args.model_dir = os.path.join("results_for_thesis", "Results", dataset, model)

    test_baseline = True
    test_ensemble = True

    # args = parser.parse_args()
    # to_list = os.path.join("data", args.model_dir)
    to_list = args.model_dir
    result_directories = os.listdir(to_list)

    print(f"{split.capitalize()} for {dataset} {model}\n")

    with (open(os.path.join("results_for_thesis", "Results", dataset, model, f"{split}_metrics_{dataset}_{model}.txt"),
               "w+") as out_file):
        for result_directory in result_directories:
            if not os.path.isdir(os.path.join(to_list, result_directory)):
                continue
            if "baseline" in result_directory.lower() and test_baseline:
                try:
                    out_file.write(f"-------------------------------------------------------------------------"
                                   f"------------------------------------------------------------------------\n"
                                   f"{split.capitalize()} from result directory \"{result_directory}\"")
                    test_metrics, args_test = test(os.path.join(to_list, result_directory), mode=split)
                    out_file.write(format_metrics(test_metrics, split=split))
                    out_file.write("\n")
                    print(format_metrics_latex(test_metrics, args_test))
                except FileNotFoundError or IOError as file_error:
                    print(f"{file_error}")

            elif ("result" in result_directory.lower() or result_directory.isdigit()) and test_ensemble:
                try:
                    out_file.write(f"-------------------------------------------------------------------------"
                                   f"------------------------------------------------------------------------\n"
                                   f"{split.capitalize()} from result directory \"{result_directory}\"")
                    paths = argparse.Namespace(
                        result_directory=result_directory,
                        config_name=os.path.join("model_files", "config_unified.json"),
                        init_config_name=os.path.join("model_files", "config_init.json"),
                        model_name=os.path.join("model_files", "unified_model.pt"),
                        dataset_path=os.path.join("data", dataset)
                    )
                    test_metrics, args_test = test(to_list, mode=split, paths=paths)
                    out_file.write(format_metrics(test_metrics, split=split))
                    out_file.write("\n")
                    print(format_metrics_latex(test_metrics, args_test))
                except FileNotFoundError as file_error:
                    print(f"{file_error}")
        out_file.write(f"-------------------------------------------------------------------------"
                       f"------------------------------------------------------------------------")
