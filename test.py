"""Evaluation script."""

import argparse
import json
import os
import pickle
import traceback

import torch

import models
from datasets.kg_dataset import KGDataset
from ensemble import util, subsampling, util_files
from utils.train import avg_both, format_metrics

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

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

    try:
        args.dataset_dir = args.dataset_dir.replace('/', os.sep)
        parent_dataset = args.dataset_dir.split('_')[0]
    except:
        parent_dataset = os.path.join("data", args.dataset)
    paths.dataset_path = parent_dataset

    try:
        print(f"Testing for sub_{args.subgraph_amount - 1:03d}")
        last_subgraph_path = os.path.join(args.dataset_dir, f"sub_{args.subgraph_amount - 1:03d}", "train.pickle")
        with open(last_subgraph_path, mode='rb') as test_file:
            test_subgraph = pickle.load(test_file)
        print(f"All subgraphs exist.")
    except Exception:
        if hasattr(paths, "init_config_name"):
            print("Subgraphs not found, resampling based on given parameters...")

            util_files.check_directory(args.dataset_dir)
            info_directory = util_files.get_info_directory_path(args.dataset_dir, args)

            parent_dataset = parent_dataset.split(os.sep)[1]
            print(f"Sampling {args.subgraph_amount} subgraphs with size {args.subgraph_size_range} and rho {args.rho} "
                  f"from {parent_dataset}.")

            subsampling.sample_graph(info_directory, parent_dataset, args.dataset_dir, args.sampling_method,
                                     subgraph_amount=args.subgraph_amount,
                                     subgraph_size_range=args.subgraph_size_range,
                                     entities_per_step=args.entities_per_step,
                                     rho=args.rho, no_progress_bar=args.no_progress_bar)

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
    model_path = os.path.join(model_dir, paths.result_directory, paths.model_name)
    print(f"The model is located at {model_path}.")
    model.load_state_dict(torch.load(model_path))

    args.batch_size = 250

    # eval
    filters = dataset.get_filters()
    if mode == "test":
        test_examples = dataset.get_examples("test")
        metrics = avg_both(*model.compute_metrics(test_examples, filters, args.sizes, batch_size=args.batch_size))
    elif mode == "valid":
        valid_examples = dataset.get_examples("valid")
        metrics = avg_both(*model.compute_metrics(valid_examples, filters, args.sizes, batch_size=args.batch_size))
    else:
        raise ValueError(f"The given mode {mode} does not exist!")
    del model

    return metrics, args


def format_metrics_latex(metrics, args):
    """Format metrics for logging."""
    if (hasattr(args, "subgraph_amount") and hasattr(args, "subgraph_size_range") and
            not isinstance(args.subgraph_size_range, str) and hasattr(args, "rho")):
        result = (f"{args.subgraph_amount}\t& "
                  f"${util.format_set(args.subgraph_size_range, first_char='(', last_char=')')}$\t& "
                  f"${args.rho if args.rho > 0 else '-'}$\t\t\t\t& ")
    else:
        result = "\\multicolumn{3}{c}{Baseline}\t\t\t& "

    for mode in ['average']:
        result += f"{metrics['MRR'][mode] * 100:.1f}\t& "
        # result += f"{metrics['MR'][mode]:.3f}\t\t\t\t& "
        result += f"{metrics['hits@[1,3,10]'][mode][0] * 100:.1f}\t& "
        result += f"{metrics['hits@[1,3,10]'][mode][1] * 100:.1f}\t& "
        result += f"{metrics['hits@[1,3,10]'][mode][2] * 100:.1f}\t& "
        result += f"{metrics['AMRI'][mode] * 100:.1f}\t"
        # result += f"{metrics['rank_deviation'][mode]:.5f}\t\t\t\\\\"

    return result


if __name__ == "__main__":
    args = argparse.Namespace()
    split = "test"
    # dataset = "FB15K-237"
    # model = "ComplEx"
    # args.model_dir = os.path.join("Results", "Rank_32")
    # args.model_dir = os.path.abspath(os.path.join("D:", "Results", "Rank_32"))
    args.model_dir = os.path.abspath(os.path.join("D:", "Results", "Rank_500"))
    # args.model_dir = os.path.abspath(os.path.join("D:", "Results", "Multi_32"))

    test_baseline = True
    test_ensemble = True
    filter_inclusive = False
    # filter_inclusive = True
    # filter_exclusive = False
    filter_exclusive = True

    filters = ["WN18RR", "Ent", "RotatE"]

    # args = parser.parse_args()
    # to_list = os.path.join("data", args.model_dir)
    to_list = args.model_dir
    result_directories = os.listdir(to_list)

    with (open(os.path.join(to_list, f"general_{split}_metrics.txt"), "w+") as out_file):
        for result_directory in result_directories:
            if not os.path.isdir(os.path.join(to_list, result_directory)):
                # skip if result_directory is no directory
                continue

            if any(filter_item in result_directory for filter_item in filters) and filter_inclusive:
                # skip if some filters are present in result_directory
                continue

            if not all(filter_item in result_directory for filter_item in filters) and filter_exclusive:
                # skip if not all filters are present in result_directory
                continue

            if "base" in result_directory.lower() and test_baseline:
                try:
                    init_str = (f"-------------------------------------------------------------------------"
                                f"------------------------------------------------------------------------\n"
                                f"{split.capitalize()} from result directory \"{result_directory}\"")
                    out_file.write(init_str)
                    print(init_str)

                    test_metrics, args_test = test(os.path.join(to_list, result_directory), mode=split)

                    out_file.write(format_metrics(test_metrics, split=split))
                    out_file.write("\n")

                    print(format_metrics(test_metrics, split=split))
                    print(format_metrics_latex(test_metrics, args_test), "\n")

                except Exception:
                    print(f"A error occurred during {split}ing of {result_directory}")
                    traceback.print_exc()

            elif test_ensemble:
                try:
                    init_str = (f"-------------------------------------------------------------------------"
                                f"------------------------------------------------------------------------\n"
                                f"{split.capitalize()} from result directory \"{result_directory}\"")
                    out_file.write(init_str)
                    print(init_str)

                    paths = argparse.Namespace(
                        result_directory=result_directory,
                        config_name=os.path.join("model_files", "config_unified.json"),
                        init_config_name=os.path.join("model_files", "config_init.json"),
                        model_name=os.path.join("model_files", "unified_model.pt"),
                        # dataset_path=os.path.join("data", dataset)
                    )
                    test_metrics, args_test = test(to_list, mode=split, paths=paths)

                    out_file.write(format_metrics(test_metrics, split=split))
                    out_file.write("\n")

                    print(format_metrics(test_metrics, split=split))
                    print(format_metrics_latex(test_metrics, args_test), "\n")

                except Exception:
                    print(f"A error occurred during {split}ing of {result_directory}")
                    traceback.print_exc()
        exit_str = (f"-------------------------------------------------------------------------"
                    f"------------------------------------------------------------------------")
        out_file.write(exit_str)
        print(exit_str)
