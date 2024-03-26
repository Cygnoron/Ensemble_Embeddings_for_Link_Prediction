"""Train Knowledge Graph embeddings for link prediction."""

import argparse
import copy
import json
import logging
import os
import time

import numpy as np
import torch
import torch.optim

import Attention_mechanism
import Constants
import KGEmb.models as models
import KGEmb.optimizers.regularizers as regularizers
import util
import util_files
from KGEmb.datasets.kg_dataset import KGDataset
from KGEmb.models import all_models
from KGEmb.optimizers.kg_optimizer import KGOptimizer
from KGEmb.utils.train import get_savedir, avg_both, format_metrics, count_params

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

try:
    DATA_PATH = "D:\\OneDrive - bwedu\\MASTER\\Masterarbeit\\Software\\Ensemble_Embedding\\KGEmb\\data"
    os.listdir(DATA_PATH)
except:
    DATA_PATH = "C:\\Users\\timbr\\OneDrive - bwedu\\MASTER\\Masterarbeit\\Software\\Ensemble_Embedding\\KGEmb\\data"

parser = argparse.ArgumentParser(
    description="Knowledge Graph Embedding"
)
parser.add_argument(
    "--dataset", default="WN18RR", help="Knowledge Graph dataset"
    # choices=["FB15K", "WN", "WN18RR", "FB237", "YAGO3-10"] (REMOVED FPR FLEXIBILITY IN DATASET NAME)
)
parser.add_argument(
    "--model", default="RotE", choices=all_models, help="Knowledge Graph embedding model"
)
parser.add_argument(
    "--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer"
)
parser.add_argument(
    "--reg", default=0, type=float, help="Regularization weight"
)
parser.add_argument(
    "--optimizer", choices=["Adagrad", "Adam", "SparseAdam"], default="Adagrad",
    help="Optimizer"
)
parser.add_argument(
    "--max_epochs", default=50, type=int, help="Maximum number of epochs to train for"
)
parser.add_argument(
    "--patience", default=10, type=int, help="Number of epochs before early stopping"
)
parser.add_argument(
    "--valid", default=3, type=float, help="Number of epochs before validation"
)
parser.add_argument(
    "--rank", default=1000, type=int, help="Embedding dimension"
)
parser.add_argument(
    "--batch_size", default=1000, type=int, help="Batch size"
)
parser.add_argument(
    "--neg_sample_size", default=50, type=int, help="Negative sample size, -1 to not use negative sampling"
)
parser.add_argument(
    "--dropout", default=0, type=float, help="Dropout rate"
)
parser.add_argument(
    "--init_size", default=1e-3, type=float, help="Initial embeddings' scale"
)
parser.add_argument(
    "--learning_rate", default=1e-1, type=float, help="Learning rate"
)
parser.add_argument(
    "--gamma", default=0, type=float, help="Margin for distance-based losses"
)
parser.add_argument(
    "--bias", default="constant", type=str, choices=["constant", "learn", "none"], help="Bias type (none for no bias)"
)
parser.add_argument(
    "--dtype", default="double", type=str, choices=["single", "double"], help="Machine precision"
)
parser.add_argument(
    "--double_neg", action="store_true",
    help="Whether to negative sample both head and tail entities"
)
parser.add_argument(
    "--debug", action="store_true",
    help="Only use 1000 examples for debugging"
)
parser.add_argument(
    "--multi_c", action="store_true", help="Multiple curvatures per relation"
)


def train(args):
    save_dir = get_savedir(args.model, args.dataset)

    # file logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "train.log")
    )

    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(save_dir))

    # create dataset
    dataset_path = os.path.join(DATA_PATH, args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()

    # load data
    logging.info("\t " + str(dataset.get_shape()))
    train_examples = dataset.get_examples("train")
    valid_examples = dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    # save config
    with open(os.path.join(save_dir, "config.json"), "w") as fjson:
        json.dump(vars(args), fjson)

    # create model
    model = getattr(models, args.model)(args)
    total = count_params(model)
    logging.info("Total number of parameters {}".format(total))
    device = "cuda"
    model.to(device)

    # get optimizer
    regularizer = getattr(regularizers, args.regularizer)(args.reg)
    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    optimizer = KGOptimizer(model, regularizer, optim_method, args.batch_size, args.neg_sample_size,
                            bool(args.double_neg))
    counter = 0
    best_mrr = None
    best_epoch = None
    logging.info("\t Start training")
    for step in range(args.max_epochs):

        # TODO Training
        # Train step
        model.train()

        # TODO Training loss metrics
        train_loss = optimizer.epoch(train_examples)
        logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))

        # Valid step
        model.eval()
        valid_loss = optimizer.calculate_valid_loss(valid_examples)
        logging.info("\t Epoch {} | average valid loss: {:.4f}".format(step, valid_loss))

        # TODO Combine
        if (step + 1) % args.valid == 0:
            valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
            logging.info(format_metrics(valid_metrics, split="valid"))

            valid_mrr = valid_metrics["MRR"]
            if not best_mrr or valid_mrr > best_mrr:
                best_mrr = valid_mrr
                counter = 0
                best_epoch = step
                logging.info("\t Saving model at epoch {} in {}".format(step, save_dir))
                torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
                model.cuda()
            else:
                counter += 1
                if counter == args.patience:
                    logging.info("\t Early stopping")
                    break
                elif counter == args.patience // 2:
                    pass
                    # logging.info("\t Reducing learning rate")
                    # optimizer.reduce_lr()

    logging.info("\t Optimization finished")
    if not best_mrr:
        torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
    else:
        logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
        model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
    model.cuda()
    model.eval()

    # Validation metrics
    valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
    logging.info(format_metrics(valid_metrics, split="valid"))

    # Test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
    logging.info(format_metrics(test_metrics, split="test"))


def own_train(info_directory, dataset="WN18RR", dataset_directory="KGEmb\\data\\WN18RR", kge_models=None,
              regularizer="N3", reg=0, optimizer="Adagrad", max_epochs=50, patience=10, valid=3, rank=1000,
              batch_size=1000, neg_sample_size=50, dropout=0, init_size=1e-3, learning_rate=1e-1, gamma=0,
              bias="constant", dtype="double", double_neg=False, debug=False, multi_c=True):
    """
    The own_train function is a wrapper for the train function in the main.py file.
    It allows us to run multiple models at once, and save them all in one folder with
    the same name as their respective model type (e.g., TransH). The own_train function
    takes all the arguments that are passed into train, but also takes an additional argument: multi_c.

    :param dataset: default="WN18RR", choices=["FB15K", "WN", "WN18RR", "WN18RR_sampled", "FB237", "YAGO3-10"], help="Knowledge Graph dataset"
    :param dataset_directory: The directory, where the dataset is located, default="KGEmb\\data\\WN18RR"
    :param kge_models: default="RotE", all allowed KGE models
    :param regularizer: choices=["N3", "F2"], default="N3", help="Regularizer"
    :param reg: default=0, type=float, help="Regularization weight"
    :param optimizer: choices=["Adagrad", "Adam", "SparseAdam"], default="Adagrad", help="Optimizer"
    :param max_epochs: default=50, type=int, help="Maximum number of epochs to train for"
    :param patience: default=10, type=int, help="Number of epochs before early stopping"
    :param valid: default=3, type=float, help="Number of epochs before validation"
    :param rank: default=1000, type=int, help="Embedding dimension"
    :param batch_size: default=1000, type=int, help="Batch size"
    :param neg_sample_size: default=50, type=int, help="Negative sample size, -1 to not use negative sampling"
    :param dropout: default=0, type=float, help="Dropout rate"
    :param init_size: default=1e-3, type=float, help="Initial embeddings' scale"
    :param learning_rate: default=1e-1, type=float, help="Learning rate"
    :param gamma: default=0, type=float, help="Margin for distance-based losses"
    :param bias: default="constant", type=str, choices=["constant", "learn", "none"], help="Bias type (none for no bias)"
    :param dtype: default="double", type=str, choices=["single", "double"], help="Machine precision"
    :param double_neg: action="store_true", help="Whether to negative sample both head and tail entities"
    :param debug: action="store_true", help="Only use 1000 examples for debugging"
    :param multi_c: action="store_true", help="Multiple curvatures per relation", needed for AttH
    """

    if kge_models is None:
        kge_models = {Constants.ROTAT_E: []}

    args = argparse.Namespace(dataset=dataset, dataset_directory=dataset_directory, regularizer=regularizer, reg=reg,
                              optimizer=optimizer, max_epochs=max_epochs, patience=patience, valid=valid, rank=rank,
                              batch_size=batch_size, neg_sample_size=neg_sample_size, dropout=dropout,
                              init_size=init_size, learning_rate=learning_rate, gamma=gamma, bias=bias, dtype=dtype,
                              double_neg=double_neg, debug=debug, multi_c=multi_c)

    # set directories and ensure that they exist
    model_setup_config_dir = util_files.check_directory(f"{info_directory}\\model_setup_configs")
    model_file_dir = util_files.check_directory(f"{info_directory}\\model_files")
    # set files and ensure that they exist
    s_e_mapping_dir = util_files.check_file(f"{info_directory}\\subgraph_embedding_mapping.json")
    e_s_mapping_dir = util_files.check_file(f"{info_directory}\\embedding_subgraph_mapping.json")

    logging.info(f"### Saving .json config files of models in: {model_setup_config_dir} ###")
    logging.info(f"### Saving .pt files of stored models in: {model_file_dir} ###")

    # create dataset object
    dataset_path = os.path.join(DATA_PATH, args.dataset)

    # --- setting up embedding models ---
    logging.info("-/\tSetting up embedding models\t\\-")
    time_start_model_creation = time.time()

    subgraph_embedding_mapping = util.assign_model_to_subgraph(kge_models, args)

    # create dataset and model objects and save them to list of dictionaries
    embedding_models = []
    models_to_load = []

    with (open(s_e_mapping_dir, 'r') as s_e_mapping_file):
        # Load subgraph_embedding_mapping from file and convert to dictionary
        try:
            s_e_mapping = dict(json.loads(s_e_mapping_file.read()))
        except:
            logging.info("No already trained models found.")
            s_e_mapping = False
            s_e_mapping_file.close()

    for subgraph_num in list(subgraph_embedding_mapping.keys()):
        args_subgraph = copy.copy(args)
        args_subgraph.model = subgraph_embedding_mapping[subgraph_num]
        args_subgraph.model_name = args_subgraph.model
        subgraph = f"sub_{subgraph_num:03d}"

        logging.info(f"-/\tCreating new model from embedding method {args_subgraph.model} for subgraph {subgraph}\t\\-")
        args_subgraph.subgraph = subgraph

        # load data
        logging.info(f"Loading data fom subgraph {subgraph}.")
        dataset = KGDataset(dataset_path + f"\\{subgraph}", args_subgraph.debug)
        train_examples = dataset.get_examples("train")
        valid_examples = dataset.get_examples("valid")
        test_examples = dataset.get_examples("test")
        filters = dataset.get_filters()
        args_subgraph.sizes = dataset.get_shape()

        # create model
        model = getattr(models, args_subgraph.model)(args_subgraph)
        total = count_params(model)
        logging.info(f"-\\\tTotal number of parameters: {total}\t/-")
        device = "cuda"
        model.to(device)

        # Skip already trained embedding models
        if s_e_mapping:
            if subgraph_embedding_mapping[subgraph_num] in s_e_mapping[str(subgraph_num)]:
                logging.info(f"Subgraph {subgraph} was already trained with embedding method "
                             f"{args_subgraph.model_name}.")

                embedding_model = {"dataset": dataset, "model": model, "subgraph": subgraph,
                                   "args": args_subgraph, "load_from_file": True}
                embedding_models.append(embedding_model)
                models_to_load.append(embedding_model)
                continue

        embedding_model = {"dataset": dataset, "model": model, "train_examples": train_examples,
                           "valid_examples": valid_examples, "test_examples": test_examples,
                           "filters": filters, "subgraph": subgraph, "args": args_subgraph,
                           "load_from_file": False}
        embedding_models.append(embedding_model)

        # save config
        with (open(f"{model_setup_config_dir}\\config_{subgraph}_{embedding_model['args'].model_name}.json", "w")
              as json_file):
            json.dump(vars(args_subgraph), json_file)

    time_stop_model_creation = time.time()
    logging.info(f"-\\\tSuccessfully created all models in "
                 f"{round(time_stop_model_creation - time_start_model_creation, 3)} seconds / "
                 f"{round((time_stop_model_creation - time_start_model_creation) / 60, 3)} minutes\t/-")

    # --- Training ---
    logging.info(f"-/\tStarting training\t\\-")
    time_start_training_total = time.time()

    for embedding_model in embedding_models:
        # skip training for already trained models
        if embedding_model['load_from_file']:
            continue

        # --- Setting up training ---
        logging.info(f"-/\tStart training {embedding_model['subgraph']} with {embedding_model['args'].model_name}\t\\-")
        subgraph = embedding_model['subgraph']

        time_start_training_sub = time.time()

        model = embedding_model['model']
        valid_examples = embedding_model['dataset'].get_examples("valid")
        test_examples = embedding_model['dataset'].get_examples("test")
        filters = embedding_model['dataset'].get_filters()

        # unpack dictionary from json file and write into args
        with (open(f"{model_setup_config_dir}\\config_{subgraph}_{embedding_model['args'].model_name}.json", "r")
              as json_file):
            args = argparse.Namespace(**json.loads(json_file.read()))

        # get optimizer
        regularizer = getattr(regularizers, embedding_model['args'].regularizer)(embedding_model['args'].reg)
        optim_method = (getattr(torch.optim, embedding_model['args'].optimizer)
                        (model.parameters(), lr=embedding_model['args'].learning_rate))
        optimizer = KGOptimizer(model, regularizer, optim_method, embedding_model['args'].batch_size,
                                embedding_model['args'].neg_sample_size, bool(embedding_model['args'].double_neg))
        counter = 0
        best_mrr = None
        best_epoch = None
        for step in range(embedding_model['args'].max_epochs):

            # Train step
            embedding_model['model'].train()
            train_loss = optimizer.epoch(embedding_model['train_examples'])
            logging.info(f"Subgraph {embedding_model['subgraph']} Training Epoch {step} | "
                         f"average train loss: {train_loss:.4f}")

            # TODO fully implement validation
            # Valid step
            embedding_model['model'].eval()
            valid_loss = optimizer.calculate_valid_loss(valid_examples)
            logging.info(f"Subgraph {embedding_model['subgraph']} Validation Epoch {step} | "
                         f"average train loss: {valid_loss:.4f}")

            if (step + 1) % embedding_model['args'].valid == 0:
                # TODO fully implement validation
                valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
                logging.info(format_metrics(valid_metrics, split="valid"))

                valid_mrr = valid_metrics["MRR"]
                if not best_mrr or valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    counter = 0
                    best_epoch = step
                logging.info(f"Saving model at epoch {step} in {info_directory}")
                torch.save(model.cpu().state_dict(),
                           f"{model_file_dir}\\model_{embedding_model['args'].subgraph}_{embedding_model['args'].model_name}.pt")
                model.cuda()

                # TODO fully implement validation
            else:
                counter += 1
                if counter == args.patience:
                    logging.info("\t Early stopping")
                    break
                elif counter == args.patience // 2:
                    pass
            logging.info("\t Reducing learning rate")
            optimizer.reduce_lr()

        time_stop_training_sub = time.time()

        if not best_mrr:
            logging.info(f"Saving new model saved at epoch {best_epoch}\n{model_file_dir}\\"
                         f"model_{embedding_model['args'].subgraph}_{embedding_model['args'].model_name}.pt")
            torch.save(model.cpu().state_dict(), f"{model_file_dir}\\model_{embedding_model['args'].subgraph}_"
                                                 f"{embedding_model['args'].model_name}.pt")
        else:
            logging.info(f"Loading best model saved at epoch {best_epoch}")
            model.load_state_dict(torch.load(f"{model_file_dir}\\model_{embedding_model['args'].subgraph}_"
                                             f"{embedding_model['args'].model_name}.pt"))

        logging.info(f"-\\\tTraining and optimization of model {embedding_model['args'].model_name} for subgraph "
                     f"{embedding_model['args'].subgraph} finished in "
                     f"{round(time_stop_training_sub - time_start_training_sub, 3)} seconds / "
                     f"{round((time_stop_training_sub - time_start_training_sub) / 60, 3)} minutes\t/-")
        model.cuda()
        model.eval()

    time_stop_training_total = time.time()

    logging.info(f"-\\\tSuccessfully finished training and optimizing all subgraphs in "
                 f"{round(time_stop_training_total - time_start_training_total, 3)} seconds / "
                 f"{round((time_stop_training_total - time_start_training_total) / 60, 3)} minutes\t/-")

    # Update mappings for already trained embeddings, in order to skip already trained subgraphs
    with (open(s_e_mapping_dir, 'r+') as s_e_mapping_file,
          open(e_s_mapping_dir, 'r+') as e_s_mapping_file):

        # Load subgraph_embedding_mapping from file and convert to dictionary
        try:
            s_e_mapping = dict(json.loads(s_e_mapping_file.read()))
        except:
            s_e_mapping = {}

        # Update s_e_mapping with subgraph_embedding_mapping
        for sub_num in subgraph_embedding_mapping:
            if str(sub_num) not in list(s_e_mapping.keys()):
                s_e_mapping[str(sub_num)] = []
            if subgraph_embedding_mapping[sub_num] not in s_e_mapping[str(sub_num)]:
                s_e_mapping[str(sub_num)].append(subgraph_embedding_mapping[sub_num])
                # TODO also change to numpy array?

        # Write updated s_e_mapping back to file
        s_e_mapping_file.seek(0)
        json.dump(s_e_mapping, s_e_mapping_file)
        logging.info(f"Updated mapping from subgraphs to embedding methods in {s_e_mapping_file.name}.")

        # Get the inverse mapping of subgraph_embedding_mapping
        inverse_mapping = util.inverse_dict(subgraph_embedding_mapping)

        try:
            e_s_mapping = dict(json.loads(e_s_mapping_file.read()))
        except:
            e_s_mapping = {}

        # Update e_s_mapping with inverse_mapping
        for sub_num in inverse_mapping:
            if str(sub_num) not in list(e_s_mapping.keys()):
                e_s_mapping[str(sub_num)] = []
            if inverse_mapping[sub_num] not in e_s_mapping[str(sub_num)]:
                e_s_mapping[str(sub_num)] = list(
                    np.unique(np.append(e_s_mapping[str(sub_num)], (inverse_mapping[sub_num]))).astype(int).astype(str))

        # Write updated e_s_mapping back to file
        e_s_mapping_file.seek(0)
        json.dump(e_s_mapping, e_s_mapping_file)
        logging.info(f"Updated mapping from embedding methods to subgraphs in {e_s_mapping_file.name}.")

    # TODO why do models load with None fields?
    #  how to initialize context_vec, so it's saved in model.pt?

    if len(models_to_load) > 0:
        logging.info(f"-/\tLoading {len(models_to_load)} models from storage.\t\\-")

        for embedding_model in models_to_load:
            args_subgraph = embedding_model['args']
            logging.info(f"Loading model {args_subgraph.model_name} for subgraph {args_subgraph.subgraph}")

            embedding_model["model"].load_state_dict(torch.load(f"{model_file_dir}\\model_{args_subgraph.subgraph}_"
                                                                f"{args_subgraph.model_name}.pt"))

        logging.info(f"-\\\tSuccessfully loaded {len(models_to_load)} models from storage.\t/-")

    # --- Create unified embedding ---

    # Calculate attention values
    for embedding_model in embedding_models:
        Attention_mechanism.calculate_self_attention(embedding_model)

    # Combine embeddings using the previously calculated attention values

    unified_embedding_ent, unified_embedding_rel = Attention_mechanism.calculate_unified_embedding(embedding_models)

    # TODO fix dimension error with AttH

    logging.info("Loading unified embeddings into all models")
    for embedding_model in embedding_models:
        model = embedding_model["model"]
        model.entity.weight.data = unified_embedding_ent
        model.rel.weight.data = unified_embedding_rel

    # --- Testing with aggregated scores ---

    # TODO fully implement validation and testing metrics
    for embedding_model in embedding_models:
        model = embedding_model["model"]
        valid_examples = embedding_model["valid_examples"]
        test_examples = embedding_model["test_examples"]
        filters = embedding_model["filters"]

        # Validation metrics
        valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
        logging.info(format_metrics(valid_metrics, split="valid"))

        # Test metrics
        test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
        logging.info(format_metrics(test_metrics, split="test"))


if __name__ == "__main__":
    train(parser.parse_args())
