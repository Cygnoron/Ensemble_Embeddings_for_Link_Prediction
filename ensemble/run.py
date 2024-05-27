import argparse
import copy
import json
import logging
import os
import time

import torch
from torch import nn

import models as models
from datasets.kg_dataset import KGDataset
from ensemble import Constants, util_files, util, Attention_mechanism, score_combination
from ensemble.score_combination import evaluate_ensemble
from models import hyperbolic
from optimizers import regularizers as regularizers, KGOptimizer
from utils.train import count_params

try:
    # path on pc
    DATA_PATH = "data"
    os.listdir(DATA_PATH)
except FileNotFoundError:
    # path on laptop
    DATA_PATH = "C:\\Users\\timbr\\Masterarbeit\\Software\\Ensemble_Embedding_for_Link_Prediction\\data"


def train(info_directory, args):
    #   subgraph_amount, args, dataset="WN18RR", dataset_directory="data\\WN18RR", kge_models=None,
    #   regularizer="N3", reg=0, optimizer="Adagrad", max_epochs=50, patience=10, valid=3, rank=1000,
    #   batch_size=1000, neg_sample_size=50, dropout=0, init_size=1e-3, learning_rate=1e-1, gamma=0,
    #   bias="constant", dtype="double", double_neg=False, debug=False, multi_c=True,
    #   aggregation_method=Constants.MAX_SCORE_AGGREGATION, theta_calculation=Constants.REGULAR_THETA):
    """
    The train function is a wrapper for the general train function in the general run.py file.
    It allows us to run multiple models at once, and save them all in one folder with
    the same name as their respective model type (e.g., TransE). The train function
    takes all the arguments that are passed into the general train, but also takes an additional argument: multi_c.


    :param info_directory: Directory for all
    :param dataset: default="WN18RR", choices=["FB15K", "WN", "WN18RR", "WN18RR_sampled", "FB237", "YAGO3-10"], help="Knowledge Graph dataset"
    :param dataset_directory: The directory, where the dataset is located, default="data\\WN18RR"
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
    time_total_start = time.time()

    # set directories and ensure that they exist
    model_setup_config_dir = util_files.check_directory(f"{info_directory}\\model_setup_configs")
    model_file_dir = util_files.check_directory(f"{info_directory}\\model_files")

    # set files and ensure that they exist
    valid_loss_file_path = util_files.check_file(f"{info_directory}\\valid_loss.csv")
    train_loss_file_path = util_files.check_file(f"{info_directory}\\train_loss.csv")
    metrics_file_path = util_files.check_file(f"{info_directory}\\metrics_valid.csv")

    with (open(valid_loss_file_path, "w") as valid_loss_file, open(train_loss_file_path, "w") as train_loss_file,
          open(metrics_file_path, "w") as metrics_file):
        subgraphs_str = ""
        for sub_num in range(args.subgraph_amount):
            subgraphs_str += f";sub_{sub_num:03d}"
        valid_loss_file.write(f"epoch;average valid loss{subgraphs_str}\n")
        train_loss_file.write(f"epoch;average train loss{subgraphs_str}\n")
        metrics_file.write(f"epoch;mode;MR;MRR;Hits@1;Hits@3;Hits@10;AMRI;MR_deviation\n")

    logging.info(f"### Saving .json config files of models in: {model_setup_config_dir} ###")
    logging.info(f"### Saving .pt files of stored models in: {model_file_dir} ###")

    # set up dataset directory
    dataset_path = os.path.join(DATA_PATH, args.dataset)

    # get original dataset name
    dataset_general = util.get_dataset_name(args.dataset)
    # create model using original dataset and sizes, use returned embeddings in new models as initialization
    embedding_general_ent, embedding_general_rel, theta_general_ent, theta_general_rel, general_dataset_shape \
        = util.generate_general_embeddings(dataset_general, args)

    subgraph_embedding_mapping = util.assign_model_to_subgraph(args.kge_models, args)

    embedding_models = setup_models(subgraph_embedding_mapping, args, info_directory, embedding_general_ent,
                                    embedding_general_rel, theta_general_ent, theta_general_rel, general_dataset_shape,
                                    model_setup_config_dir, dataset_path)

    # --- Training ---
    logging.info(f"-/\tStarting training\t\\-")
    time_start_training_total = time.time()

    valid_args = argparse.Namespace(counter=0, best_mrr=None, best_epoch=None, epoch=0, valid=args.valid,
                                    patience=args.patience)

    for embedding_model in embedding_models:
        # --- Setting up training ---

        args = embedding_model['args']
        model = embedding_model['model']

        logging.info(f"-/\tStart training {embedding_model['subgraph']} with {args.model_name}\t\\-")

        # Get optimizer
        regularizer = (getattr(regularizers, args.regularizer)(args.reg))
        optim_method = (getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate))
        embedding_model['optimizer'] = KGOptimizer(model, regularizer, optim_method, args.batch_size,
                                                   args.neg_sample_size, bool(args.double_neg))

    train_losses = {}
    valid_losses = {}
    # Iterate over epochs
    for epoch in range(args.max_epochs):
        time_start_training_sub = time.time()

        # initialize dict for train losses
        train_losses[epoch] = {}
        # Iterate over models
        for embedding_model in embedding_models:
            args = embedding_model['args']
            model = embedding_model['model']

            logging.info(f"Training subgraph {embedding_model['subgraph']} in epoch {epoch} with model "
                         f"{args.model_name}")

            # Train step
            model.train()
            train_loss = embedding_model['optimizer'].epoch(embedding_model['train_examples'])
            logging.info(f"Subgraph {embedding_model['subgraph']} Training Epoch {epoch} | "
                         f"average train loss: {train_loss:.4f}")
            train_losses[epoch][embedding_model['args'].subgraph] = train_loss

            # debugging messages
            logging.debug(f"Entity size epoch {epoch}: {model.entity.weight.data.size()}")
            logging.debug(f"Relation size epoch {epoch}: {model.rel.weight.data.size()}")
            if model.theta_ent is not None:
                logging.debug(f"Theta_ent size epoch {epoch}: {model.theta_ent.weight.data.size()}")
            if model.theta_rel is not None:
                logging.debug(f"Theta_rel size epoch {epoch}: {model.theta_rel.weight.data.size()}")

            model.cuda()
            model.eval()

        # Calculate unified embedding
        cands_att_dict = Attention_mechanism.calculate_self_attention(embedding_models, args.theta_calculation)
        Attention_mechanism.calculate_and_apply_unified_embedding(embedding_general_ent,
                                                                  embedding_general_rel,
                                                                  embedding_models,
                                                                  cands_att_dict,
                                                                  args.theta_calculation)
        # print training losses to file
        util_files.print_loss_to_file(train_loss_file_path, epoch, train_losses[epoch])

        # Valid step
        valid_loss, valid_losses[epoch] = score_combination.calculate_valid_loss(embedding_models)

        # print validation losses
        util_files.print_loss_to_file(valid_loss_file_path, epoch, valid_losses[epoch])
        logging.debug(f"Validation Epoch {epoch} per subgraph | average valid losses: {valid_losses[epoch]}")
        logging.info(f"Validation Epoch {epoch} | average valid loss: {valid_loss:.4f}")

        if (epoch + 1) % valid_args.valid == 0:

            valid_metrics = evaluate_ensemble(embedding_models, args.aggregation_method, mode="valid",
                                              metrics_file_path=metrics_file_path, epoch=epoch)

            valid_mrr = valid_metrics["MRR"]
            if not valid_args.best_mrr or valid_mrr > valid_args.best_mrr:
                valid_args.best_mrr = valid_mrr
                valid_args.counter = 0
                valid_args.best_epoch = epoch
                logging.info(f"Saving models at epoch {epoch} in {model_file_dir}")
                for embedding_model in embedding_models:
                    args = embedding_model['args']
                    torch.save(embedding_model['model'].cpu().state_dict(), f"{model_file_dir}\\model_{args.subgraph}_"
                                                                            f"{args.model_name}.pt")
                    embedding_model['model'].cuda()

            else:
                valid_args.counter += 1
                if valid_args.counter == valid_args.patience:
                    logging.info("\t Early stopping")
                    break
                elif valid_args.counter == (valid_args.patience // 2):
                    pass
                    logging.info(f"\t Reducing learning rate")
                    for embedding_model in embedding_models:
                        embedding_model['optimizer'].reduce_lr()

        time_stop_training_sub = time.time()
        logging.info(f"-\\\tTraining and optimization of epoch {epoch} finished in "
                     f"{util.format_time(time_start_training_sub, time_stop_training_sub)}\t/-")

    # load or save best models after completed training
    util_files.save_load_trained_models(embedding_models, valid_args, model_file_dir)

    time_stop_training_total = time.time()

    logging.info(f"-\\\tSuccessfully finished training and optimizing all subgraphs in "
                 f"{util.format_time(time_start_training_total, time_stop_training_total)}\t/-")

    # --- Testing with aggregated scores ---

    evaluate_ensemble(embedding_models, aggregation_method=args.aggregation_method, metrics_file_path=metrics_file_path)

    time_total_end = time.time()
    logging.info(f"Finished ensemble training and testing in {util.format_time(time_total_start, time_total_end)}.")


def setup_models(subgraph_embedding_mapping, args, info_directory, embedding_general_ent, embedding_general_rel,
                 theta_general_ent, theta_general_rel, general_dataset_shape, model_setup_config_dir, dataset_path):
    # --- setting up embedding models ---
    logging.info("-/\tSetting up embedding models\t\\-")
    time_start_model_creation = time.time()

    # create dataset and model objects and save them to list of dictionaries
    embedding_models = []
    rank = args.rank

    # --- Training preparation ---
    for subgraph_num in list(subgraph_embedding_mapping.keys()):
        args_subgraph = copy.copy(args)
        args_subgraph.model = subgraph_embedding_mapping[subgraph_num]
        args_subgraph.model_name = args_subgraph.model
        args_subgraph.subgraph_num = subgraph_num
        subgraph = f"sub_{subgraph_num:03d}"

        logging.info(f"-/\tCreating new model from embedding method {args_subgraph.model} for subgraph {subgraph}\t\\-")
        args_subgraph.subgraph = subgraph

        # load data
        logging.info(f"Loading data for subgraph {subgraph}.")
        dataset = KGDataset(dataset_path + f"\\{subgraph}", args_subgraph.debug, info_directory=info_directory)
        train_examples = dataset.get_examples("train")
        valid_examples = dataset.get_examples("valid")
        test_examples = dataset.get_examples("test")
        filters = dataset.get_filters()
        # args_subgraph.sizes = dataset.get_shape()
        # set shape of dataset to that of the general
        #  -> calculation of size in KGDataset is with max id
        #  -> if max id is 500 and the highest sampled id was 499, shape would be 499
        args_subgraph.sizes = general_dataset_shape

        logging.debug(f"Sizes: old {dataset.get_shape()}\tnew {args_subgraph.sizes}")

        # create model
        model = getattr(models, args_subgraph.model)(args_subgraph)
        total = count_params(model)

        entity_set, relation_name_set = dataset.get_entities_relation_names(args_subgraph.sizes, double_relations=True)

        # set embeddings
        model.entity = nn.Embedding(embedding_general_ent.num_embeddings, embedding_general_ent.embedding_dim)
        # initialize with zeros and set present entities to one
        model.entity.weight.data = torch.zeros(embedding_general_ent.weight.data.size())
        model.entity.weight.data[entity_set] = torch.rand((len(entity_set), rank))
        logging.debug(f"Entity size: {model.entity.weight.data.size()}")

        if args_subgraph.model_name in hyperbolic.HYP_MODELS:
            model.rel = nn.Embedding(embedding_general_rel.num_embeddings, embedding_general_rel.embedding_dim)
            # initialize with zeros and set present relation names to one
            model.rel.weight.data = torch.rand(args_subgraph.sizes[0], args_subgraph.rank * 2)
            # model.rel.weight.data[relation_name_set] = torch.rand((len(relation_name_set), rank * 2))
        else:
            model.rel = nn.Embedding(embedding_general_rel.num_embeddings, embedding_general_rel.embedding_dim)
            # initialize with zeros and set present relation names to one
            model.rel.weight.data = torch.zeros(embedding_general_rel.weight.data.size())
            model.rel.weight.data[relation_name_set] = torch.rand((len(relation_name_set), rank))
        logging.debug(f"Relation size: {model.rel.weight.data.size()}")

        # set context vectors
        set_context_vectors(model, rank, args.theta_calculation, theta_general_ent, theta_general_rel, entity_set,
                            relation_name_set)

        logging.info(f"-\\\tTotal number of parameters: {total}\t/-")
        device = "cuda"
        model.to(device)

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
                 f"{util.format_time(time_start_model_creation, time_stop_model_creation)}\t/-")

    return embedding_models


def set_context_vectors(model, rank, theta_calculation, theta_general_ent, theta_general_rel, entity_set,
                        relation_name_set):
    if theta_calculation[0] == Constants.NO_THETA[0]:
        return

    theta_ent_set = None
    theta_rel_set = None

    if theta_calculation[0] == Constants.REGULAR_THETA[0]:
        theta_ent_set = entity_set
        theta_rel_set = relation_name_set

    elif theta_calculation[0] == Constants.REVERSED_THETA[0]:
        theta_ent_set = relation_name_set
        theta_rel_set = entity_set

    elif theta_calculation[0] == Constants.RELATION_THETA[0]:
        theta_ent_set = relation_name_set
        theta_rel_set = relation_name_set

    elif theta_calculation[0] == Constants.MULTIPLIED_THETA[0]:
        theta_ent_set = entity_set
        theta_rel_set = relation_name_set

    model.theta_ent = nn.Embedding(theta_general_ent.num_embeddings, theta_general_ent.embedding_dim)
    # initialize with zeros and set present entities to one
    model.theta_ent.weight.data = torch.zeros(theta_general_ent.weight.data.size())
    model.theta_ent.weight.data[theta_ent_set] = torch.rand((len(theta_ent_set), rank))
    logging.debug(f"Theta_ent size: {model.theta_ent.weight.data.size()}")

    model.theta_rel = nn.Embedding(theta_general_rel.num_embeddings, theta_general_rel.embedding_dim)
    # initialize with zeros and set present relation names to one
    model.theta_rel.weight.data = torch.zeros(theta_general_rel.weight.data.size())
    model.theta_rel.weight.data[theta_rel_set] = torch.rand((len(theta_rel_set), rank))
    logging.debug(f"Theta_rel size: {model.theta_rel.weight.data.size()}")
