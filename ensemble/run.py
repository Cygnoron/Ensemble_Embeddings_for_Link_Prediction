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

DATA_PATH = "data"


def train(info_directory, args):
    """
    The train function is a wrapper for the general train function in the general run.py file.
    It allows us to run multiple models at once, and save them all in one folder with
    the same name as their respective model type (e.g., TransE). The train function
    takes all the arguments that are passed into the general train, but also takes an additional argument: multi_c.


    """
    time_total_start = time.time()

    # set directories and ensure that they exist
    model_setup_config_dir = util_files.check_directory(os.path.join(info_directory, "model_setup_configs"))
    model_file_dir = util_files.check_directory(os.path.join(info_directory, "model_files"))
    test_valid_file_dir = util_files.check_directory(os.path.join(args.dataset_dir, "data"))

    # set files and ensure that they exist
    valid_loss_file_path = util_files.check_file(os.path.join(info_directory, "valid_loss.csv"))
    train_loss_file_path = util_files.check_file(os.path.join(info_directory, "train_loss.csv"))
    metrics_file_path = util_files.check_file(os.path.join(info_directory, "metrics_valid.csv"))

    with (open(valid_loss_file_path, "w") as valid_loss_file, open(train_loss_file_path, "w") as train_loss_file,
          open(metrics_file_path, "w") as metrics_file):
        subgraphs_str = ""
        for sub_num in range(args.subgraph_amount):
            subgraphs_str += f";sub_{sub_num:03d}"
        valid_loss_file.write(f"epoch;average valid loss{subgraphs_str}\n")
        train_loss_file.write(f"epoch;average train loss{subgraphs_str}\n")
        metrics_file.write(f"epoch;mode;metric_type;MR;MRR;Hits@1;Hits@3;Hits@10;AMRI;rank_deviation\n")

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

    embedding_models = setup_models(subgraph_embedding_mapping, args, test_valid_file_dir, embedding_general_ent,
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
                                                   args.neg_sample_size, bool(args.double_neg), args.no_progress_bar)

    train_losses = {}
    valid_losses = {}
    # Iterate over epochs
    for epoch in range(args.max_epochs):
        time_start_training_sub = time.time()

        # initialize dict for train losses
        train_losses[epoch] = {}
        # Iterate over models
        for index, embedding_model in enumerate(embedding_models):
            args = embedding_model['args']
            model = embedding_model['model']

            if args.model_dropout:
                train_losses[epoch][embedding_model['args'].subgraph] = "dropout"
                continue

            logging.info(
                f"Training subgraph {embedding_model['subgraph']} (ensemble step {index + 1}/{len(embedding_models)}) "
                f"in epoch {epoch} with model {args.model_name}")

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
        logging.debug(f"Validation Epoch {epoch} per subgraph | average valid losses:\n"
                      f"{util.format_dict(valid_losses[epoch])}")
        logging.info(f"Validation Epoch {epoch} | average valid loss: {valid_loss:.4f}")

        check_model_dropout(embedding_models, valid_losses[epoch])

        if (epoch + 1) % valid_args.valid == 0:

            valid_metrics = evaluate_ensemble(embedding_models, args.aggregation_method, mode="valid",
                                              metrics_file_path=metrics_file_path, epoch=epoch)

            valid_mrr = valid_metrics["MRR"]['average']
            if not valid_args.best_mrr or valid_mrr > valid_args.best_mrr:
                valid_args.best_mrr = valid_mrr
                valid_args.counter = 0
                valid_args.best_epoch = epoch
                logging.info(f"Saving models at epoch {epoch} in {model_file_dir}")
                for embedding_model in embedding_models:
                    args = embedding_model['args']
                    torch.save(embedding_model['model'].cpu().state_dict(),
                               os.path.join(model_file_dir, f"model_{args.subgraph}_{args.model_name}.pt"))
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


def setup_models(subgraph_embedding_mapping, args, test_valid_file_dir, embedding_general_ent, embedding_general_rel,
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
        dataset = KGDataset(os.path.join(dataset_path, subgraph), args_subgraph.debug,
                            test_valid_file_dir=test_valid_file_dir)
        train_examples = dataset.get_examples("train")
        valid_examples = dataset.get_examples("valid")
        test_examples = dataset.get_examples("test")
        filters = dataset.get_filters()

        dtype = torch.float
        if args.dtype == 'double':
            dtype = torch.double
        logging.debug(f"dtype: {dtype}")
        # args_subgraph.sizes = dataset.get_shape()
        # set shape of dataset to that of the general
        #  -> calculation of size in KGDataset is with max id
        #  -> if max id is 500 and the highest sampled id was 499, shape would be 499
        args_subgraph.sizes = general_dataset_shape

        logging.debug(f"Sizes: old {dataset.get_shape()}\tnew {args_subgraph.sizes}")

        args_subgraph.model_dropout = False

        # create model
        model = getattr(models, args_subgraph.model)(args_subgraph)
        total = count_params(model)

        entity_set, relation_name_set = dataset.get_entities_relation_names(args_subgraph.sizes, double_relations=True)

        # set embeddings
        model.entity = nn.Embedding(embedding_general_ent.num_embeddings, embedding_general_ent.embedding_dim,
                                    dtype=dtype)
        model.rel = nn.Embedding(embedding_general_rel.num_embeddings, embedding_general_rel.embedding_dim,
                                 dtype=dtype)
        if dtype == torch.double:
            # initialize with zeros and set present entities to random number
            model.entity.weight.data = torch.zeros(embedding_general_ent.weight.data.size()).double()
            model.entity.weight.data[entity_set] = torch.rand((len(entity_set), rank)).double()
            logging.debug(f"Entity size: {model.entity.weight.data.size()}")

            if args_subgraph.model_name in hyperbolic.HYP_MODELS:
                # initialize with zeros and set present relation names to one
                model.rel.weight.data = torch.rand(args_subgraph.sizes[0], args_subgraph.rank * 2).double()
                # model.rel.weight.data[relation_name_set] = torch.rand((len(relation_name_set), rank * 2))
            else:
                # initialize with zeros and set present relation names to one
                model.rel.weight.data = torch.zeros(embedding_general_rel.weight.data.size()).double()
                model.rel.weight.data[relation_name_set] = torch.rand((len(relation_name_set), rank)).double()
        else:
            # initialize with zeros and set present entities to random number
            model.entity.weight.data = torch.zeros(embedding_general_ent.weight.data.size()).float()
            model.entity.weight.data[entity_set] = torch.rand((len(entity_set), rank)).float()
            logging.debug(f"Entity size: {model.entity.weight.data.size()}")

            if args_subgraph.model_name in hyperbolic.HYP_MODELS:
                # initialize with zeros and set present relation names to one
                model.rel.weight.data = torch.rand(args_subgraph.sizes[0], args_subgraph.rank * 2).float()
                # model.rel.weight.data[relation_name_set] = torch.rand((len(relation_name_set), rank * 2))
            else:
                # initialize with zeros and set present relation names to one
                model.rel.weight.data = torch.zeros(embedding_general_rel.weight.data.size()).float()
                model.rel.weight.data[relation_name_set] = torch.rand((len(relation_name_set), rank)).float()

        logging.debug(f"Relation size: {model.rel.weight.data.size()}")

        # set context vectors
        set_context_vectors(model, rank, args.theta_calculation, theta_general_ent, theta_general_rel, entity_set,
                            relation_name_set)

        logging.info(f"-\\\tTotal number of parameters: {total}\t/-")

        # Wrap the model in DataParallel to use multiple GPUs
        # if torch.cuda.device_count() > 1:
        #     logging.info(f"Using {torch.cuda.device_count()} GPUs.")
        #     model = nn.DataParallel(model)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        embedding_model = {"dataset": dataset, "model": model, "train_examples": train_examples,
                           "valid_examples": valid_examples, "test_examples": test_examples,
                           "filters": filters, "subgraph": subgraph, "args": args_subgraph, "data_type": dtype,
                           "first_valid_loss": None, "size_rhs": None, "size_lhs": None}
        embedding_models.append(embedding_model)

        # save config
        with (open(os.path.join(model_setup_config_dir, f"config_{subgraph}_{embedding_model['args'].model_name}.json"),
                   "w") as json_file):
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


def check_model_dropout(embedding_models, valid_losses):
    for embedding_model in embedding_models:
        args = embedding_model['args']
        subgraph = embedding_model['subgraph']

        if valid_losses[subgraph] == "dropout":
            continue

        if embedding_model['first_valid_loss'] is None:
            # if "1" in subgraph:
            #     embedding_model['first_valid_loss'] = float(valid_losses[subgraph] / 20)
            # else:
            embedding_model['first_valid_loss'] = float(valid_losses[subgraph])

        if (valid_losses[subgraph] >=
                embedding_model['first_valid_loss'] * args.model_dropout_factor):
            # exclude model from ensemble if valid loss was to high, indicating diverging behaviour
            logging.info(f"Excluding model {subgraph} from ensemble since {valid_losses[subgraph]} is larger than "
                         f"{args.model_dropout_factor} times the first validation loss "
                         f"{embedding_model['first_valid_loss']}")

            # "hard dropout" -> completly exclude model if it diverged once
            args.model_dropout = True
