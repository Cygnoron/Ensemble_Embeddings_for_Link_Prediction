import argparse
import json
import logging
import os
import time

import torch
from torch import nn

import models as models
from datasets.kg_dataset import KGDataset
from ensemble import Constants, util_files, util
from optimizers import regularizers as regularizers, KGOptimizer
from utils.train import count_params, avg_both, format_metrics

DATA_PATH = "data"


def train(info_directory, args):
    """
    The train function is a wrapper for the general train function in the general run.py file.
    It allows us to run multiple embedding_models at once, and save them all in one folder with
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
        # for sub_num in range(args.subgraph_amount):
        #     subgraphs_str += f";sub_{sub_num:03d}"
        valid_loss_file.write(f"epoch;valid loss{subgraphs_str}\n")
        train_loss_file.write(f"epoch;train loss{subgraphs_str}\n")
        metrics_file.write(f"epoch;mode;metric_type;MR;MRR;Hits@1;Hits@3;Hits@10;AMRI;rank_deviation\n")

    logging.info(f"### Saving .json config files of embedding_models in: {model_setup_config_dir} ###")
    logging.info(f"### Saving .pt files of stored embedding_models in: {model_file_dir} ###")

    # set up dataset directory
    dataset_path = os.path.join(DATA_PATH, args.dataset)

    # get original dataset name
    dataset_general = util.get_dataset_name(args.dataset)

    dataset = KGDataset(os.path.join("data", dataset_general), args.debug, test_valid_file_dir=test_valid_file_dir)
    train_examples = dataset.get_examples("train")
    valid_examples = dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()
    general_dataset_shape = dataset.get_shape()
    # create model using original dataset and sizes, use returned embeddings in new embedding_models as initialization
    # embedding_general_ent, embedding_general_rel, theta_general_ent, theta_general_rel, general_dataset_shape \
    #     = util.generate_general_embeddings(dataset_general, util.get_args(args, "general_args"))

    subgraph_embedding_mapping = util.assign_model_to_subgraph(args.kge_models, args)

    unified_model, args_unified = setup_models(subgraph_embedding_mapping, args, test_valid_file_dir,
                                               general_dataset_shape, model_setup_config_dir, dataset_path)
    cands_att_dict = None
    run_diverged = False

    # --- Training ---
    logging.info(f"-/\tStarting training\t\\-")
    time_start_training_total = time.time()

    valid_args = argparse.Namespace(counter=0, best_mrr=None, best_epoch=None, epoch=0, valid=args.valid,
                                    patience=args.patience)

    # TODO rework training loop
    # --- Setting up training ---

    # Get optimizer
    regularizer = (getattr(regularizers, args_unified.regularizer)(args_unified.reg))
    optim_method = (getattr(torch.optim, args_unified.optimizer)(unified_model.parameters(),
                                                                 lr=args_unified.learning_rate))
    optimizer = KGOptimizer(unified_model, regularizer, optim_method, args_unified.batch_size,
                            args_unified.neg_sample_size, bool(args_unified.double_neg), args_unified.no_progress_bar)

    # Iterate over epochs
    for epoch in range(args.max_epochs):
        time_start_training_sub = time.time()
        print("Epoch:", epoch)
        # logging.info(f"Training subgraph {embedding_model['subgraph']} (ensemble step {index + 1}/"
        #              f"{len(embedding_models)}) in epoch {epoch} with model {args.model_name}")

        # Train step
        unified_model.train()
        train_loss = optimizer.epoch(train_examples)
        logging.info(f"Training Epoch {epoch} | average train loss: {train_loss:.4f}")

        # debugging messages
        logging.debug(f"Entity size epoch {epoch}: {unified_model.entity.weight.data.size()}")
        logging.debug(f"Relation size epoch {epoch}: {unified_model.rel.weight.data.size()}")
        if unified_model.theta_ent is not None:
            logging.debug(f"Theta_ent size epoch {epoch}: {unified_model.theta_ent.weight.data.size()}")
        if unified_model.theta_rel is not None:
            logging.debug(f"Theta_rel size epoch {epoch}: {unified_model.theta_rel.weight.data.size()}")

        unified_model.cuda()
        unified_model.eval()

        # Calculate unified embedding
        # cands_att_dict = Attention_mechanism.calculate_self_attention(embedding_models, args.theta_calculation)
        # cands_att_dict = Attention_mechanism.calculate_and_apply_unified_embedding(embedding_general_ent,
        #                                                                            embedding_general_rel,
        #                                                                            embedding_models,
        #                                                                            cands_att_dict,
        #                                                                            args.theta_calculation)
        # # print training losses to file
        util_files.print_loss_to_file(train_loss_file_path, epoch, train_loss)

        # Valid step
        # valid_loss = score_combination.calculate_valid_loss(embedding_models)
        valid_loss = optimizer.calculate_valid_loss(valid_examples)

        # print validation losses
        # run_diverged = check_model_dropout(unified_model.embedding_models, unified_model.active_models)
        util_files.print_loss_to_file(valid_loss_file_path, epoch, valid_loss)

        logging.info(f"Validation Epoch {epoch} | average valid loss: {valid_loss:.4f}")

        if run_diverged:
            logging.critical(f"The run diverged and is now stopped. "
                             f"Please try a other combination of embedding methods or a higher model dropout factor.")
            break

        if (epoch + 1) % valid_args.valid == 0:
            valid_metrics = avg_both(*unified_model.compute_metrics(valid_examples, filters, args.sizes), epoch=epoch,
                                     active_subgraphs=unified_model.active_models)
            logging.info(format_metrics(valid_metrics, split="valid"))
            # valid_metrics = evaluate_ensemble(embedding_models, args.aggregation_method, mode="valid",
            #                                   metrics_file_path=metrics_file_path, epoch=epoch,
            #                                   attention={'ent': cands_att_dict['att_weights_ent'],
            #                                              'rel': cands_att_dict['att_weights_rel']})

            valid_mrr = valid_metrics["MRR"]['average']
            if not valid_args.best_mrr or valid_mrr > valid_args.best_mrr:
                valid_args.best_mrr = valid_mrr
                valid_args.counter = 0
                valid_args.best_epoch = epoch
                logging.info(f"Saving embedding_models at epoch {epoch} in {model_file_dir}")

                # torch.save(cands_att_dict['att_weights_ent'], os.path.join(model_file_dir, "attention_ent.pt"))
                # torch.save(cands_att_dict['att_weights_rel'], os.path.join(model_file_dir, "attention_rel.pt"))

                torch.save(unified_model.cpu().state_dict(), os.path.join(model_file_dir, f"model_unified.pt"))

            else:
                valid_args.counter += 1
                if valid_args.counter == valid_args.patience:
                    logging.info("\t Early stopping")
                    break
                elif valid_args.counter == (valid_args.patience // 2):
                    pass
                    logging.info(f"\t Reducing learning rate")
                    optimizer.reduce_lr()

        time_stop_training_sub = time.time()
        logging.info(f"-\\\tTraining and optimization of epoch {epoch} finished in "
                     f"{util.format_time(time_start_training_sub, time_stop_training_sub)}\t/-")

    if run_diverged:
        return "The run diverged and was aborted."

    # load or save best embedding_models after completed training
    # cands_att_dict = util_files.save_load_trained_models(embedding_models, valid_args, model_file_dir, cands_att_dict)

    time_stop_training_total = time.time()

    logging.info(f"-\\\tSuccessfully finished training and optimizing all subgraphs in "
                 f"{util.format_time(time_start_training_total, time_stop_training_total)}\t/-")

    # --- Testing with aggregated scores ---
    valid_metrics = avg_both(*unified_model.compute_metrics(valid_examples, filters, args.sizes),
                             epoch=args.max_epochs + 10, active_subgraphs=unified_model.active_models)
    logging.info(format_metrics(valid_metrics, split="valid"))

    valid_metrics = avg_both(*unified_model.compute_metrics(test_examples, filters, args.sizes),
                             epoch=args.max_epochs + 20, active_subgraphs=unified_model.active_models)
    logging.info(format_metrics(valid_metrics, split="test"))

    time_total_end = time.time()
    logging.info(f"Finished ensemble training and testing in {util.format_time(time_total_start, time_total_end)}.")


def setup_models(subgraph_embedding_mapping, args, test_valid_file_dir, general_dataset_shape, model_setup_config_dir,
                 dataset_path):
    # --- setting up embedding embedding_models ---
    logging.info("-/\tSetting up embedding embedding_models\t\\-")
    time_start_model_creation = time.time()

    # create dataset and model objects and save them to list of dictionaries
    embedding_models = []
    args.sizes = general_dataset_shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Training preparation ---
    for subgraph_num in list(subgraph_embedding_mapping.keys()):
        model = subgraph_embedding_mapping[subgraph_num]
        args_subgraph = util.get_args(args, model)

        args_subgraph.model = model
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

        args_subgraph.model_dropout = False

        args_subgraph.entities, args_subgraph.relation_names = dataset.get_entities_relation_names(
            args_subgraph.sizes, double_relations=True)

        # create model
        model = getattr(models, args_subgraph.model)(args_subgraph)
        total = count_params(model)

        # Get optimizer
        regularizer = (getattr(regularizers, args_subgraph.regularizer)(args_subgraph.reg))
        optim_method = (getattr(torch.optim, args_subgraph.optimizer)(model.parameters(),
                                                                      lr=args_subgraph.learning_rate))
        optimizer = KGOptimizer(model, regularizer, optim_method, args_subgraph.batch_size,
                                args_subgraph.neg_sample_size, bool(args_subgraph.double_neg),
                                args_subgraph.no_progress_bar)

        # set embeddings
        # initialize with zeros and set present entities to random number
        # model.entity.weight.data = torch.zeros(args.sizes[0], rank).to(dtype)
        # model.entity.weight.data[entity_set] = torch.rand((len(entity_set), rank)).to(dtype)
        # logging.debug(f"Entity size: {model.entity.weight.data.size()}")
        #
        # if args_subgraph.model_name in hyperbolic.HYP_MODELS:
        #     # initialize with zeros and set present relation names to one
        #     model.rel.weight.data = torch.rand(args_subgraph.sizes[1], args_subgraph.rank * 2).to(dtype)
        #     # model.rel.weight.data[relation_name_set] = torch.rand((len(relation_name_set), rank * 2))
        # else:
        #     # initialize with zeros and set present relation names to one
        #     model.rel.weight.data = torch.zeros(args.sizes[1], rank).to(dtype)
        #     model.rel.weight.data[relation_name_set] = torch.rand((len(relation_name_set), rank)).to(dtype)

        logging.debug(f"Relation size: {model.rel.weight.data.size()}")

        logging.info(f"-\\\tTotal number of parameters: {total}\t/-")

        model.to(device)

        embedding_model = {"dataset": dataset, "model": model, "train_examples": train_examples,
                           "valid_examples": valid_examples, "test_examples": test_examples, "optimizer": optimizer,
                           "filters": filters, "subgraph": subgraph, "args": args_subgraph, "data_type": dtype,
                           "first_valid_loss": None, "size_rhs": None, "size_lhs": None}
        embedding_models.append(embedding_model)

        # save config
        with (open(os.path.join(model_setup_config_dir, f"config_{subgraph}_{embedding_model['args'].model_name}.json"),
                   "w") as json_file):
            json.dump(vars(args_subgraph), json_file)

    # create unified model
    unified_model = "Unified"
    args_unified = util.get_args(args, unified_model)
    unified_model = getattr(models, unified_model)(args_unified, embedding_models)
    unified_model.to(device)

    time_stop_model_creation = time.time()
    logging.info(f"-\\\tSuccessfully created all embedding_models in "
                 f"{util.format_time(time_start_model_creation, time_stop_model_creation)}\t/-")

    return unified_model, args_unified


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
            valid_losses[subgraph] = "dropout"
            # "hard dropout" -> completely exclude model if it diverged once
            args.model_dropout = True

    run_diverged = True
    for embedding_model in embedding_models:
        if not embedding_model["args"].model_dropout:
            run_diverged = False

    return run_diverged, valid_losses
