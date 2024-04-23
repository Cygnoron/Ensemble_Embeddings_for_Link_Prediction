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
from optimizers import regularizers as regularizers, KGOptimizer
from utils.train import count_params, avg_both, format_metrics

try:
    # path on pc
    DATA_PATH = "data"
    os.listdir(DATA_PATH)
except:
    # path on laptop
    DATA_PATH = "C:\\Users\\timbr\\Masterarbeit\\Software\\Ensemble_Embedding_for_Link_Prediction\\data"


def train(info_directory, dataset="WN18RR", dataset_directory="data\\WN18RR", kge_models=None,
          regularizer="N3", reg=0, optimizer="Adagrad", max_epochs=50, patience=10, valid=3, rank=1000,
          batch_size=1000, neg_sample_size=50, dropout=0, init_size=1e-3, learning_rate=1e-1, gamma=0,
          bias="constant", dtype="double", double_neg=False, debug=False, multi_c=True):
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

    # set up dataset directory
    dataset_path = os.path.join(DATA_PATH, args.dataset)

    # get original dataset name
    dataset_general = util.get_dataset_name(dataset)
    # create model using original dataset and sizes, use returned embeddings in new models as initialization
    buffer_var = util.generate_general_embeddings(dataset_general, args)
    embedding_general_ent, embedding_general_rel, theta_general_ent, theta_general_rel = buffer_var

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
        args_subgraph.subgraph_num = subgraph_num
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

        # set embeddings
        model.entity = nn.Embedding(embedding_general_ent.num_embeddings, embedding_general_ent.embedding_dim)
        model.entity.weight.data = embedding_general_ent.weight.data.clone()
        model.rel = nn.Embedding(embedding_general_rel.num_embeddings, embedding_general_rel.embedding_dim)
        model.rel.weight.data = embedding_general_rel.weight.data.clone()

        # set context vectors
        model.theta_ent = nn.Embedding(theta_general_ent.num_embeddings, theta_general_ent.embedding_dim)
        model.theta_ent.weight.data = theta_general_ent.weight.data.clone()
        model.theta_rel = nn.Embedding(theta_general_rel.num_embeddings, theta_general_rel.embedding_dim)
        model.theta_rel.weight.data = theta_general_rel.weight.data.clone()

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
        # Skip training setup for already trained models
        if embedding_model['load_from_file']:
            continue

        # --- Setting up training ---
        args = embedding_model['args']
        logging.info(f"-/\tStart training {embedding_model['subgraph']} with {args.model_name}\t\\-")

        model = embedding_model['model']

        # Get optimizer
        regularizer = (getattr(regularizers, args.regularizer)(args.reg))
        optim_method = (getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate))
        embedding_model['optimizer'] = KGOptimizer(model, regularizer, optim_method, args.batch_size,
                                                   args.neg_sample_size, bool(args.double_neg))

        args.counter = 0
        args.best_mrr = None
        args.best_epoch = None

    # Iterate over epochs
    for epoch in range(max_epochs):
        time_start_training_sub = time.time()

        # Iterate over models
        for embedding_model in embedding_models:
            # TODO check feasibility
            #  Skip training for already trained models
            if embedding_model['load_from_file']:
                continue

            args = embedding_model['args']
            model = embedding_model['model']
            logging.info(f"Training subgraph {embedding_model['subgraph']} in epoch {epoch} with model "
                         f"{args.model_name}")

            # Train step
            model.train()
            train_loss = embedding_model['optimizer'].epoch(embedding_model['train_examples'])
            logging.info(f"Subgraph {embedding_model['subgraph']} Training Epoch {epoch} | "
                         f"average train loss: {train_loss:.4f}")

            # TODO fully implement validation -> correct datasets and maybe adapting process
            #  validate all models together -> implement validation and testing process for all models
            # Valid step
            # model.eval()
            # valid_loss = embedding_model['optimizer'].calculate_valid_loss(embedding_model["valid_examples"])
            # logging.info(f"Subgraph {embedding_model['subgraph']} Validation Epoch {epoch} | "
            #              f"average train loss: {valid_loss:.4f}")

            if (epoch + 1) % args.valid == 0:
                # valid_metrics = avg_both(*model.compute_metrics(embedding_model["valid_examples"],
                #                                                 embedding_model['filters']))
                # logging.info(format_metrics(valid_metrics, split="valid"))
                #
                # valid_mrr = valid_metrics["MRR"]
                # if not args.best_mrr or valid_mrr > args.best_mrr:
                #     args.best_mrr = valid_mrr
                #     args.counter = 0
                #     args.best_epoch = epoch
                logging.info(f"Saving model at epoch {epoch} in {info_directory}")
                torch.save(embedding_model['model'].cpu().state_dict(),
                           f"{model_file_dir}\\model_{args.subgraph}_"f"{args.model_name}.pt")
                embedding_model['model'].cuda()

            else:
                args.counter += 1
                if args.counter == args.patience:
                    logging.info("\t Early stopping")
                    break
                elif args.counter == args.patience // 2:
                    pass
                    # logging.info(f"\t Reducing learning rate")
                    # embedding_model['optimizer'].reduce_lr()

            if not args.best_mrr:
                logging.info(f"Saving new model saved at epoch {args.best_epoch}\n{model_file_dir}\\model_"
                             f"{args.subgraph}_{args.model_name}.pt")
                torch.save(embedding_model['model'].cpu().state_dict(),
                           f"{model_file_dir}\\model_{args.subgraph}_"
                           f"{args.model_name}.pt")
            else:
                logging.info(f"Loading best model saved at epoch {args.best_epoch}")
                embedding_model['model'].load_state_dict(torch.load(f"{model_file_dir}\\model_{args.subgraph}_"
                                                                    f"{args.model_name}.pt"))

            embedding_model['model'].cuda()
            embedding_model['model'].eval()

        # validation step

        time_stop_training_sub = time.time()
        logging.info(f"-\\\tTraining and optimization of epoch {epoch} finished in "
                     f"{round(time_stop_training_sub - time_start_training_sub, 3)} seconds / "
                     f"{round((time_stop_training_sub - time_start_training_sub) / 60, 3)} minutes\t/-")

        Attention_mechanism.calculate_self_attention(embedding_models)

        Attention_mechanism.calculate_and_apply_unified_embedding(embedding_general_ent,
                                                                  embedding_general_rel,
                                                                  embedding_models)

    # for embedding_model in embedding_models:
    #     util.difference_embeddings(embedding_general_ent.weight.data,
    #                                embedding_model['model'].entity.weight.data,
    #                                info_directory, ent=True, file_identifier=embedding_model['args'].subgraph)
    #     util.difference_embeddings(embedding_general_rel.weight.data,
    #                                embedding_model['model'].rel.weight.data,
    #                                info_directory, rel=True, file_identifier=embedding_model['args'].subgraph)

    time_stop_training_total = time.time()

    logging.info(f"-\\\tSuccessfully finished training and optimizing all subgraphs in "
                 f"{round(time_stop_training_total - time_start_training_total, 3)} seconds / "
                 f"{round((time_stop_training_total - time_start_training_total) / 60, 3)} minutes\t/-")

    # TODO adapt to new training process
    # Update mappings for already trained embeddings, in order to skip already trained subgraphs
    # with (open(s_e_mapping_dir, 'r+') as s_e_mapping_file,
    #       open(e_s_mapping_dir, 'r+') as e_s_mapping_file):
    #
    #     # Load subgraph_embedding_mapping from file and convert to dictionary
    #     try:
    #         s_e_mapping = dict(json.loads(s_e_mapping_file.read()))
    #     except:
    #         s_e_mapping = {}
    #
    #     # Update s_e_mapping with subgraph_embedding_mapping
    #     for sub_num in subgraph_embedding_mapping:
    #         if str(sub_num) not in list(s_e_mapping.keys()):
    #             s_e_mapping[str(sub_num)] = []
    #         if subgraph_embedding_mapping[sub_num] not in s_e_mapping[str(sub_num)]:
    #             s_e_mapping[str(sub_num)].append(subgraph_embedding_mapping[sub_num])
    #             # TODO also change to numpy array?
    #
    #     # Write updated s_e_mapping back to file
    #     s_e_mapping_file.seek(0)
    #     json.dump(s_e_mapping, s_e_mapping_file)
    #     logging.info(f"Updated mapping from subgraphs to embedding methods in {s_e_mapping_file.name}.")
    #
    #     # Get the inverse mapping of subgraph_embedding_mapping
    #     inverse_mapping = util.inverse_dict(subgraph_embedding_mapping)
    #
    #     try:
    #         e_s_mapping = dict(json.loads(e_s_mapping_file.read()))
    #     except:
    #         e_s_mapping = {}
    #
    #     # Update e_s_mapping with inverse_mapping
    #     for sub_num in inverse_mapping:
    #         if str(sub_num) not in list(e_s_mapping.keys()):
    #             e_s_mapping[str(sub_num)] = []
    #         if inverse_mapping[sub_num] not in e_s_mapping[str(sub_num)]:
    #             e_s_mapping[str(sub_num)] = list(
    #                 np.unique(np.append(e_s_mapping[str(sub_num)], (inverse_mapping[sub_num]))).astype(int).astype(str))
    #
    #     # Write updated e_s_mapping back to file
    #     e_s_mapping_file.seek(0)
    #     json.dump(e_s_mapping, e_s_mapping_file)
    #     logging.info(f"Updated mapping from embedding methods to subgraphs in {e_s_mapping_file.name}.")
    #
    # if len(models_to_load) > 0:
    #     logging.info(f"-/\tLoading {len(models_to_load)} models from storage.\t\\-")
    #
    #     for embedding_model in models_to_load:
    #         args_subgraph = embedding_model['args']
    #         logging.info(f"Loading model {args_subgraph.model_name} for subgraph {args_subgraph.subgraph}")
    #
    #         embedding_model["model"].load_state_dict(torch.load(f"{model_file_dir}\\model_{args_subgraph.subgraph}_"
    #                                                             f"{args_subgraph.model_name}.pt"))
    #
    #     logging.info(f"-\\\tSuccessfully loaded {len(models_to_load)} models from storage.\t/-")

    # TODO fix dimension error with AttH

    # # TODO fully implement validation and testing metrics -> correct datasets and maybe adapting process
    # for embedding_model in embedding_models:
    #     model = embedding_model["model"]
    #     valid_examples = embedding_model["valid_examples"]
    #     test_examples = embedding_model["test_examples"]
    #     filters = embedding_model["filters"]
    #
    #     # Validation metrics
    #     valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
    #     logging.info(format_metrics(valid_metrics, split="valid"))
    #
    #     # Test metrics
    #     test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
    #     logging.info(format_metrics(test_metrics, split="test"))

    model = embedding_models[0]["model"]
    valid_examples = embedding_models[0]["valid_examples"]
    test_examples = embedding_models[0]["test_examples"]
    filters = embedding_models[0]["filters"]

    # Validation metrics
    # valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
    # logging.info(format_metrics(valid_metrics, split="valid"))

    # Test metrics
    # test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
    # logging.info(format_metrics(test_metrics, split="test"))

    # --- Testing with aggregated scores ---

    test_ensemble(embedding_models)


def test_ensemble(embedding_models, aggregation_method=Constants.MAX_SCORE, mode="test", batch_size=500):
    # TODO move filter, test_examples, valid_examples files to general location

    if mode == "test":
        logging.info(f"Testing the ensemble with the score aggregation method \"{aggregation_method[1]}\".")
        examples = embedding_models[0]["test_examples"]
    elif mode == "valid":
        logging.info(f"Validating the ensemble with the score aggregation method \"{aggregation_method[1]}\".")
        examples = embedding_models[0]["valid_examples"]
    else:
        logging.error(f"The given mode \"{mode}\" does not exist!")
        return

    # calculate scores for all models
    embedding_models = score_combination.calculate_scores(embedding_models, examples, batch_size=batch_size)

    # set target scores from first embedding model, as these are all the same
    targets = {'rhs': embedding_models[0]['targets']['rhs'], 'lhs': embedding_models[0]['targets']['lhs']}

    # combine the calculated scores from all models, according to the given aggregation method
    aggregated_scores = score_combination.combine_scores(embedding_models, aggregation_method)

    # compute the ranks for all queries
    filters = embedding_models[0]["filters"]
    ranks = score_combination.compute_ranks(embedding_models, examples, filters, targets, aggregated_scores,
                                            batch_size=batch_size)

    # calculate metrics from the ranks
    metrics = avg_both(*score_combination.compute_metrics_from_ranks(ranks))
    logging.info(format_metrics(metrics, split=mode))

    return
