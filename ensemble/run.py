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
from optimizers import regularizers as regularizers, KGOptimizer
from utils.train import count_params

try:
    # path on pc
    DATA_PATH = "data"
    os.listdir(DATA_PATH)
except FileNotFoundError:
    # path on laptop
    DATA_PATH = "C:\\Users\\timbr\\Masterarbeit\\Software\\Ensemble_Embedding_for_Link_Prediction\\data"


def train(info_directory, subgraph_amount, dataset="WN18RR", dataset_directory="data\\WN18RR", kge_models=None,
          regularizer="N3", reg=0, optimizer="Adagrad", max_epochs=50, patience=10, valid=3, rank=1000,
          batch_size=1000, neg_sample_size=50, dropout=0, init_size=1e-3, learning_rate=1e-1, gamma=0,
          bias="constant", dtype="double", double_neg=False, debug=False, multi_c=True,
          aggregation_method=Constants.MAX_SCORE_AGGREGATION, save_models=False, load_pretrained_models=False):
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

    if kge_models is None:
        kge_models = {Constants.ROTAT_E: []}

    args = argparse.Namespace(dataset=dataset, dataset_directory=dataset_directory, regularizer=regularizer, reg=reg,
                              optimizer=optimizer, max_epochs=max_epochs, patience=patience, valid=valid, rank=rank,
                              batch_size=batch_size, neg_sample_size=neg_sample_size, dropout=dropout,
                              init_size=init_size, learning_rate=learning_rate, gamma=gamma, bias=bias, dtype=dtype,
                              double_neg=double_neg, debug=debug, multi_c=multi_c, subgraph_amount=subgraph_amount)

    # set directories and ensure that they exist
    model_setup_config_dir = util_files.check_directory(f"{info_directory}\\model_setup_configs")
    model_file_dir = util_files.check_directory(f"{info_directory}\\model_files")
    # set files and ensure that they exist
    s_e_mapping_dir = util_files.check_file(f"{info_directory}\\subgraph_embedding_mapping.json")
    e_s_mapping_dir = util_files.check_file(f"{info_directory}\\embedding_subgraph_mapping.json")

    valid_loss_file_path = util_files.check_file(f"{info_directory}\\valid_loss.csv")
    train_loss_file_path = util_files.check_file(f"{info_directory}\\train_loss.csv")
    metrics_file_path = util_files.check_file(f"{info_directory}\\metrics_valid.csv")

    with (open(valid_loss_file_path, "w") as valid_loss_file, open(train_loss_file_path, "w") as train_loss_file,
          open(metrics_file_path, "w") as metrics_file):
        subgraphs_str = ""
        for sub_num in range(subgraph_amount):
            subgraphs_str += f";sub_{sub_num:03d}"
        valid_loss_file.write(f"epoch;average valid loss{subgraphs_str}\n")
        train_loss_file.write(f"epoch{subgraphs_str}\n")
        metrics_file.write(f"epoch;mode;MR;MRR;Hits@1;Hits@3;Hits@10;AMRI;MR_deviation\n")

    logging.info(f"### Saving .json config files of models in: {model_setup_config_dir} ###")
    logging.info(f"### Saving .pt files of stored models in: {model_file_dir} ###")

    # set up dataset directory
    dataset_path = os.path.join(DATA_PATH, args.dataset)

    # get original dataset name
    dataset_general = util.get_dataset_name(dataset)
    # create model using original dataset and sizes, use returned embeddings in new models as initialization
    embedding_general_ent, embedding_general_rel, theta_general_ent, theta_general_rel, general_dataset_shape \
        = util.generate_general_embeddings(dataset_general, args)

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
        except Exception:
            logging.info("No already trained models found.")
            s_e_mapping = False
            s_e_mapping_file.close()

    # --- Training preparation ---
    for subgraph_num in list(subgraph_embedding_mapping.keys()):
        # TODO perform preparation in external function
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

        # set embeddings
        model.entity = nn.Embedding(embedding_general_ent.num_embeddings, embedding_general_ent.embedding_dim)
        model.entity.weight.data = torch.rand(embedding_general_ent.weight.data.size())
        model.rel = nn.Embedding(embedding_general_rel.num_embeddings, embedding_general_rel.embedding_dim)
        model.rel.weight.data = torch.rand(embedding_general_rel.weight.data.size())

        # set context vectors
        model.theta_ent = nn.Embedding(theta_general_ent.num_embeddings, theta_general_ent.embedding_dim)
        model.theta_ent.weight.data = torch.rand(theta_general_ent.weight.data.size())
        model.theta_rel = nn.Embedding(theta_general_rel.num_embeddings, theta_general_rel.embedding_dim)
        model.theta_rel.weight.data = torch.rand(theta_general_rel.weight.data.size())

        logging.debug(f"Entity size: {model.entity.weight.data.size()}")
        logging.debug(f"Relation size: {model.rel.weight.data.size()}")
        logging.debug(f"Theta_ent size: {model.theta_ent.weight.data.size()}")
        logging.debug(f"Theta_rel size: {model.theta_rel.weight.data.size()}")

        logging.info(f"-\\\tTotal number of parameters: {total}\t/-")
        device = "cuda"
        model.to(device)

        # Handle already trained embedding models
        if s_e_mapping:
            if subgraph_embedding_mapping[subgraph_num] in s_e_mapping[str(subgraph_num)]:
                logging.info(f"Subgraph {subgraph} was already trained with embedding method "
                             f"{args_subgraph.model_name}.")

                embedding_model = {"dataset": dataset, "model": model, "subgraph": subgraph,
                                   "args": args_subgraph, "load_from_file": load_pretrained_models}

                if load_pretrained_models:
                    # file_name = f"model_{args.subgraph}_{args.model_name}_epoch{epoch}.pt"
                    embedding_model["pretrained_model_path"] = (f"{model_file_dir}\\"
                                                                f"model_{args.subgraph}_{args.model_name}")

                if embedding_model['load_from_file']:
                    file_path = f"{model_setup_config_dir}\\config_{subgraph}_{embedding_model['args'].model_name}.json"
                    # Open the JSON file for reading
                    with open(file_path, "r") as json_file:
                        # Load the JSON data into a Python dictionary
                        config_data = json.load(json_file)

                embedding_model["args"] = argparse.Namespace(**config_data)

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
                 f"{util.format_time(time_start_model_creation, time_stop_model_creation)}\t/-")

    # --- Training ---
    logging.info(f"-/\tStarting training\t\\-")
    time_start_training_total = time.time()

    valid_args = argparse.Namespace(counter=0, best_mrr=None, best_epoch=None, epoch=0, valid=valid, patience=patience)

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

    # Iterate over epochs
    for epoch in range(max_epochs):
        time_start_training_sub = time.time()

        # Iterate over models
        for embedding_model in embedding_models:
            args = embedding_model['args']
            model = embedding_model['model']

            # Load pretrained model for current epoch
            if embedding_model['load_from_file'] and load_pretrained_models:
                # TODO test if this works
                path_to_model = os.path.join(embedding_model["pretrained_model_path"],
                                             f"model_{args.subgraph}_{args.model_name}_epoch{epoch}.pt")
                model.load_state_dict(torch.load(path_to_model))

            logging.info(f"Training subgraph {embedding_model['subgraph']} in epoch {epoch} with model "
                         f"{args.model_name}")

            # Train step
            model.train()
            train_loss = embedding_model['optimizer'].epoch(embedding_model['train_examples'])
            logging.info(f"Subgraph {embedding_model['subgraph']} Training Epoch {epoch} | "
                         f"average train loss: {train_loss:.4f}")
            util_files.print_loss_to_file(train_loss_file_path, train_loss, epoch, "train",
                                          args.subgraph_num, subgraph_amount)

            # save embeddings of the model each epoch for later training processes
            if epoch % 1 == 0 or epoch == max_epochs - 1:
                # save model if the current epoch is the last one
                if epoch == max_epochs - 1:
                    torch.save(model.cpu().state_dict(), os.path.join(model_file_dir, f"model_{args.subgraph}_"
                                                                                      f"{args.model_name}.pt"))
                    logging.debug(f"Saved final model for {args.subgraph} ({args.model}).")

                # save model during training
                elif save_models:
                    file_dir = f"{model_file_dir}\\model_{args.subgraph}_{args.model_name}"
                    util_files.check_directory(file_dir)
                    torch.save(model.cpu().state_dict(), os.path.join(file_dir, f"model_{args.subgraph}_"
                                                                                f"{args.model_name}_epoch{epoch}.pt"))
                    logging.debug(f"Saved model for {args.subgraph} ({args.model}) at epoch {epoch}.")

            logging.debug(f"Entity size epoch {epoch}: {model.entity.weight.data.size()}")
            logging.debug(f"Relation size epoch {epoch}: {model.rel.weight.data.size()}")
            logging.debug(f"Theta_ent size epoch {epoch}: {model.theta_ent.weight.data.size()}")
            logging.debug(f"Theta_rel size epoch {epoch}: {model.theta_rel.weight.data.size()}")

            model.cuda()
            model.eval()

        # Calculate unified embedding
        cands_att_dict = Attention_mechanism.calculate_self_attention(embedding_models)
        Attention_mechanism.calculate_and_apply_unified_embedding(embedding_general_ent,
                                                                  embedding_general_rel,
                                                                  embedding_models,
                                                                  cands_att_dict)

        # Valid step
        valid_loss, valid_loss_dict = score_combination.calculate_valid_loss(embedding_models, valid_loss_file_path,
                                                                             epoch)

        logging.debug(f"Validation Epoch {epoch} per subgraph | average valid losses: {valid_loss_dict}")
        logging.info(f"Validation Epoch {epoch} | average valid loss: {valid_loss:.4f}")

        if (epoch + 1) % valid_args.valid == 0:

            valid_metrics = evaluate_ensemble(embedding_models, aggregation_method, mode="valid",
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

    evaluate_ensemble(embedding_models, metrics_file_path=metrics_file_path)

    time_total_end = time.time()
    logging.info(f"Finished ensemble training and testing in {util.format_time(time_total_start, time_total_end)}.")

    # TODO adapt to new training process
    #  -> make function
    #  -> update mapping based on existing files
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

    # model = embedding_models[0]["model"]
    # valid_examples = embedding_models[0]["valid_examples"]
    # test_examples = embedding_models[0]["test_examples"]
    # filters = embedding_models[0]["filters"]

    # Validation metrics
    # valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
    # logging.info(format_metrics(valid_metrics, split="valid"))

    # Test metrics
    # test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
    # logging.info(format_metrics(test_metrics, split="test"))

