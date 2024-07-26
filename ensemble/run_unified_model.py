import argparse
import logging
import os
import time

import torch

import models as models
from datasets.kg_dataset import KGDataset
from ensemble import util_files, util
from optimizers import regularizers as regularizers, KGOptimizer
from utils.train import avg_both, format_metrics

DATA_PATH = "data"

# TODO tidy up
def train(info_directory, args):

    time_total_start = time.time()

    # set directories and ensure that they exist
    model_setup_config_dir = util_files.check_directory(os.path.join(info_directory, "model_setup_configs"))
    model_file_dir = util_files.check_directory(os.path.join(info_directory, "model_files"))
    test_valid_file_dir = util_files.check_directory(os.path.join(args.dataset_dir, "data"))

    # set files and ensure that they exist
    valid_loss_file_path = util_files.check_file(os.path.join(info_directory, "valid_loss.csv"))
    train_loss_file_path = util_files.check_file(os.path.join(info_directory, "train_loss.csv"))
    metrics_file_path = util_files.check_file(os.path.join(info_directory, "metrics_valid.csv"))

    # print header into loss and metric files
    with (open(valid_loss_file_path, "w") as valid_loss_file, open(train_loss_file_path, "w") as train_loss_file,
          open(metrics_file_path, "w") as metrics_file):
        valid_loss_file.write(f"epoch;valid loss;change;change in %\n")
        train_loss_file.write(f"epoch;train loss;change;change in %\n")
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

    subgraph_embedding_mapping = util.assign_model_to_subgraph(args.kge_models, args)

    unified_model, args_unified = setup_unified_model(args, test_valid_file_dir, model_setup_config_dir,
                                                      dataset_general, subgraph_embedding_mapping, dataset_path)

    previous_valid_loss = 0
    previous_train_loss = 0
    previous_metrics = {}
    run_diverged = False

    # --- Training ---
    logging.info(f"-/\tStarting training\t\\-")
    time_start_training_total = time.time()

    valid_args = argparse.Namespace(counter=0, best_mrr=None, best_epoch=None, epoch=0, valid=args.valid,
                                    patience=args.patience)

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
        # Train step
        unified_model.train()
        train_loss = optimizer.epoch(train_examples, epoch=epoch)

        previous_train_loss, train_loss_change = util.get_loss_change(train_loss, previous_train_loss)
        util_files.print_loss_to_file(train_loss_file_path, epoch,
                                      [train_loss, train_loss_change[0], train_loss_change[1]])
        logging.info(f"Training\tEpoch {epoch} | average train loss: {train_loss:.4f} | "
                     f"change to last epoch: {train_loss_change[0]:.4f} ({train_loss_change[1]:.3f}%)")

        # debugging messages
        logging.debug(f"Entity size epoch {epoch}: {unified_model.entity.weight.data.size()}")
        logging.debug(f"Relation size epoch {epoch}: {unified_model.rel.weight.data.size()}")
        if unified_model.theta_ent is not None:
            logging.debug(f"Theta_ent size epoch {epoch}: {unified_model.theta_ent.weight.data.size()}")
        if unified_model.theta_rel is not None:
            logging.debug(f"Theta_rel size epoch {epoch}: {unified_model.theta_rel.weight.data.size()}")

        # Valid step
        valid_loss = optimizer.calculate_valid_loss(valid_examples)

        # print validation losses
        previous_valid_loss, valid_loss_change = util.get_loss_change(valid_loss, previous_valid_loss)
        util_files.print_loss_to_file(valid_loss_file_path, epoch,
                                      [valid_loss, valid_loss_change[0], valid_loss_change[1]])

        logging.info(f"Validation\tEpoch {epoch} | average valid loss: {valid_loss:.4f} | "
                     f"change to last epoch: {valid_loss_change[0]:.4f} ({valid_loss_change[1]:.3f}%)")

        if (epoch + 1) % valid_args.valid == 0:
            valid_metrics = avg_both(*unified_model.compute_metrics(valid_examples, filters, args.sizes), epoch=epoch)
            previous_metrics, metrics_change_absolut, metrics_change_percent = (
                util.get_metrics_change(valid_metrics, previous_metrics))

            logging.info(format_metrics(valid_metrics, split="valid", metrics_change_absolut=metrics_change_absolut,
                                        metrics_change_percent=metrics_change_percent))
            util_files.print_metrics_to_file(metrics_file_path, valid_metrics, epoch=epoch, mode="valid")

            valid_mrr = valid_metrics["MRR"]['average']
            if not valid_args.best_mrr or valid_mrr > valid_args.best_mrr:
                valid_args.best_mrr = valid_mrr
                valid_args.counter = 0
                valid_args.best_epoch = epoch
                logging.info(f"Saving embedding_models at epoch {epoch} in {model_file_dir}")

                torch.save(unified_model.cpu().state_dict(), os.path.join(model_file_dir, f"unified_model.pt"))
                unified_model.cuda()

            else:
                valid_args.counter += 1
                if not valid_args.counter == valid_args.patience:
                    logging.info(f"Early stopping in {valid_args.patience - valid_args.counter} validation steps.")

                if valid_args.counter == valid_args.patience:
                    logging.info("\t Early stopping")
                    break
                elif valid_args.counter == (valid_args.patience // 2):
                    pass
                    logging.info(f"\t Reducing learning rate")
                    optimizer.reduce_lr()

        unified_model.cuda()
        unified_model.eval()

        time_stop_training_sub = time.time()
        logging.info(f"-\\\tTraining and optimization of epoch {epoch} finished in "
                     f"{util.format_time(time_start_training_sub, time_stop_training_sub)}\t/-")

    if not valid_args.best_mrr:
        torch.save(unified_model.cpu().state_dict(), os.path.join(model_file_dir, "unified_model.pt"))
    else:
        logging.info(f"\t Loading best model saved at epoch {valid_args.best_epoch}")
        unified_model.load_state_dict(torch.load(os.path.join(model_file_dir, "unified_model.pt")))
    unified_model.cuda()
    unified_model.eval()

    time_stop_training_total = time.time()

    logging.info(f"-\\\tSuccessfully finished training and optimizing all subgraphs in "
                 f"{util.format_time(time_start_training_total, time_stop_training_total)}\t/-")

    # --- Final testing with best model ---
    valid_metrics = avg_both(*unified_model.compute_metrics(valid_examples, filters, args.sizes),
                             epoch=args.max_epochs + 10)
    logging.info(format_metrics(valid_metrics, split="valid"))
    util_files.print_metrics_to_file(metrics_file_path, valid_metrics, epoch=args.max_epochs + 10, mode="valid")

    test_metrics = avg_both(*unified_model.compute_metrics(test_examples, filters, args.sizes),
                            epoch=args.max_epochs + 20)
    logging.info(format_metrics(test_metrics, split="test"))
    util_files.print_metrics_to_file(metrics_file_path, test_metrics, epoch=args.max_epochs + 20, mode="test")

    time_total_end = time.time()
    logging.info(f"Finished ensemble training and testing in {util.format_time(time_total_start, time_total_end)}.")


def setup_unified_model(args, test_valid_file_dir, model_setup_config_dir, general_dataset,
                        subgraph_embedding_mapping, dataset_path):
    device = "cuda"

    # create unified model
    init_args = argparse.Namespace(
        test_valid_file_dir=test_valid_file_dir,
        model_setup_config_dir=model_setup_config_dir,
        no_progress_bar=args.no_progress_bar,
        subgraph_embedding_mapping=subgraph_embedding_mapping,
        device=device,
        subgraph_amount=args.subgraph_amount,
        general_dataset=general_dataset,  # parent dataset name
        dataset_path=dataset_path  # direct path
    )

    # general_dataset
    dataset_general = KGDataset(os.path.abspath(os.path.join("data", init_args.general_dataset)), args.debug)
    args.sizes = dataset_general.get_shape()

    unified_model = "Unified"
    args.model = unified_model
    args_unified = util.get_args(args, unified_model)
    unified_model = getattr(models, unified_model)(args, init_args, args_unified)
    unified_model.to(init_args.device)

    return unified_model, args_unified
