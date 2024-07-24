"""Train Knowledge Graph embeddings for link prediction."""

import argparse
import json
import logging
import os

import torch
import torch.optim

import models as models
import optimizers.regularizers as regularizers
import wandb
from datasets.kg_dataset import KGDataset
from ensemble import util, Constants
from models import all_models
from optimizers.kg_optimizer import KGOptimizer
from utils.train import get_savedir, avg_both, format_metrics, count_params

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

try:
    # path on pc
    DATA_PATH = "data"
    os.listdir(DATA_PATH)
except:
    # path on laptop
    DATA_PATH = "C:\\Users\\timbr\\OneDrive - bwedu\\MASTER\\Masterarbeit\\Software\\Ensemble_Embedding_for_Link_Prediction\\data"

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
    util.setup_logging(save_dir, "train.log", logging_level="info")

    # create dataset
    dataset_path = os.path.join(DATA_PATH, args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()
    args.theta_calculation = Constants.NO_THETA

    if Constants.LOG_WANDB:
        wandb.init(project=Constants.PROJECT_NAME, config=args)

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
                            bool(args.double_neg), no_progress_bar=args.no_progress_bar)
    counter = 0
    best_mrr = None
    best_epoch = None
    logging.info("\t Start training")
    for step in range(args.max_epochs):

        # Train step
        model.train()

        train_loss = optimizer.epoch(train_examples, epoch=step)
        logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))

        # Valid step
        model.eval()
        valid_loss = optimizer.calculate_valid_loss(valid_examples)
        logging.info("\t Epoch {} | average valid loss: {:.4f}".format(step, valid_loss))

        if (step + 1) % args.valid == 0:
            valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters, args.sizes), epoch=step)
            logging.info(format_metrics(valid_metrics, split="valid"))

            valid_mrr = valid_metrics["MRR"]["average"]
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
    valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters, args.sizes), epoch=args.max_epochs + 10)
    logging.info(format_metrics(valid_metrics, split="valid"))

    # Test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters, args.sizes), epoch=args.max_epochs + 20)
    logging.info(format_metrics(test_metrics, split="test"))


if __name__ == "__main__":
    train(parser.parse_args())
