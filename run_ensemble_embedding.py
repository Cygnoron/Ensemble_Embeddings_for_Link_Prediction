import argparse
import logging
import os
import time
import traceback

import wandb
from ensemble import Constants, util_files, util, subsampling, run_unified_model
from run import train

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

parser = argparse.ArgumentParser(
    description="Ensemble Approaches for Link Prediction"
)
# --- Arguments for embedding embedding_models ---
#   - General hyperparameters -
parser.add_argument(
    "--dataset", default="WN18RR", help="Knowledge Graph dataset"
)
parser.add_argument(
    "--max_epochs", default=50, type=int, help="Maximum number of epochs to train for"
)
parser.add_argument(
    "--rank", default=1000, type=int, help="Embedding dimension"
)
parser.add_argument(
    "--patience", default=10, type=int, help="Number of epochs before early stopping"
)
parser.add_argument(
    "--valid", default=3, type=float, help="Number of epochs before validation"
)
parser.add_argument(
    "--dtype", default="double", type=str, choices=["single", "double"], help="Machine precision"
)
parser.add_argument(
    "--debug", action="store_true",
    help="Only use 1000 examples for debugging"
)
#   - Individually specifiable hyperparameters -
parser.add_argument(
    '--model', type=str, default="{TransE:[\'all\']}",
    help='JSON string of the mapping from embedding methods to subgraphs.\n'
         '- <subgraph number> in a mapping sets the specified subgraphs to this class_name\n'
         '- \'all\' in a mapping sets all subgraphs to this class_name. This has the same effect as --model <MODEL_NAME>\n'
         '- \'rest\' in a mapping allows all unmapped subgraphs to be embedded by this class_name. '
         'If nothing was specified in the mapping, all subgraphs can be embedded by the given embedding class_name.'

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
    "--double_neg", action="store_true",
    help="Whether to negative sample both head and tail entities"
)
parser.add_argument(
    "--multi_c", action="store_true", help="Multiple curvatures per relation"
)
# --- Parameters for ensemble methods ---

#   - Sampling parameters -
parser.add_argument(
    "--subgraph_amount", default=10, type=int, help="The amount of subgraphs, that will be present in the ensemble"
)
parser.add_argument(
    "--subgraph_size_range", default=(0.3, 0.7), help="A tuple (min, normal) with the relative subgraph sizes, where \n"
                                                      "- \'min\' is the minimal relative subgraph size that will be "
                                                      "used with Feature sampling\n"
                                                      "- \'normal\' is the normal relative subgraph size, that needs "
                                                      "to be reached under normal conditions"
)
parser.add_argument(
    "--sampling_method", default="Entity", choices=["Entity", "Feature"],
    help="The sampling class_name, that should be used"
)
parser.add_argument(
    "--rho", default=1, type=float,
    help="Factor for Feature sampling, which specifies, how many relation names should be "
         "present in the subgraph, which is calculated by the formula \'rho * √|Relation Names|\'"
)
parser.add_argument(
    "--sampling_seed", default="42",
    help="The seed for sampling subgraphs. Type \'random\' for a random seed."
)

#   - Model parameters -
parser.add_argument(
    "--aggregation_method", default="average", choices=["max", "average", "attention"],
    help="The class_name by which all scores from the ensemble are aggregated."
)
parser.add_argument(
    "--model_dropout_factor", default=10, type=int,
    help="The factor, which determines, if a model is considered diverged and therefore removed from the ensemble. "
         "A model is considered diverged, if the current validation loss exceeds the first validation loss times the "
         "model dropout factor"
)
parser.add_argument(
    "--theta_method", default="regular", choices=["no", "regular", "reversed", "relation", "multiplied"],
    help="The class_name by which the context vectors (cv) are used in order to calculate attention values for the unified "
         "embedding:\n"
         "- \'no\' deactivates the calculation of an unified embedding\n"
         "- \'regular\' cv for entities is only influenced by entity embeddings, "
         "cv for relation names is only influenced by relation name embeddings\n"
         "- \'reversed\' cv for entities is only influenced by relation name embeddings, "
         "cv for relation names is only influenced by entity embeddings\n"
         "- \'relation\' cv for entities is only influenced by relation name embeddings, "
         "cv for relation names is only influenced by relation name embeddings\n"
         "- \'multiplied\' cv for entities is influenced by entity embeddings as well as relation name embeddings, "
         "cv for relation names is only influenced by relation name embeddings\n"
)

#   - System parameters -
parser.add_argument(
    "--logging", default="info", choices=['critical', 'error', 'warning', 'info', 'debug', 'data'],
    help="Determines the level of logging.\n"
         "- \'info\': Contains information about the progress\n"
         "- \'debug\': Also contains information about variables, e.g. tensor sizes\n"
         "- \'data\': Also contains embedding weights and other data from variables, "
         "which is printed directly to the log\'"
)
parser.add_argument(
    "--wandb_project", default="False",
    help="Turn on logging of metrics via Weights&Biases and synchronize with the given project name."
)
parser.add_argument(
    "--no_sampling", action='store_true', help="Turn off sampling"
)
parser.add_argument(
    "--no_training", action='store_true', help="Turn off training"
)
parser.add_argument(
    "--no_progress_bar", action='store_true', help="Turn off all progress bars"
)
parser.add_argument(
    "--no_time_dependent_file_path", action='store_true', help="Turn off specifying the current time in file path, "
                                                               "when creating logs and other files"
)


def run_baseline(args):
    try:
        Constants.get_wandb(args.wandb_project)
    except:
        pass
    train(args)


def run_embedding(args):
    Constants.get_wandb(args.wandb_project)

    time_process_start = time.time()

    args.sampling_method = util.handle_methods(args.sampling_method, "sampling")
    args.aggregation_method = util.handle_methods(args.aggregation_method, "aggregation")
    args.theta_calculation = util.handle_methods(args.theta_method, "theta")
    args.subgraph_size_range = util.handle_methods(args.subgraph_size_range, "size_range")

    dataset_out = ""
    if args.sampling_method == Constants.FEATURE_SAMPLING:
        dataset_out = (
            f"{args.dataset}_{args.sampling_method[2]}_N{args.subgraph_amount}_rho{args.rho}"
            f"_min{args.subgraph_size_range[0]}_max{args.subgraph_size_range[1]}")
    elif args.sampling_method == Constants.ENTITY_SAMPLING:
        dataset_out = (f"{args.dataset}_{args.sampling_method[2]}_N{args.subgraph_amount}"
                       f"_min{args.subgraph_size_range[0]}_max{args.subgraph_size_range[1]}")

    dataset_out_dir = os.path.join("data", dataset_out)
    dataset_in = args.dataset
    args.dataset = dataset_out
    args.dataset_dir = dataset_out_dir

    info_directory = os.path.abspath(f"{dataset_out_dir}")
    util_files.check_directory(info_directory)
    info_directory = util_files.get_info_directory_path(dataset_out_dir, args)

    util.setup_logging(info_directory, "Ensemble_Embedding_for_Link_Prediction.log",
                       logging_level=args.logging)

    if not args.no_sampling:
        subsampling.sample_graph(info_directory, dataset_in, dataset_out_dir, args.sampling_method,
                                 subgraph_amount=args.subgraph_amount,
                                 subgraph_size_range=args.subgraph_size_range,
                                 rho=args.rho, no_progress_bar=args.no_progress_bar, random_seed=args.sampling_seed)

    error = False
    try:
        if not args.no_training:
            args.kge_models = util.get_embedding_methods(args.model)

            wandb.init(project=Constants.PROJECT_NAME, config=args)

            # run.train(info_directory, args)
            run_unified_model.train(info_directory, args)

    except Exception:
        logging.error(traceback.format_exc())
        error = True

    time_process_end = time.time()

    if not error:
        logging.info(f"The entire process including sampling, training and testing took "
                     f"{util.format_time(time_process_start, time_process_end)}.")
    else:
        logging.info(f"The process ended with an error after "
                     f"{util.format_time(time_process_start, time_process_end)}.")


def run_embedding_manual():
    # dataset_in = "Debug"
    dataset_in = "WN18RR"
    # dataset_in = "FB15K"
    # dataset_in = "NELL-995"
    subgraph_amount = 4
    subgraph_size_range = (0.03, 0.7)
    rho = 2
    model_dropout_factor = 10

    args = argparse.Namespace(no_sampling=True, no_training=False, no_time_dependent_file_path=False,
                              no_progress_bar=False, subgraph_amount=subgraph_amount, wandb_project="False",
                              subgraph_size_range=subgraph_size_range, rho=rho,
                              sampling_method=Constants.ENTITY_SAMPLING,
                              # sampling_method=Constants.FEATURE_SAMPLING,
                              aggregation_method=Constants.AVERAGE_SCORE_AGGREGATION,
                              # aggregation_method=Constants.ATTENTION_SCORE_AGGREGATION,
                              # aggregation_method=Constants.MAX_SCORE_AGGREGATION,
                              theta_calculation=Constants.REGULAR_THETA, model_dropout_factor=model_dropout_factor)

    args.wandb_project = "Experiments"

    Constants.get_wandb(args.wandb_project)

    time_process_start = time.time()

    dataset_out = ""
    if args.sampling_method == Constants.FEATURE_SAMPLING:
        dataset_out = (f"{dataset_in}_{args.sampling_method[2]}_N{subgraph_amount}_rho{args.rho}"
                       f"_min{subgraph_size_range[0]}_max{subgraph_size_range[1]}")
    elif args.sampling_method == Constants.ENTITY_SAMPLING:
        dataset_out = (f"{dataset_in}_{args.sampling_method[2]}_N{subgraph_amount}"
                       f"_min{subgraph_size_range[0]}_max{subgraph_size_range[1]}")
    dataset_out_dir = os.path.join("data", dataset_out)

    args.dataset = dataset_out
    args.dataset_dir = dataset_out_dir

    info_directory = os.path.abspath(f"{dataset_out_dir}")
    util_files.check_directory(info_directory)
    info_directory = util_files.get_info_directory_path(dataset_out_dir, args)

    args.info_directory = info_directory

    util.setup_logging(info_directory, "Ensemble_Embedding_for_Link_Prediction.log",
                       logging_level="info")

    # --- if a new debugging dataset was created ---
    # for split in ["train", "test", "valid"]:
    #     util_files.csv_to_file(os.path.join("ensemble", "Random_Triples.csv"), os.path.join("data", "Debug", split),
    #                            only_unique=True)

    # process_specific_dataset("Debug", "data")

    # util_files.pickle_to_csv(f"D:\\Masterarbeit\\Software\\Ensemble_Embedding_for_Link_Prediction\\"
    #                          f"data\\{dataset_in}\\train.pickle",
    #                          f"D:\\Masterarbeit\\Software\\Ensemble_Embedding_for_Link_Prediction\\"
    #                          f"data\\{dataset_in}\\train_readable.csv", ';')

    # --- create files for insights in dataset ---

    # util_files.create_entity_and_relation_name_set_file(os.path.join("data",dataset_in))

    # for dataset in os.scandir("data"):
    #     print(dataset.name)
    #     util.create_entity_and_relation_name_set_file(dataset.name)

    # --- sampling process ---
    if not args.no_sampling:
        subsampling.sample_graph(info_directory, dataset_in, dataset_out_dir, args.sampling_method,
                                 subgraph_amount=args.subgraph_amount, subgraph_size_range=args.subgraph_size_range,
                                 rho=args.rho, no_progress_bar=args.no_progress_bar)

    # --- setup for model training ---

    allowed_kge_models = {Constants.TRANS_E: list(range(0, 5)),
                          Constants.COMPL_EX: [20],
                          Constants.DIST_MULT: ['all']
                          # Constants.DIST_MULT: list(range(5, 10))
                          }

    # --- training process ---

    error = False
    try:
        if not args.no_training:
            args.kge_models = allowed_kge_models

            args.max_epochs = 50
            args.rank = 32
            args.patience = 15
            args.valid = 5
            args.dtype = "single"
            args.batch_size = 1000
            args.debug = True

            args.learning_rate = {'TransE': 0.001, 'DistMult': 0.1, 'ComplEx': 0.1, 'RotatE': 0.001, 'AttE': 0.001,
                                  'AttH': 0.001}
            args.reg = {'ComplEx': 0.05, 'TransE': 0.0, 'DistMult': 0.05, 'rest': 0.0}
            args.optimizer = {"TransE": "Adam", 'DistMult': "Adagrad", 'ComplEx': "Adagrad", 'RotatE': "Adam",
                              'AttE': "Adam", 'AttH': "Adam", 'Unified': "Adam"}
            args.neg_sample_size = {"TransE": -1, 'DistMult': -1, 'ComplEx': -1, 'RotatE': 250, 'AttE': -1,
                                    'AttH': 250}
            args.bias = {'ComplEx': "none", 'TransE': "learn", 'DistMult': "none", 'AttE': "learn", 'rest': "none"}
            args.double_neg = {'ComplEx': True, 'TransE': True, 'DistMult': True, 'AttE': False}
            args.multi_c = {'AttE': True, 'AttH': True, 'rest': False}

            args.regularizer = {'all': "N3"}
            args.init_size = {'all': 0.001}
            args.dropout = {'all': 0}
            args.gamma = {'all': 0}

            if Constants.LOG_WANDB:
                wandb.init(project=Constants.PROJECT_NAME, config=vars(args))
                wandb.login()

            # run.train(info_directory, args)
            run_unified_model.train(info_directory, args)

    except Exception:
        logging.error(traceback.format_exc())
        error = True

    time_process_end = time.time()

    if not error:
        logging.info(f"The entire process including sampling, training and testing took "
                     f"{util.format_time(time_process_start, time_process_end)}.")
    else:
        logging.info(f"The process ended with an error after "
                     f"{util.format_time(time_process_start, time_process_end)}")


if __name__ == "__main__":
    # Function to run via command prompt
    # run_embedding(parser.parse_args())
    # Function to run manual via IDE
    run_embedding_manual()

    # Function to run baseline
    args = parser.parse_args()

    args.model = "DistMult"
    args.dataset = "WN18RR"
    args.max_epochs = 50
    args.rank = 32
    args.patience = 15
    args.valid = 5
    args.dtype = "single"
    args.batch_size = 100
    args.debug = False

    args.learning_rate = 0.1
    args.reg = 0.5
    args.optimizer = "Adagrad"
    args.neg_sample_size = -1
    args.bias = "none"
    args.double_neg = False
    args.multi_c = False

    args.regularizer = "N3"
    args.init_size = 0.001
    args.gamma = 0.0
    args.dropout = 0.0

    args.no_progress_bar = False
    args.entities = None
    args.relation_names = None
    args.wandb_project = "Experiments"

    # run_baseline(args)
