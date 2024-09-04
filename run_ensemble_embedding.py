import argparse
import logging
import os
import random
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
    "--batch_size", default=1000, type=int, help="Batch size"
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
    '--model', type=str, default="{\"TransE\":[\"all\"]}",
    help='JSON string of the mapping from embedding methods to subgraphs.\n'
         '- <subgraph number> in a mapping sets the specified subgraphs to this method\n'
         '- \'all\' in a mapping sets all subgraphs to this method. This has the same effect as --model <MODEL_NAME>\n'
         '- \'rest\' in a mapping allows all unmapped subgraphs to be embedded by this method. '
         'If nothing was specified in the mapping, all subgraphs can be embedded by the given embedding method.'

)
parser.add_argument(
    "--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer"
)
parser.add_argument(
    "--reg", default=0, help="Regularization weight"
)
parser.add_argument(
    "--optimizer", default="Adagrad",  # , choices=["Adagrad", "Adam", "SparseAdam"]
    help="Optimizer"
)
parser.add_argument(
    "--neg_sample_size", default=50, help="Negative sample size, -1 to not use negative sampling"
)
parser.add_argument(
    "--dropout", default=0, help="Dropout rate"
)
parser.add_argument(
    "--init_size", default=1e-3, help="Initial embeddings' scale"
)
parser.add_argument(
    "--learning_rate", default=1e-1, help="Learning rate"
)
parser.add_argument(
    "--gamma", default=0, help="Margin for distance-based losses"
)
parser.add_argument(
    "--bias", default="constant", help="Bias type (none for no bias)"  # , choices=["constant", "learn", "none"]
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
    "--subgraph_amount", default=10, type=int, help="The amount of subgraphs, that will be present in the ensemble."
)
parser.add_argument(
    "--subgraph_size_range", default="(0.6, 0.7)",
    help="A tuple (min, normal) with the relative subgraph sizes, where \n"
         "- \'min\' is the minimal relative subgraph size that will be "
         "used with Feature sampling.\n"
         "- \'normal\' is the normal relative subgraph size, that needs "
         "to be reached under normal conditions."
)
parser.add_argument(
    "--sampling_method", default="Entity", choices=["Entity", "Feature"],
    help="The sampling method, that will be used."
)
parser.add_argument(
    "--rho", default=1, type=float,
    help="Factor for Feature sampling, which specifies, how many relation names should be "
         "present in the subgraph, which is calculated by the formula \'rho * √|Relation Names|\'."
)
parser.add_argument(
    "--random_seed", default="42",
    help="The seed for random processes. Type \'random\' for a random seed."
)

parser.add_argument(
    "--entities_per_step", default=1, type=int,
    help="The amount of entities, that will be selected per sampling step."
)
parser.add_argument(
    "--enforcement", default=1, type=int,
    help="Enforcement level for ensuring the inclusion of all entities and relation names."
)

#   - Model parameters -
parser.add_argument(
    "--aggregation_method", default="average", choices=["max", "average"],
    help="The method by which all scores from the ensemble are aggregated."
)
parser.add_argument(
    "--model_dropout_factor", default=10, type=int,
    help="The multiplier for the first validation loss in order to disregard a model as diverged."
)

#   - System parameters -
parser.add_argument(
    "--logging", default="info", choices=['critical', 'error', 'warning', 'info', 'debug', 'data'],
    help="Determines the level of logging.\n"
         "- \'info\': Contains information about the progress.\n"
         "- \'debug\': Also contains information about variables, e.g. tensor sizes.\n"
         "- \'data\': Also contains embedding weights and other data from variables, "
         "which is printed directly to the log.\'"
)
parser.add_argument(
    "--wandb_project", default=False,
    help="Turn on logging of metrics via Weights&Biases and synchronize with the given project name."
)
parser.add_argument(
    "--only_valid", action='store_true', help="Only calculate metrics on the validation set."
)
parser.add_argument(
    "--no_sampling", action='store_true', help="Turn off sampling."
)
parser.add_argument(
    "--no_training", action='store_true', help="Turn off training."
)
parser.add_argument(
    "--no_progress_bar", action='store_true', help="Turn off all progress bars."
)
parser.add_argument(
    "--no_time_dependent_file_path", action='store_true', help="Turn off specifying the current time in file path, "
                                                               "when creating logs and other files."
)


def run_baseline():
    """
    Run the baseline configuration for the knowledge graph embedding model.

    This function sets up the arguments with a predefined baseline configuration and
    initiates the training process using these arguments.
    """

    # --- Setup args ---
    args = parser.parse_args()

    args.model = "TransE"
    args.dataset = "NELL-995-h100"
    args.max_epochs = 500
    args.rank = 32
    args.patience = 15
    args.valid = 5
    args.dtype = "single"
    args.batch_size = 500
    args.debug = False

    args.learning_rate = 0.001
    args.reg = 0.0
    args.optimizer = "Adam"
    args.neg_sample_size = -1
    args.bias = "learn"
    args.double_neg = False
    args.multi_c = False

    args.regularizer = "N3"
    args.init_size = 0.001
    args.gamma = 0.0
    args.dropout = 0.0

    args.no_progress_bar = False
    args.entities = None
    args.relation_names = None
    args.model_name = args.model
    args.wandb_project = "Experiments"

    # --- Training ---
    Constants.get_wandb(args.wandb_project)
    train(args)


def run_embedding(args):
    """
    Run the embedding model with the specified arguments.

    This function performs the following steps:
    1. Sets up Weights & Biases (wandb) logging if specified.
    2. Handles method-specific arguments for sampling, aggregation, and theta calculation.
    3. Configures the dataset and creates necessary directories.
    4. Sets up logging.
    5. Samples the graph if sampling is not turned off.
    6. Trains the model if training is not turned off.
    7. Logs the total time taken for the process.
    """

    time_process_start = time.time()

    # Setup wandb logging if needed
    Constants.get_wandb(args.wandb_project)

    args.sampling_method = util.handle_methods(args.sampling_method, "sampling")
    args.aggregation_method = util.handle_methods(args.aggregation_method, "aggregation")
    args.subgraph_size_range = util.handle_methods(args.subgraph_size_range, "size_range")

    dataset_out = ""
    if args.sampling_method == Constants.FEATURE_SAMPLING:
        dataset_out = (f"{args.dataset}_{args.sampling_method[2]}_N{args.subgraph_amount}_rho{args.rho}"
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

    logging.debug(f"Args: {args}")

    if type(args.random_seed) is str:
        try:
            args.random_seed = int(args.random_seed)
            logging.debug(f"Converting seed {args.random_seed} to int.")
        except ValueError:
            args.random_seed = random.randint(0, 2 ** 32 - 1)
            logging.debug(f"Selecting a random sampling seed: Seed {args.random_seed}")

    random.seed(args.random_seed)
    logging.info(f"Seed for random processes: {args.random_seed}")

    if not args.no_sampling:
        subsampling.sample_graph(info_directory, dataset_in, dataset_out_dir, args.sampling_method,
                                 subgraph_amount=args.subgraph_amount,
                                 subgraph_size_range=args.subgraph_size_range, entities_per_step=args.entities_per_step,
                                 rho=args.rho, no_progress_bar=args.no_progress_bar)

    error = False
    try:
        if not args.no_training:
            args.kge_models = util.get_embedding_methods(args.model)

            if Constants.LOG_WANDB:
                wandb.login()
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
    """
    Run the embedding model with a manual configuration.

    This function sets up the arguments with a predefined manual configuration,
    performs necessary setups such as directory creation and logging, and initiates
    the sampling and training processes based on the configuration.
    """
    # torch.autograd.set_detect_anomaly(True)
    time_process_start = time.time()

    # --- Setup parameters and args ---

    # dataset_in = "Debug"
    # dataset_in = "WN18RR"
    # dataset_in = "FB15K-237"
    dataset_in = "NELL-995-h100"
    # subgraph_amount = 30
    # subgraph_amount = 10
    subgraph_amount = 5
    # subgraph_size_range = (0.6, 0.7)
    subgraph_size_range = (0.2, 0.3)
    # rho = 2.0
    # rho = 1.0
    rho = 0.5
    model_dropout_factor = 10
    entities_per_step = 1

    args = argparse.Namespace(no_sampling=True, no_training=False, no_time_dependent_file_path=False,
                              no_progress_bar=False, subgraph_amount=subgraph_amount, wandb_project="False",
                              subgraph_size_range=subgraph_size_range, rho=rho, random_seed=42,
                              entities_per_step=entities_per_step, only_valid=False,
                              sampling_method=Constants.ENTITY_SAMPLING,
                              # sampling_method=Constants.FEATURE_SAMPLING,
                              # aggregation_method=Constants.MAX_SCORE_AGGREGATION,
                              aggregation_method=Constants.AVERAGE_SCORE_AGGREGATION,
                              model_dropout_factor=model_dropout_factor)

    # --- Setup wandb ---
    # args.wandb_project = "Experiments"
    Constants.get_wandb(args.wandb_project)

    # --- Setup directories ---
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

    # util_files.create_entity_and_relation_name_set_file(f"data\\{dataset_in}")

    if type(args.random_seed) is str:
        try:
            args.random_seed = int(args.random_seed)
            logging.debug(f"Converting seed {args.random_seed} to int.")
        except ValueError:
            args.random_seed = random.randint(0, 2 ** 32 - 1)
            logging.debug(f"Selecting a random sampling seed: Seed {args.random_seed}")

    random.seed(args.random_seed)
    logging.info(f"Seed for random processes: {args.random_seed}")

    # --- Sampling process ---
    if not args.no_sampling:
        subsampling.sample_graph(info_directory, dataset_in, dataset_out_dir, args.sampling_method,
                                 subgraph_amount=args.subgraph_amount, subgraph_size_range=args.subgraph_size_range,
                                 entities_per_step=args.entities_per_step, rho=args.rho,
                                 no_progress_bar=args.no_progress_bar)

    # allowed_kge_models = ({
    #     Constants.TRANS_E: [0],
    #     Constants.DIST_MULT: [1],
    #     Constants.COMPL_EX: [2],
    #     Constants.ROTAT_E: [3],
    #     Constants.ATT_E: [4]
    # })
    # ,)
    # Constants.ATT_H: []}

    # --- Training process ---
    error = False
    try:
        if not args.no_training:
            # args.kge_models = util.get_embedding_methods("{\"TransE\": [\"0:3\", 8], \"DistMult\": [\"7:9\"], "
            #                                              "\"ComplEx\": [2], \"AttE\": [3]}")
            args.kge_models = {
                Constants.DIST_MULT: ["0:9", 4],
                Constants.TRANS_E: list(range(20, 30)),
                Constants.SEPA: ['all'],
                Constants.COMPL_EX: list(range(10, 20)),
            }

            # general parameters
            args.max_epochs = 50
            args.rank = 32
            args.patience = 15
            args.valid = 5
            args.dtype = "single"
            args.batch_size = 1000
            args.debug = False

            # individually settable parameters
            args.learning_rate = {'TransE': 0.1, 'DistMult': 0.1,
                                  'ComplEx': 0.1, 'RotatE': 0.001,
                                  'AttE': 0.001, 'AttH': 0.001,
                                  'SEA': 0.001, 'SEPA': 0.001}
            args.reg = {'TransE': 0.0, 'DistMult': 0.05,
                        'ComplEx': 0.05,
                        'rest': 0.0}
            args.optimizer = {"TransE": "Adam", 'DistMult': "Adagrad",
                              'ComplEx': "Adagrad", 'RotatE': "SparseAdam",
                              'AttE': "Adam", 'AttH': "Adam",
                              'SEA': "Adam", 'SEPA': "Adagrad",
                              'Unified': "Adagrad"}

            args.neg_sample_size = {"TransE": -1, 'DistMult': -1,
                                    'ComplEx': -1, 'RotatE': 250,
                                    'AttE': -1, 'AttH': 250,
                                    'SEA': 250, 'SEPA': -1}
            args.bias = {'TransE': "learn", 'DistMult': "none",
                         'ComplEx': "none",
                         'AttE': "learn",
                         'SEA': "learn", 'SEPA': "none",
                         'Unified': "learn",
                         'rest': "none"}
            args.double_neg = {'TransE': True, 'DistMult': False,
                               'ComplEx': True,
                               'AttE': False, 'AttH': False,
                               'SEA': False, 'SEPA': False}
            args.multi_c = {'AttE': True, 'AttH': True,
                            'SEPA': True,
                            'Unified': True,
                            'rest': False}

            args.regularizer = {'all': "N3"}
            args.init_size = {'all': 0.001}
            args.dropout = {'all': 0}
            args.gamma = {'all': 0}

            logging.debug(f"Args: {args}")

            if Constants.LOG_WANDB:
                wandb.init(project=Constants.PROJECT_NAME, config=vars(args))
                wandb.login()

            run_unified_model.train(info_directory, args)

    except Exception:
        logging.error(traceback.format_exc())
        error = True

    time_process_end = time.time()

    # --- Final output regarding process ---
    if not error:
        logging.info(f"The entire process including sampling, training and testing took "
                     f"{util.format_time(time_process_start, time_process_end)}.")
    else:
        logging.info(f"The process ended with an error after "
                     f"{util.format_time(time_process_start, time_process_end)}")


if __name__ == "__main__":
    # Function to run via command prompt
    run_embedding(parser.parse_args())

    # Function to run manual via IDE
    # run_embedding_manual()

    # Function to run baseline
    # run_baseline()
