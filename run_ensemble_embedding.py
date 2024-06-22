import argparse
import logging
import os
import time
import traceback

import wandb
from ensemble import Constants, util_files, util, run, subsampling
from run import train

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

parser = argparse.ArgumentParser(
    description="Ensemble methods for Knowledge Graph Embedding"
)
# --- Arguments for embedding models ---
parser.add_argument(
    "--dataset", default="WN18RR", help="Knowledge Graph dataset"
)
# parser.add_argument(
#     "--model", default="RotE", choices=all_models, help="Knowledge Graph embedding model"
# )
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
# --- Parameters for ensemble methods ---
parser.add_argument(
    "--no_sampling", action='store_true', help="Turn off sampling, if subgraphs already exist"
)
parser.add_argument(
    "--no_training", action='store_true', help="Turn off training, if only sampling should be done"
)
parser.add_argument(
    "--no_progress_bar", action='store_true', help="Turn off all progress bars"
)
parser.add_argument(
    "--no_time_dependent_file_path", action='store_true', help="Specify current time in file path, when "
                                                               "creating logs and other files"
)
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
    "--rho", default=1, type=float,
    help="Factor for Feature sampling, which specifies, how many relation names should be "
         "present in the subgraph, which is calculated by the formula \'rho * √|Relation Names|\'"
)
parser.add_argument(
    "--sampling_method", default="Entity", choices=["Entity", "Feature"],
    help="The sampling method, that should be used"
)
parser.add_argument(
    "--aggregation_method", default="average", choices=["max", "average", "attention"],
    help="The method by which all scores from the ensemble are combined."
)
parser.add_argument(
    "--model_dropout_factor", default=1000, type=int,
    help="The method by which all scores from the ensemble are combined."
)
parser.add_argument(
    "--theta_method", default="regular", choices=["no", "regular", "reversed", "relation", "multiplied"],
    help="The method by which the context vectors (cv) are used in order to calculate attention values for the unified "
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
parser.add_argument(
    '--model', type=str, default="{TransE:[\'all\']}",
    help='JSON string of the mapping from embedding methods to subgraphs.\n'
         '- \'all\' in a mapping sets all subgraphs to this method\n'
         '- \'rest\' in a mapping allows all unmapped subgraphs to be embedded by this method. '
         'If no \'rest\' was specified, all subgraphs can be embedded by all given embedding methods.'
)
parser.add_argument(
    "--logging", default="info", help="Determines the level of logging.\n"
                                      "- \'info\': Contains information about the progress"
                                      "- \'debug: Also contains information about variables, e.g. tensor sizes\'"
                                      "- \'data: Also contains embedding weights and other data from variables, "
                                      "which is printed directly to the log\'"
)


def run_baseline(args):
    train(args)


def run_embedding(args):
    time_process_start = time.time()

    args.sampling_method = util.handle_methods(args.sampling_method, "sampling")
    args.aggregation_method = util.handle_methods(args.aggregation_method, "aggregation")
    args.theta_calculation = util.handle_methods(args.theta_method, "theta")

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
                                 relation_name_amount=args.rho, no_progress_bar=args.no_progress_bar)

    error = False
    try:
        if not args.no_training:
            args.kge_models = util.get_embedding_methods(args.model)

            wandb.init(project=Constants.PROJECT_NAME, config=args)

            run.train(info_directory, args)

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
    # dataset_in = "YAGO3-10"
    # dataset_in = "NELL-995"
    subgraph_amount = 3
    subgraph_size_range = (0.2, 0.25)
    relation_name_amount = 0.5
    model_dropout_factor = 10

    args = argparse.Namespace(no_sampling=True, no_training=False, no_time_dependent_file_path=True, wandb_log=False,
                              no_progress_bar=False, subgraph_amount=subgraph_amount,
                              subgraph_size_range=subgraph_size_range, relation_name_amount=relation_name_amount,
                              sampling_method=Constants.ENTITY_SAMPLING,
                              aggregation_method=Constants.AVERAGE_SCORE_AGGREGATION,
                              theta_calculation=Constants.REGULAR_THETA, model_dropout_factor=model_dropout_factor)

    subgraph_size_range_list = [subgraph_size_range]
    # for i in range(25, 70, 5):
    #     subgraph_size_range_list.append((i / 100, 0.7))

    for subgraph_size_range in subgraph_size_range_list:
        time_process_start = time.time()

        dataset_out = ""
        if args.sampling_method == Constants.FEATURE_SAMPLING:
            dataset_out = (f"{dataset_in}_{args.sampling_method[2]}_N{subgraph_amount}_rho{args.rho}"
                           f"_min{subgraph_size_range[0]}_max{subgraph_size_range[1]}")
        elif args.sampling_method == Constants.ENTITY_SAMPLING:
            dataset_out = (f"{dataset_in}_{args.sampling_method[2]}_N{subgraph_amount}"
                           f"_min{subgraph_size_range[0]}_max{subgraph_size_range[1]}")
        dataset_out_dir = f"data\\{dataset_out}"

        args.dataset = dataset_out
        args.dataset_dir = dataset_out_dir

        info_directory = os.path.abspath(f"{dataset_out_dir}")
        util_files.check_directory(info_directory)
        info_directory = util_files.get_info_directory_path(dataset_out_dir, args)

        args.info_directory = info_directory

        util.setup_logging(info_directory, "Ensemble_Embedding_for_Link_Prediction.log",
                           logging_level="critical")

        # --- if a new debugging dataset was created ---

        # util_files.csv_to_file("D:\\Masterarbeit\\Software\\Ensemble_Embedding_for_Link_Prediction\\"
        #                        "ensemble\\Random_Triples.csv",
        #                        "D:\\Masterarbeit\\Software\\Ensemble_Embedding_for_Link_Prediction\\"
        #                        "data\\Debug\\train", only_unique=True)
        #
        # util_files.pickle_to_csv(f"D:\\Masterarbeit\\Software\\Ensemble_Embedding_for_Link_Prediction\\"
        #                          f"data\\{dataset_in}\\train.pickle",
        #                          f"D:\\Masterarbeit\\Software\\Ensemble_Embedding_for_Link_Prediction\\"
        #                          f"data\\{dataset_in}\\train_readable.csv", ';')

        # --- create files for insights in dataset ---

        # util.create_entity_and_relation_name_set_file(f"data\\{dataset_in}")

        # for dataset in os.scandir("data"):
        #     print(dataset.name)
        #     util.create_entity_and_relation_name_set_file(dataset.name)

        # --- sampling process ---
        if not args.no_sampling:
            subsampling.sample_graph(info_directory, dataset_in, dataset_out_dir, args.sampling_method,
                                     subgraph_amount=args.subgraph_amount, subgraph_size_range=args.subgraph_size_range,
                                     relation_name_amount=args.rho, no_progress_bar=args.no_progress_bar)

        # sampling_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120,
        #                   150, 200, 300, 400, 500]
        # for i in sampling_sizes:
        #     subsampling.sample_graph(info_directory, dataset_in, dataset_out_dir, sampling_method,
        #                              subgraph_amount=subgraph_amount, subgraph_size_range=subgraph_size_range,
        #                              relation_name_amount=relation_name_amount,
        #                              entities_per_step=i)

        # --- create .graphml files for dataset visualization ---

        # create .graphml files for visualizing the subgraphs
        # plotting.create_graphml(info_directory, os.path.abspath(dataset_out_dir))

        # --- setup for model training ---

        # allowed_kge_models = {Constants.TRANS_E: [1,2], Constants.DIST_MULT: [3, 4], Constants.ROTAT_E: [5, 6],
        #                       Constants.COMPL_EX: [7, 8], Constants.ATT_E: [9, 0]}
        # allowed_kge_models = {Constants.TRANS_E: [0, 1], Constants.DIST_MULT: [2], Constants.ROTAT_E: [9],
        #                       Constants.COMPL_EX: [3, 4], Constants.ATT_E: [5, 6, 7, 8]}
        # allowed_kge_models = {Constants.TRANS_E: [0, 1], Constants.DIST_MULT: [2, 3], Constants.ROTAT_E: ["rest"],
        #                       Constants.COMPL_EX: [], Constants.ATT_E: ["rest"], Constants.ATT_H: [5]}

        allowed_kge_models = [{Constants.TRANS_E: [0, "rest"], Constants.DIST_MULT: [1], Constants.ROTAT_E: [2],
                               Constants.COMPL_EX: [3, "all"], Constants.ATT_E: [4, "rest"], Constants.ATT_H: [5]}]

        # allowed_kge_models = [{Constants.TRANS_E: [1, 0, "rest"], Constants.DIST_MULT: [13], Constants.ROTAT_E: [21],
        #                        Constants.COMPL_EX: [2, 3, "rest"], Constants.ATT_E: [50]}]

        # allowed_kge_models = [{Constants.TRANS_E: [], Constants.DIST_MULT: [], Constants.ROTAT_E: [],
        #                        Constants.COMPL_EX: [], Constants.ATT_E: [], Constants.ATT_H: []}]

        # --- training process ---

        # for i in range(1):
        #     run.own_train(info_directory, dataset=dataset_out, dataset_directory=dataset_out_dir,
        #                     learning_rate=0.01, kge_models=allowed_kge_models, max_epochs=3, batch_size=100,
        #                     rank=32, debug=False)

        # allowed_kge_models = [{Constants.TRANS_E: []}, {Constants.DIST_MULT: []}, {Constants.ROTAT_E: []},
        #                       {Constants.COMPL_EX: []}, {Constants.ATT_E: []}, {Constants.ATT_H: []}]

        # allowed_kge_models = [allowed_kge_models[0]]

        error = False
        for models in allowed_kge_models:
            try:
                if not args.no_training:
                    args.kge_models = models

                    args.max_epochs = 50
                    args.batch_size = 2500
                    args.rank = 32
                    args.learning_rate = 0.1
                    args.reg = 0.05
                    args.regularizer = "N3"
                    args.optimizer = "Adagrad"
                    args.patience = 15
                    args.valid = 5
                    args.neg_sample_size = -1
                    args.dropout = 0
                    args.init_size = 0.001
                    args.gamma = 0
                    args.bias = "learn"
                    args.dtype = "double"
                    args.debug = False
                    args.multi_c = True
                    args.double_neg = True

                    if Constants.LOG_WANDB:
                        wandb.init(project=Constants.PROJECT_NAME, config=vars(args))
                        wandb.login()

                    run.train(info_directory, args)

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

    # Function to run baseline
    args = parser.parse_args()

    args.model = "ComplEx"
    args.dataset = "NELL-995"
    args.rank = 32
    args.regularizer = "N3"
    args.reg = 0.05
    args.optimizer = "Adagrad"
    args.max_epochs = 500
    args.patience = 15
    args.valid = 1
    args.batch_size = 2500
    args.neg_sample_size = -1
    args.init_size = 0.001
    args.learning_rate = 0.001
    args.gamma = 0.0
    args.bias = "none"
    args.dtype = "single"
    args.debug = True
    args.double_neg = False

    # run_baseline(args)

    # Function to run manual via IDE
    run_embedding_manual()
