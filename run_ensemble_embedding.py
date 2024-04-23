import logging
import os
import traceback

from ensemble import Constants, util_files, util, run, subsampling

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

if __name__ == "__main__":
    # dataset_in = "Debug"
    dataset_in = "WN18RR"
    # dataset_in = "YAGO3-10"
    # dataset_in = "NELL-995"
    subgraph_amount = 10
    subgraph_size_range = (0.1, 0.7)
    sampling_method = Constants.ENTITY_SAMPLING
    # sampling_method = Constants.FEATURE_SAMPLING
    relation_name_amount = 0.5
    max_indices_per_step = "max"

    subgraph_size_range_list = [subgraph_size_range]
    # for i in range(25, 70, 5):
    #     subgraph_size_range_list.append((i / 100, 0.7))

    for subgraph_size_range in subgraph_size_range_list:
        # try:

        if sampling_method == Constants.FEATURE_SAMPLING:
            dataset_out = (f"{dataset_in}_{sampling_method[2]}_N{subgraph_amount}_rho{relation_name_amount}"
                           f"_min{subgraph_size_range[0]}")
        else:
            dataset_out = f"{dataset_in}_{sampling_method[2]}_N{subgraph_amount}_min{subgraph_size_range[0]}"
        dataset_out_dir = f"data\\{dataset_out}"

        info_directory = os.path.abspath(f"{dataset_out_dir}")
        util_files.check_directory(info_directory)
        info_directory = os.path.abspath(f"{dataset_out_dir}\\results")
        util_files.check_directory(info_directory)

        util.setup_logging(info_directory, "Ensemble_Embedding_for_Link_Prediction.log",
                           logging_level="debug")

        # util_files.csv_to_file("D:\\Masterarbeit\\Software\\Ensemble_Embedding_for_Link_Prediction\\"
        #                        "ensemble\\Random_Triples.csv",
        #                        "D:\\Masterarbeit\\Software\\Ensemble_Embedding_for_Link_Prediction\\"
        #                        "data\\Debug\\train", only_unique=True)
        #
        # util_files.pickle_to_csv(f"D:\\Masterarbeit\\Software\\Ensemble_Embedding_for_Link_Prediction\\"
        #                          f"data\\{dataset_in}\\train.pickle",
        #                          f"D:\\Masterarbeit\\Software\\Ensemble_Embedding_for_Link_Prediction\\"
        #                          f"data\\{dataset_in}\\train_readable.csv", ';')
        #
        # util.create_entity_and_relation_name_set_file(f"data\\{dataset_in}")

        # for dataset in os.scandir("data"):
        #     print(dataset.name)
        #     util.create_entity_and_relation_name_set_file(dataset.name)
        #
        # subsampling.sample_graph(info_directory, dataset_in, dataset_out_dir, sampling_method,
        #                          subgraph_amount=subgraph_amount, subgraph_size_range=subgraph_size_range,
        #                          relation_name_amount=relation_name_amount,
        #                          max_indices_per_step=max_indices_per_step)

        # create .graphml files for visualizing the subgraphs
        # plotting.create_graphml(info_directory, os.path.abspath(dataset_out_dir))

        # allowed_kge_models = {Constants.TRANS_E: [1,2], Constants.DIST_MULT: [3, 4], Constants.ROTAT_E: [5, 6],
        #                       Constants.COMPL_EX: [7, 8], Constants.ATT_E: [9, 0]}
        # allowed_kge_models = {Constants.TRANS_E: [0, 1], Constants.DIST_MULT: [2], Constants.ROTAT_E: [9],
        #                       Constants.COMPL_EX: [3, 4], Constants.ATT_E: [5, 6, 7, 8]}
        # allowed_kge_models = {Constants.TRANS_E: [0, 1], Constants.DIST_MULT: [2, 3], Constants.ROTAT_E: ["rest"],
        #                       Constants.COMPL_EX: [], Constants.ATT_E: ["rest"], Constants.ATT_H: [5]}

        # allowed_kge_models = [{Constants.TRANS_E: [], Constants.DIST_MULT: [], Constants.ROTAT_E: [],
        #                        Constants.COMPL_EX: [], Constants.ATT_E: [], Constants.ATT_H: ["all"]}]

        allowed_kge_models = [{Constants.TRANS_E: [0], Constants.DIST_MULT: [1], Constants.ROTAT_E: [2],
                               Constants.COMPL_EX: [3], Constants.ATT_E: [4]}]
        # # for i in range(1):
        # #     run.own_train(info_directory, dataset=dataset_out, dataset_directory=dataset_out_dir,
        # #                     learning_rate=0.01, kge_models=allowed_kge_models, max_epochs=3, batch_size=100,
        # #                     rank=32, debug=False)
        #
        # allowed_kge_models = [{Constants.TRANS_E: []}, {Constants.DIST_MULT: []}, {Constants.ROTAT_E: []},
        #                       # {Constants.COMPL_EX: []}, {Constants.ATT_E: []}, {Constants.ATT_H: []}]
        #                       {Constants.COMPL_EX: []}, {Constants.ATT_E: []}]
        # #
        for models in allowed_kge_models:
            try:
                run.train(info_directory, dataset=dataset_out, dataset_directory=dataset_out_dir,
                          learning_rate=0.01, kge_models=models, max_epochs=2, batch_size=800,
                          rank=32, debug=False)
            except RuntimeError:
                logging.error(traceback.format_exc())
