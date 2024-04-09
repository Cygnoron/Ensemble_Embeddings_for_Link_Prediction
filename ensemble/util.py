import logging
import os
import pickle
import random
from collections import defaultdict

import models
from datasets.kg_dataset import KGDataset
from ensemble import Constants


def create_entity_and_relation_name_set_file(dataset):
    """
    Calculates and writes entity and relation name sets for dataset to csv files

    :param dataset: Name of the input dataset
    """
    logging.debug(f"Creating csv files containing the entity and relation name sets for dataset {dataset}")

    with (open(os.path.abspath(f"data\\{dataset}\\train.pickle"), 'rb') as pickle_file,
          open(f"data\\{dataset}\\entity_set.csv", 'w') as entity_set_file,
          open(f"data\\{dataset}\\relation_name_set.csv", 'w') as relation_name_set_file):

        logging.debug("Loading data from .pickle file")
        # load data from train.pickle file
        data = pickle.load(pickle_file)
        total_triples = len(data)

        logging.debug("Creating entity and relation name sets")
        # calculate entity and relation name sets
        entity_set, relation_name_set = get_unique_triple_ids(data, h=True, r=True, t=True)

        # sort entities by total amount of triples
        sorted_entities = sorted(entity_set.items(), key=lambda x: len(x[1]), reverse=True)

        logging.debug("Writing entity set to csv file")
        # write header for entity set file
        entity_set_file.write(f"entity_id;total_amount_of_triples;relative_amount_of_triples\n")
        # write entity id and amount of triples for each entity
        for entity, triples in sorted_entities:
            entity_set_file.write(f"{entity};{len(triples)};{round(len(triples) / total_triples * 100, 3)}%\n")
        entity_set_file.write(f"total_triples;{total_triples};100%")

        # sort relation names by total amount of triples
        sorted_relation_names = sorted(relation_name_set.items(), key=lambda x: len(x[1]), reverse=True)

        logging.debug("Writing relation name set to csv file")
        # write header for relation name file
        relation_name_set_file.write(f"relation_name_id;total_amount_of_triples;relative_amount_of_triples\n")
        # write relation name id and amount of triples for each relation name
        for relation_name, triples in sorted_relation_names:
            relation_name_set_file.write(
                f"{relation_name};{len(triples)};{round(len(triples) / total_triples * 100, 3)}%\n")
        relation_name_set_file.write(f"total_triples;{total_triples};100%")


def get_unique_triple_ids(dataset, h=False, r=False, t=False):
    """
    The get_unique_triple_ids function takes in a dataset and returns a dictionary of unique entities or relations
        with their corresponding indices in the dataset.

    :param dataset: The dataset, which was loaded from a .pickle file
    :param h: Indicate whether to check for head entities
    :param r: Indicate whether to check for relation names
    :param t: Indicate whether to check for tail entities
    :return: Depending on h, r, t return the respective dictionaries
    """

    logging.debug(f"Calculating sets of unique ids for entities and relation names")
    entity_ids = {}
    relation_name_ids = {}
    for index, triple in enumerate(dataset):
        head, relation_name, tail = triple
        # If entity or relation name is not in triple_ids, add it and then append index to list
        if h:
            if head not in entity_ids:
                entity_ids[head] = set()
            entity_ids[head].add(index)
        if r:
            if relation_name not in relation_name_ids:
                relation_name_ids[relation_name] = set()
            relation_name_ids[relation_name].add(index)
        if t:
            if tail not in entity_ids:
                entity_ids[tail] = set()
            entity_ids[tail].add(index)

    if (h or t) and r:
        # logging.debug(f"Entity ids:\n{entity_ids}")
        # logging.debug(f"Relation name ids:\n{relation_name_ids}")
        return entity_ids, relation_name_ids
    elif (h or t) and not r:
        # logging.debug(f"Entity ids:\n{entity_ids}")
        return entity_ids
    elif r and not (h or t):
        # logging.debug(f"Relation name ids:\n{relation_name_ids}")
        return relation_name_ids


def unique_unsorted(arr):
    """
    The unique_unsorted function takes an array and returns a list which contains the first occurrence of each element in that order.

    :param arr: The array, which will be processed
    :return: List containing the first occurrence of each element
    """
    unique_list = []
    # iterate through input array
    for element in arr:
        # if element is not already contained in unique_list, append it, else continue
        if element not in unique_list:
            unique_list.append(element)

    return unique_list


def inverse_dict(dictionary):
    """
        Return an inverted dictionary where keys become values and values become keys.

        Args:
            dictionary (dict): The input dictionary to be inverted.

        Returns:
            dict: An inverted dictionary where the values of the input dictionary become
                  keys, and the keys become values. Each key in the input dictionary maps
                  to a list of values in the output dictionary.

        Example:
            >>> dict_a = {0: 'TransE', 1: 'TransE', 2: 'RotatE', 3: 'AttE', 4: 'RotatE', 5: 'ComplEx'}
            >>> inverse_dict(dict_a)
            {'TransE': [0, 1], 'RotatE': [2, 4], 'AttE':[3], 'ComplEx': [5]}
        """
    inverse_dictionary = defaultdict(list)

    for key, value in dictionary.items():
        inverse_dictionary[value].append(key)

    return dict(inverse_dictionary)


def assign_model_to_subgraph(kge_models, args):
    """
    Assign embedding models to subgraphs based on the given dictionary of kge_models.

    Args:
        kge_models (dict): A dictionary where keys are embedding models (constants) and values are lists of subgraphs
                           to be embedded using the corresponding model.
        args: Additional arguments containing dataset_directory and other necessary configurations.

    Returns:
        dict: A mapping of subgraphs to their assigned embedding models.

    Examples:
        >>> kge_models = {Constants.TRANS_E: [0, 1], Constants.DIST_MULT: [2, 3], Constants.ROTAT_E: [],
            Constants.COMPL_EX: [], Constants.ATT_E: [], Constants.ATT_H: [5]}
            ->  subgraph 0, 1 embedded by TRANS_E, subgraph 2,3 by DIST_MULT, subgraph 5 by ATT_H,
                other subgraphs randomly

        >>> kge_models = {Constants.TRANS_E: [0, 1,"rest"], Constants.DIST_MULT: [2, 3], Constants.ROTAT_E: [],
            Constants.COMPL_EX: [], Constants.ATT_E: ["rest"], Constants.ATT_H: [5]}
            ->  subgraph 0, 1 embedded by TRANS_E, subgraph 2,3 by DIST_MULT, subgraph 5 by ATT_H,
                other subgraphs by TRANS_E or ATT_E randomly

        >>> kge_models = {Constants.TRANS_E: [0, 1,"rest"], Constants.DIST_MULT: [2, 3], Constants.ROTAT_E: ["all"],
            Constants.COMPL_EX: [], Constants.ATT_E: ["rest"], Constants.ATT_H: [5]}
            -> All subgraphs are embedded by ROTAT_E
        """
    # TODO include possibility for ration of embedding methods?
    #   e.g. Constants.TransE: [0.2] -> 20% of subgraphs with TransE, remaining subgraphs random

    # Initial setup
    subgraph_embedding_mapping = {}
    kge_models_adjusted = list(kge_models.keys()).copy()

    logging.debug("Checking if 'all' or 'rest' was specified")
    for embedding_model in list(kge_models.keys()):
        # Handle case "all" if specified in kge_models
        if kge_models[embedding_model] == ["all"]:
            logging.debug(f"'all' was found for embedding model {embedding_model}")
            logging.info(f"All subgraphs will be embedded by {kge_models_adjusted[0]}")
            # Clear mapping, if some subgraphs were already mapped (-> "all" overrides all other specifications)
            subgraph_embedding_mapping.clear()

            for subgraph in os.listdir(os.path.abspath(args.dataset_directory)):
                # Ignore folders which don't contain a subgraph
                if "sub_" not in subgraph:
                    continue

                # Get subgraph number and set mapping and necessary args
                subgraph_num = int(subgraph.split(sep='_')[1])  # subgraph = sub_XXX
                subgraph_embedding_mapping[subgraph_num] = embedding_model
                args.model = embedding_model
                args.subgraph = subgraph

                logging.info(f"Setting {args.model} as embedding method for subgraph {args.subgraph}.")

            logging.info(f"Mapping from embedding methods to subgraphs: {inverse_dict(subgraph_embedding_mapping)}")
            return subgraph_embedding_mapping

        # Handle case "rest" if specified in kge_models
        elif "rest" in kge_models[embedding_model]:
            logging.debug(f"'rest' was found for embedding model {embedding_model}")

            if len(kge_models_adjusted) == len(list(kge_models.keys())):
                # Clear embedding methods to choose from
                kge_models_adjusted.clear()
            # Only add embedding methods with "rest" included to list of allowed embedding methods
            kge_models_adjusted.append(embedding_model)

    # Iterate through kge_models to get specific mappings
    logging.debug("Checking for direct mappings")
    for embedding_model in kge_models:
        for subgraph in kge_models[embedding_model]:
            if subgraph == "rest":
                continue
            # Set specific embedding method for the subgraph, if specified
            subgraph_embedding_mapping[subgraph] = embedding_model
    # List containing only specific mappings for subgraphs
    mapped_subgraphs = list(subgraph_embedding_mapping.keys())

    logging.info(f"Subgraphs with fixed embedding: {subgraph_embedding_mapping}\n\t\t"
                 f"Remaining subgraphs will be embedded by random embedding methods {kge_models_adjusted}")

    for subgraph in os.listdir(os.path.abspath(args.dataset_directory)):
        # Ignore folders which don't contain a subgraph
        if "sub_" not in subgraph:
            continue

        # Get subgraph number and set mapping and necessary args
        subgraph_num = int(subgraph.split(sep='_')[1])  # subgraph = sub_XXX
        args.subgraph = subgraph

        # Case: select specified embedding method
        if subgraph_num in mapped_subgraphs:
            args.model = subgraph_embedding_mapping[subgraph_num]
            logging.info(f"Found mapping, using embedding method {args.model} for subgraph {subgraph}")

        # Case: select random embedding method
        else:
            args.model = random.choice(kge_models_adjusted)
            subgraph_embedding_mapping[subgraph_num] = args.model
            logging.info(f"Randomly selected embedding method {args.model} for subgraph {subgraph}")

    logging.info(f"Mapping from embedding methods to subgraphs: {inverse_dict(subgraph_embedding_mapping)}")
    return subgraph_embedding_mapping


def setup_logging(info_directory: str, log_file_name: str, logging_level="info"):
    """
    Set up logging configuration.

    Args:
        info_directory (str): Directory where log files will be saved.
        log_file_name (str): Name of the log file.
        logging_level (str): Logging level for the logger, will be converted to logging class if available.
    """
    # logging levels:
    #  CRITICAL - 50
    #  ERROR - 40
    #  WARNING - 30
    #  INFO - 20
    #  DEBUG - 10
    #  DATA - 5
    logging.addLevelName(Constants.DATA_LEVEL, "DATA")
    logging_level = logging_level.lower()
    if logging_level == "critical":
        logging_level = logging.CRITICAL
    elif logging_level == "error":
        logging_level = logging.ERROR
    elif logging_level == "warning":
        logging_level = logging.WARNING
    elif logging_level == "info":
        logging_level = logging.INFO
    elif logging_level == "debug":
        logging_level = logging.DEBUG
    elif logging_level == "data":
        logging_level = Constants.DATA_LEVEL
    else:
        raise ValueError("Invalid logging level")


    # file logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging_level,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(info_directory, log_file_name),
        filemode="w"
    )

    # setup log for subgraph sampling
    console = logging.StreamHandler()
    console.setLevel(logging_level)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.log(Constants.DATA_LEVEL, "### Logging data ###")
    logging.debug(f"### Debug messages are enabled ###")
    logging.info(f"### Saving logs in: {info_directory} ###")


def get_dataset_name(dataset: str):
    """
        Extract the original dataset name from a possibly sampled dataset name.

        Args:
            dataset (str): The dataset name, which may have been sampled.

        Returns:
            str: The original name of the dataset.
        """
    if Constants.ENTITY_SAMPLING[2] in dataset:
        dataset_name = dataset.split(f"_{Constants.ENTITY_SAMPLING[2]}")[0]
        logging.debug(f"Given dataset was sampled by {Constants.ENTITY_SAMPLING[1]}, "
                      f"returning original name {dataset_name}")
        return dataset_name
    elif Constants.FEATURE_SAMPLING[2] in dataset:
        dataset_name = dataset.split(f"_{Constants.FEATURE_SAMPLING[2]}")[0]
        logging.debug(f"Given dataset was sampled by {Constants.FEATURE_SAMPLING[1]}, "
                      f"returning original name {dataset_name}")
        return dataset_name
    else:
        logging.debug(f"Given dataset has not been sampled. Returning dataset name {dataset}")
        return dataset


def generate_general_embeddings(general_dataset: str, args):
    logging.debug("Creating dataset and model for later unified embeddings.")
    # load data
    dataset = KGDataset(os.path.abspath(f"data\\{general_dataset}"), args.debug)
    args.sizes = dataset.get_shape()
    model_to_use = Constants.ATT_E
    logging.debug(f"Using {model_to_use} for general embedding")
    args.model = model_to_use

    # create model
    logging.debug(f"Dataset: {dataset}")
    model = getattr(models, args.model)(args)
    device = "cuda"
    model.to(device)

    return model
