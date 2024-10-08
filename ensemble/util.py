import copy
import json
import logging
import os
import random
from collections import defaultdict

import torch

from ensemble import Constants
from ensemble.Constants import EMBEDDING_METHODS


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

            >>> dict_b = {'TransE': [0, 1], 'RotatE': [2, 4], 'AttE':[3], 'ComplEx': [5]}
            >>> inverse_dict(dict_b)
            {0: 'TransE', 1: 'TransE', 2: 'RotatE', 3: 'AttE', 4: 'RotatE', 5: 'ComplEx'}
        """
    inverse_dictionary = defaultdict(list)

    for key, value in dictionary.items():
        if isinstance(value, list):
            for item in value:
                inverse_dictionary[item].append(key)
        else:
            inverse_dictionary[value].append(key)

    return dict(inverse_dictionary)


def assign_model_to_subgraph(kge_models, args):
    """
    Assign embedding embedding_models to subgraphs based on the given dictionary of kge_models.

    Args:
        kge_models (dict): A dictionary where keys are embedding embedding_models (constants) and values are lists of subgraphs
                           to be embedded using the corresponding model.
        args: Additional arguments containing dataset_directory and other necessary configurations.

    Returns:
        dict: A mapping of subgraphs to their assigned embedding embedding_models.

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
    # Initial setup
    subgraph_embedding_mapping = {}
    kge_models_adjusted = list(kge_models.keys()).copy()

    logging.debug(f"kge_models before checking for illegal assignments:\n{format_dict(kge_models)}")
    # check for illegal subgraph assignment and remove if any were found
    for embedding_model in list(kge_models.keys()):
        for assignment in kge_models[embedding_model].copy():
            if type(assignment) is int:
                if assignment >= args.subgraph_amount:
                    logging.debug(f"Removed illegal assignment \"{assignment}\"")
                    kge_models[embedding_model].remove(assignment)
            elif type(assignment) is str:
                if assignment != "all" and assignment != "rest" and ":" not in assignment:
                    logging.debug(f"Removed illegal assignment \"{assignment}\"")
                    kge_models[embedding_model].remove(assignment)
    logging.debug(f"kge_models after checking for illegal assignments:\n{format_dict(kge_models)}")

    logging.debug("Checking if 'all' or 'rest' was specified")
    for embedding_model in list(kge_models.keys()):
        # Handle case "all" if specified in kge_models
        if "all" in kge_models[embedding_model]:
            logging.debug(f"'all' was found for embedding model {embedding_model}")
            logging.info(f"All subgraphs will be embedded by {embedding_model}")
            # Clear mapping, if some subgraphs were already mapped (-> "all" overrides all other specifications)
            subgraph_embedding_mapping.clear()

            for subgraph in os.listdir(os.path.abspath(args.dataset_dir)):
                # Ignore folders which don't contain a subgraph
                if "sub_" not in subgraph:
                    continue

                # Get subgraph number and set mapping and necessary args
                subgraph_num = int(subgraph.split(sep='_')[1])  # subgraph = sub_XXX
                subgraph_embedding_mapping[subgraph_num] = embedding_model
                # args.model = embedding_model
                # args.subgraph = subgraph

                # logging.info(f"Setting {args.model} as embedding method for subgraph {args.subgraph}.")

            logging.info(f"Mapping from embedding methods to subgraphs:\n"
                         f"{format_dict(inverse_dict(subgraph_embedding_mapping))}")
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
            elif isinstance(subgraph, str):
                if ":" in subgraph:
                    start_end = subgraph.split(":")
                    for subgraph_num in range(int(start_end[0]), int(start_end[1]) + 1):
                        if subgraph_num not in subgraph_embedding_mapping.keys():
                            subgraph_embedding_mapping[subgraph_num] = embedding_model
                        else:
                            logging.info(f"The mapping from subgraph to embedding model "
                                         f"({subgraph_num} -> {embedding_model}) could not be applied due to the existing "
                                         f"mapping ({subgraph_num} -> {subgraph_embedding_mapping[subgraph_num]}).")
            else:
                # Set specific embedding class_name for the subgraph, if specified
                if subgraph not in subgraph_embedding_mapping.keys():
                    subgraph_embedding_mapping[subgraph] = embedding_model
                else:
                    logging.info(f"The mapping from subgraph to embedding model ({subgraph} -> {embedding_model}) "
                                 f"could not be applied due to the existing mapping "
                                 f"({subgraph} -> {subgraph_embedding_mapping[subgraph]}).")
    # List containing only specific mappings for subgraphs

    mapped_subgraphs = list(subgraph_embedding_mapping.keys())

    logging.info(f"Subgraphs with fixed embedding:\n{format_dict(subgraph_embedding_mapping)}\n\t\t"
                 f"Remaining subgraphs will be embedded by random embedding methods {kge_models_adjusted}")

    for subgraph in os.listdir(os.path.abspath(args.dataset_dir)):
        # Ignore folders which don't contain a subgraph
        if "sub_" not in subgraph:
            continue

        # Get subgraph number and set mapping and necessary args
        subgraph_num = int(subgraph.split(sep='_')[1])  # subgraph = sub_XXX
        # args.subgraph = subgraph

        # Case: select specified embedding class_name
        if subgraph_num in mapped_subgraphs:
            # args.model = subgraph_embedding_mapping[subgraph_num]
            logging.info(f"Found mapping, using embedding method {subgraph_embedding_mapping[subgraph_num]} "
                         f"for subgraph {subgraph}")

        # Case: select random embedding class_name
        else:
            # args.model =
            subgraph_embedding_mapping[subgraph_num] = random.choice(kge_models_adjusted)
            logging.info(f"Randomly selected embedding method {subgraph_embedding_mapping[subgraph_num]} "
                         f"for subgraph {subgraph}")

    dict(sorted(subgraph_embedding_mapping.items()))

    list(subgraph_embedding_mapping.keys()).sort()

    logging.info(f"Mapping from embedding methods to subgraphs:\n{format_dict(inverse_dict(subgraph_embedding_mapping))}")
    return subgraph_embedding_mapping


def setup_logging(info_directory: str, log_file_name: str, logging_level="info"):
    """
    Set up logging configuration.

    Args:
        info_directory (str): Directory where log files will be saved.
        log_file_name (str): Name of the log file.
        logging_level (str): Logging level for the logger, will be converted to logging class (e.g. logging.INFO) if
         available.
    """

    logging.addLevelName(Constants.DATA_LEVEL_LOGGING, "DATA")
    logging_level = logging_level.lower()
    if logging_level == "critical":
        #  CRITICAL - 50
        logging_level = logging.CRITICAL
    elif logging_level == "error":
        #  ERROR - 40
        logging_level = logging.ERROR
    elif logging_level == "warning":
        #  WARNING - 30
        logging_level = logging.WARNING
    elif logging_level == "info":
        #  INFO - 20
        logging_level = logging.INFO
    elif logging_level == "debug":
        #  DEBUG - 10
        logging_level = logging.DEBUG
    elif logging_level == "data":
        #  DATA - 5
        logging_level = Constants.DATA_LEVEL_LOGGING
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
    logging.info(f"### Saving logs in: {info_directory} ###")
    logging.debug(f"### Debug messages are enabled ###")
    logging.log(Constants.DATA_LEVEL_LOGGING, "### Logging data ###")


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


def format_time(time_total_start, time_total_end, divisor=1, multiplier=1, precision=2):
    """
    Format the total time elapsed between two given time points into hours, minutes, and seconds.

    Parameters:
        time_total_start (float): The start time in seconds.
        time_total_end (float): The end time in seconds.
        divisor (float, optional): Divisor to scale the time difference. Defaults to 1.
        multiplier (float, optional): Multiplier to scale the time difference. Defaults to 1.
        precision (int, optional): Number of decimal places of seconds that will be displayed. Defaults to 2.

    Returns:
        str: A string representing the formatted total time in hours, minutes, and seconds.

    Example:
        # Example 1: Default case
        total_time_str = format_time(1000, 1050)
        print(total_time_str)  # Output: "0 hours, 0 minutes, 50 seconds"

        # Example 2: Using a divisor to scale time (e.g., converting milliseconds to seconds)
        total_time_str = format_time(1000, 3000, divisor=1000)
        print(total_time_str)  # Output: "0 hours, 0 minutes, 2 seconds"

        # Example 3: Using a multiplier to scale time (e.g., counting time in half)
        total_time_str = format_time(1000, 1050, multiplier=0.5)
        print(total_time_str)  # Output: "0 hours, 0 minutes, 25 seconds"

        # Example 4: Combining divisor and multiplier
        total_time_str = format_time(1000, 2000, divisor=1000, multiplier=2)
        print(total_time_str)  # Output: "0 hours, 0 minutes, 2 seconds"
    """

    # Handle division by zero error
    if divisor == 0:
        divisor = 1

    # Calculate time difference including divisor and multiplier
    total_time_seconds = ((time_total_end - time_total_start) / divisor) * multiplier

    # Calculate hours, minutes, and seconds
    total_time_hours, total_time_seconds = divmod(total_time_seconds, 3600)
    total_time_minutes, total_time_seconds = divmod(total_time_seconds, 60)

    # Initialize the output string
    output_time_str = ""

    # Formatting hours
    format_previous = False
    if total_time_hours > 0:
        format_previous = True
        # Set to whole number
        total_time_hours = int(total_time_hours)
        # Check for case singular
        if total_time_hours == 1:
            output_time_str += f"{total_time_hours} hour"
        else:
            output_time_str += f"{total_time_hours} hours"

    # Formatting minutes
    if total_time_minutes > 0:
        # Add separator if previous time is displayed
        if format_previous:
            output_time_str += ", "
        format_previous = True
        # Set to whole number
        total_time_minutes = int(total_time_minutes)
        # Check for case singular
        if total_time_minutes == 1:
            output_time_str += f"{total_time_minutes} minute"
        else:
            output_time_str += f"{total_time_minutes} minutes"

    # Formatting seconds
    if total_time_seconds > 0:
        # Add separator if previous time is displayed
        if format_previous:
            output_time_str += ", "
        total_time_seconds = round(total_time_seconds, precision)
        # Check for case singular
        if total_time_seconds == 1:
            output_time_str += f"{total_time_seconds} second"
        else:
            output_time_str += f"{total_time_seconds} seconds"

    return output_time_str


def format_dict(dictionary):
    """
        Formats a dictionary into a string representation.

        Args:
            dictionary (dict): A dictionary to be formatted.

        Returns:
            str: A string representation of the dictionary, where each key-value pair is formatted as follows:
                - Key is enclosed in single quotes.
                - Values are separated by commas.
                - If a value is a string, it is enclosed in single quotes.
                - Each key-value pair is separated by a tab ('\t').
                - Each key-value pair is terminated by a newline character ('\n').
    """
    out_str = ""
    # iterate through all keys
    for key_index, key in enumerate(dictionary):
        # add key to string
        out_str += f"'{key}':\t"

        if type(dictionary[key]) is list:

            if len(dictionary[key]) == 0:
                out_str += "\n"
                continue
            # iterate through values
            for value_index, value in enumerate(dictionary[key]):
                # add single quotes to value, if value is string
                if type(value) is str:
                    value = f"'{value}'"
                # add value to string and end with newline if it is the last value
                if value_index == len(dictionary[key]) - 1:
                    if key_index == len(dictionary) - 1:
                        out_str += f"{value}"
                    else:
                        out_str += f"{value}\n"
                else:
                    out_str += f"{value}, "
        else:
            value = dictionary[key]
            if type(value) is str:
                out_str += f"'{value}'\n"
            else:
                out_str += f"{value}\n"

    return out_str.rstrip("\n")


def format_set(set_to_format, delimiter=", ", first_char="", last_char=""):
    """
    Formats a set of elements into a string, with elements separated by a specified delimiter.

    Args:
        set_to_format (set): The set of elements to format.
        delimiter (str): The string to use as a delimiter between elements.

    Returns:
        str: A formatted string with elements separated by the delimiter.
    """
    out_str = first_char
    # Iterate through the set and append each element followed by the delimiter
    for element in set_to_format:
        out_str += f"{element}{delimiter}"
    # Remove the trailing delimiter and add last char
    out_str = out_str.rstrip(delimiter) + last_char
    return out_str


def get_embedding_methods(mapping_json_str):
    """
    Parses a JSON string to extract embedding methods. Handles both single and multiple model inputs.

    Args:
        mapping_json_str (str): JSON string containing embedding methods or a single method string.

    Returns:
        dict: A dictionary with embedding methods and their associated parameters.

    Raises:
        ValueError: If the provided embedding method does not exist in EMBEDDING_METHODS.
    """

    kge_mapping = None
    # Differentiate between inputs as dict and direct model inputs
    try:
        logging.debug(f"Multiple model input: {mapping_json_str}")
        kge_mapping = json.loads(mapping_json_str)
    except json.decoder.JSONDecodeError:
        logging.debug(f"Single model input: {mapping_json_str}")
        if mapping_json_str not in EMBEDDING_METHODS:
            raise ValueError(f"The given embedding method \'{mapping_json_str}\' does not exist!\n"
                             f"Please check if your spelling was correct, if the method should exist.")
        return {mapping_json_str: []}

    # Validate each embedding model in the parsed JSON
    for embedding_model in list(kge_mapping.keys()):
        if embedding_model not in EMBEDDING_METHODS:
            raise ValueError(f"The given embedding method \'{embedding_model}\' does not exist!\n"
                             f"Please check if your spelling was correct, if the method should exist.")

    return kge_mapping


def handle_methods(method_str: str, mode):
    """
    Handles different methods based on the provided mode. Supports sampling, aggregation, and size range methods.

    Args:
        method_str (str): The method string to handle.
        mode (str): The mode to determine the method list to use. Options: 'sampling', 'aggregation', 'size_range'.

    Returns:
        Union[tuple, str]: A tuple of floats for size range, or a method string for sampling/aggregation.

    Raises:
        ValueError: If the provided method does not exist in the corresponding method list.
    """
    method_list = None
    # Select the appropriate method list based on the mode
    if mode == "sampling":
        method_list = Constants.SAMPLING_METHODS
    elif mode == "aggregation":
        method_list = Constants.AGGREGATION_METHODS
    elif mode == "size_range":
        # Parse the size range method string into a tuple of floats
        method_str = method_str.lstrip("(").rstrip(")").split(",")
        method_str = (float(method_str[0]), float(method_str[1]))
        return method_str

    # Check if the method string exists in the selected method list
    for candidate_method in method_list:
        if method_str.lower() in candidate_method[1].lower():
            return candidate_method
    raise ValueError(f"The given sampling method \'{method_str}\' does not exist!\n"
                     f"Please check if your spelling was correct, if the method should exist.")


def get_args(args, model):
    """
    Processes and modifies arguments for a specific model. Handles conversion of string arguments to integers and
    applies model-specific mappings.

    Args:
        args (argparse.Namespace): The arguments to process.
        model (str): The model name to use for specific mappings.

    Returns:
        argparse.Namespace: The modified arguments.
    """

    args_subgraph = copy.copy(args)
    # List of argument keys to process
    args_list = ["bias",
                 "double_neg",
                 "dropout",
                 "gamma",
                 "init_size",
                 "learning_rate",
                 "multi_c",
                 "neg_sample_size",
                 "optimizer",
                 "regularizer",
                 "reg"]
    counter = 0

    # Convert string arguments to float if possible
    for key in vars(args):
        try:
            if isinstance(vars(args)[key], str):
                vars(args)[key] = float(vars(args)[key])
        except ValueError:
            logging.debug(f"Couldn't parse {key} to float")
        logging.debug(f"{key}: {vars(args)[key]} ({type(vars(args)[key])})")

    # Apply model-specific mappings to the arguments
    for key in vars(args):
        if key in args_list:
            value = vars(args)[key]
            try:
                try:
                    if isinstance(value, str):
                        vars(args)[key] = float(vars(args)[key])
                        value = vars(args)[key]
                except Exception:
                    value = json.loads(value)
                    logging.debug(f"Dict {value} for key {key}")
            except Exception:
                pass
            logging.debug(f"{key}: {value}, (type: {type(value)})")

            if isinstance(value, dict):
                if "all" in list(value.keys()):
                    try:
                        vars(args_subgraph)[key] = float(value['all'])
                    except ValueError:
                        vars(args_subgraph)[key] = value['all']
                elif model in list(value.keys()):
                    try:
                        vars(args_subgraph)[key] = float(value[model])
                    except ValueError:
                        vars(args_subgraph)[key] = value[model]
                elif "rest" in list(value.keys()):
                    try:
                        vars(args_subgraph)[key] = float(value['rest'])
                    except ValueError:
                        vars(args_subgraph)[key] = value['rest']
                else:
                    first_key = next(iter(value.keys()))
                    try:
                        if isinstance(value[first_key], bool):
                            vars(args_subgraph)[key] = bool(value[first_key])
                        else:
                            vars(args_subgraph)[key] = float(value[first_key])
                    except ValueError:
                        vars(args_subgraph)[key] = value[first_key]
                counter += 1

        else:
            args_subgraph = args

    logging.debug(f"{counter} parameters were changed due to specific mapping.")
    return args_subgraph


def get_loss_change(valid_loss, previous_valid_loss):
    """
    Calculates the change in validation loss and the percentage change.

    Args:
        valid_loss (float): The current validation loss.
        previous_valid_loss (float): The previous validation loss.

    Returns:
        tuple: A tuple containing the current validation loss and a tuple with the absolute and percentage change.
    """

    valid_loss_change = valid_loss - previous_valid_loss
    percentage = (valid_loss_change / previous_valid_loss) * 100
    valid_loss_change = (valid_loss_change, percentage)

    return valid_loss, valid_loss_change


def get_metrics_change(current_metrics, previous_metrics):
    """
    Calculates the absolute and percentage change in metrics from previous to current.

    Args:
        current_metrics (dict): The current metrics.
        previous_metrics (dict): The previous metrics.

    Returns:
        tuple: A tuple containing the current metrics, absolute change in metrics, and percentage change in metrics.
    """

    metrics_change_absolut = {}
    metrics_change_percent = {}

    # If there are no previous metrics, initialize changes to current metrics
    if len(previous_metrics) == 0:
        for metric in current_metrics:
            metrics_change_absolut[metric] = {}
            metrics_change_percent[metric] = {}
            for mode in current_metrics[metric]:
                metrics_change_absolut[metric][mode] = current_metrics[metric][mode]
                metrics_change_percent[metric][mode] = [100, 100, 100] if "hits@" in metric else 100
    else:
        for metric in current_metrics:
            metrics_change_absolut[metric] = {}
            metrics_change_percent[metric] = {}
            for mode in current_metrics[metric]:
                if "hits@" in metric:
                    metrics_change_absolut[metric][mode] = []
                    metrics_change_percent[metric][mode] = []
                    for hits in range(len(current_metrics[metric][mode])):
                        current_value = current_metrics[metric][mode][hits].item() if isinstance(
                            current_metrics[metric][mode][hits], torch.Tensor) else current_metrics[metric][mode][hits]
                        previous_value = previous_metrics[metric][mode][hits].item() if isinstance(
                            previous_metrics[metric][mode][hits], torch.Tensor) else previous_metrics[metric][mode][
                            hits]
                        absolute_change = current_value - previous_value
                        percent_change = (absolute_change / (previous_value + 1e-6)) * 100
                        metrics_change_absolut[metric][mode].append(absolute_change)
                        metrics_change_percent[metric][mode].append(percent_change)
                else:
                    current_value = current_metrics[metric][mode].item() if isinstance(current_metrics[metric][mode],
                                                                                       torch.Tensor) else \
                        current_metrics[metric][mode]
                    previous_value = previous_metrics[metric][mode].item() if isinstance(previous_metrics[metric][mode],
                                                                                         torch.Tensor) else \
                        previous_metrics[metric][mode]
                    absolute_change = current_value - previous_value
                    percent_change = (absolute_change / (previous_value + 1e-6)) * 100
                    metrics_change_absolut[metric][mode] = absolute_change
                    metrics_change_percent[metric][mode] = percent_change

    return current_metrics, metrics_change_absolut, metrics_change_percent
