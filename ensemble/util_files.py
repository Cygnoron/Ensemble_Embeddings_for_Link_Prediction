import argparse
import csv
import logging
import os
import pickle
import shutil
from datetime import datetime

import numpy as np
import torch

from ensemble import Constants
from ensemble.util import get_unique_triple_ids


def check_directory(directory):
    """
        Check if the given directory exists, and if not, create it along with its parent directories if necessary.

        Args:
            directory (str): The directory path to check or create.

        Returns:
            str: The input directory path.

        Raises:
            FileNotFoundError: If the directory or its parent directory does not exist and cannot be created.

        Examples:
            >>> check_directory('/path/to/directory')  # Existing directory
            '/path/to/directory'

            >>> check_directory('/path/to/new_directory')  # New directory
            '/path/to/new_directory'
        """
    if os.path.isdir(directory):  # Check if the directory already exists
        try:
            os.scandir(os.path.dirname(directory))  # Check if parent directory exists
        except FileNotFoundError:
            os.mkdir(os.path.dirname(directory))  # Create the parent directory
    else:
        try:
            os.scandir(directory)  # Check if the directory exists
        except FileNotFoundError:
            os.mkdir(directory)  # Create the directory
    return directory


def check_file(file_path, argument='r', clear_file=False):
    """
    Check if the given file exists, and if not, create it along with its parent directories if necessary.

    Args:
        file_path (str): The path to the file to check or create.
        argument (str, optional): The mode to open the file with. Defaults to 'r'.
        clear_file (bool, optional): If True, clear the file contents if it exists. Defaults to False.

    Returns:
        str: The absolute path to the file.

    Raises:
        FileNotFoundError: If the file or its parent directory does not exist and cannot be created.

    Examples:
        >>> check_file('/path/to/file.txt')  # Existing file
        '/path/to/file.txt'

        >>> check_file('/path/to/new_file.txt', argument='w', clear_file=True)  # New file
        '/path/to/new_file.txt'
    """
    file = os.path.abspath(file_path)  # Get the absolute path of the file
    try:
        if clear_file:
            open(file, argument)  # Try to open the file with the specified mode
        else:
            open(file, 'r')  # Try to open the file in read mode
        return file  # Return the file path if successfully opened
    except NotADirectoryError:
        # Create directory and try again
        logging.debug(f"File {file} was not found, creating folders")
        check_directory(file)  # Create directory if it does not exist
        return check_file(file, argument=argument, clear_file=clear_file)  # Recursively check file
    except FileNotFoundError:
        try:
            # Create file and try to open
            file = os.path.abspath(file_path)
            open(file, 'w+')  # Open file in write mode to create it
            return file  # Return the file path if successfully created
        except FileNotFoundError:
            return file  # Return the file path if creation failed


def delete_paths(paths, to_skip=None):
    """
    Recursively delete the specified directories or files if they exist, excluding the specified file.

    Args:
        paths (str or list): A path or a list of paths to the directories or files to be deleted.
        to_skip (str, optional): The name of the file to skip during deletion. Defaults to None.

    Returns:
        None
    """
    logging.debug("Deleting unwanted files and directories")
    if isinstance(paths, str):  # Convert single path to list
        paths = [paths]

    for path in paths:
        for path2 in os.listdir(path):  # Iterate over contents of the directory
            if path2 == to_skip:  # Check if the current item matches the file to skip
                continue  # Skip deletion if the item matches the file to skip

            path2 = os.path.abspath(os.path.join(path, path2))  # Get the absolute path of the item
            if os.path.isdir(path2):  # Check if the item is a directory
                if os.path.exists(path2):  # Check if the directory exists
                    shutil.rmtree(path2)  # Recursively delete the directory and its contents
            elif os.path.isfile(path2):  # Check if the item is a file
                if os.path.exists(path2):  # Check if the file exists
                    os.remove(path2)  # Delete the file


def pickle_to_csv(input_pickle_path, output_csv_path, delim=';'):
    """
    Converts a pickle file to a csv file.
    :param input_pickle_path: Directory to the pickle file.
    :param output_csv_path: Directory to the output csv file.
    :param delim: Delimiter for the csv file. Defaults to ';'.
    """
    logging.debug(f"Converting pickle file {input_pickle_path} to csv file {output_csv_path}")
    with (open(input_pickle_path, 'rb') as pickle_file, open(output_csv_path, 'w', newline='') as csv_file):
        # load pickle fil
        data = pickle.load(pickle_file)

        # initialize csv writer
        csv_writer = csv.writer(csv_file, delimiter=delim)

        # iterate through data and write to csv
        for index, triple in enumerate(data):
            head, relation_name, tail = triple
            csv_writer.writerow([head, relation_name, tail])


def csv_to_file(input_csv_path, output_pickle_path, delim=';', only_unique=False):
    """
    Converts a CSV file containing triples (head, relation, tail) into a pickle file.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_pickle_path (str): Path to the output pickle file.
        delim (str, optional): Delimiter used in the CSV file. Defaults to ';'.
        only_unique (bool, optional): If True, only unique triples will be included in the output. Defaults to False.
    """

    # Log the start of the conversion process
    logging.debug(f"Converting CSV file {input_csv_path} to pickle file {output_pickle_path}")

    # Open input and output files
    with open(input_csv_path, 'r') as csv_file, open(output_pickle_path, 'w') as output_file:
        data = csv.reader(csv_file, delimiter=delim)

        # Process data based on whether only unique triples are required
        if only_unique:
            unique_triples = []
            last_row = 0
            for index, triple in enumerate(data):
                if triple not in unique_triples:
                    unique_triples.append(triple)
                # Log the processed triple and its index
                logging.log(Constants.DATA_LEVEL_LOGGING, f"Triple: {triple}\tIndex: {index}")
                last_row = index + 1

            # Log information about excluded duplicates
            if last_row - len(unique_triples) > 0:
                logging.debug(f"Excluded {last_row - len(unique_triples)} duplicates.")
            else:
                logging.log(Constants.DATA_LEVEL_LOGGING, "No duplicate triples found")

            # Write unique triples to output file
            for triple in unique_triples:
                output_file.write(f"{str(triple[0])}\t{str(triple[1])}\t{str(triple[2])}\n")
        else:
            # Write all triples to output file
            for index, triple in enumerate(data):
                output_file.write(f"{str(triple[0])}\t{str(triple[1])}\t{str(triple[2])}\n")

    # Process the created plain text file and create a pickle file
    examples = []
    with open(output_pickle_path, "r") as plain_text_file, \
            open(f"{output_pickle_path}.pickle", 'wb') as output_pickle_file:
        for line in plain_text_file:
            head, rel, tail = line.strip().split("\t")
            try:
                examples.append([head, rel, tail])
            except ValueError:
                continue
        # Dump examples to pickle file after converting to numpy array of int64
        pickle.dump(np.array(examples).astype("int64"), output_pickle_file)


def copy_test_valid_filter_data(dataset_in: str, info_directory: str):
    """
      Copy necessary files from the input dataset directory to the info directory for all subgraphs.

      Args:
          dataset_in (str): The name of the dataset containing the files to be copied.
          info_directory (str): The directory where the files will be copied.

      Raises:
          FileNotFoundError: If any of the required files are not found in the dataset directory or fail to copy.

      Returns:
          None
      """
    # List of files to copy from the dataset directory
    files_to_copy = ["test.pickle", "valid.pickle", "to_skip.pickle"]
    # List to store names of successfully copied files
    copied_files = []
    # Source directory where files are located within the dataset
    source_dir = f"data\\{dataset_in}"
    # Target directory where files will be copied to
    target_dir = f"{info_directory}\\data"
    # Create the target directory if it doesn't exist
    check_directory(target_dir)

    # Iterate over files in the source directory
    for file_name in os.listdir(source_dir):
        # Check if the file is in the list of files to copy
        if file_name in files_to_copy:
            # Copy the file from the source directory to the target directory
            shutil.copy(os.path.join(source_dir, file_name),
                        os.path.join(target_dir, file_name))
            # Add the file to the list of successfully copied files
            copied_files += [file_name]

    # Check if all required files were copied successfully
    for file_to_copy in files_to_copy:
        if file_to_copy not in copied_files:
            # Raise an error if any required file was not copied successfully
            raise FileNotFoundError(f"The file \"{file_to_copy}\" was not correctly copied to \"{target_dir}\", "
                                    f"or did not exist in \"{source_dir}\"")


def save_load_trained_models(embedding_models, valid_args: argparse.Namespace, model_file_dir):
    """
    Save or load trained models based on the validation arguments provided.

    Parameters:
        embedding_models (list): List of dictionaries containing trained embedding models and their arguments.
        valid_args (argparse.Namespace): Validation arguments saved in Namespace object.
        model_file_dir (str): Directory where the models will be saved or loaded from.
    """
    if not valid_args.best_mrr:
        logging.info(f"Saving new models saved at epoch {valid_args.best_epoch}.")
        # Iterate over models
        for embedding_model in embedding_models:
            args = embedding_model["args"]
            # Save the model
            torch.save(embedding_model['model'].cpu().state_dict(),
                       os.path.join(model_file_dir, f"model_{args.subgraph}_{args.model_name}.pt"))
    else:
        logging.info(f"Loading best models saved at epoch {valid_args.best_epoch}")
        # Iterate over models
        for embedding_model in embedding_models:
            args = embedding_model["args"]
            # Load the best model
            embedding_model["model"].load_state_dict(
                torch.load(os.path.join(model_file_dir, f"model_{args.subgraph}_{args.model_name}.pt")))


def print_loss_to_file(loss_file_path, epoch, loss_dict):
    with open(loss_file_path, 'a+') as loss_file:
        keys_sorted = list(loss_dict.keys())
        logging.debug(f"Output keys unsorted: {keys_sorted}")
        keys_sorted.sort()
        logging.debug(f"Output keys sorted: {keys_sorted}")

        average_loss = 0.
        for subgraph in loss_dict:
            average_loss += loss_dict[subgraph]
        average_loss /= len(keys_sorted)

        out_str = f"{epoch};{average_loss}"
        for subgraph in keys_sorted:
            out_str += f";{loss_dict[subgraph]}"

        loss_file.write(out_str)
        loss_file.write("\n")


def print_metrics_to_file(metrics_file_path, metrics, epoch, mode):
    with open(f"{metrics_file_path}", 'a') as metrics_file:
        logging.debug(f"Printing metrics to {metrics_file_path}.")
        metrics_file.write(f"{epoch};{mode};{metrics['MR']};{metrics['MRR']};{metrics['hits@[1,3,10]'][0]};"
                           f"{metrics['hits@[1,3,10]'][1]};{metrics['hits@[1,3,10]'][2]};{metrics['AMRI']};"
                           f"{metrics['MR_deviation']}")

        metrics_file.write("\n")


def create_entity_and_relation_name_set_file(dataset):
    """
    Calculates and writes entity and relation name sets for dataset to csv files

    :param dataset: Name of the input dataset
    """
    logging.debug(f"Creating csv files containing the entity and relation name sets for dataset {dataset}")

    with (open(os.path.abspath(os.path.join(dataset, "train.pickle")), 'rb') as pickle_file,
          open(os.path.join(dataset, "entity_set.csv"), 'w') as entity_set_file,
          open(os.path.join(dataset, "relation_name_set.csv"), 'w') as relation_name_set_file):

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


def get_info_directory_path(dataset_out_dir, args):
    hyper_param_str = f"results_{args.aggregation_method[2]}_{args.theta_calculation[2]}"

    if not args.no_time_dependent_file_path:
        hyper_param_str += datetime.now().strftime('_%m.%d_%H_%M')

    info_directory = os.path.join(dataset_out_dir, hyper_param_str)
    check_directory(info_directory)
    return info_directory
