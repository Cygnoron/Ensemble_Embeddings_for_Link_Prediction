import csv
import os
import pickle
import shutil

import numpy as np


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
    with (open(input_pickle_path, 'rb') as pickle_file, open(output_csv_path, 'w', newline='') as csv_file):
        # load pickle file
        data = pickle.load(pickle_file)

        # initialize csv writer
        csv_writer = csv.writer(csv_file, delimiter=delim)

        # iterate through data and write to csv
        for index, triple in enumerate(data):
            head, relation_name, tail = triple
            csv_writer.writerow([head, relation_name, tail])


def csv_to_file(input_csv_path, output_path, delim=';'):
    """
    Converts a csv file to a pickle file.

    :param input_csv_path: Directory to the csv file.
    :param output_path: Directory to the output pickle file.
    :param delim: Delimiter for the csv file. Defaults to ';'.
    """
    with (open(input_csv_path, 'r') as csv_file, open(output_path, 'w') as output_file):
        data = csv.reader(csv_file, delimiter=delim)
        for index, triple in enumerate(data):
            output_file.write(f"{str(triple[0])}\t{str(triple[1])}\t{str(triple[2])}\n")

    examples = []
    with (open(output_path, "r") as plain_text_file, open(f"{output_path}.pickle", 'wb') as output_pickle_file):
        for line in plain_text_file:
            head, rel, tail = line.strip().split("\t")
            try:
                examples.append([head, rel, tail])
            except ValueError:
                continue
        pickle.dump(np.array(examples).astype("int64"), output_pickle_file)
