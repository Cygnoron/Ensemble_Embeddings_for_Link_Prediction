import os

import networkx as nx

import util_files

TRAIN = "train"
TEST = "test"
VALID = "valid"


def create_graphml(info_directory, directory, to_convert=None):
    """
    The create_graphml function takes a file path as input and creates a .graphml representation of the KG.
        The function reads in triples from the specified file, creates an empty graph, adds nodes and edges based on
        triples, then writes the Graph to a .graphml file. The IDs of nodes and edges are written as data label.

    :param directory: The path to the dataset with the files, that should be converted into a .graphml file
    :param to_convert:
    """
    if to_convert is None:
        to_convert = [TRAIN]
    directory_split = directory.split("\\")
    info_directory = util_files.check_directory(os.path.abspath(os.path.join(info_directory, "graphml")))

    # scan given directory for either subgraph directory or data files
    for file_dir in os.listdir(directory):
        if "sub" in file_dir:
            # move into subgraph folder and search for data files
            for subgraph in os.listdir(directory + "\\" + file_dir):
                for data in to_convert:
                    if subgraph in data:
                        print(f"create .graphml representation of {os.path.join(directory, file_dir)} {subgraph} ...")

                        # Read the Triples File
                        triples_file = os.path.join(directory, file_dir, subgraph)
                        with open(triples_file, "r") as file:
                            triples = [line.strip().split() for line in file]

                        # Create an empty graph
                        graph = nx.Graph()

                        # Add nodes and edges based on triples
                        for triple in triples:
                            subject, predicate, obj = triple
                            graph.add_node(subject, label=subject)
                            graph.add_node(obj, label=obj)
                            graph.add_edge(subject, obj, label=predicate)

                        # Write the Graph to a .graphml file
                        directory_split = directory.split("\\")
                        output_graphml_file = (f"KGEmb\\data\\{directory_split[len(directory_split) - 1]}\\"
                                               f"results\\graphml\\{file_dir}_{subgraph}.graphml")

                        nx.write_graphml(graph, output_graphml_file)

                        print(f"Graph converted and saved to {output_graphml_file}")
        else:
            for data in to_convert:
                if file_dir in data:
                    print(f"create .graphml representation of {directory}: {file_dir} ...")

                    # Read the Triples File
                    triples_file = f"{directory}\\{file_dir}"
                    with open(triples_file, "r") as file:
                        triples = [line.strip().split() for line in file]

                    # Create an empty graph
                    graph = nx.Graph()

                    # Add nodes and edges based on triples
                    for triple in triples:
                        subject, predicate, obj = triple
                        graph.add_node(subject, label=subject)
                        graph.add_node(obj, label=obj)
                        graph.add_edge(subject, obj, label=predicate)

                    # Write the Graph to a .graphml file
                    # Write the Graph to a .graphml file
                    directory_split = directory.split("\\")
                    output_graphml_file = (f"KGEmb\\data\\{directory_split[len(directory_split) - 1]}\\"
                                           f"results\\graphml\\{file_dir}.graphml")
                    nx.write_graphml(graph, output_graphml_file)

                    print(f"Graph converted and saved to {output_graphml_file}")


def combine_files(directory):
    """
    The combine_files function takes in a directory of datasets and combines the train, test and validation files into one file.
    It returns the path to this combined file.

    :param directory: Specify the dataset directory where the files are located
    :return: The file path of the combined_data file
    """
    print("combining train, test and validation file ...")
    output_file_dir = os.path.abspath(directory) + "\\combined_data"
    input_files = [os.path.abspath(directory) + "\\train", os.path.abspath(directory) + "\\test",
                   os.path.abspath(directory) + "\\valid"]
    try:
        open(output_file_dir, 'w+')
    except FileNotFoundError:
        print(f"File {output_file_dir} was not found.")

    for input_file_dir in input_files:
        try:
            with open(input_file_dir, 'r') as input_file, open(output_file_dir, 'a+') as output_file:
                for line in input_file:
                    output_file.write(line)

        except FileNotFoundError:
            print(f"File {input_file_dir} was not found.")
    return output_file_dir
