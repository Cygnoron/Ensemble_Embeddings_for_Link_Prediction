import logging
import math
import os
import pickle
import random
import shutil
import time
import traceback
from builtins import round

import numpy as np
from tqdm import tqdm

from ensemble import Constants, util, util_files
from ensemble.util import get_unique_triple_ids

# from util import get_unique_triple_ids

removed_ids_counter = 0


def sample_graph(info_directory: str, dataset_in: str, dataset_out_dir: str, sampling_method, subgraph_amount=10,
                 subgraph_size_range=(0.1, 0.7), relation_name_amount=-1, max_indices_per_step="max"):
    """
    The sample_graph function takes in a dataset name, an output directory, and two optional parameters:
    subgraph_amount (default 10) - the number of subgraphs to create from the original dataset
    subgraph_size (default 0.7) - the percentage of triples that will be kept in each sampled graph

    :param dataset_in: str: Specify the dataset that should be sampled
    :param dataset_out_dir: str: Specify the directory where the sampled datasets are to be stored
    :param sampling_method: The sampling method, which is used to create Delta from the input dataset
    :param info_directory: Directory for saving info, log and config files
    :param subgraph_amount: Determine how many subgraphs should be created
    :param subgraph_size_range: Determine the range for the percentage of triples that will be kept in the subgraph
    :param relation_name_amount: how many entities should be selected compared to relation names, only applied in FEATURE_SAMPLING
    :param max_indices_per_step: The max amount of indices that may be sampled per step. (int or "max")
    """

    global removed_ids_counter

    util_files.delete_paths(f"{info_directory}", "Ensemble_Embedding_for_Link_Prediction.log")
    dataset_in_train = f"data\\{dataset_in}\\train"

    logging.info(f"Sampling {subgraph_amount} subgraphs with a relative size {subgraph_size_range} from "
                 f"{dataset_in} with {sampling_method[1]}")

    # check if input file exist
    util_files.check_file(dataset_in_train)

    # start timer for whole subgraph creation
    time_start = time.time()

    # columns for .csv file with stats about the subgraphs
    config_directory = f"{info_directory}\\subgraph_config.csv"
    with open(config_directory, 'w') as config_file:
        config_file.write(f"original_kg;sampling_method;relation_names_rho;directory;sampling_time;"
                          f"subgraph_size_range;subgraph_size_rel;subgraph_num;subgraph_triples_amount;"
                          f"subgraph_entities_amount;subgraph_relation_names_amount;subgraph_relation_names")

    # bool to check if init was successful for corresponding error message
    init_successful = False

    try:
        # -- initial setup --

        input_file = open(os.path.abspath(f"{dataset_in_train}.pickle"), 'rb')
        # load data from train.pickle file
        data = pickle.load(input_file)

        # get entity and relation name ids with corresponding triple indices
        entity_set, relation_name_set = get_unique_triple_ids(data, h=True, r=True, t=True)
        entity_ids_unused, relation_name_ids_unused = set(entity_set.keys()), set(relation_name_set.keys())

        # calculate number of relation names for the case Feature Sampling
        hyperparameter_str = ""
        if sampling_method == Constants.FEATURE_SAMPLING:
            relation_name_amount, relation_name_amount_str = get_relation_name_amount(relation_name_set,
                                                                                      relation_name_amount)
            hyperparameter_str = f", rho = {relation_name_amount_str} "
        else:
            relation_name_amount = len(relation_name_set.keys())

        # init was successful
        init_successful = True

        # -- subgraph creation --
        for subgraph_num in range(subgraph_amount):
            entity_set, relation_name_set = get_unique_triple_ids(data, h=True, r=True, t=True)
            # start timer for individual subgraph creation
            time_start_sub = time.time()

            # create directory for subgraph, if non-existent
            os.makedirs(f"{dataset_out_dir}\\sub_{subgraph_num:03d}", exist_ok=True)

            logging.info(f"-/\tSampling subgraph {dataset_out_dir}\\sub_{subgraph_num:03d} from {dataset_in} with "
                         f"params:\n\t\t{sampling_method[1]}{hyperparameter_str} and {subgraph_size_range} "
                         f"relative subgraph size.\t\\-")

            with (open(os.path.abspath(f"{dataset_out_dir}\\sub_{subgraph_num:03d}\\train.pickle"),
                       'wb') as output_file_pickle,
                  open(os.path.abspath(f"{dataset_out_dir}\\sub_{subgraph_num:03d}\\train"), 'w') as output_file_raw):

                # Create Delta
                # -- Special case: subgraph_num ==  --
                # check if all entity and relation name ids are used
                # -> force them to be present if missing
                buffer_var = calculate_delta(subgraph_size_range, data, subgraph_num, subgraph_amount, entity_set,
                                             relation_name_set, entity_ids_unused, relation_name_ids_unused,
                                             sampling_method, relation_name_amount, max_indices_per_step)
                delta_triples, entity_ids_unused, relation_name_ids_unused, subgraph_relation_names = buffer_var

                # Initialize a mask with False values
                mask = np.zeros(len(data), dtype=bool)
                # Set mask to True for all indices in delta_triples
                mask[list(delta_triples)] = True
                # Use the mask to select only the wanted triples
                sampled_data = [data[i] for i in range(len(data)) if mask[i]]

                # Write sampled data to new pickle file
                pickle.dump(sampled_data, output_file_pickle)
                for triple in sampled_data:
                    output_file_raw.write(f"{str(triple[0])}\t{str(triple[1])}\t{str(triple[2])}\n")

                # copy necessary files to the folder of the subgraph
                files_to_copy = ["test.pickle", "valid.pickle", "to_skip.pickle"]
                source_dir = f"data\\{dataset_in}"
                for file_name in os.listdir(source_dir):
                    if file_name in files_to_copy:
                        shutil.copy(os.path.join(source_dir, file_name),
                                    os.path.join(f"{dataset_out_dir}\\sub_{subgraph_num:03d}", file_name))

                # get used entity and relation name ids for the current subgraph
                used_entity_set, used_relation_name_set = get_unique_triple_ids(sampled_data, h=True, r=True, t=True)

                # stop timer for individual subgraph creation
                time_stop_sub = time.time()

                # save statistics of subgraph to config file:
                # ['original_kg', 'sampling_method', 'relation_names_rho', 'directory', 'sampling_time',
                #   'subgraph_size_range', 'subgraph_size_rel', 'subgraph_num', 'subgraph_triples_amount',
                #   'subgraph_entities_amount', 'subgraph_relation_names_amount', 'embedded_by']

                with open(config_directory, 'a') as config_file:
                    config_file.write(f"\n{dataset_in};{sampling_method[1]};{relation_name_amount};"
                                      f"{dataset_out_dir}\\sub_{subgraph_num:03d};"
                                      f"{round(time_stop_sub - time_start_sub, 3)};{subgraph_size_range};"
                                      f"{round(len(delta_triples) / len(data), 3)};{subgraph_num};"
                                      f"{len(delta_triples)};{len(used_entity_set)};{len(used_relation_name_set)};"
                                      f"{subgraph_relation_names}")
                    logging.critical(f"Length of delta: {len(delta_triples)}\t"
                                     f"Unnecessary sampling steps: {removed_ids_counter}")
                    logging.info(f"-\\\tUpdated config file {config_directory} for subgraph {subgraph_num:03d}, "
                                 f"sampling took {round(time_stop_sub - time_start_sub, 3)} sec\t/-\n")
                    removed_ids_counter = 0

    except FileNotFoundError:
        if not init_successful:
            logging.error(f"File {dataset_in_train}.pickle was not found!")
        else:
            logging.error(f"File {dataset_out_dir}\\sub_{subgraph_num:03d}\\train was not found!")
    except IndexError:
        logging.error(f"The selected sampling method {sampling_method[1]} is not yet implemented or doesn't exist!")
        traceback.print_exc()

    # stop timer for whole subgraph creation
    time_stop = time.time()
    logging.info(f"### Successfully created {str(subgraph_amount)} subgraphs of the relative size {subgraph_size_range}"
                 f" from the original KG \"{dataset_in}\" in a total of {round(time_stop - time_start, 3)} seconds "
                 f"(avg: {round((time_stop - time_start) / subgraph_amount, 3)} sec) ###")


def calculate_delta(subgraph_size_range, dataset, subgraph_num, subgraph_amount, entity_set, relation_name_set,
                    entity_ids_unused, relation_name_ids_unused, sampling_method, relation_name_amount,
                    max_indices_per_step):
    """
    The calculate_delta function calculates, which triples will be deleted from the input dataset and returns the
    indices of these triples as array.

    :param subgraph_size_range: The relative size of the subgraph compared to the input dataset
    :param dataset: The original dataset
    :param subgraph_num: The number of the subgraph, which is currently created
    :param subgraph_amount: The total amount of subgraphs that will be created
    :param entity_set: Dictionary containing all entity ids with corresponding triple indices
    :param relation_name_set: Dictionary containing all relation name ids with corresponding triple indices
    :param entity_ids_unused: List containing all entity ids, which are not yet present in any subgraph
    :param relation_name_ids_unused: List containing all relation name ids, which are not yet present in any subgraph
    :param sampling_method: The sampling method, which is used to create Delta from the input dataset
    :param relation_name_amount: How many different relation names should be present in the subgraph, only applied in RANDOM_FOREST_SAMPLING
    :param max_indices_per_step: The max amount of indices that may be sampled per step.
    :return: Array of indices, which represent the triples, that are to be deleted from the input dataset
    """

    # initialize general variables
    # delta = []
    delta = set()
    samples = []
    enforce_relation_names = False
    enforce_entities = False
    subgraph_relation_names = []

    # set subgraph_size to standard high value
    subgraph_size = subgraph_size_range[1]

    # -- delta creation for Entity Sampling --
    if sampling_method[0] == Constants.ENTITY_SAMPLING[0]:
        # select entities with respect to subgraph size:
        # select e_1 -> check size of effected triples in dataset
        # size > subgraph_size?: (yes)->select e_2, repeat; (no)->end selection process

        # reset missing relation names for every subgraph
        relation_name_ids_unused = set(relation_name_set.keys())
        progress_bar = None
        # -- sampling process --
        while len(delta) / len(dataset) <= subgraph_size:
            # check whether relation names are missing after most samples were selected
            if subgraph_amount - subgraph_num <= 3:
                # if entities are missing, enforce missing entities and set flag
                if len(entity_ids_unused) > 0:
                    enforce_entities = True
                    logging.info(f"Enforcing {len(entity_ids_unused)} entities to be present in the subgraph.")
                # if all entities are selected, reset flag
                else:
                    enforce_entities = False

            # if relation names are missing, enforce missing relation names and set flag
            if (len(relation_name_ids_unused) > 0) and (len(delta) / len(dataset) >= (subgraph_size * 0.9)):
                enforce_relation_names = True
                logging.info(f"Enforcing {len(relation_name_ids_unused)} relation names to be present "
                             f"in the subgraph.")
            # if all relation names are selected, reset flag
            else:
                enforce_relation_names = False

            # initialize progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=math.ceil(len(dataset) * subgraph_size),
                                    desc=f"Sampling Progress for subgraph {subgraph_num}", unit=" triple")

            # sample entities
            buffer_var = sampling(entity_set, relation_name_set, entity_ids_unused, relation_name_ids_unused,
                                  dataset, delta, sampling_method, enforce_relation_names, enforce_entities,
                                  progress_bar, subgraph_relation_names, max_indices_per_step)
            entity_set, relation_name_set, delta, entity_ids_unused, relation_name_ids_unused, subgraph_relation_names = buffer_var

        progress_bar.close()
        logging.info(f"Current subgraph size {len(delta) / len(dataset)} reached or exceeded target size "
                     f"{subgraph_size}.")
        logging.info(f"Created subgraph {subgraph_num} with {len(entity_ids_unused)} entities and "
                     f"{len(relation_name_ids_unused)} relation names left unused and a relative subgraph size "
                     f"{len(delta) / len(dataset)}.")
        return delta, entity_ids_unused, relation_name_ids_unused, subgraph_relation_names

    # -- delta creation for Feature Sampling --
    elif sampling_method[0] == Constants.FEATURE_SAMPLING[0]:
        # ratio_ent_rel ={2*\sqrt{len(relation_name_set)},\sqrt{len(relation_name_set)},\sqrt{len(relation_name_set)}/2}
        # select entities with respect to subgraph size:
        # select e_1 -> check size of effected triples in dataset
        # size > subgraph_size?: (yes)->select e_2, repeat; (no)->end selection process

        logging.info(f"Missing Relation Names: {relation_name_ids_unused}")

        # after most subgraphs are sampled, check if entities or relation names are missing
        if subgraph_amount - subgraph_num <= 3:
            # if entities are missing, enforce missing entities and set flag
            if len(entity_ids_unused) > 0:
                enforce_entities = entity_ids_unused
                logging.info(f"Enforcing {len(entity_ids_unused)} entities to be present in the subgraph.")
            # if all entities are selected, reset flag
            else:
                enforce_entities = False

        # create ranked list for relation names from missing ids combined with all other relation names
        ranked_relation_names = np.array(list(relation_name_ids_unused))
        relation_name_ids = list(relation_name_set.keys())

        # shuffle
        np.random.shuffle(ranked_relation_names)
        np.random.shuffle(relation_name_ids)

        ranked_relation_names = util.unique_unsorted(np.append(ranked_relation_names, relation_name_ids).astype(int))
        logging.info(f"Ranked Relation Names: {ranked_relation_names}")

        relation_name_set_safe = relation_name_set.copy()
        max_relation_name_index = relation_name_amount
        # take the top relation_name_amount relations based on ranking
        top_ranked_relations = ranked_relation_names[:relation_name_amount]
        # insert top ranked relations into new dictionary
        relation_name_set = {k: relation_name_set[k] for k in top_ranked_relations}
        entity_set = get_entity_set(relation_name_set, dataset)
        logging.info(f"Using relation names {list(relation_name_set.keys())} for subgraph {subgraph_num}, "
                     f"creating new entity set.")

        progress_bar = None
        while len(delta) / len(dataset) < subgraph_size:

            # initialize progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=math.ceil(len(dataset) * subgraph_size),
                                    desc=f"Sampling subgraph {subgraph_num}", unit=" triple")

            # sample entities and relation names
            buffer_var = sampling(entity_set, relation_name_set, entity_ids_unused, relation_name_ids_unused,
                                  dataset, delta, sampling_method, enforce_relation_names, enforce_entities,
                                  progress_bar, subgraph_relation_names, max_indices_per_step,
                                  relation_names_amount=relation_name_amount)
            (entity_set, relation_name_set, delta, entity_ids_unused, relation_name_ids_unused, relation_name_amount,
             subgraph_relation_names) = buffer_var

            if (len(relation_name_set) == 0) or (len(entity_set) == 0):
                if subgraph_size != subgraph_size_range[0]:
                    subgraph_size = subgraph_size_range[0]
                    logging.info(f"Changing relative subgraph size to minimum subgraph size {subgraph_size}.")

                if len(delta) / len(dataset) >= subgraph_size:
                    progress_bar.close()
                    logging.info(f"Current subgraph size {len(delta) / len(dataset)} reached or exceeded target size "
                                 f"{subgraph_size}.")
                    logging.info(f"Created subgraph {subgraph_num} with {len(entity_ids_unused)} entities and "
                                 f"{len(relation_name_ids_unused)} relation names left unused and a relative subgraph "
                                 f"size {len(delta) / len(dataset)}.")
                    return delta, entity_ids_unused, relation_name_ids_unused, subgraph_relation_names
                else:
                    # sample additional relation_name until small subgraph_size is reached
                    next_relation_name = ranked_relation_names[max_relation_name_index]
                    relation_name_set.clear()
                    relation_name_set[next_relation_name] = relation_name_set_safe[next_relation_name]
                    logging.info(f"No relation names left to sample (subgraph size "
                                 f"{round(len(delta) / len(dataset), 3)}/{subgraph_size}). Adding additional relation "
                                 f"name {ranked_relation_names[max_relation_name_index]}")
                    entity_set = get_entity_set(relation_name_set, dataset)
                    max_relation_name_index += 1

        progress_bar.close()
        logging.info(f"Current subgraph size {len(delta) / len(dataset)} reached or exceeded target size "
                     f"{subgraph_size}.")
        logging.info(f"Created subgraph {subgraph_num} with {len(entity_ids_unused)} entities and "
                     f"{len(relation_name_ids_unused)} relation names left unused and a relative subgraph size "
                     f"{len(delta) / len(dataset)}.")
        return delta, entity_ids_unused, relation_name_ids_unused, subgraph_relation_names

    # If no legal sampling method was given, return -1
    logging.error("Given argument for sampling method is not supported")
    return -1


def sampling(entity_set, relation_name_set, entity_ids_unused, relation_name_ids_unused, dataset, delta,
             sampling_method, enforce_relation_names, enforce_entities, progress_bar, subgraph_relation_names,
             max_indices_per_step, relation_names_amount=-1):
    """
    Function for sampling from a sampling set using various methods.

    :param entity_set: A dictionary representing the sampling set.
    :param relation_name_set: Dataset containing triples, in order to enforce remaining relation names.
    :param entity_ids_unused: A set of entity IDs that have not been used.
    :param relation_name_ids_unused: A set of relation name IDs that have not been used.
    :param dataset: The original dataset containing all triples.
    :param delta: An array representing the indices of the samples.
    :param sampling_method: A string representing the sampling method to be used.
    :param enforce_relation_names: A boolean indicating whether to enforce missing relation names.
    :param enforce_entities: A boolean indicating whether to enforce missing entities.
    :param progress_bar: A progress_bar, which shows the progress of the sampling.
    :param subgraph_relation_names: A set containing all relation names used in the current subgraph.
    :param relation_names_amount: Hyperparameter to set the ratio of relation names for Feature Sampling.
    :param max_indices_per_step: The max amount of indices that may be sampled per step.
    :return: The updated samples, delta, entity_ids_unused, and relation_name_ids_unused.
    """

    global removed_ids_counter
    unique_before = 0
    unique_after = 0

    try:
        # -- enforce relation names in case of Entity sampling --
        if (enforce_relation_names and sampling_method == Constants.ENTITY_SAMPLING and
                len(relation_name_ids_unused) > 0):
            # copy relation_name_ids_unused to avoid RuntimeError: Set changed size during iteration
            relation_name_ids_iterator_set = relation_name_ids_unused.copy()
            # sample random triple for each missing relation name
            for enforced_relation_name_id in relation_name_ids_iterator_set:
                # sample = random.choice(list(relation_name_set[enforced_relation_name_id]))
                sample = relation_name_set[enforced_relation_name_id].pop()
                # add sampled triple index to delta
                delta.add(sample)

                # get head entity, relation name and tail entity of sampled triple
                head_entity, relation_name, tail_entity = dataset[sample]

                if sample in relation_name_set[relation_name]:
                    relation_name_set[relation_name].discard(sample)

                # remove head entity from list of unused entities
                if head_entity in entity_ids_unused:
                    entity_ids_unused.discard(head_entity)
                # remove relation name from list of unused relation names or add to list of contained relation names
                if (relation_name in relation_name_ids_unused) or (relation_name not in subgraph_relation_names):
                    relation_name_ids_unused.discard(relation_name)
                    subgraph_relation_names.append(relation_name)
                # remove tail entity from list of unused entities
                if tail_entity in entity_ids_unused:
                    entity_ids_unused.discard(tail_entity)

        # -- enforce entities --
        if enforce_entities and len(entity_ids_unused) > 0:
            # copy entity_ids_unused to avoid RuntimeError: Set changed size during iteration
            entity_ids_iterator_set = entity_ids_unused.copy()
            # sample random triple for each missing entity
            for enforced_entity_id in entity_ids_iterator_set:
                if enforced_entity_id in entity_set:
                    # sample = random.choice(list(entity_set[enforced_entity_id]))
                    sample = entity_set[enforced_entity_id].pop()

                    # add sampled triple index to delta
                    delta.add(sample)

                    # get head entity, relation name and tail entity of sampled triple
                    head_entity, relation_name, tail_entity = dataset[sample]

                    if sample in relation_name_set[relation_name]:
                        relation_name_set[relation_name].discard(sample)

                    # remove head entity from list of unused entities
                    if head_entity in entity_ids_unused:
                        entity_ids_unused.discard(head_entity)
                    # remove relation name from list of unused relation names or add to list of contained relation names
                    if (relation_name in relation_name_ids_unused) or (
                            relation_name not in subgraph_relation_names):
                        relation_name_ids_unused.discard(relation_name)
                        subgraph_relation_names.append(relation_name)
                    # remove tail entity from list of unused entities
                    if tail_entity in entity_ids_unused:
                        entity_ids_unused.discard(tail_entity)

        # -- regular sampling process --
        # if not (enforce_relation_names or enforce_entities):
        if len(entity_set) == 0:
            raise IndexError("No entity in entity set.")
        sampling_set = entity_set
        entity_sampling = True

        # take a sample from the sampling set
        sample = random.choice(list(sampling_set.keys()))

        sample_triple_indices = []
        # set all entries of sampling_set[sample] as sampled triples and clear the key "sample" in the dict
        if max_indices_per_step == "max":
            sample_triple_indices = sampling_set[sample]
            sampling_set[sample] = set()
        else:
            while len(sample_triple_indices) < max_indices_per_step and len(sampling_set[sample]) > 0:
                sample_triple_indices.append(sampling_set[sample].pop())

        if relation_names_amount > 0 and sampling_method == Constants.FEATURE_SAMPLING:
            relation_names_amount -= 1
        if len(sampling_set[sample]) == 0 or sampling_set[sample] == set():
            del sampling_set[sample]
        # add all indices of the sample to delta
        unique_before = len(delta)
        delta.update(sample_triple_indices)
        unique_after = len(delta)

        # remove sample from entity_ids_unused and relation_name_ids_unused
        for index, triple in enumerate(dataset[list(sample_triple_indices)]):
            # get head entity, relation name and tail entity of sampled triple
            head_entity, relation_name, tail_entity = triple

            # remove remaining triple indices from entity and relation name sets
            if head_entity in entity_set:
                entity_set[head_entity].discard(sample)
            if relation_name in relation_name_set:
                relation_name_set[relation_name].discard(sample)
            if tail_entity in entity_set:
                entity_set[tail_entity].discard(sample)

            # remove head entity from list of unused entities
            if head_entity in entity_ids_unused:
                entity_ids_unused.discard(head_entity)
            # remove relation name from list of unused relation names or add to list of contained relation names
            if (relation_name in relation_name_ids_unused) or (relation_name not in subgraph_relation_names):
                relation_name_ids_unused.discard(relation_name)
                subgraph_relation_names.append(relation_name)
            # remove tail entity from list of unused entities
            if tail_entity in entity_ids_unused:
                entity_ids_unused.discard(tail_entity)

        if entity_sampling:
            entity_set = sampling_set
        else:
            relation_name_set = sampling_set

    except IndexError:
        logging.error("Received IndexError")
    # except KeyError as key:
    #     logging.error(f"Received KeyError with key {key}")

    except Exception as error:
        logging.error(f"{error}\t{traceback.format_exc()}")

    # ensure delta only has unique triple indices

    if unique_before == unique_after:
        removed_ids_counter += 1
        # logging.critical(f"No indices were added to delta")

    # update progress bar
    progress_bar.n = len(delta)
    progress_bar.refresh()
    # return all changed lists for the different sampling methods
    if sampling_method == Constants.FEATURE_SAMPLING:
        return (entity_set, relation_name_set, delta, entity_ids_unused, relation_name_ids_unused,
                relation_names_amount, subgraph_relation_names)
    else:
        return (entity_set, relation_name_set, delta, entity_ids_unused, relation_name_ids_unused,
                subgraph_relation_names)


def get_relation_name_amount(relation_name_set, factor):
    """
    Calculate the ratio of relation names for Feature Sampling.
    :param relation_name_set: Set of relation names.
    :param factor: Multiplier for square root of the number of relation names.
    :return: Amount of relation names to be included in Feature Sampling.
    """
    if factor == -1:
        relation_name_amount = len(relation_name_set.keys())
        return relation_name_amount, f"|Relation Names| ({relation_name_amount})"
    else:
        relation_name_amount = math.ceil(factor * math.sqrt(len(relation_name_set.keys())))
        return relation_name_amount, f"{factor} * √|Relation Names| ({relation_name_amount})"


def get_entity_set(relation_name_set, dataset):
    entity_set = {}
    for index_list in list(relation_name_set.values()):
        for triple_index in index_list:
            head, relation_name, tail = dataset[triple_index]
            if head not in entity_set:
                entity_set[head] = set()
            entity_set[head].add(triple_index)

            if tail not in entity_set:
                entity_set[tail] = set()
            entity_set[tail].add(triple_index)
    return entity_set
