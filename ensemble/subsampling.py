import logging
import math
import os
import pickle
import random
import time
import traceback
from builtins import round

import numpy as np
from tqdm import tqdm

from ensemble import Constants, util, util_files
from ensemble.util import get_unique_triple_ids


def sample_graph(info_directory: str, dataset_in: str, dataset_out_dir: str, sampling_method, subgraph_amount=10,
                 subgraph_size_range=(0.1, 0.7), rho=-1, entities_per_step=1, no_progress_bar=False, random_seed=42):
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
    :param rho: how many entities should be selected compared to relation names, only applied in FEATURE_SAMPLING
    :param entities_per_step: The max amount of indices that may be sampled per step. (int or "max")
    """

    util_files.delete_paths(f"{info_directory}", "Ensemble_Embedding_for_Link_Prediction.log")
    dataset_in_train = os.path.join("data", dataset_in, "train")

    logging.info(f"Sampling {subgraph_amount} subgraphs with a relative size {subgraph_size_range} from "
                 f"{dataset_in} with {sampling_method[1]}")

    # check if input file exist
    util_files.check_file(dataset_in_train)

    # start timer for whole subgraph creation
    time_start = time.time()

    # columns for .csv file with stats about the subgraphs
    config_directory = os.path.join(info_directory, "subgraph_config.csv")
    with open(config_directory, 'w') as config_file:
        config_file.write(f"original_kg;sampling_method;relation_names_rho;directory;sampling_time;"
                          f"subgraph_size_range;subgraph_size_rel;subgraph_num;subgraph_triples_amount;"
                          f"triples_deviation;subgraph_entities_amount;entities_per_step;"
                          f"subgraph_relation_names_amount;subgraph_relation_names")

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
            rho, relation_name_amount_str = get_relation_name_amount(relation_name_set,
                                                                     rho)
            hyperparameter_str = f", rho = {relation_name_amount_str} "
        else:
            rho = len(relation_name_set.keys())

        if type(random_seed) is str:
            try:
                random_seed = int(random_seed)
                logging.debug(f"Converting seed {random_seed} to int.")
            except ValueError:
                random_seed = random.randint(0, 2 ** 32 - 1)
                logging.debug(f"Selecting a random sampling seed: Seed {random_seed}")

        random.seed(random_seed)
        logging.info(f"Seed for subgraph sampling: {random_seed}")

        # init was successful
        init_successful = True

        # -- subgraph creation --
        for subgraph_num in range(subgraph_amount):
            entity_set, relation_name_set = get_unique_triple_ids(data, h=True, r=True, t=True)
            # start timer for individual subgraph creation
            time_start_sub = time.time()

            # create directory for subgraph, if non-existent
            os.makedirs(os.path.join(dataset_out_dir, f"sub_{subgraph_num:03d}"), exist_ok=True)

            logging.info(f"-/\tSampling subgraph {dataset_out_dir}\\sub_{subgraph_num:03d} from {dataset_in} with "
                         f"params:\n\t\t\t\t\t\t\t {sampling_method[1]}{hyperparameter_str} and {subgraph_size_range} "
                         f"relative subgraph size.\t\\-")

            with (open(os.path.abspath(os.path.join(dataset_out_dir, f"sub_{subgraph_num:03d}", "train.pickle")),
                       'wb') as output_file_pickle,
                  open(os.path.abspath(os.path.join(dataset_out_dir, f"sub_{subgraph_num:03d}", "train")),
                       'w') as output_file_raw):

                # Create Delta
                # -- Special case: subgraph_num ==  --
                # check if all entity and relation name ids are used
                # -> force them to be present if missing
                delta_triples, entity_ids_unused, relation_name_ids_unused, subgraph_relation_names = calculate_delta(
                    subgraph_size_range, data, subgraph_num, subgraph_amount, entity_set,
                    relation_name_set, entity_ids_unused, relation_name_ids_unused,
                    sampling_method, rho, entities_per_step, no_progress_bar=no_progress_bar)

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

                # get used entity and relation name ids for the current subgraph
                used_entity_set, used_relation_name_set = get_unique_triple_ids(sampled_data, h=True, r=True, t=True)

                # stop timer for individual subgraph creation
                time_stop_sub = time.time()

                # save statistics of subgraph to config file:
                # original_kg sampling_method relation_names_rho directory sampling_time subgraph_size_range
                # subgraph_size_rel subgraph_num subgraph_triples_amount triples_deviation subgraph_entities_amount
                # entities_per_step subgraph_relation_names_amount subgraph_relation_names

                with open(config_directory, 'a') as config_file:
                    triples_deviation = len(delta_triples) - math.ceil(len(data) * subgraph_size_range[1])
                    config_file.write(f"\n{dataset_in};{sampling_method[1]};{rho};"
                                      f"{os.path.join(dataset_out_dir, f'sub_{subgraph_num:03d}')};"
                                      f"{round(time_stop_sub - time_start_sub, 3)};{subgraph_size_range};"
                                      f"{round(len(delta_triples) / len(data), 3)};{subgraph_num};"
                                      f"{len(delta_triples)};{triples_deviation};{len(used_entity_set)};"
                                      f"{entities_per_step};{len(used_relation_name_set)};{subgraph_relation_names}")
                    logging.info(f"Length of delta: {len(delta_triples)}\t\t"
                                 f"Deviation from target size: {triples_deviation}")
                    logging.info(f"-\\\tUpdated config file \"{config_directory}\" for subgraph {subgraph_num:03d}, "
                                 f"sampling took {util.format_time(time_start_sub, time_stop_sub)}\t/-\n")

        # copy necessary files
        util_files.copy_test_valid_filter_data(dataset_in, dataset_out_dir)

    except FileNotFoundError:
        if not init_successful:
            logging.error(f"File {dataset_in_train}.pickle was not found!")
        else:
            logging.error(f"File {os.path.join(dataset_out_dir, f'sub_{subgraph_num:03d}', 'train')} was not found!")
    except IndexError:
        logging.error(f"The selected sampling method {sampling_method[1]} is not yet implemented or doesn't exist!")
        traceback.print_exc()

    # stop timer for whole subgraph creation
    time_stop = time.time()
    logging.info(f"### Successfully created {str(subgraph_amount)} subgraphs of the relative size {subgraph_size_range}"
                 f" from the original KG \"{dataset_in}\" in a total of {util.format_time(time_start, time_stop)} "
                 f"(avg: {util.format_time(time_start, time_stop, divisor=subgraph_amount)}) ###")


def calculate_delta(subgraph_size_range, dataset, subgraph_num, subgraph_amount, entity_set, relation_name_set,
                    entity_ids_unused, relation_name_ids_unused, sampling_method, relation_name_amount,
                    entities_per_step, enforcement=2, no_progress_bar=False):
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
    :param entities_per_step: The amount triples that will be sampled each step.
    :return: Array of indices, which represent the triples, that are to be deleted from the input dataset
    """

    # initialize general variables
    delta = set()
    enforce_relation_names = False
    enforce_entities = False
    subgraph_relation_names = []
    sampled_entity_ids = set()

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
            if subgraph_amount - subgraph_num <= enforcement:
                # if entities are missing, enforce missing entities and set flag
                if len(entity_ids_unused) > 0:
                    enforce_entities = True

                # if all entities are selected, reset flag
                else:
                    enforce_entities = False

            # if relation names are missing, enforce missing relation names and set flag
            if (len(relation_name_ids_unused) > 0) and (len(delta) / len(dataset) >= (subgraph_size * 0.9)):
                enforce_relation_names = True

            # if all relation names are selected, reset flag
            else:
                enforce_relation_names = False

            # initialize progress bar
            if progress_bar is None and not no_progress_bar:
                progress_bar = tqdm(total=math.ceil(len(dataset) * subgraph_size),
                                    desc=f"Sampling Progress for subgraph {subgraph_num}", unit=" triple")

            # sample entities
            buffer_var = sampling(entity_set, relation_name_set, entity_ids_unused, relation_name_ids_unused,
                                  dataset, delta, sampling_method, enforce_relation_names, enforce_entities,
                                  progress_bar, no_progress_bar, subgraph_relation_names, sampled_entity_ids,
                                  entities_per_step)
            (entity_set, relation_name_set, delta, entity_ids_unused, relation_name_ids_unused,
             subgraph_relation_names, sampled_entity_ids) = buffer_var

        if not no_progress_bar:
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

        logging.info(f"Missing Relation Names: {list(relation_name_ids_unused)}")

        # after most subgraphs are sampled, check if entities or relation names are missing
        if subgraph_amount - subgraph_num <= enforcement:
            # if entities are missing, enforce missing entities and set flag
            if len(entity_ids_unused) > 0:
                enforce_entities = True

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
        # take the top rho relations based on ranking
        top_ranked_relations = ranked_relation_names[:relation_name_amount]
        # insert top ranked relations into new dictionary
        relation_name_set = {k: relation_name_set[k] for k in top_ranked_relations}
        entity_set = get_entity_set(relation_name_set, dataset)
        logging.info(f"Using relation names {list(relation_name_set.keys())} for subgraph {subgraph_num}, "
                     f"creating new entity set.")

        progress_bar = None
        while len(delta) / len(dataset) < subgraph_size:

            # initialize progress bar
            if progress_bar is None and not no_progress_bar:
                progress_bar = tqdm(total=math.ceil(len(dataset) * subgraph_size),
                                    desc=f"Sampling subgraph {subgraph_num}", unit=" triple")

            # sample entities and relation names
            buffer_var = sampling(entity_set, relation_name_set, entity_ids_unused, relation_name_ids_unused,
                                  dataset, delta, sampling_method, enforce_relation_names, enforce_entities,
                                  progress_bar, no_progress_bar, subgraph_relation_names, sampled_entity_ids,
                                  entities_per_step,
                                  relation_names_amount=relation_name_amount)
            (entity_set, relation_name_set, delta, entity_ids_unused, relation_name_ids_unused, relation_name_amount,
             subgraph_relation_names, sampled_entity_ids) = buffer_var

            if (len(relation_name_set) == 0) or (len(entity_set) == 0):
                if subgraph_size != subgraph_size_range[0]:
                    subgraph_size = subgraph_size_range[0]
                    logging.info(f"Changing relative subgraph size to minimum subgraph size {subgraph_size}.")
                    if not no_progress_bar:
                        progress_bar.total = math.ceil(len(dataset) * subgraph_size)

                if len(delta) / len(dataset) >= subgraph_size:
                    if not no_progress_bar:
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

        if not no_progress_bar:
            progress_bar.close()
        logging.info(f"Current subgraph size {len(delta) / len(dataset)} reached or exceeded target size "
                     f"{subgraph_size}.")
        logging.info(f"Created subgraph {subgraph_num} with {len(entity_ids_unused)} entities and "
                     f"{len(relation_name_ids_unused)} relation names left unused and a relative subgraph size "
                     f"{len(delta) / len(dataset)}.")
        return delta, entity_ids_unused, relation_name_ids_unused, subgraph_relation_names

    elif sampling_method[0] == Constants.DEBUG_SAMPLING[0]:
        logging.critical(f"{Constants.DEBUG_SAMPLING[1]} wasn't implemented yet!")
        return delta, entity_ids_unused, relation_name_ids_unused, subgraph_relation_names

    # If no legal sampling method was given, return -1
    logging.error("Given argument for sampling method is not supported")
    return -1


def sampling(entity_set, relation_name_set, entity_ids_unused, relation_name_ids_unused, dataset, delta,
             sampling_method, enforce_relation_names, enforce_entities, progress_bar, no_progress_bar,
             subgraph_relation_names, sampled_entity_ids, entities_per_step, relation_names_amount=-1):
    """
    Perform sampling of entities and relation names for subgraph creation.

    Args:
        entity_set (dict): Dictionary containing entity IDs with corresponding triple indices.
        relation_name_set (dict): Dictionary containing relation name IDs with corresponding triple indices.
        entity_ids_unused (set): Set containing IDs of unused entities.
        relation_name_ids_unused (set): Set containing IDs of unused relation names.
        dataset (list): Original dataset containing triples.
        delta (set): Set representing the sampled triples.
        sampling_method (tuple): Tuple representing the sampling method and its description.
        enforce_relation_names (bool): Flag indicating whether to enforce missing relation names.
        enforce_entities (bool): Flag indicating whether to enforce missing entities.
        progress_bar (tqdm.tqdm): Progress bar for sampling progress visualization.
        subgraph_relation_names (list): List to store relation names present in the subgraph.
        sampled_entity_ids (set): Set containing IDs of sampled entities.
        entities_per_step (int): Maximum number of entities to sample per step.
        relation_names_amount (int, optional): Number of relation names to include in Feature Sampling. Defaults to -1.

    Returns:
        tuple: Tuple containing updated entity and relation name sets, delta set, lists of unused entity and relation name IDs,
               list of relation names in the subgraph, and set of sampled entity IDs.
    """

    samples = set()
    # samples an entity id as sampled_entity_id
    # -- enforce relation names in case of Entity sampling --
    if (enforce_relation_names and sampling_method == Constants.ENTITY_SAMPLING and
            len(relation_name_ids_unused) > 0):
        logging.info(f"Enforcing {len(relation_name_ids_unused)} relation names to be present in the subgraph.")
        for missing_relation_name in relation_name_ids_unused:
            candidates = relation_name_set[missing_relation_name]
            # iterate through all candidate triples
            for candidate in candidates:
                head, _, tail = dataset[candidate]
                # logging.info(f"({head},\t{_},\t{tail})")
                if head == tail:
                    pass
                # Add tail entity to samples if head was already sampled
                if head in sampled_entity_ids:
                    samples.add(tail)
                    break
                # Add head entity to samples if tail was already sampled
                elif tail in sampled_entity_ids:
                    samples.add(head)
                    break
                else:
                    samples.add(head)
                    samples.add(tail)

    # -- enforce entities --
    if enforce_entities and len(entity_ids_unused) > 0:
        logging.info(f"Enforcing {len(entity_ids_unused)} entities to be present in the subgraph")
        # Add all unused entity ids to samples
        while len(entity_ids_unused) > 0:
            samples.add(entity_ids_unused.pop())
        logging.log(Constants.DATA_LEVEL_LOGGING, f"Enforced the following entities:\n{samples}")

    # only sample additional entities, if more entities for this sampling step are allowed
    if entities_per_step > len(samples):
        # add entities_per_step times sampled_entity_id to sampled_entity_id_set
        for i in range(entities_per_step):
            sampled_entity_id = random.choice(list(entity_set.keys()))
            samples.add(sampled_entity_id)
            # logging.log(Constants.DATA_LEVEL_LOGGING, f"Sampled entity id {sampled_entity_id}.")

    safety_counter = 0
    for sampled_entity_id in samples:
        # check if sampled_entity_id is still in entity_set
        if sampled_entity_id in entity_set:
            candidates_list = entity_set[sampled_entity_id]
        else:
            # raise error if trapped in else statement
            safety_counter += 1
            if safety_counter >= 5:
                continue
                # raise KeyError(f"There were entities enforced, which are not present in the subgraph. "
                #                f"Try increasing the relative subgraph size.\n"
                #                f"Sample {sampled_entity_id} was attempted {safety_counter} times, without success!")
            continue

        # iterate through all candidate triples
        for candidate in candidates_list:
            head, relation, tail = dataset[candidate]
            if ((head in sampled_entity_ids) or (tail in sampled_entity_ids) or
                    ((head == tail) and (head not in sampled_entity_ids))):
                delta.add(candidate)
                relation_name_ids_unused.discard(relation)
                if relation not in subgraph_relation_names:
                    subgraph_relation_names.append(relation)

        # add sampled_entity_id to set of sampled_entity_ids, and remove it from unused entity ids
        sampled_entity_ids.add(sampled_entity_id)
        entity_ids_unused.discard(sampled_entity_id)
        try:
            del entity_set[sampled_entity_id]
        except KeyError:
            pass

    # update progress bar
    if not no_progress_bar:
        progress_bar.n = len(delta)
        progress_bar.refresh()

    # return all changed lists for the different sampling methods
    if sampling_method == Constants.FEATURE_SAMPLING:
        return (entity_set, relation_name_set, delta, entity_ids_unused, relation_name_ids_unused,
                relation_names_amount, subgraph_relation_names, sampled_entity_ids)
    else:
        return (entity_set, relation_name_set, delta, entity_ids_unused, relation_name_ids_unused,
                subgraph_relation_names, sampled_entity_ids)


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
