import logging
import random
import traceback

from ensemble import Constants


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
    except KeyError as key:
        logging.error(f"Received KeyError with key {key}")

    except Exception as error:
        logging.error(f"{error}\t{traceback.format_exc()}")

    # ensure delta only has unique triple indices

    if unique_before == unique_after:
        removed_ids_counter += 1
        # logging.debug(f"No indices were added to delta")

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
