"""Dataset class for loading and processing KG datasets."""
import logging
import os
import pickle as pkl

import numpy as np
import torch


class KGDataset(object):
    """Knowledge Graph dataset class."""

    def __init__(self, data_path, debug, test_valid_file_dir=""):
        """Creates KG dataset object for data loading.

        Args:
             data_path: Path to directory containing train/valid/test pickle files produced by process.py
             debug: boolean indicating whether to use debug mode or not
             if true, the dataset will only contain 1000 examples for debugging.
             info_directory: Possible path to valid/test and to_skip pickle files created from sampling process
        """
        self.relation_name_set = []
        self.entity_set = []
        self.test_valid_file_dir = test_valid_file_dir
        self.data_path = data_path
        self.debug = debug
        self.data = {}
        for split in ["train", "test", "valid"]:
            try:
                file_path = os.path.join(self.data_path, split + ".pickle")
                with open(file_path, "rb") as in_file:
                    self.data[split] = pkl.load(in_file)
            except FileNotFoundError:
                file_path = os.path.join(test_valid_file_dir, split + ".pickle")
                with open(file_path, "rb") as in_file:
                    self.data[split] = pkl.load(in_file)
        try:
            filters_file = open(os.path.join(self.data_path, "to_skip.pickle"), "rb")
        except FileNotFoundError:
            filters_file = open(os.path.join(test_valid_file_dir, "to_skip.pickle"), "rb")
        self.to_skip = pkl.load(filters_file)
        filters_file.close()
        max_axis = np.max(self.data["train"], axis=0)
        self.n_entities = int(max(max_axis[0], max_axis[2]) + 1)
        self.n_predicates = int(max_axis[1] + 1) * 2

    def get_examples(self, split, rel_idx=-1):
        """Get examples in a split.

        Args:
            split: String indicating the split to use (train/valid/test)
            rel_idx: integer for relation index to keep (-1 to keep all relation)

        Returns:
            examples: torch.LongTensor containing KG triples in a split
        """
        examples = self.data[split]
        if split == "train":
            copy = np.copy(examples)
            tmp = np.copy(copy[:, 0])
            copy[:, 0] = copy[:, 2]
            copy[:, 2] = tmp
            copy[:, 1] += self.n_predicates // 2
            examples = np.vstack((examples, copy))
        if rel_idx >= 0:
            examples = examples[examples[:, 1] == rel_idx]
        if self.debug:
            examples = examples[:1000]
        return torch.from_numpy(examples.astype("int64"))

    def get_filters(self, ):
        """Return filter dict to compute ranking metrics in the filtered setting."""
        return self.to_skip

    def get_shape(self):
        """Returns KG dataset shape."""
        return self.n_entities, self.n_predicates, self.n_entities

    def get_entities_relation_names(self, sizes, double_relations=False):
        """Returns lists with all entity and all relation name ids"""
        # Calculate sets, if they are None
        if (len(self.entity_set) == 0) or (len(self.relation_name_set) == 0):
            logging.debug(f"Calculating new entity and relation name from the dataset.")
            entity_set = set()
            relation_name_set = set()
            for triple in self.data['train']:
                head, relation, tail = triple
                entity_set.add(head)
                relation_name_set.add(relation)
                entity_set.add(tail)

            # return sets of entities and relation names
            if double_relations:
                relation_name_set = list(relation_name_set)
                buffer = []
                for relation in relation_name_set:
                    buffer.append(relation + int(sizes[1] / 2))
                relation_name_set += buffer
                self.entity_set = list(entity_set)
                self.relation_name_set = list(relation_name_set)
                return list(entity_set), list(relation_name_set)
            else:
                self.entity_set = list(entity_set)
                self.relation_name_set = list(relation_name_set)
                return list(entity_set), list(relation_name_set)
        else:
            return self.entity_set, self.relation_name_set
