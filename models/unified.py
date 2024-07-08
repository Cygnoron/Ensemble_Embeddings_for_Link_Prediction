import importlib
import logging

import torch
from torch import nn

from ensemble import Constants
from models import COMPLEX_MODELS
from models import EUC_MODELS
from models import HYP_MODELS
from models import KGModel

UNIFIED_MODELS = ["Unified"]


class Unified(KGModel):
    def __init__(self, args, embedding_models):
        super(Unified, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                      args.init_size, args.model, args.theta_calculation,
                                      subgraph_amount=args.subgraph_amount,
                                      batch_size=args.batch_size, aggregation_method=args.aggregation_method,
                                      embedding_models=embedding_models)

        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

        self.theta_ent_unified = self.theta_ent_unified.to('cuda')
        self.theta_rel_unified = self.theta_rel_unified.to('cuda')
        self.cands_ent = self.cands_ent.to('cuda')
        self.cands_rel = self.cands_rel.to('cuda')
        self.active_models = list(range(self.subgraph_amount))
        self.embedding_methods = set()
        self.rel_last_epoch = None
        self.single_train_loss = {}
        self.sim = None
        self.collect_embedding_methods()

    def get_class(self, class_name, base_method=False):
        if class_name in EUC_MODELS:
            euclidean_method = importlib.import_module("models.euclidean")
            if base_method:
                class_name = getattr(euclidean_method, "BaseE")
            else:
                class_name = getattr(euclidean_method, class_name)
        elif class_name in COMPLEX_MODELS:
            complex_method = importlib.import_module("models.complex")
            if base_method:
                class_name = getattr(complex_method, "BaseC")
            else:
                class_name = getattr(complex_method, class_name)
        elif class_name in HYP_MODELS:
            hyperbolic_method = importlib.import_module("models.hyperbolic")
            if base_method:
                class_name = getattr(hyperbolic_method, "BaseH")
            else:
                class_name = getattr(hyperbolic_method, class_name)
        else:
            raise ValueError(f"The method {class_name} is not in the list of knowledge graph embedding "
                             f"methods. Please add it to the list in the models module.")

        return class_name

    def get_queries(self, queries):
        method = None
        for index, embedding_method in self.embedding_methods:
            method = self.get_class(embedding_method)
        lhs_e, lhs_biases = getattr(method, "get_queries")(self, queries)

        # TODO handle multiple methods in forward
        #  make lhs_e and lhs_biases lists for multiple methods

        return lhs_e, lhs_biases

    def get_rhs(self, queries, eval_mode):
        method = None
        for index, embedding_method in self.embedding_methods:
            method = self.get_class(embedding_method, base_method=True)

        rhs_e, rhs_biases = getattr(method, "get_rhs")(self, queries, eval_mode)

        # TODO handle multiple methods in forward
        #  make rhs_e and rhs_biases lists for multiple methods

        return rhs_e, rhs_biases

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        method = None
        for index, embedding_method in self.embedding_methods:
            method = self.get_class(embedding_method, base_method=True)
            self.set_sim(embedding_method)
        score = getattr(method, "similarity_score")(self, lhs_e, rhs_e, eval_mode)

        # TODO handle multiple methods in forward
        #  make score list for multiple methods

        return score

    def set_sim(self, embedding_method):
        if embedding_method not in EUC_MODELS:
            return

        methods_dist = ["TransE", "DistMult", "MurE", "RotE", "RefE", "AttE"]
        methods_dot = ["CP"]

        if embedding_method in methods_dot:
            self.sim = "dot"
        elif embedding_method in methods_dist:
            self.sim = "dist"
        else:
            raise ValueError(f"There was no sim specified for the given embedding method {embedding_method}.")

    def train_single_models(self, queries, epoch_step):

        denominator = epoch_step + 1

        for embedding_model in self.embedding_models:
            if embedding_model['args'].subgraph not in self.single_train_loss.keys():
                self.single_train_loss[embedding_model['args'].subgraph] = 0

            optimizer = embedding_model['optimizer']

            single_model_loss = optimizer.calculate_loss(queries)
            optimizer.optimizer.zero_grad()
            single_model_loss.backward()
            optimizer.optimizer.step()

            self.single_train_loss[embedding_model['args'].subgraph] = (
                    (self.single_train_loss[embedding_model['args'].subgraph] * (denominator - 1) + single_model_loss)
                    / denominator)
            pass
        pass

    def forward_unified(self, queries, eval_mode=False):

        self.stack_theta_cands(queries)
        self.calculate_attention(queries)

        predictions, factors = self.single_model_forward(queries)
        return predictions, factors

    def aggregated_score(self, queries):
        aggregated_scores = []
        aggregated_targets = []
        candidates = None
        for embedding_method in self.embedding_methods:
            model = self.embedding_models[embedding_method[0]]['model']

            if candidates is None:
                candidates = model.get_rhs(queries, eval_mode=True)
            q = model.get_queries(queries)
            rhs = model.get_rhs(queries, eval_mode=False)

            aggregated_scores += [model.score(q, candidates, eval_mode=True)]
            aggregated_targets += [model.score(q, rhs, eval_mode=False)]

        if len(self.embedding_methods) == 1:
            return aggregated_scores[0], aggregated_targets[0]

        aggregated_scores = torch.stack(aggregated_scores, dim=-1)
        aggregated_targets = torch.stack(aggregated_targets, dim=-1)

        if self.aggregation_method[0] == Constants.MAX_SCORE_AGGREGATION[0]:
            aggregated_scores, _ = torch.max(aggregated_scores, dim=-1)
            aggregated_targets, _ = torch.max(aggregated_targets, dim=-1)

        elif self.aggregation_method[0] == Constants.AVERAGE_SCORE_AGGREGATION[0]:
            aggregated_scores = torch.mean(aggregated_scores, dim=-1)
            aggregated_targets = torch.mean(aggregated_targets, dim=-1)

        elif self.aggregation_method[0] == Constants.ATTENTION_SCORE_AGGREGATION[0]:
            raise ValueError(f"The aggregation method \"{self.aggregation_method[1]}\" has not been implemented yet!")
        else:
            raise ValueError(f"The aggregation method \"{self.aggregation_method[1]}\" does not exist!")

        return aggregated_scores.to('cuda'), aggregated_targets.to('cuda')

    def calculate_attention(self, queries):
        logging.debug(f"Attention")
        # TODO implement attention
        #  multiple functions for different attentions?

        self.stack_theta_cands(queries)

        if self.theta_calculation[0] == Constants.REGULAR_THETA[0]:
            # EXAMPLE WN18RR dimensions (40943 entities, 11*2 relation names, N subgraphs)
            # GOAL: att_weights_ent.size() = [batch, rank, N]     att_weights_ent.size() = [batch, rank, N]
            # INPUT: theta_ent_unified = [batch, rank, N]         theta_rel_unified = [batch, rank, N]
            #        cands_ent = [batch, rank, N]                 cands_rel = [batch, rank, N]

            # self.att_ent[queries[:, 0]] = torch.sum(self.theta_ent_unified * self.cands_ent, dim=1)
            # self.att_rel[queries[:, 1]] = torch.sum(self.theta_rel_unified * self.cands_rel, dim=1)
            self.att_rel = self.theta_rel_unified.weight.data * self.cands_rel
            self.att_ent = self.theta_ent_unified.weight.data * self.cands_ent

        elif self.theta_calculation[0] == Constants.REVERSED_THETA[0]:
            raise ValueError(f"The method \"{self.theta_calculation[1]}\" has not been implemented yet!")
            # EXAMPLE WN18RR dimensions (40943 entities, 11*2 relation names, N subgraphs)
            # GOAL: att_weights_ent.size() = [40943, 32, N]     att_weights_ent.size() = [22, 32, N]
            # INPUT: theta_ent = [   22, 32, N]                 theta_rel = [40943, 32, N]
            #        cands_ent = [40943, 32, N]                 cands_rel = [   22, 32, N]

            # theta_ent = theta_ent.unsqueeze(0)  # Shape: [1, 22, 32, N]
            # theta_ent = theta_ent.expand(cands_ent.size(0), -1, -1, -1)  # Shape: [40943, 22, 32, N]
            # att_weights_ent = torch.sum(theta_ent * cands_ent.unsqueeze(1), dim=1)  # Shape: [40943, 32, N]
            # logging.debug(f"att_weights_ent size: {att_weights_ent.size()}")
            #
            # theta_rel = theta_rel.unsqueeze(0)  # Shape: [1, 40943, 32, N]
            # theta_rel = theta_rel.expand(cands_rel.size(0), -1, -1, -1)  # Shape: [22, 40943, 32, N]
            # att_weights_rel = torch.sum(theta_rel * cands_rel.unsqueeze(1), dim=1)  # Shape: [22, 32, N]
            # logging.debug(f"att_weights_rel size: {att_weights_rel.size()}")

        elif self.theta_calculation[0] == Constants.RELATION_THETA[0]:
            raise ValueError(f"The method \"{self.theta_calculation[1]}\" has not been implemented yet!")
            # EXAMPLE WN18RR dimensions (40943 entities, 11*2 relation names, N subgraphs)
            # GOAL: att_weights_ent.size() = [40943, 32, N]     att_weights_ent.size() = [22, 32, N]
            # INPUT: theta_ent = [   22, 32, N]                 theta_rel = [22, 32, N]
            #        cands_ent = [40943, 32, N]                 cands_rel = [22, 32, N]

            # theta_ent = theta_ent.unsqueeze(0)  # Shape: [1, 22, 32, N]
            # theta_ent = theta_ent.expand(cands_ent.size(0), -1, -1, -1)  # Shape: [40943, 22, 32, N]
            # att_weights_ent = torch.sum(theta_ent * cands_ent.unsqueeze(1), dim=1)  # Shape: [40943, 32, N]
            # logging.debug(f"att_weights_ent size: {att_weights_ent.size()}")
            #
            # att_weights_rel = torch.sum(theta_rel * cands_rel, dim=-1, keepdim=True)
            # logging.debug(f"att_weights_rel size: {att_weights_rel.size()}")

        elif self.theta_calculation[0] == Constants.MULTIPLIED_THETA[0]:
            raise ValueError(f"The method \"{self.theta_calculation[1]}\" has not been implemented yet!")
            # EXAMPLE WN18RR dimensions (40943 entities, 11*2 relation names, N subgraphs)
            # GOAL: att_weights_ent.size() = [40943, 32, N]     att_weights_ent.size() = [22, 32, N]
            # INPUT: theta_ent = [40943, 32, N]                 theta_rel = [22, 32, N]
            #        cands_ent = [40943, 32, N]                 cands_rel = [22, 32, N]

        # apply activation function
        self.att_ent = self.act(self.att_ent)
        self.att_rel = self.act(self.att_rel)

    def stack_theta_cands(self, queries):
        logging.debug(f"Stacking")
        cands_ent_temp = []
        cands_rel_temp = []
        theta_ent_temp = []
        theta_rel_temp = []

        for embedding_model in self.embedding_models:
            model = embedding_model["model"]
            args = embedding_model["args"]
            model_name = args.model_name

            if args.model_dropout:
                logging.debug(f"Ignoring subgraph {args.subgraph_num} in calculation due to model dropout.")
                if args.subgraph_num in self.active_models:
                    self.active_models.remove(args.subgraph_num)
                continue

            embedding_ent = model.entity.weight.data
            embedding_rel = model.rel.weight.data
            if model_name in COMPLEX_MODELS:
                embedding_ent = model.embeddings[0].weight.data
                embedding_rel = model.embeddings[1].weight.data
            elif model_name in HYP_MODELS:
                # TODO check hyperbolic case
                embedding_rel = torch.chunk(embedding_rel, 2, dim=1)

            # cands_rel_temp.append(append_entries(queries, embedding_rel[queries[:, 1]],
            #                                      self.cands_rel.clone(memory_format=torch.preserve_format)))
            # cands_ent_temp.append(append_entries(queries, embedding_rel[queries[:, 0]],
            #                                      self.cands_ent.clone(memory_format=torch.preserve_format)))

            # buffer_e0 = torch.zeros(self.sizes[0], self.rank).unsqueeze(-1)
            # buffer_e = model.theta_ent.weight.data[queries[:, 0]].unsqueeze(-1)
            # theta_ent_temp.append(buffer_e)
            theta_ent_temp.append(model.theta_ent.weight.data[queries[:, 0]])
            cands_ent_temp.append(embedding_ent[queries[:, 0]])
            theta_rel_temp.append(model.theta_rel.weight.data[queries[:, 1]])
            cands_rel_temp.append(embedding_rel[queries[:, 1]])

        # cands_ent: [40943, 32, N]     cands_rel: [22, 32, N]
        self.cands_ent[queries[:, 0]] = torch.stack(cands_ent_temp, dim=-1)
        self.cands_rel[queries[:, 1]] = torch.stack(cands_rel_temp, dim=-1)

        # theta_ent_unified: [40943, 32, N]     theta_rel_unified: [22, 32, N]
        self.theta_ent_unified.weight.data[queries[:, 0]] = torch.stack(theta_ent_temp, dim=-1).to('cuda')
        self.theta_rel_unified.weight.data[queries[:, 1]] = torch.stack(theta_rel_temp, dim=-1).to('cuda')

    def single_model_forward(self, queries):
        logging.debug("Forward pass")
        predictions = []
        factors_h = []
        factors_r = []
        factors_t = []
        for embedding_model in self.embedding_models:
            args = embedding_model['args']
            logging.debug(f"Forward pass model {args.subgraph}")

            # Create masks for entities and relation names
            entity_mask = torch.isin(queries[:, 0], torch.tensor(args.entities).cuda())
            relation_mask = torch.isin(queries[:, 1], torch.tensor(args.relation_names).cuda())

            query_mask = entity_mask & relation_mask
            model_queries = queries[query_mask]

            prediction = torch.zeros(len(queries), self.sizes[0], dtype=self.data_type).cuda()

            factor_h = torch.zeros(len(queries), self.rank, dtype=self.data_type).cuda()
            factor_r = torch.zeros(len(queries), self.rank, dtype=self.data_type).cuda()
            factor_t = torch.zeros(len(queries), self.rank, dtype=self.data_type).cuda()

            optimizer = embedding_model["optimizer"]
            # prediction.size() = [batch, 40943]    factor_X.size() = [batch, 32]
            prediction[query_mask], factor = optimizer.model.forward(model_queries, eval_mode=True)

            if embedding_model['model'].model_name in COMPLEX_MODELS:
                factor_buffer_h, factor_buffer_r, factor_buffer_t = factor
                # factor_x[:, self.rank // 2:] = factor_x[:, :self.rank // 2]
                factor_buffer_h = torch.cat([factor_buffer_h, factor_buffer_h], dim=-1)
                factor_buffer_r = torch.cat([factor_buffer_r, factor_buffer_r], dim=-1)
                factor_buffer_t = torch.cat([factor_buffer_t, factor_buffer_t], dim=-1)
                factor = (factor_buffer_h, factor_buffer_r, factor_buffer_t)

            predictions += [prediction]

            factor_h[query_mask], factor_r[query_mask], factor_t[query_mask] = factor
            factors_h += [factor_h]
            factors_r += [factor_r]
            factors_t += [factor_t]

        # predictions.size() = [batch, 40943, N]    factors_X.size() = [batch, 32, N]
        # att_ent.size()     = [40943,    32, N]    att_rel.size() = [22   , 32, N]
        # TODO get correct attention
        predictions = torch.stack(predictions, dim=-1)
        factors_h = torch.stack(factors_h, dim=-1)
        factors_r = torch.stack(factors_r, dim=-1)
        factors_t = torch.stack(factors_t, dim=-1)

        use_attention = False
        if use_attention:
            predictions = torch.sum(torch.mean(self.att_ent, dim=1).unsqueeze(0) * predictions, dim=-1)
            factors_h = torch.sum(self.att_ent[queries[:, 0]] * factors_h, dim=-1)
            factors_r = torch.sum(self.att_rel[queries[:, 1]] * factors_r, dim=-1)
            factors_t = torch.sum(self.att_ent[queries[:, 0]] * factors_t, dim=-1)
        else:
            predictions = torch.mean(predictions, dim=-1)
            factors_h = torch.mean(factors_h, dim=-1)
            factors_r = torch.mean(factors_r, dim=-1)
            factors_t = torch.mean(factors_t, dim=-1)

        factors = (factors_h, factors_r, factors_t)
        return predictions, factors

    def update_single_models(self, queries, loss):
        b_begin = 0
        # logging.info(f"Updating single models...")
        # write new unified embedding into all embedding_models
        for index, embedding_model in enumerate(self.embedding_models):
            logging.debug(f"Updating subgraph {index:03d}")
            if ('entities' not in embedding_model.keys()) or ('relation_names' not in embedding_model.keys()):
                entities, relation_names = embedding_model['dataset'].get_entities_relation_names(self.sizes,
                                                                                                  double_relations=True)
                embedding_model['entities'] = entities
                embedding_model['relation_names'] = relation_names

            model = embedding_model['model'].cuda()
            entities = embedding_model['entities']
            relation_names = embedding_model['relation_names']
            optimizer = embedding_model['optimizer']

            use_attention = True
            if use_attention:

                # self.theta_ent_unified = [40943, 32, N], self.entity = [40943, 32]
                self.att_ent[queries[:, 0], :, index] = (self.theta_ent_unified.weight.data[queries[:, 0], :, index] *
                                                         model.entity.weight.data[queries[:, 0]])

                # self.theta_rel_unified = [batch, 32, N], self.rel = [22, 32]
                self.att_rel[queries[:, 1], :, index] = (self.theta_rel_unified.weight.data[queries[:, 1], :, index] *
                                                         model.rel.weight.data[queries[:, 1]])

                unified_att_ent = self.act(torch.stack([model.att_ent_single,
                                                        torch.mean(self.att_ent[:, :, index], dim=1)], dim=1))
                unified_att_rel = self.act(torch.stack([model.att_rel_single,
                                                        torch.mean(self.att_rel[:, :, index], dim=1)], dim=1))

                optimizer.model.entity.weight.data[entities] = (model.entity.weight.data[entities] *
                                                                unified_att_ent[entities][:, 0].unsqueeze(-1) +
                                                                self.entity.weight.data[entities] *
                                                                unified_att_ent[entities][:, 1].unsqueeze(-1))
                optimizer.model.rel.weight.data[relation_names] = (model.rel.weight.data[relation_names] *
                                                                   unified_att_rel[relation_names][:, 0].unsqueeze(-1) +
                                                                   self.rel.weight.data[relation_names] *
                                                                   unified_att_rel[relation_names][:, 1].unsqueeze(-1))
            else:
                optimizer.model.entity.weight.data[relation_names] = (model.entity.weight.data[relation_names] * 0.1 +
                                                                      self.entity.weight.data[relation_names] * 0.9)
                optimizer.model.rel.weight.data[relation_names] = (model.rel.weight.data[relation_names] * 0.1 +
                                                                   self.rel.weight.data[relation_names] * 0.9)

    def collect_embedding_methods(self):
        for index, embedding_model in enumerate(self.embedding_models):
            embedding_method = embedding_model['args'].model_name

            if not any(embedding_method == method for _, method in self.embedding_methods):
                self.embedding_methods.add((index, embedding_method))

        logging.info(f"Found the following embedding methods:\n{self.embedding_methods}")

    def log_rel_emb_change(self, b_begin, total_loss):
        step = (b_begin // self.batch_size) * self.batch_size
        model = self.embedding_models[0]['model']
        if self.rel_last_epoch is None:
            self.rel_last_epoch = model.rel.weight.data.clone(memory_format=torch.preserve_format)
            return

        difference = torch.abs(self.rel_last_epoch - model.rel.weight.data)
        difference_sum = torch.sum(torch.sum(difference, dim=-1), dim=-1)
        difference_mean = torch.mean(torch.mean(difference, dim=-1), dim=-1)

        logging.info(f"After {step} examples: Total / Average change {difference_sum:.3f} / {difference_mean:.6f} "
                     f"with loss {total_loss:.3f}")

        self.rel_last_epoch = model.rel.weight.data.clone(memory_format=torch.preserve_format)


def append_entries(queries, query_tensor, embedding_tensor):
    # Extract indices from queries[:, 1]

    indices = queries[:, 1]
    # Initialize buffer_r0 with correct dimensions

    embedding_tensor = [[] for _ in range(len(embedding_tensor))]
    # Append entries from buffer_r to buffer_r0 at the correct indices

    for i, idx in enumerate(indices):
        embedding_tensor[idx].append(query_tensor[i])
    # Determine the maximum number of entries at any index in buffer_r0

    max_entries = max(len(entries) for entries in embedding_tensor)
    # Initialize buffer_r0_final to store concatenated tensors

    embedding_tensor_final = []
    # Concatenate tensors along a new dimension (unsqueeze)

    for tensors in embedding_tensor:
        if tensors:
            padded_tensors = torch.stack(tensors, dim=0)
            # Pad with zeros if there are fewer entries than max_entries
            if len(tensors) < max_entries:
                padding_shape = (max_entries - len(tensors), query_tensor.shape[1])
                zeros_padding = torch.zeros(padding_shape, dtype=padded_tensors.dtype, device=padded_tensors.device)
                padded_tensors = torch.cat([padded_tensors, zeros_padding], dim=0)
        else:
            padded_tensors = torch.zeros(max_entries, query_tensor.shape[1], dtype=query_tensor.dtype,
                                         device=query_tensor.device)

        embedding_tensor_final.append(padded_tensors)
    # Stack all tensors in buffer_r0_final along dim=0
    embedding_tensor_final = torch.stack(embedding_tensor_final, dim=0)

    return embedding_tensor_final
