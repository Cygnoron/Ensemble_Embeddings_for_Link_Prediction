import importlib
import json
import logging
import os
import time

import torch
from torch import nn
from tqdm import tqdm

import models as models
from datasets.kg_dataset import KGDataset
from ensemble import Constants, util
from models import COMPLEX_MODELS
from models import EUC_MODELS
from models import HYP_MODELS
from models import KGModel
from optimizers import regularizers, KGOptimizer
from utils.train import count_params

UNIFIED_MODELS = ["Unified"]


class Unified(KGModel):
    def __init__(self, args, init_args):
        super(Unified, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                      args.init_size, args.model, args.theta_calculation,
                                      subgraph_amount=args.subgraph_amount,
                                      batch_size=args.batch_size, aggregation_method=args.aggregation_method,
                                      embedding_models="Unified")

        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

        self.theta_ent_unified = self.theta_ent_unified.to('cuda')
        self.theta_rel_unified = self.theta_rel_unified.to('cuda')
        self.cands_ent = self.cands_ent.to('cuda')
        self.cands_rel = self.cands_rel.to('cuda')

        # list of single model optimizers
        self.single_models = []
        self.single_model_args = []
        self.init_single_models(args, init_args)

        # self.active_models = list(range(self.subgraph_amount))
        self.embedding_methods = set()
        self.single_train_loss = {}
        self.sim = None
        self.validation = False
        self.collect_embedding_methods()

    def init_single_models(self, args, init_args):
        logging.info("-/\tCreating single embedding models\t\\-")
        time_start_model_creation = time.time()

        init_progress_bar = None
        if not init_args.no_progress_bar:
            init_progress_bar = tqdm(total=init_args.subgraph_amount, desc=f"Creating single embedding models",
                                     unit=" model(s) ", position=0, leave=True)

        for subgraph_num in list(init_args.subgraph_embedding_mapping.keys()):
            model = init_args.subgraph_embedding_mapping[subgraph_num]

            # load args
            args_subgraph = util.get_args(args, model)
            args_subgraph.model_name = model
            args_subgraph.model = model
            args_subgraph.subgraph = f"sub_{subgraph_num:03d}"
            args_subgraph.model_dropout = False

            # load data
            logging.debug(f"Loading data for subgraph {args_subgraph.subgraph}.")
            dataset_subgraph = KGDataset(os.path.join(init_args.dataset_path, args_subgraph.subgraph),
                                         args_subgraph.debug, test_valid_file_dir=init_args.test_valid_file_dir)
            args_subgraph.entities, args_subgraph.relation_names = dataset_subgraph.get_entities_relation_names(
                args_subgraph.sizes, double_relations=True)

            # create model
            model = getattr(models, args_subgraph.model)(args_subgraph)
            total = count_params(model)
            logging.debug(f"Total number of parameters of {args_subgraph.subgraph}: {total}")

            # Get optimizer
            regularizer = (getattr(regularizers, args_subgraph.regularizer)(args_subgraph.reg))
            optim_method = (getattr(torch.optim, args_subgraph.optimizer)(model.parameters(),
                                                                          lr=args_subgraph.learning_rate))
            optimizer = KGOptimizer(model, regularizer, optim_method, args_subgraph.batch_size,
                                    args_subgraph.neg_sample_size, bool(args_subgraph.double_neg),
                                    args_subgraph.no_progress_bar)
            model.to('cuda')
            self.single_model_args.append(args_subgraph)
            self.single_models.append(optimizer)

            # save config
            with (open(os.path.join(init_args.model_setup_config_dir,
                                    f"config_{args_subgraph.subgraph}_{args_subgraph.model_name}.json"),
                       "w") as json_file):
                json.dump(vars(args_subgraph), json_file)

            if not init_args.no_progress_bar:
                # Update progress bar
                init_progress_bar.n = subgraph_num + 1
                init_progress_bar.refresh()

        time_stop_model_creation = time.time()

        # Close progress bar
        if not init_args.no_progress_bar:
            init_progress_bar.close()
        logging.info(f"-\\\tSuccessfully created all embedding_models in "
                     f"{util.format_time(time_start_model_creation, time_stop_model_creation)}\t/-")

    def collect_embedding_methods(self):
        for single_model in self.single_models:
            embedding_method = single_model.model.model_name

            if not any(embedding_method == method for method in self.embedding_methods):
                self.embedding_methods.add(embedding_method)

        logging.info(f"Found the following embedding methods:\n{util.format_set(self.embedding_methods)}")

    def forward(self, queries, eval_mode=False):
        # run single model
        self.train_single_models(queries)
        # calculate cross model attention
        self.calculate_cross_model_attention(queries)
        # combine single embeddings into unified embedding
        self.combine_embeddings(queries)
        # get queries on unified embedding
        lhs_e, lhs_biases = self.get_queries(queries)
        # get rhs on unified embedding
        rhs_e, rhs_biases = self.get_rhs(queries, eval_mode)
        # get predictions on unified embedding
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases), eval_mode=eval_mode)
        # get factors on unified embedding
        factors = self.get_factors(queries)

        self.update_single_models(queries)

        return predictions, factors

    def train_single_models(self, queries):
        if self.validation:
            pass
        else:
            for single_model in self.single_models:
                # actual_queries= get_actual_queries(queries, single_model=single_model)
                actual_queries = queries.cuda()

                l = single_model.calculate_loss(actual_queries).cuda()
                single_model.optimizer.zero_grad()
                l.backward()
                single_model.optimizer.step()

    def calculate_cross_model_attention(self, queries):
        theta_ent_temp = []
        cands_ent_temp = []
        theta_rel_temp = []
        cands_rel_temp = []

        for single_model in self.single_models:
            model = single_model.model
            # args =model
            #
            # if args.model_dropout:
            #     logging.debug(f"Ignoring subgraph {args.subgraph_num} in calculation due to model dropout.")
            #     if args.subgraph_num in self.active_models:
            #         self.active_models.remove(args.subgraph_num)
            #     continue

            # TODO try with actual queries
            actual_queries, query_mask = get_actual_queries(queries, single_model=single_model)
            embedding_ent = torch.zeros(len(queries), self.rank).cuda()
            embedding_rel = torch.zeros(len(queries), self.rank).cuda()

            embedding_ent[query_mask] = model.entity.weight.data[actual_queries[:, 0]]
            embedding_rel[query_mask] = model.rel.weight.data[actual_queries[:, 1]]
            if model.model_name in COMPLEX_MODELS:
                embedding_ent = model.embeddings[0].weight.data[actual_queries[:, 0]]
                embedding_rel = model.embeddings[1].weight.data[actual_queries[:, 1]]
            elif model.model_name in HYP_MODELS:
                # TODO check hyperbolic case
                embedding_rel = torch.chunk(embedding_rel[actual_queries[:, 1]], 2, dim=1)

            # Stack cands and theta for cross model attention
            theta_ent_temp.append(model.theta_ent.weight.data[queries[:, 0]])
            cands_ent_temp.append(embedding_ent)
            theta_rel_temp.append(model.theta_rel.weight.data[queries[:, 1]])
            cands_rel_temp.append(embedding_rel)

        # cands_ent: [40943, 32, N]     cands_rel: [22, 32, N]
        self.cands_ent[queries[:, 0]] = torch.stack(cands_ent_temp, dim=-1).cuda()
        self.cands_rel[queries[:, 1]] = torch.stack(cands_rel_temp, dim=-1).cuda()
        del cands_ent_temp, cands_rel_temp

        # theta_ent_unified: [40943, 32, N]     theta_rel_unified: [22, 32, N]
        self.theta_ent_unified.weight.data[queries[:, 0]] = torch.stack(theta_ent_temp, dim=-1).cuda()
        self.theta_rel_unified.weight.data[queries[:, 1]] = torch.stack(theta_rel_temp, dim=-1).cuda()
        del theta_ent_temp, theta_rel_temp

        # cross model attention
        self.att_ent_cross_model[queries[:, 0]] = self.calculate_attention(
            self.theta_ent_unified.weight.data[queries[:, 0]], self.cands_ent[queries[:, 0]])

        self.att_rel_cross_model[queries[:, 1]] = self.calculate_attention(
            self.theta_rel_unified.weight.data[queries[:, 1]], self.cands_rel[queries[:, 1]])

    def calculate_attention(self, theta, cands, dim=-1):
        softmax = self.act
        if not dim == -1:
            softmax = nn.Softmax(dim=dim)

        logging.debug(f"Sizes for attention:\nTheta\t{theta.size()}\nCands:\t{cands.size()}")
        return softmax(theta * cands)

    def combine_ranks(self, attention, dim_softmax=-1, rank_dim=-1):
        sizes = attention.size()
        softmax = self.act
        if dim_softmax != -1:
            softmax = nn.Softmax(dim=dim_softmax)

        if rank_dim == -1:
            for dimension_index, size in enumerate(sizes):
                if size == self.rank:
                    rank_dim = dimension_index
                    break
            # No rank dimension was found
            if rank_dim == -1:
                raise RuntimeError(f"Rank dimension {rank_dim} is not defined")

        return softmax(torch.mean(attention, dim=rank_dim))

    def combine_embeddings(self, queries):
        ent_emb_temp = []
        rel_emb_temp = []
        rel_diag_emb_temp = []

        for single_model in self.single_models:
            if single_model.model.model_name in EUC_MODELS:
                ent_emb_temp.append(single_model.model.entity.weight.data[queries[:, 0]])
                rel_emb_temp.append(single_model.model.rel.weight.data[queries[:, 1]])
            elif single_model.model.model_name in COMPLEX_MODELS:
                ent_emb_temp.append(single_model.embeddings[0].weight.data[queries[:, 0]])
                rel_emb_temp.append(single_model.embeddings[1].weight.data[queries[:, 1]])

            try:
                rel_diag_emb_temp.append(single_model.model.rel.weight.data[queries[:, 1]])
            except ValueError:
                rel_diag_emb_temp = []

        ent_emb_temp = torch.stack(ent_emb_temp, dim=-1).cuda()
        rel_emb_temp = torch.stack(rel_emb_temp, dim=-1).cuda()
        rel_diag_emb_temp = torch.stack(rel_diag_emb_temp, dim=-1).cuda()

        self.entity.weight.data[queries[:, 0]] = torch.sum(ent_emb_temp * self.att_ent_cross_model[queries[:, 0]],
                                                           dim=-1).cuda()
        self.rel.weight.data[queries[:, 1]] = torch.sum(rel_emb_temp * self.att_rel_cross_model[queries[:, 1]],
                                                        dim=-1).cuda()
        self.rel_diag.weight.data[queries[:, 1]] = torch.sum(
            rel_diag_emb_temp * self.att_rel_cross_model[queries[:, 1]], dim=-1).cuda()

    def get_queries(self, queries):
        lhs_e_list = []
        lhs_biases_list = []
        for embedding_method in self.embedding_methods:
            if embedding_method == Constants.ATT_E:
                lhs_e, lhs_biases = self.get_queries_AttE(queries)
            else:
                method = get_class(embedding_method)
                lhs_e, lhs_biases = getattr(method, "get_queries")(self, queries)
            lhs_e_list.append(lhs_e)
            lhs_biases_list.append(lhs_biases)

        lhs_e = torch.mean(torch.stack(lhs_e_list, dim=-1), dim=-1).to('cuda')
        lhs_biases = torch.mean(torch.stack(lhs_biases_list, dim=-1), dim=-1).to('cuda')

        return lhs_e, lhs_biases

    def get_rhs(self, queries, eval_mode):
        rhs_e_list = []
        rhs_biases_list = []
        for embedding_method in self.embedding_methods:
            if embedding_method in COMPLEX_MODELS:
                rhs_e, rhs_biases = self.get_rhs_complex(queries, eval_mode)
            else:
                method = get_class(embedding_method, base_method=True)
                rhs_e, rhs_biases = getattr(method, "get_rhs")(self, queries, eval_mode)
            rhs_e_list.append(rhs_e)
            rhs_biases_list.append(rhs_biases)

        rhs_e = torch.mean(torch.stack(rhs_e_list, dim=-1), dim=-1).to('cuda')
        rhs_biases = torch.mean(torch.stack(rhs_biases_list, dim=-1), dim=-1).to('cuda')

        return rhs_e, rhs_biases

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        score_list = []
        for embedding_method in self.embedding_methods:
            if embedding_method in COMPLEX_MODELS:
                score = self.similarity_score_complex(lhs_e, rhs_e, eval_mode)
            else:
                method = get_class(embedding_method, base_method=True)
                self.set_sim(embedding_method)
                score = getattr(method, "similarity_score")(self, lhs_e, rhs_e, eval_mode)
            score_list.append(score)

        # TODO include aggregation method
        score = torch.mean(torch.stack(score_list, dim=-1), dim=-1).to('cuda')

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

    def get_factors(self, queries):

        factors_h_list = []
        factors_r_list = []
        factors_t_list = []

        for embedding_method in self.embedding_methods:
            method = get_class(embedding_method, base_method=True)
            factors_h, factors_r, factors_t = getattr(method, "get_factors")(self, queries)
            if embedding_method in COMPLEX_MODELS:
                factors_h_list.append(torch.cat((factors_h, factors_h), dim=-1))
                factors_r_list.append(torch.cat((factors_r, factors_r), dim=-1))
                factors_t_list.append(torch.cat((factors_t, factors_t), dim=-1))
            else:
                factors_h_list.append(factors_h)
                factors_r_list.append(factors_r)
                factors_t_list.append(factors_t)

        factors_h = torch.mean(torch.stack(factors_h_list, dim=-1), dim=-1).to('cuda')
        factors_r = torch.mean(torch.stack(factors_r_list, dim=-1), dim=-1).to('cuda')
        factors_t = torch.mean(torch.stack(factors_t_list, dim=-1), dim=-1).to('cuda')

        return factors_h, factors_r, factors_t

    def update_single_models(self, queries):
        for index, single_model in enumerate(self.single_models):
            actual_queries ,_= get_actual_queries(queries, single_model=single_model)
            single_model.model.entity.weight.data[actual_queries[:, 0]] = self.entity.weight.data[actual_queries[:, 0]]
            single_model.model.rel.weight.data[actual_queries[:, 1]] = self.rel.weight.data[actual_queries[:, 1]]

            single_model.model.theta_ent.weight.data[actual_queries[:, 0]] = \
                self.theta_ent_unified.weight.data[actual_queries[:, 0]][:, :, index]
            single_model.model.theta_rel.weight.data[actual_queries[:, 1]] = \
                self.theta_rel_unified.weight.data[actual_queries[:, 1]][:, :, index]


def get_actual_queries(queries, single_model=None, entities=None, relation_names=None):
    if single_model is not None:
        entity_set = single_model.model.entities
        relation_name_set = single_model.model.relation_names
    elif entities is not None and relation_names is not None:
        entity_set = entities
        relation_name_set = relation_names
    else:
        raise ValueError(f"There was no filter specified!")

    # Create masks for entities and relation names
    entity_mask = torch.isin(queries[:, 0].to('cuda'), torch.tensor(entity_set).to('cuda')).to('cuda')
    relation_mask = torch.isin(queries[:, 1].to('cuda'), torch.tensor(relation_name_set).to('cuda')).to('cuda')

    query_mask = entity_mask & relation_mask

    return queries[query_mask], query_mask


def get_class(class_name, base_method=False):
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
