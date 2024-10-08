import importlib
import json
import logging
import os
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import models as models
from datasets.kg_dataset import KGDataset
from ensemble import Constants, util
from models import EUC_MODELS, COMPLEX_MODELS, HYP_MODELS, KGModel
from optimizers import regularizers, KGOptimizer
from utils.euclidean import givens_reflection, givens_rotations
from utils.train import count_params

UNIFIED_MODELS = ["Unified"]


class Unified(KGModel):

    def __init__(self, args, init_args, unified_args):
        """
        Initializes the Unified model with the given arguments and settings.

        Args:
            args: Arguments for the model.
            init_args: Initialization arguments.
            unified_args: Unified model specific arguments.
        """

        self.args = args
        self.rank_rel = unified_args.rank
        # change dimensions of relation embeddings if hyperbolic methods are contained
        self.init_if_hyperbolic(unified_args)

        super(Unified, self).__init__(unified_args.sizes, unified_args.rank, unified_args.dropout, unified_args.gamma,
                                      unified_args.dtype, unified_args.bias, unified_args.init_size, unified_args.model,
                                      subgraph_amount=unified_args.subgraph_amount, batch_size=unified_args.batch_size,
                                      aggregation_method=unified_args.aggregation_method)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank_rel), dtype=self.data_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank_rel)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank_rel), dtype=self.data_type) - 1.0

        self.theta_ent_unified = self.theta_ent_unified.to('cuda')
        self.theta_rel_unified = self.theta_rel_unified.to('cuda')
        self.cands_ent = self.cands_ent.to('cuda')
        self.cands_rel = self.cands_rel.to('cuda')

        # list of single model optimizers
        self.single_models = []
        self.single_model_args = []
        self.active_models = []
        self.init_single_models(args, init_args)

        self.embedding_methods = set()
        self.sim = None
        self.validation = False
        self.collect_embedding_methods()

    def init_if_hyperbolic(self, unified_args):
        for method in list(self.args.kge_models.keys()):
            if method in HYP_MODELS:
                self.rank_rel = 2 * unified_args.rank
                break

    def init_single_models(self, args, init_args):
        """
        Initializes single embedding models based on the provided arguments.

        Args:
            args: General arguments for single models.
            init_args: Initialization arguments specific to single models.
        """

        logging.info("-/\tCreating single embedding models\t\\-")
        time_start_model_creation = time.time()

        init_progress_bar = None
        if not init_args.no_progress_bar:
            init_progress_bar = tqdm(total=init_args.subgraph_amount, desc=f"Creating single embedding models",
                                     unit=" model(s) ", position=0, leave=True)

        counter = 0
        for subgraph_num in list(init_args.subgraph_embedding_mapping.keys()):
            model = init_args.subgraph_embedding_mapping[subgraph_num]

            # load args
            args_subgraph = util.get_args(args, model)
            args_subgraph.model_name = model
            args_subgraph.model = model
            args_subgraph.subgraph = f"sub_{int(subgraph_num):03d}"

            logging.debug(f"Args for {args_subgraph.subgraph}:\n{args_subgraph}")

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

            model.to('cuda')
            # Get optimizer
            regularizer = (getattr(regularizers, args_subgraph.regularizer)(args_subgraph.reg))
            optim_method = (getattr(torch.optim, args_subgraph.optimizer)(model.parameters(),
                                                                          lr=args_subgraph.learning_rate))
            optimizer = KGOptimizer(model, regularizer, optim_method, args_subgraph.batch_size,
                                    args_subgraph.neg_sample_size, bool(args_subgraph.double_neg),
                                    args_subgraph.no_progress_bar)

            self.single_model_args.append(args_subgraph)
            self.single_models.append(optimizer)
            self.active_models.append(subgraph_num)

            # save config
            if not hasattr(args, "test_mode"):
                with (open(os.path.join(init_args.model_setup_config_dir,
                                        f"config_{args_subgraph.subgraph}_{args_subgraph.model_name}.json"),
                           "w") as json_file):
                    json.dump(vars(args_subgraph), json_file)

            args_subgraph.entities = torch.tensor(args_subgraph.entities, dtype=torch.int).cuda()
            args_subgraph.relation_names = torch.tensor(args_subgraph.relation_names, dtype=torch.int).cuda()

            if not init_args.no_progress_bar:
                # Update progress bar
                counter += 1
                init_progress_bar.n = counter
                init_progress_bar.refresh()

        time_stop_model_creation = time.time()

        # Close progress bar
        if not init_args.no_progress_bar:
            init_progress_bar.close()
        logging.info(f"-\\\tSuccessfully created all embedding_models in "
                     f"{util.format_time(time_start_model_creation, time_stop_model_creation)}\t/-")

    def collect_embedding_methods(self):
        """
        Collects the embedding methods used by the single models.
        """

        for single_model in self.single_models:
            embedding_method = single_model.model.model_name

            if not any(embedding_method == method for method in self.embedding_methods):
                self.embedding_methods.add(embedding_method)
                if embedding_method in HYP_MODELS and not hasattr(self, "multi_c"):
                    self.multi_c = self.args.multi_c
                    if self.multi_c:
                        c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
                    else:
                        c_init = torch.ones((1, 1), dtype=self.data_type)
                    self.c = nn.Parameter(c_init, requires_grad=True)

                    self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
                    self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank),
                                                               dtype=self.data_type) - 1.0
                    self.context_vec = nn.Embedding(self.sizes[1], self.rank)
                    self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank),
                                                                                dtype=self.data_type)
                    self.act = nn.Softmax(dim=1)
                    if self.data_type == "double":
                        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
                    else:
                        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

        logging.info(f"Found the following embedding methods:\n{util.format_set(self.embedding_methods)}")

        # self.embedding_methods = list(self.embedding_methods)

        created_embeddings = False
        for embedding_method in self.embedding_methods:
            if embedding_method == Constants.SEA or embedding_method == Constants.SEPA and not created_embeddings:
                buffer = None
                if embedding_method == Constants.SEA:
                    from models.euclidean import SEA
                    buffer = SEA(self.args)
                elif embedding_method == Constants.SEPA:
                    from models.hyperbolic import SEPA
                    buffer = SEPA(self.args)
                self.ref = buffer.ref
                self.rot = buffer.rot
                self.tr = buffer.tr
                self.dm = buffer.dm
                self.cp = buffer.cp
                self.context_vec = buffer.context_vec
                created_embeddings = True

    def forward(self, queries, eval_mode=False):
        """
        Forward pass of the Unified model.

        Args:
            queries: The input queries.
            eval_mode: Flag indicating evaluation mode.

        Returns:
            predictions: Predictions from the model.
            factors: Factors used in the predictions.
        """

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

    def check_model_dropout(self, queries):
        run_diverged = True
        for single_model in self.single_models:
            if single_model.model.model_dropout:
                continue

            run_diverged = False
            actual_queries, _ = get_actual_queries(queries, single_model=single_model)

            valid_loss = single_model.calculate_valid_loss(actual_queries.to('cuda')).to('cuda')
            logging.debug(f"Valid loss for {single_model.model.subgraph}: {valid_loss:.2f}")

            if not hasattr(single_model.model, "first_loss"):
                single_model.model.first_loss = valid_loss
            else:
                if valid_loss > single_model.model.first_loss * self.args.model_dropout_factor:
                    logging.critical(f"Model {single_model.model.subgraph} was excluded, since the current loss "
                                     f"{valid_loss:.2f} exceeded {self.args.model_dropout_factor} times the first loss "
                                     f"{single_model.model.first_loss:.2f}.")
                    single_model.model.model_dropout = True
                    logging.debug(f"Removing {single_model.model.subgraph_num} from {self.active_models}")
                    self.active_models.remove(single_model.model.subgraph_num)

        if run_diverged:
            raise RuntimeError("The run diverged and is now stopped. Please try a other combination of embedding "
                               "methods, other hyperparameters, or a higher model dropout factor.")

    def train_single_models(self, queries):
        """
        Trains single models using the provided queries.

        Args:
            queries: The input queries.
        """
        if self.validation:
            pass
        else:
            for single_model in self.single_models:
                if single_model.model.model_dropout:
                    logging.debug(f"Ignoring model {single_model.model.subgraph}.")
                    continue

                # actual_queries, _ = get_actual_queries(queries, single_model=single_model)
                actual_queries = queries.cuda()

                l = single_model.calculate_loss(actual_queries.to('cuda')).to('cuda')

                single_model.optimizer.zero_grad()
                l.backward()
                single_model.optimizer.step()

                logging.debug(f"{single_model.model.subgraph}: {l:.2f}")

    def calculate_cross_model_attention(self, queries):
        """
        Calculates cross-model attention for the given queries.

        Args:
            queries: The input queries.
        """

        theta_ent_temp = []
        cands_ent_temp = []
        theta_rel_temp = []
        cands_rel_temp = []

        for single_model in self.single_models:
            model = single_model.model

            actual_queries, query_mask = get_actual_queries(queries, single_model=single_model)
            embedding_ent = torch.zeros(len(queries), self.rank, dtype=self.data_type).cuda()
            embedding_rel = torch.zeros(len(queries), self.rank_rel, dtype=self.data_type).cuda()

            if not single_model.model.model_dropout:
                if model.model_name in COMPLEX_MODELS:
                    embedding_ent[query_mask] = model.embeddings[0].weight.data[actual_queries[:, 0]]
                    embedding_rel[query_mask] = model.embeddings[1].weight.data[actual_queries[:, 1]]
                else:
                    embedding_ent[query_mask] = model.entity.weight.data[actual_queries[:, 0]]
                    embedding_rel[query_mask] = model.rel.weight.data[actual_queries[:, 1]]

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
        """
        Calculates attention scores based on theta and candidate embeddings.

        Args:
            theta: The theta embeddings.
            cands: The candidate embeddings.
            dim: Dimension to apply softmax.

        Returns:
            attention: Attention scores.
        """

        softmax = self.act
        if not dim == -1:
            softmax = nn.Softmax(dim=dim)

        logging.debug(f"Sizes for attention:\nTheta\t{theta.size()}\nCands:\t{cands.size()}")
        attention = torch.zeros(theta.size(), dtype=self.data_type).to('cuda')
        attention[:, :, self.active_models] = softmax(theta[:, :, self.active_models] * cands[:, :, self.active_models])
        return attention

    def combine_ranks(self, attention, dim_softmax=-1, rank_dim=-1):
        """
        Combines ranks of different models using attention scores.

        Args:
            attention: Attention scores.
            dim_softmax: Dimension for softmax.
            rank_dim: Dimension for rank.

        Returns:
            combined: Combined ranks.
        """

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
        """
        Combines embeddings from different models using attention scores.

        Args:
            queries: The input queries.
        """

        ent_emb_temp = []
        rel_emb_temp = []
        rel_diag_emb_temp = []

        SEA_SEPA = False
        if (Constants.SEA or Constants.SEPA) in self.embedding_methods:
            SEA_SEPA = True
            ref_emb_temp = []
            rot_emb_temp = []
            tr_emb_temp = []
            dm_emb_temp = []
            cp_emb_temp = []
            context_emb_temp = []

        for single_model in self.single_models:
            # if single_model.model.model_dropout:
            #     logging.debug(f"Ignoring model {single_model.model.subgraph}.")
            #     continue
            if SEA_SEPA and single_model.model.model_name == (Constants.SEA or Constants.SEPA):
                ref_emb_temp.append(single_model.model.ref.weight.data[queries[:, 1]])
                rot_emb_temp.append(single_model.model.rot.weight.data[queries[:, 1]])
                tr_emb_temp.append(single_model.model.tr.weight.data[queries[:, 1]])
                dm_emb_temp.append(single_model.model.dm.weight.data[queries[:, 1]])
                cp_emb_temp.append(single_model.model.cp.weight.data[queries[:, 1]])
                context_emb_temp.append(single_model.model.context_vec.weight.data[queries[:, 1]])

            if single_model.model.model_name in EUC_MODELS:
                ent_emb_temp.append(single_model.model.entity.weight.data[queries[:, 0]])
                rel_emb_temp.append(single_model.model.rel.weight.data[queries[:, 1]])
            elif single_model.model.model_name in COMPLEX_MODELS:
                ent_emb_temp.append(single_model.model.embeddings[0].weight.data[queries[:, 0]])
                rel_emb_temp.append(single_model.model.embeddings[1].weight.data[queries[:, 1]])
            elif single_model.model.model_name in HYP_MODELS:
                ent_emb_temp.append(single_model.model.entity.weight.data[queries[:, 0]])
                rel_emb_temp.append(single_model.model.rel.weight.data[queries[:, 1]])
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
        self.rel_diag.weight.data[queries[:, 1]] = torch.sum(rel_diag_emb_temp *
                                                             self.att_rel_cross_model[queries[:, 1]], dim=-1).cuda()
        if SEA_SEPA:
            ref_emb_temp = torch.stack(ref_emb_temp, dim=-1).cuda()
            rot_emb_temp = torch.stack(rot_emb_temp, dim=-1).cuda()
            tr_emb_temp = torch.stack(tr_emb_temp, dim=-1).cuda()
            dm_emb_temp = torch.stack(dm_emb_temp, dim=-1).cuda()
            cp_emb_temp = torch.stack(cp_emb_temp, dim=-1).cuda()
            context_emb_temp = torch.stack(context_emb_temp, dim=-1).cuda()

            self.ref.weight.data[queries[:, 1]] = torch.sum(ref_emb_temp * self.att_rel_cross_model[queries[:, 1]],
                                                            dim=-1).cuda()
            self.rot.weight.data[queries[:, 1]] = torch.sum(rot_emb_temp * self.att_rel_cross_model[queries[:, 1]],
                                                            dim=-1).cuda()
            self.tr.weight.data[queries[:, 1]] = torch.sum(tr_emb_temp * self.att_rel_cross_model[queries[:, 1]],
                                                           dim=-1).cuda()
            self.dm.weight.data[queries[:, 1]] = torch.sum(dm_emb_temp * self.att_rel_cross_model[queries[:, 1]],
                                                           dim=-1).cuda()
            self.cp.weight.data[queries[:, 1]] = torch.sum(cp_emb_temp * self.att_rel_cross_model[queries[:, 1]],
                                                           dim=-1).cuda()
            self.context_vec.weight.data[queries[:, 1]] = torch.sum(context_emb_temp *
                                                                    self.att_rel_cross_model[queries[:, 1]],
                                                                    dim=-1).cuda()

    def get_queries(self, queries):
        """
        Retrieves the query embeddings and biases.

        Args:
            queries: The input queries.

        Returns:
            lhs_e: Query embeddings.
            lhs_biases: Query biases.
        """

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

        if isinstance(lhs_e_list[0], tuple):
            # handle hyperbolic case
            lhs_e_list_buffer = []
            c_list_buffer = []
            for query_tuple in lhs_e_list:
                lhs_e, c = query_tuple
                lhs_e_list_buffer.append(lhs_e)
                c_list_buffer.append(c)
            lhs_e_list_buffer = torch.stack(lhs_e_list_buffer, dim=-1).to('cuda')
            c_list_buffer = torch.stack(c_list_buffer, dim=-1).to('cuda')
            lhs_e_list = (lhs_e_list_buffer, c_list_buffer)

        else:
            lhs_e_list = torch.stack(lhs_e_list, dim=-1).to('cuda')
        lhs_biases_list = torch.stack(lhs_biases_list, dim=-1)

        # if self.aggregation_method[0] == Constants.MAX_SCORE_AGGREGATION[0]:
        #     if isinstance(lhs_e_list, tuple):
        #         lhs_e = (torch.max(lhs_e_list[0], dim=-1)[0].to('cuda'), torch.max(lhs_e_list[1], dim=-1)[0].to('cuda'))
        #     else:
        #         lhs_e, _ = torch.max(lhs_e_list, dim=-1)
        #         lhs_e = lhs_e.to('cuda')
        #     lhs_biases, _ = torch.max(lhs_biases_list, dim=-1)
        #     lhs_biases = lhs_biases.to('cuda')
        #
        # elif self.aggregation_method[0] == Constants.AVERAGE_SCORE_AGGREGATION[0]:
        #     if isinstance(lhs_e_list, tuple):
        #         lhs_e = (torch.mean(lhs_e_list[0], dim=-1).to('cuda'), torch.mean(lhs_e_list[1], dim=-1).to('cuda'))
        #     else:
        #         lhs_e = torch.mean(lhs_e_list, dim=-1).to('cuda')
        #     lhs_biases = torch.mean(lhs_biases_list, dim=-1).to('cuda')
        #
        # else:
        #     raise ValueError(f'Aggregation method {self.aggregation_method} in get_queries not supported.')

        return lhs_e_list, lhs_biases_list

    def get_rhs(self, queries, eval_mode):
        """
        Retrieves the right-hand side embeddings and biases.

        Args:
            queries: The input queries.
            eval_mode: Flag indicating evaluation mode.

        Returns:
            rhs_e: Right-hand side embeddings.
            rhs_biases: Right-hand side biases.
        """

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

        rhs_e_list = torch.stack(rhs_e_list, dim=-1).to('cuda')
        rhs_biases_list = torch.stack(rhs_biases_list, dim=-1)
        # if self.aggregation_method[0] == Constants.MAX_SCORE_AGGREGATION[0]:
        #     rhs_e, _ = torch.max(rhs_e_list, dim=-1)
        #     rhs_biases, _ = torch.max(rhs_biases_list, dim=-1)
        #     rhs_e = rhs_e.to('cuda')
        #     rhs_biases = rhs_biases.to('cuda')
        # elif self.aggregation_method[0] == Constants.AVERAGE_SCORE_AGGREGATION[0]:
        #     rhs_e = torch.mean(rhs_e_list, dim=-1).to('cuda')
        #     rhs_biases = torch.mean(rhs_biases_list, dim=-1).to('cuda')
        # else:
        #     raise ValueError(f'Aggregation method {self.aggregation_method} in get_rhs not supported.')

        return rhs_e_list, rhs_biases_list

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """
        Compute similarity scores or queries against targets in embedding space.

        Args:
            lhs_e: Left-hand side embeddings.
            rhs_e: Right-hand side embeddings.
            eval_mode: Flag indicating evaluation mode.

        Returns:
            score: Similarity scores.
        """

        score_list = []
        for index, embedding_method in enumerate(self.embedding_methods):
            if embedding_method in COMPLEX_MODELS:
                score = self.similarity_score_complex(lhs_e[:, :, index], rhs_e[:, :, index], eval_mode)
            elif embedding_method in HYP_MODELS:
                method = get_class(embedding_method, base_method=True)
                lhs_e_hyp = (lhs_e[0][:, :, index], lhs_e[1][:, :, index])
                score = getattr(method, "similarity_score")(self, lhs_e_hyp, rhs_e[:, :, index], eval_mode)
            else:
                method = get_class(embedding_method, base_method=True)
                self.set_sim(embedding_method)
                score = getattr(method, "similarity_score")(self, lhs_e[:, :, index], rhs_e[:, :, index], eval_mode)
            score_list.append(score)

        score = self.aggregate_scores(score_list)

        return score

    def aggregate_scores(self, score_list, stacking=True):
        if stacking:
            score_list = torch.stack(score_list, dim=-1).to('cuda')
        if self.aggregation_method[0] == Constants.MAX_SCORE_AGGREGATION[0]:
            score, _ = torch.max(score_list, dim=-1)
            score = score.to('cuda')
        elif self.aggregation_method[0] == Constants.AVERAGE_SCORE_AGGREGATION[0]:
            score = torch.mean(score_list, dim=-1).to('cuda')
        else:
            raise ValueError(f'Aggregation method {self.aggregation_method} in similarity_score not supported.')

        return score

    # def normalize(self, score, dim):
    #
    #     activation = nn.Softmax(dim=dim)
    #     score = activation(score)
    #
    #     return score
    #
    # def min_max_normalize(self, tensor, dim):
    #     """
    #     Normalize the entries of a PyTorch tensor along a given dimension to the interval [0, 1].
    #
    #     Args:
    #         tensor (torch.Tensor): The input tensor.
    #         dim (int): The dimension along which to normalize.
    #
    #     Returns:
    #         torch.Tensor: The normalized tensor.
    #     """
    #     # return tensor
    #
    #     min_vals, _ = tensor.min(dim=dim, keepdim=True)
    #     max_vals, _ = tensor.max(dim=dim, keepdim=True)
    #     range_vals = max_vals - min_vals
    #
    #     # Avoid division by zero
    #     normalized_tensor = (tensor - min_vals) / range_vals.clamp(min=1e-6)
    #     return normalized_tensor

    def set_sim(self, embedding_method):
        """
        Sets the similarity metric for the embedding method.

        Args:
            embedding_method: The embedding method to set the similarity metric for.
        """

        if embedding_method not in EUC_MODELS:
            return

        methods_dist = ["TransE", "DistMult", "MurE", "RotE", "RefE", "AttE", "SEA"]
        methods_dot = ["CP"]

        if embedding_method in methods_dot:
            self.sim = "dot"
        elif embedding_method in methods_dist:
            self.sim = "dist"
        else:
            raise ValueError(f"There was no sim specified for the given embedding method {embedding_method}.")

    def get_factors(self, queries):
        """
        Retrieves factors for the given queries.

        Args:
            queries: The input queries.

        Returns:
            factors_h: Head entity factors.
            factors_r: Relation factors.
            factors_t: Tail entity factors.
        """

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

        if self.aggregation_method[0] == Constants.MAX_SCORE_AGGREGATION[0]:
            factors_h, _ = torch.max(torch.stack(factors_h_list, dim=-1), dim=-1)
            factors_r, _ = torch.max(torch.stack(factors_r_list, dim=-1), dim=-1)
            factors_t, _ = torch.max(torch.stack(factors_t_list, dim=-1), dim=-1)

            factors_h = factors_h.to('cuda')
            factors_r = factors_r.to('cuda')
            factors_t = factors_t.to('cuda')

        elif self.aggregation_method[0] == Constants.AVERAGE_SCORE_AGGREGATION[0]:
            factors_h = torch.mean(torch.stack(factors_h_list, dim=-1), dim=-1).to('cuda')
            factors_r = torch.mean(torch.stack(factors_r_list, dim=-1), dim=-1).to('cuda')
            factors_t = torch.mean(torch.stack(factors_t_list, dim=-1), dim=-1).to('cuda')

        else:
            raise ValueError(f'Aggregation method {self.aggregation_method} in get_factors not supported.')

        return factors_h, factors_r, factors_t

    def update_single_models(self, queries):
        """
        Updates the single models with the unified embeddings.

        Args:
            queries: The input queries.
        """

        for index, single_model in enumerate(self.single_models):
            if single_model.model.model_dropout:
                logging.debug(f"Ignoring model {single_model.model.subgraph}.")
                continue

            actual_queries, _ = get_actual_queries(queries, single_model=single_model)

            if single_model.model.model_name in COMPLEX_MODELS:
                single_model.model.embeddings[0].weight.data[actual_queries[:, 0]] = self.entity.weight.data[
                    actual_queries[:, 0]]
                single_model.model.embeddings[1].weight.data[actual_queries[:, 1]] = self.rel.weight.data[
                    actual_queries[:, 1]]
            else:
                single_model.model.entity.weight.data[actual_queries[:, 0]] = self.entity.weight.data[
                    actual_queries[:, 0]]
                single_model.model.rel.weight.data[actual_queries[:, 1]] = self.rel.weight.data[actual_queries[:, 1]]

            if hasattr(single_model.model, "rel_diag"):
                single_model.model.rel_diag.weight.data[actual_queries[:, 1]] = self.rel_diag.weight.data[
                    actual_queries[:, 1]]

            single_model.model.theta_ent.weight.data[actual_queries[:, 0]] = \
                self.theta_ent_unified.weight.data[actual_queries[:, 0]][:, :, index]
            single_model.model.theta_rel.weight.data[actual_queries[:, 1]] = \
                self.theta_rel_unified.weight.data[actual_queries[:, 1]][:, :, index]

            if single_model.model.model_name in [Constants.SEA, Constants.SEPA]:
                single_model.model.ref.weight.data[actual_queries[:, 1]] = self.ref.weight.data[actual_queries[:, 1]]
                single_model.model.rot.weight.data[actual_queries[:, 1]] = self.rot.weight.data[actual_queries[:, 1]]
                single_model.model.tr.weight.data[actual_queries[:, 1]] = self.tr.weight.data[actual_queries[:, 1]]
                single_model.model.dm.weight.data[actual_queries[:, 1]] = self.dm.weight.data[actual_queries[:, 1]]
                single_model.model.cp.weight.data[actual_queries[:, 1]] = self.cp.weight.data[actual_queries[:, 1]]

    # --- required functions for the case of complex embeddings ---

    def get_rhs_complex(self, queries, eval_mode):
        """
        Get embeddings and biases of target entities for complex models.

        Args:
            queries: The input queries.
            eval_mode: Flag indicating evaluation mode.

        Returns:
            rhs_e: Right-hand side embeddings.
            rhs_biases: Right-hand side biases.
        """

        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def get_complex_embeddings(self, queries):
        """
        Get complex embeddings of queries.

        Args:
            queries: The input queries.

        Returns:
            head_e: Head entity embeddings.
            rel_e: Relation embeddings.
            rhs_e: Right-hand side entity embeddings.
        """

        rank = self.rank // 2
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        rhs_e = self.entity(queries[:, 2])
        head_e = head_e[:, :rank], head_e[:, rank:]
        rel_e = rel_e[:, :rank], rel_e[:, rank:]
        rhs_e = rhs_e[:, :rank], rhs_e[:, rank:]
        return head_e, rel_e, rhs_e

    def similarity_score_complex(self, lhs_e, rhs_e, eval_mode):
        """
        Compute similarity scores or queries against targets in embedding space for complex models.

        Args:
            lhs_e: Left-hand side embeddings.
            rhs_e: Right-hand side embeddings.
            eval_mode: Flag indicating evaluation mode.

        Returns:
            score: Similarity scores.
        """

        rank = self.rank // 2
        lhs_e = lhs_e[:, :rank], lhs_e[:, rank:]
        rhs_e = rhs_e[:, :rank], rhs_e[:, rank:]
        if eval_mode:
            return lhs_e[0] @ rhs_e[0].transpose(0, 1) + lhs_e[1] @ rhs_e[1].transpose(0, 1)
        else:
            return torch.sum(lhs_e[0] * rhs_e[0] + lhs_e[1] * rhs_e[1], 1, keepdim=True)

    # --- required functions for the case of AttE ---

    def get_queries_AttE(self, queries):
        """
        Retrieves query embeddings and biases for AttE models.

        Args:
            queries: The input queries.

        Returns:
            lhs_e: Query embeddings.
            lhs_biases: Query biases.
        """

        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()
        lhs_ref_e, lhs_rot_e, context_vec = self.get_reflection_rotation_context(queries)

        # self-attention mechanism
        cands = torch.cat([lhs_ref_e, lhs_rot_e], dim=1)
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        lhs_e = torch.sum(att_weights * cands, dim=1) + self.rel(queries[:, 1])

        return lhs_e, self.bh(queries[:, 0])

    def get_reflection_rotation_context(self, queries):
        """
        Retrieves reflection, rotation, and context vectors for AttE models.

        Args:
            queries: The input queries.

        Returns:
            lhs_ref_e: Reflection embeddings.
            lhs_rot_e: Rotation embeddings.
            context_vec: Context vectors.
        """

        reflection = []
        rotation = []
        context_vec = []
        for single_model in self.single_models:
            if single_model.model.model_dropout:
                logging.debug(f"Ignoring model {single_model.model.subgraph}.")
                continue

            if single_model.model.model_name == Constants.ATT_E:
                reflection.append(single_model.model.ref(queries[:, 1]).view((-1, 1, self.rank)))
                rotation.append(single_model.model.rot(queries[:, 1]).view((-1, 1, self.rank)))
                context_vec.append(single_model.model.context_vec(queries[:, 1]).view((-1, 1, self.rank)))

        reflection = torch.mean(torch.stack(reflection, dim=-1), dim=-1).to('cuda')
        rotation = torch.mean(torch.stack(rotation, dim=-1), dim=-1).to('cuda')
        context_vec = torch.mean(torch.stack(context_vec, dim=-1), dim=-1).to('cuda')

        lhs_ref_e = givens_reflection(reflection, self.entity(queries[:, 0])).view((-1, 1, self.rank))
        lhs_rot_e = givens_rotations(rotation, self.entity(queries[:, 0])).view((-1, 1, self.rank))
        return lhs_ref_e, lhs_rot_e, context_vec

    # --- required functions for the case of SEA and SEPA ---

    def get_reflection_queries(self, queries):
        lhs_ref_e = givens_reflection(
            self.ref(queries[:, 1]), self.entity(queries[:, 0])
        )
        return lhs_ref_e

    def get_rotation_queries(self, queries):
        lhs_rot_e = givens_rotations(
            self.rot(queries[:, 1]), self.entity(queries[:, 0])
        )
        return lhs_rot_e

    def get_transe_queries(self, queries):
        tr = self.tr(queries[:, 1])
        h = self.entity(queries[:, 0])
        lhs_tr_e = h + tr
        return lhs_tr_e

    def get_complex_queries(self, queries):
        cp = self.cp(queries[:, 1])
        cp = cp[:, :self.rank // 2], cp[:, self.rank // 2:]
        h = self.entity(queries[:, 0])
        h = h[:, :self.rank // 2], h[:, self.rank // 2:]
        lhse_cp_e = h[0] * cp[0] - h[1] * cp[1], h[0] * cp[1] + h[1] * cp[0]
        lhs_cp_e = torch.cat((lhse_cp_e[0], lhse_cp_e[1]), dim=1)
        return lhs_cp_e

    def get_distmult_queries(self, queries):
        dm = self.dm(queries[:, 1])
        h = self.entity(queries[:, 0])
        lhs_dm_e = h * dm
        return lhs_dm_e


def get_actual_queries(queries, single_model=None, entities=None, relation_names=None):
    """
    Filters queries based on the provided single model or entity and relation sets.

    Args:
        queries: The input queries.
        single_model: Single model to use for filtering.
        entities: Set of entities to filter.
        relation_names: Set of relation names to filter.

    Returns:
        filtered_queries: Filtered queries.
        query_mask: Mask indicating valid queries.
    """

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

    return queries[query_mask.cpu()], query_mask


def get_class(class_name, base_method=False):
    """
    Retrieves the class definition for the given class name.

    Args:
        class_name: Name of the class to retrieve.
        base_method: Flag indicating if the base method should be returned.

    Returns:
        class_def: Class definition.
    """

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
