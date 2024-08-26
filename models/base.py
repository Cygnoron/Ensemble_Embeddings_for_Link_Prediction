"""Base Knowledge Graph embedding model."""
import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn


class KGModel(nn.Module, ABC):
    """Base Knowledge Graph Embedding model class.

        Attributes:
            sizes: Tuple[int, int, int] with (n_entities, n_relations, n_entities)
            rank: integer for embedding dimension
            dropout: float for dropout rate
            gamma: torch.nn.Parameter for margin in ranking-based loss
            data_type: torch.dtype for machine precision (single or double)
            bias: string for whether to learn or fix bias (none for no bias)
            init_size: float for embeddings' initialization scale
            entity: torch.nn.Embedding with entity embeddings
            rel: torch.nn.Embedding with relation embeddings
            bh: torch.nn.Embedding with head entity bias embeddings
            bt: torch.nn.Embedding with tail entity bias embeddings
        """

    def __init__(self, sizes, rank, dropout, gamma, data_type, bias, init_size, model, subgraph=None,
                 entities=None, relation_names=None, subgraph_amount=None, batch_size=None, aggregation_method=None):
        """Initialize KGModel."""
        super(KGModel, self).__init__()
        if data_type == 'double':
            self.data_type = torch.double
        else:
            self.data_type = torch.float
        self.subgraph = subgraph
        self.subgraph_num = int(subgraph.split('_')[1]) if subgraph is not None else None
        self.model_name = model
        self.sizes = sizes
        self.rank = rank
        self.dropout = dropout
        self.bias = bias
        self.init_size = init_size
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.entity = nn.Embedding(sizes[0], rank)
        self.rel = nn.Embedding(sizes[1], rank)
        self.bh = nn.Embedding(sizes[0], 1)
        self.bh.weight.data = torch.zeros((sizes[0], 1), dtype=self.data_type)
        self.bt = nn.Embedding(sizes[0], 1)
        self.bt.weight.data = torch.zeros((sizes[0], 1), dtype=self.data_type)

        # check if model is unified model
        from models.unified import UNIFIED_MODELS
        if self.model_name in UNIFIED_MODELS:
            self.is_unified_model = True
            self.subgraph_amount = subgraph_amount
            self.batch_size = batch_size
            self.aggregation_method = aggregation_method

            self.rel = nn.Embedding(sizes[1], self.rank_rel)

            self.cands_ent = None
            self.cands_rel = None

            self.att_ent_cross_model = None
            self.att_rel_cross_model = None

            self.theta_ent_unified = None
            self.theta_rel_unified = None

        else:
            self.is_unified_model = False
            self.model_dropout = False
            if entities is None or relation_names is None:
                self.is_in_ensemble = False
            else:
                self.is_in_ensemble = True
                self.att_ent_single = None
                self.att_rel_single = None
                self.entities = entities
                self.relation_names = relation_names

        if not hasattr(self, "is_in_ensemble"):
            self.is_in_ensemble = False

        # ensemble attention
        if self.is_in_ensemble or self.is_unified_model :
            self.theta_ent = None
            self.theta_rel = None
            self.init_theta()
        self.act = nn.Softmax(dim=-1)
        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    @abstractmethod
    def get_queries(self, queries):
        """Compute embedding and biases of queries.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
             lhs_e: torch.Tensor with queries' embeddings (embedding of head entities and relations)
             lhs_biases: torch.Tensor with head entities' biases
        """
        pass

    @abstractmethod
    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
             rhs_e: torch.Tensor with targets' embeddings
                    if eval_mode=False returns embedding of tail entities (n_queries x rank)
                    else returns embedding of all possible entities in the KG dataset (n_entities x rank)
             rhs_biases: torch.Tensor with targets' biases
                         if eval_mode=False returns biases of tail entities (n_queries x 1)
                         else returns biases of all possible entities in the KG dataset (n_entities x 1)
        """
        pass

    @abstractmethod
    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space.

        Args:
            lhs_e: torch.Tensor with queries' embeddings
            rhs_e: torch.Tensor with targets' embeddings
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            scores: torch.Tensor with similarity scores of queries against targets
        """
        pass

    def score(self, lhs, rhs, eval_mode):
        """Scores queries against targets

        Args:
            lhs: Tuple[torch.Tensor, torch.Tensor] with queries' embeddings and head biases
                 returned by get_queries(queries)
            rhs: Tuple[torch.Tensor, torch.Tensor] with targets' embeddings and tail biases
                 returned by get_rhs(queries, eval_mode)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            score: torch.Tensor with scores of queries against targets
                   if eval_mode=True, returns scores against all possible tail entities, shape (n_queries x n_entities)
                   else returns scores for triples in batch (shape n_queries x 1)
        """
        lhs_e, lhs_biases = lhs
        rhs_e, rhs_biases = rhs
        score = self.similarity_score(lhs_e, rhs_e, eval_mode)
        if self.bias == 'constant':
            return self.gamma.item() + score
        elif self.bias == 'learn':
            if eval_mode:
                return lhs_biases + rhs_biases.t() + score
            else:
                return lhs_biases + rhs_biases + score
        else:
            return score

    def get_factors(self, queries):
        """Computes factors for embeddings' regularization.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] with embeddings to regularize
        """
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        rhs_e = self.entity(queries[:, 2])
        return head_e, rel_e, rhs_e

    def forward(self, queries, eval_mode=False):
        """KGModel forward pass.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            predictions: torch.Tensor with triples' scores
                         shape is (n_queries x 1) if eval_mode is false
                         else (n_queries x n_entities)
            factors: embeddings to regularize
        """
        # get embeddings and similarity scores
        lhs_e, lhs_biases = self.get_queries(queries)
        # queries = F.dropout(queries, self.dropout, training=self.training)
        rhs_e, rhs_biases = self.get_rhs(queries, eval_mode)
        # candidates = F.dropout(candidates, self.dropout, training=self.training)
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases), eval_mode)

        # get factors for regularization
        factors = self.get_factors(queries)
        return predictions, factors

    def get_ranking(self, queries, filters, batch_size=1000):
        """Compute filtered ranking of correct entity for evaluation.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            filters: filters[(head, relation)] gives entities to ignore (filtered setting)
            batch_size: int for evaluation batch size

        Returns:
            ranks_opt: torch.Tensor with optimistic ranks or correct entities
            rank_deviation: torch.Tensor with rank_deviation metric
        """

        rank_deviation = 0
        ranks = torch.ones(len(queries)).to('cuda')

        with torch.no_grad():
            b_begin = 0

            candidates = self.get_rhs(queries, eval_mode=True)

            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size].cuda()

                q = self.get_queries(these_queries)
                rhs = self.get_rhs(these_queries, eval_mode=False)

                scores = self.score(q, candidates, eval_mode=True)
                targets = self.score(q, rhs, eval_mode=False)

                # set filtered and true scores to -1e6 to be ignored
                for i, query in enumerate(these_queries):
                    filter_out = filters[(query[0].item(), query[1].item())]
                    filter_out += [queries[b_begin + i, 2].item()]

                    if query[0].item() != query[2].item():
                        filter_out += [queries[b_begin + i, 0].item()]

                    scores[i, torch.LongTensor(filter_out)] = -1e6

                # Calculate optimistic rank
                ranks[b_begin:b_begin + batch_size] += torch.sum((scores >= targets).to(self.data_type), dim=1)

                # Calculate rank_deviation
                rank_deviation_buffer = scores.to(self.data_type) == targets.to(self.data_type)
                rank_deviation_buffer = torch.sum(rank_deviation_buffer, dim=1)

                rank_deviation += torch.sum(torch.abs(rank_deviation_buffer))
                logging.debug(f"rank deviation step {b_begin}/{len(queries)}: {rank_deviation}")

                b_begin += batch_size

            rank_deviation = (rank_deviation / len(queries)).to(self.data_type)

        return ranks, rank_deviation

    def compute_metrics(self, examples, filters, sizes, batch_size=500):
        """Compute ranking-based evaluation metrics.
    
        Args:
            examples: torch.LongTensor of size n_examples x 3 containing triples' indices
            filters: Dict with entities to skip per query for evaluation in the filtered setting
            sizes: tuple with the sizes of the embeddings
            batch_size: integer for batch size to use to compute scores

        Returns:
            Evaluation metrics (mean rank, mean reciprocical rank, hits, amri and rank_deviation)
        """
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}
        amri = {}
        rank_deviation = {}

        for m in ["rhs", "lhs"]:
            q = examples.clone()
            if m == "lhs":
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.sizes[1] // 2

            ranks_opt, rank_deviation[m] = self.get_ranking(q, filters[m], batch_size=batch_size)

            mean_rank[m] = torch.mean(ranks_opt).item()
            mean_reciprocal_rank[m] = torch.mean(1. / ranks_opt).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks_opt <= x).float()).item(),
                (1, 3, 10)
            ))))

            # Calculate AMRI
            sum_ranks = torch.sum(ranks_opt - 1)
            sum_scores = ranks_opt.size()[0] * sizes[0]
            amri[m] = 1 - (2 * sum_ranks) / sum_scores

        return mean_rank, mean_reciprocal_rank, hits_at, amri, rank_deviation

    def init_theta(self):
        """
        Initialize embeddings and tensors for the calculation of attention between all models.
        """
        if self.is_unified_model:
            self.att_ent_cross_model = torch.zeros(self.sizes[0], self.rank, self.subgraph_amount,
                                                   dtype=self.data_type).to('cuda')
            self.att_rel_cross_model = torch.zeros(self.sizes[1], self.rank_rel, self.subgraph_amount,
                                                   dtype=self.data_type).to('cuda')

            self.theta_ent_unified = nn.Embedding(self.sizes[0], self.rank, dtype=self.data_type)
            self.theta_rel_unified = nn.Embedding(self.sizes[1], self.rank_rel, dtype=self.data_type)

            self.theta_ent_unified.weight.data = torch.rand(self.sizes[0], self.rank, self.subgraph_amount,
                                                            dtype=self.data_type).to('cuda')
            self.theta_rel_unified.weight.data = torch.rand(self.sizes[1], self.rank_rel, self.subgraph_amount,
                                                            dtype=self.data_type).to('cuda')

            self.cands_ent = torch.zeros(self.sizes[0], self.rank, self.subgraph_amount,
                                         dtype=self.data_type).to('cuda')
            self.cands_rel = torch.zeros(self.sizes[1], self.rank_rel, self.subgraph_amount,
                                         dtype=self.data_type).to('cuda')

        else:
            self.theta_ent = nn.Embedding(self.sizes[0], self.rank, dtype=self.data_type)
            from models import HYP_MODELS
            if self.model_name in HYP_MODELS:
                self.theta_rel = nn.Embedding(self.sizes[1], 2 * self.rank, dtype=self.data_type)
            else:
                self.theta_rel = nn.Embedding(self.sizes[1], self.rank, dtype=self.data_type)
            self.att_ent_single = torch.zeros(self.sizes[0], self.rank, dtype=self.data_type).cuda()
            self.att_rel_single = torch.zeros(self.sizes[1], self.rank, dtype=self.data_type).cuda()
