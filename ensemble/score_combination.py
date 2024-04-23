import logging

import torch
from tqdm import tqdm

from ensemble import Constants


def calculate_scores(embedding_models, examples, batch_size=500):
    # -- calculate scores for all queries and all models --
    progress_bar_testing = tqdm(total=len(embedding_models), desc=f"Calculating test scores", unit=" embedding models")

    for embedding_model in embedding_models:
        model = embedding_model['model']
        args = embedding_model['args']
        embedding_model['scores'] = {}
        embedding_model['targets'] = {}

        progress_bar_testing.n = args.subgraph_num
        progress_bar_testing.refresh()

        # calculate scores, adapted from base.compute_metrics() and base.get_rankings()
        for mode in ["rhs", "lhs"]:
            queries = examples.clone()

            if mode == "lhs":
                tmp = torch.clone(queries[:, 0])
                queries[:, 0] = queries[:, 2]
                queries[:, 2] = tmp
                queries[:, 1] += model.sizes[1] // 2

            with torch.no_grad():
                b_begin = 0
                candidates = model.get_rhs(queries, eval_mode=True)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size].cuda()

                    q = model.get_queries(these_queries)
                    rhs = model.get_rhs(these_queries, eval_mode=False)

                    scores = model.score(q, candidates, eval_mode=True)
                    targets = model.score(q, rhs, eval_mode=False)

                    b_begin += batch_size

                embedding_model['scores'][mode] = scores
                embedding_model['targets'][mode] = targets

    progress_bar_testing.n = len(embedding_models)
    progress_bar_testing.refresh()
    progress_bar_testing.close()

    return embedding_models


def combine_scores(embedding_models, aggregation_method=Constants.MAX_SCORE):
    aggregated_scores = {}

    logging.info(f"Combining scores of all models with {aggregation_method[1]}.")

    for mode in ["rhs", "lhs"]:
        # aggregate saved scores based on aggregation_method
        if aggregation_method[0] == Constants.MAX_SCORE[0]:
            # select maximum score between all models
            model_scores = []
            for embedding_model in embedding_models:
                model_scores += [embedding_model['scores'][mode]]
            stacked_scores = torch.stack(model_scores, dim=1)
            logging.log(Constants.DATA_LEVEL, f"Stacked scores:\n{stacked_scores}")

            aggregated_scores[mode] = torch.max(stacked_scores, dim=1)
            logging.log(Constants.DATA_LEVEL, f"Size:\t{aggregated_scores[mode].size()}\t\t"
                                              f"Aggregated scores:\n{aggregated_scores[mode]}")

        elif aggregation_method[0] == Constants.AVERAGE_SCORE[0]:
            logging.error(f"Aggregation method {aggregation_method[1]} isn't implemented yet!")
            # TODO implement average score
            # average the score across all models
            pass

        elif aggregation_method[0] == Constants.ATTENTION_SCORE[0]:
            logging.error(f"Aggregation method {aggregation_method[1]} isn't implemented yet!")
            # TODO implement attention score
            # calculate attention between all models and average scores based on this attention
            pass

        else:
            logging.error(f"Selected aggregation method does not exist!")

    return aggregated_scores


def compute_ranks(embedding_models, examples, filters, targets, aggregated_scores, batch_size=500):
    queries = examples.clone()
    ranks = torch.ones(len(queries))

    for mode in ["rhs", "lhs"]:
        with torch.no_grad():
            b_begin = 0
            queries = examples.clone()
            if mode == "lhs":
                tmp = torch.clone(queries[:, 0])
                queries[:, 0] = queries[:, 2]
                queries[:, 2] = tmp
                queries[:, 1] += embedding_models[0]['model'].sizes[1] // 2

            while b_begin < len(queries):
                try:
                    these_queries = queries[b_begin:b_begin + batch_size].cuda()

                    # set filtered and true scores to -1e6 to be ignored
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        aggregated_scores[mode][i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (aggregated_scores[mode] >= targets[mode]).float(),
                        dim=1).cpu()
                except KeyError:
                    print(f"ERROR\t{b_begin}")

                b_begin += batch_size

        # logging.debug(f"Mode:\t{mode}\t\tSize:\t{ranks.size()}\t\tRanks:\n{ranks}")
    return ranks


def compute_metrics_from_ranks(ranks_dict):
    mean_rank = {}
    mean_reciprocal_rank = {}
    hits_at = {}
    amri = {}
    mr_deviation = {}
    for mode in ["rhs", "lhs"]:
        ranks = ranks_dict[mode]

        mean_rank[mode] = torch.mean(ranks).item()
        mean_reciprocal_rank[mode] = torch.mean(1. / ranks).item()
        hits_at[mode] = torch.FloatTensor((list(map(
            lambda x: torch.mean((ranks <= x).float()).item(),
            (1, 3, 10)
        ))))

        # Calculate AMRI
        expected_rank = torch.mean(ranks - 1)  # Expectation of MR - 1
        amri[mode] = 1 - ((mean_rank[mode] - 1) / expected_rank)

        # Calculate MR_deviation
        optimistic_rank = torch.min(ranks) - 1
        pessimistic_rank = torch.max(ranks) - 1
        mr_deviation[mode] = optimistic_rank - pessimistic_rank

    return mean_rank, mean_reciprocal_rank, hits_at, amri, mr_deviation
