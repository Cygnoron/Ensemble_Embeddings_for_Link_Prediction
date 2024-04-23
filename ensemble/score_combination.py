import logging
import traceback

import torch
from tqdm import tqdm

from ensemble import Constants


def calculate_scores(embedding_models, examples, batch_size=500):
    # -- calculate scores for all queries and all models --
    progress_bar_testing = tqdm(total=len(embedding_models), desc=f"Calculating test scores", unit=" embedding models")

    targets_lhs = None
    targets_rhs = None
    candidate_answers = None

    for embedding_model in embedding_models:
        model = embedding_model['model']
        args = embedding_model['args']
        candidate_answers = len(model.get_rhs(examples, eval_mode=True)[0])
        scores_rhs = torch.zeros((len(examples), candidate_answers))
        scores_lhs = torch.zeros((len(examples), candidate_answers))

        targets_rhs = torch.zeros((len(examples), candidate_answers))
        targets_lhs = torch.zeros((len(examples), candidate_answers))

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

                    if mode == "lhs":
                        scores_lhs[b_begin:b_begin + batch_size] = scores
                        targets_lhs[b_begin:b_begin + batch_size] = targets
                        logging.log(Constants.DATA_LEVEL, f"From {b_begin} to {b_begin + batch_size}:\n"
                                                          f"{scores_lhs[b_begin:b_begin + batch_size]}")
                    else:
                        scores_rhs[b_begin:b_begin + batch_size] = scores
                        targets_rhs[b_begin:b_begin + batch_size] = targets
                        logging.log(Constants.DATA_LEVEL, f"From {b_begin} to {b_begin + batch_size}:\n"
                                                          f"{scores_rhs[b_begin:b_begin + batch_size]}")

                    b_begin += batch_size

        embedding_model['scores_lhs'] = scores_lhs
        embedding_model['scores_rhs'] = scores_rhs
        embedding_model['targets_lhs'] = targets_lhs
        embedding_model['targets_rhs'] = targets_rhs
        embedding_model['candidate_answers'] = candidate_answers

    targets = {'rhs': targets_rhs, 'lhs': targets_lhs}

    progress_bar_testing.n = progress_bar_testing.total
    progress_bar_testing.refresh()
    progress_bar_testing.close()

    return embedding_models, targets


def combine_scores(embedding_models, aggregation_method=Constants.MAX_SCORE, batch_size=500):
    aggregated_scores = {}
    progress_bar_combination = tqdm(total=embedding_models[0]['candidate_answers'], desc=f"Combine test scores",
                                    unit=" queries")
    logging.info(f"Combining scores of all models with {aggregation_method[1]}.")
    # TODO FIX
    # TODO include batch_size, also include progressbar

    for mode in ["rhs", "lhs"]:
        b_begin = 0

        # update progressbar
        if mode == 'lhs':
            progress_bar_combination.n = b_begin + embedding_models[0]['candidate_answers']
        else:
            progress_bar_combination.n = b_begin
        progress_bar_combination.refresh()

        while b_begin < len(embedding_models[0]['scores_lhs']):

            # aggregate saved scores based on aggregation_method
            if aggregation_method[0] == Constants.MAX_SCORE[0]:
                # select maximum score between all models
                model_scores = []
                for embedding_model in embedding_models:
                    if mode == "lhs":
                        model_scores += [embedding_model['scores_lhs'][b_begin:b_begin + batch_size]]
                    else:
                        model_scores += [embedding_model['scores_rhs'][b_begin:b_begin + batch_size]]
                try:
                    stacked_scores = torch.stack(model_scores, dim=1)
                    logging.log(Constants.DATA_LEVEL, f"Stacked scores:\n{stacked_scores}")

                    aggregated_scores[mode][b_begin:b_begin + batch_size], _ = torch.max(stacked_scores, dim=1)
                    logging.log(Constants.DATA_LEVEL, f"Size:\t{aggregated_scores[mode].size()}\t\t"
                                                      f"Aggregated scores:\n{aggregated_scores[mode]}")
                except Exception as e:
                    logging.error(traceback.format_exception(e))
                    for index, score in enumerate(model_scores):
                        logging.error(f"Mode: {mode}\tSize of score set {index}: {score.size()}")


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

            b_begin += batch_size

    progress_bar_combination.n = progress_bar_combination.total
    progress_bar_combination.refresh()
    progress_bar_combination.close()

    logging.error(f"Aggregated scores: {aggregated_scores['rhs'].size()}\n\n{aggregated_scores['lhs'].size()}")
    return aggregated_scores


def compute_ranks(embedding_models, examples, filters, targets, aggregated_scores, batch_size=500):
    queries = examples.clone()
    ranks = {'rhs': torch.ones(len(queries)), 'lhs': torch.ones(len(queries))}

    progress_bar_ranking = tqdm(total=len(queries) * 2, desc=f"Computing ranking", unit=" queries")

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
                # update progressbar
                if mode == 'lhs':
                    progress_bar_ranking.n = b_begin + len(queries)
                else:
                    progress_bar_ranking.n = b_begin
                progress_bar_ranking.refresh()

                # try:
                these_queries = queries[b_begin:b_begin + batch_size].cuda()

                # set filtered and true scores to -1e6 to be ignored
                for i, query in enumerate(these_queries):
                    filter_out = filters[mode][(query[0].item(), query[1].item())]
                    filter_out += [queries[b_begin + i, 2].item()]
                    aggregated_scores[mode][i, torch.LongTensor(filter_out)] = -1e6
                ranks[mode][b_begin:b_begin + batch_size] += torch.sum(
                    (aggregated_scores[mode][b_begin:b_begin + batch_size] >=
                     targets[mode][b_begin:b_begin + batch_size]).float(),
                    dim=1).cpu()
                # except KeyError:
                #     print(f"ERROR\t{b_begin}")

                b_begin += batch_size

        progress_bar_ranking.n = progress_bar_ranking.total
        progress_bar_ranking.refresh()
        progress_bar_ranking.close()

        logging.log(Constants.DATA_LEVEL, f"Mode:\t{mode}\t\tSize:\t{ranks[mode].size()}\t\tRanks:\n{ranks[mode]}")
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
