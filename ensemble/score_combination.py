import logging
import time
import traceback

import torch
from tqdm import tqdm

from ensemble import Constants, util
from utils.train import avg_both, format_metrics


def evaluate_ensemble(embedding_models, aggregation_method=Constants.MAX_SCORE_AGGREGATION, mode="test",
                      batch_size=500):
    mode_str = ""
    if mode == "test":
        logging.info(f"-/\tTesting the ensemble with the score aggregation method \"{aggregation_method[1]}\".\t\\-")
        examples = embedding_models[0]["test_examples"]
        mode_str = "Test"
    elif mode == "valid":
        logging.info(f"-/\tValidating the ensemble with the score aggregation method \"{aggregation_method[1]}\".\t\\-")
        examples = embedding_models[0]["valid_examples"]
        mode_str = "Validat"
    else:
        logging.error(f"The given mode \"{mode}\" does not exist!")
        return

    time_eval_start = time.time()

    # calculate scores for all models
    embedding_models, targets = calculate_scores(embedding_models, examples, batch_size=batch_size, eval_mode=mode)

    # combine the calculated scores from all models, according to the given aggregation method
    aggregated_scores = combine_scores(embedding_models, aggregation_method, batch_size=batch_size, eval_mode=mode)

    # compute the ranks for all queries
    filters = embedding_models[0]["filters"]
    ranks = compute_ranks(embedding_models, examples, filters, targets, aggregated_scores,
                          batch_size=batch_size, eval_mode=mode)

    # calculate metrics from the ranks
    metrics = avg_both(*compute_metrics_from_ranks(ranks))
    logging.info(format_metrics(metrics, split=mode))

    time_eval_stop = time.time()

    logging.info(f"-\\\tFinished {mode_str.lower()}ing the ensemble in "
                 f"{util.format_time(time_eval_start, time_eval_stop)}\t/-")

    return metrics


def calculate_scores(embedding_models, examples, batch_size=500, eval_mode="test"):
    """
       Calculate scores for all queries and models provided.

       Args:
           embedding_models (list): A list of dictionaries, each containing an embedding model
               and its corresponding arguments.
           examples (torch.Tensor): Tensor containing query examples.
           batch_size (int, optional): Batch size for processing queries. Defaults to 500.

       Returns:
           tuple: A tuple containing:
               - embedding_models (list): The updated list of embedding models, each containing
                 the calculated scores.
               - targets (dict): Dictionary containing target scores for both lhs and rhs directions.
    """

    # Initialize progress bar for tracking progress
    progress_bar_testing = tqdm(total=len(embedding_models), desc=f"Calculating {eval_mode} scores",
                                unit=" embedding models")

    # Initialize variables to store target scores
    targets_lhs = None
    targets_rhs = None

    # Iterate over each embedding model
    for embedding_model in embedding_models:
        model = embedding_model['model']
        args = embedding_model['args']

        # Get the number of candidate answers
        candidate_answers = len(model.get_rhs(examples, eval_mode=True)[0])
        logging.debug(f"Candidate answers for {args.subgraph}: {candidate_answers}")

        # Initialize tensors to store scores and targets
        scores_rhs = torch.zeros((len(examples), candidate_answers))
        scores_lhs = torch.zeros((len(examples), candidate_answers))

        targets_rhs = torch.zeros((len(examples), candidate_answers))
        targets_lhs = torch.zeros((len(examples), candidate_answers))

        # Update progress bar
        progress_bar_testing.n = args.subgraph_num
        progress_bar_testing.refresh()

        # calculate scores, adapted from base.compute_metrics() and base.get_rankings()
        # Iterate over each mode (rhs and lhs)
        for mode in ["rhs", "lhs"]:
            queries = examples.clone()

            # Swap subject and object for lhs mode
            if mode == "lhs":
                tmp = torch.clone(queries[:, 0])
                queries[:, 0] = queries[:, 2]
                queries[:, 2] = tmp
                queries[:, 1] += model.sizes[1] // 2

            # Disable gradient computation for inference
            with torch.no_grad():
                b_begin = 0
                candidates = model.get_rhs(queries, eval_mode=True)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size].cuda()

                    q = model.get_queries(these_queries)
                    rhs = model.get_rhs(these_queries, eval_mode=False)

                    scores = model.score(q, candidates, eval_mode=True)
                    targets = model.score(q, rhs, eval_mode=False)

                    # Store scores and targets based on mode
                    if mode == "lhs":
                        scores_lhs[b_begin:b_begin + batch_size] = scores
                        targets_lhs[b_begin:b_begin + batch_size] = targets
                        logging.log(Constants.DATA_LEVEL_LOGGING, f"From {b_begin} to {b_begin + batch_size} lhs:\n"
                                                                  f"{scores_lhs[b_begin:b_begin + batch_size]}")
                    else:
                        scores_rhs[b_begin:b_begin + batch_size] = scores
                        targets_rhs[b_begin:b_begin + batch_size] = targets
                        logging.log(Constants.DATA_LEVEL_LOGGING, f"From {b_begin} to {b_begin + batch_size} rhs:\n"
                                                                  f"{scores_rhs[b_begin:b_begin + batch_size]}")

                    b_begin += batch_size
        # Update embedding model dictionary with calculated scores
        embedding_model['scores_lhs'] = scores_lhs
        embedding_model['scores_rhs'] = scores_rhs
        embedding_model['targets_lhs'] = targets_lhs
        embedding_model['targets_rhs'] = targets_rhs
        embedding_model['candidate_answers'] = candidate_answers

    # Aggregate target scores into a dictionary
    targets = {'rhs': targets_rhs, 'lhs': targets_lhs}

    # Complete progress bar and close
    progress_bar_testing.n = progress_bar_testing.total
    progress_bar_testing.refresh()
    progress_bar_testing.close()

    return embedding_models, targets


def combine_scores(embedding_models, aggregation_method=Constants.MAX_SCORE_AGGREGATION, batch_size=500,
                   eval_mode="test"):
    """
        Combine scores from multiple models using the specified aggregation method.

        Args:
            embedding_models (list): A list of dictionaries, each containing an embedding model
                and its corresponding arguments. Each dictionary should have the following keys:
                    - 'scores_lhs': Tensor containing scores for lhs direction.
                    - 'scores_rhs': Tensor containing scores for rhs direction.
            aggregation_method (tuple, optional): A tuple containing the aggregation method and its name.
                Defaults to Constants.MAX_SCORE_AGGREGATION.
            batch_size (int, optional): Batch size for processing queries. Defaults to 500.

        Returns:
            dict: Dictionary containing aggregated scores for both lhs and rhs directions.

    """

    logging.info(f"Combining scores of all models with {aggregation_method[1]}.")

    # Get the size of scores for rhs and lhs directions
    size = {'rhs': embedding_models[0]['scores_rhs'].size(),
            'lhs': embedding_models[0]['scores_lhs'].size()}

    # Initialize progress bar for tracking progress
    progress_bar_combination = tqdm(total=size['rhs'][0] + size['lhs'][0], desc=f"Combine {eval_mode} scores",
                                    unit=" scores")

    # Initialize dictionary to store aggregated scores
    aggregated_scores = {'rhs': torch.zeros(size['rhs']),
                         'lhs': torch.zeros(size['lhs'])}

    # Iterate over both directions (rhs and lhs)
    for mode in ["rhs", "lhs"]:
        logging.debug(f"Aggregating {mode}, with a size {size[mode]}")
        b_begin = 0

        # Iterate over each batch of queries
        while b_begin < len(embedding_models[0]['scores_lhs']):
            # Update progress bar
            if mode == 'lhs':
                progress_bar_combination.n = b_begin + size['rhs'][0]
            else:
                progress_bar_combination.n = b_begin
            progress_bar_combination.refresh()

            # Collect scores from all models
            model_scores = []
            for embedding_model in embedding_models:
                if mode == "lhs":
                    model_scores += [embedding_model['scores_lhs'][b_begin:b_begin + batch_size]]
                else:
                    model_scores += [embedding_model['scores_rhs'][b_begin:b_begin + batch_size]]

            # Aggregate scores based on the specified method
            if aggregation_method[0] == Constants.MAX_SCORE_AGGREGATION[0]:
                # select maximum score between all models
                try:
                    stacked_scores = torch.stack(model_scores, dim=1)
                    logging.log(Constants.DATA_LEVEL_LOGGING, f"Stacked scores:\n{stacked_scores}")

                    aggregated_scores[mode][b_begin:b_begin + batch_size], _ = torch.max(stacked_scores, dim=1)
                    logging.log(Constants.DATA_LEVEL_LOGGING, f"Size:\t{aggregated_scores[mode].size()}\t\t"
                                                              f"Aggregated scores:\n{aggregated_scores[mode]}")
                except Exception as e:
                    for index, score in enumerate(model_scores):
                        logging.error(f"Aggregation method: {aggregation_method[1]}\tMode: {mode}\t"
                                      f"Size of score set {index}: {score.size()}")
                    logging.error(traceback.format_exception(e))
                    return

            elif aggregation_method[0] == Constants.AVERAGE_SCORE_AGGREGATION[0]:
                # average the score across all models
                try:
                    stacked_scores = torch.stack(model_scores, dim=1)
                    logging.log(Constants.DATA_LEVEL_LOGGING, f"Stacked scores:\n{stacked_scores}")

                    aggregated_scores[mode][b_begin:b_begin + batch_size], _ = torch.mean(stacked_scores, dim=1)
                    logging.log(Constants.DATA_LEVEL_LOGGING, f"Size:\t{aggregated_scores[mode].size()}\t\t"
                                                              f"Aggregated scores:\n{aggregated_scores[mode]}")
                except Exception as e:
                    for index, score in enumerate(model_scores):
                        logging.error(f"Aggregation method: {aggregation_method[1]}\tMode: {mode}\t"
                                      f"Size of score set {index}: {score.size()}")
                    logging.error(traceback.format_exception(e))

            elif aggregation_method[0] == Constants.ATTENTION_SCORE_AGGREGATION[0]:
                # calculate attention between all models and average scores based on this attention
                logging.error(f"Aggregation method {aggregation_method[1]} isn't implemented yet!")
                # TODO implement attention score
                pass

            else:
                logging.error(f"Selected aggregation method does not exist!")

            b_begin += batch_size

    # Update progress bar and close
    progress_bar_combination.n = progress_bar_combination.total
    progress_bar_combination.refresh()
    progress_bar_combination.close()

    logging.info(f"Successfully aggregated all scores with aggregation method {aggregation_method[1]}.")

    return aggregated_scores


def compute_ranks(embedding_models, examples, filters, targets, aggregated_scores, batch_size=500, eval_mode="test"):
    """
        Compute ranks for each query based on the aggregated scores, targets, and filters.

        Args:
            embedding_models (list): A list of dictionaries, each containing an embedding model
                and its corresponding arguments. Each dictionary should have the following keys:
                    - 'scores_lhs': Tensor containing scores for lhs direction.
                    - 'scores_rhs': Tensor containing scores for rhs direction.
            examples (torch.Tensor): Tensor containing query examples.
            filters (dict): Dictionary containing filters for lhs and rhs directions.
            targets (dict): Dictionary containing target scores for lhs and rhs directions.
            aggregated_scores (dict): Dictionary containing aggregated scores for lhs and rhs directions.
            batch_size (int, optional): Batch size for processing queries. Defaults to 500.

        Returns:
            dict: Dictionary containing computed ranks for both lhs and rhs directions.

    """

    # Initialize dictionary for ranks
    queries = examples.clone()
    ranks = {'rhs': torch.zeros(len(queries)),
             'lhs': torch.zeros(len(queries))}

    # Initialize progress bar for tracking progress
    progress_bar_ranking = tqdm(total=len(queries) * 2, desc=f"Computing {eval_mode} ranking", unit=" queries")

    # Iterate over both directions (rhs and lhs)
    for mode in ["rhs", "lhs"]:
        # Disable gradient computation for inference
        with torch.no_grad():
            b_begin = 0
            queries = examples.clone()

            # Swap subject and object for lhs mode
            if mode == "lhs":
                tmp = torch.clone(queries[:, 0])
                queries[:, 0] = queries[:, 2]
                queries[:, 2] = tmp
                queries[:, 1] += embedding_models[0]['model'].sizes[1] // 2

            # Iterate over each batch of queries
            while b_begin < len(queries):
                # Update progress bar
                if mode == 'lhs':
                    progress_bar_ranking.n = b_begin + len(queries)
                else:
                    progress_bar_ranking.n = b_begin
                progress_bar_ranking.refresh()

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

                b_begin += batch_size

        # Complete progress bar and close
        progress_bar_ranking.n = progress_bar_ranking.total
        progress_bar_ranking.refresh()
        progress_bar_ranking.close()

        logging.log(Constants.DATA_LEVEL_LOGGING,
                    f"Mode:\t{mode}\t\tSize:\t{ranks[mode].size()}\t\tRanks:\n{ranks[mode]}")
    return ranks


def compute_metrics_from_ranks(ranks_dict):
    """
        Compute various evaluation metrics based on the computed ranks.

        Args:
            ranks_dict (dict): Dictionary containing computed ranks for both lhs and rhs directions.

        Returns:
            tuple: A tuple containing:
                - mean_rank (dict): Mean rank for lhs and rhs directions.
                - mean_reciprocal_rank (dict): Mean reciprocal rank for lhs and rhs directions.
                - hits_at (dict): Dictionary containing hits at different positions for lhs and rhs directions.
                - amri (dict): Average Mean Rank Improvement for lhs and rhs directions.
                - mr_deviation (dict): Mean Rank Deviation for lhs and rhs directions.

    """

    # Initialize dictionaries for metrics
    mean_rank = {}
    mean_reciprocal_rank = {}
    hits_at = {}
    amri = {}
    mr_deviation = {}

    # TODO check/correct calculation of MRR AMRI and MR_Deviation
    # Iterate over both directions (rhs and lhs)
    for mode in ["rhs", "lhs"]:
        ranks = ranks_dict[mode]

        # Compute mean rank
        mean_rank[mode] = torch.mean(ranks).item()

        # Compute mean reciprocal rank
        mean_reciprocal_rank[mode] = torch.mean(1. / ranks).item()

        # Compute hits@1, hits@3 and hits@10
        hits_at[mode] = torch.FloatTensor((list(map(
            lambda x: torch.mean((ranks <= x).float()).item(), (1, 3, 10)
        ))))

        # Compute AMRI
        expected_rank = torch.mean(ranks - 1)  # Expectation of MR - 1
        amri[mode] = 1 - ((mean_rank[mode] - 1) / expected_rank)

        # Compute MR_deviation
        optimistic_rank = torch.min(ranks) - 1
        pessimistic_rank = torch.max(ranks) - 1
        mr_deviation[mode] = optimistic_rank - pessimistic_rank

    return mean_rank, mean_reciprocal_rank, hits_at, amri, mr_deviation


def calculate_valid_loss(embedding_models):
    valid_loss_dict = {}
    valid_loss = 0.0
    # Iterate over all embedding models
    for embedding_model in embedding_models:
        # Setup variables
        model = embedding_model["model"]
        optimizer = embedding_model["optimizer"]
        valid_examples = embedding_model["valid_examples"]
        subgraph = embedding_model["subgraph"]

        # calculate single validation loss
        model.eval()
        # save individual valid losses for display
        valid_loss_sub = optimizer.calculate_valid_loss(valid_examples)
        # sum up valid losses
        valid_loss += valid_loss_sub
        valid_loss_dict[subgraph] = valid_loss_sub.item()

    # average valid loss over all "len(embedding_models)" models
    valid_loss /= len(embedding_models)

    return valid_loss, valid_loss_dict
