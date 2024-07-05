import logging
import time
import traceback

import torch
from torch import nn
from tqdm import tqdm

from ensemble import Constants, util, util_files
from utils.train import avg_both, format_metrics


def evaluate_ensemble(embedding_models, aggregation_method=Constants.MAX_SCORE_AGGREGATION, mode="test",
                      metrics_file_path="", batch_size=20, epoch=None, attention=None):
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

    # calculate scores for all embedding_models
    aggregated_scores, aggregated_targets = calculate_and_combine_scores(embedding_models, examples,
                                                                         aggregation_method, attention, eval_mode=mode,
                                                                         batch_size=batch_size)

    # compute the ranks for all queries
    filters = embedding_models[0]["filters"]
    args = embedding_models[0]["args"]
    model = embedding_models[0]["model"]

    if epoch is None:
        if mode == "test":
            epoch = args.max_epochs + 20
        elif mode == "valid":
            epoch = args.max_epochs + 10

    active_subgraphs = []
    for embedding_model in embedding_models:
        if not embedding_model['args'].model_dropout:
            active_subgraphs.append(embedding_model['subgraph'])

    # calculate metrics from the ranks
    metrics = avg_both(*model.compute_metrics(examples, filters, args.sizes, (aggregated_scores, aggregated_targets),
                                              batch_size=batch_size), epoch=epoch, active_subgraphs=active_subgraphs)
    logging.info(format_metrics(metrics, split=mode))

    if metrics_file_path != "":
        util_files.print_metrics_to_file(metrics_file_path, metrics, epoch, mode)

    for embedding_model in embedding_models:
        for i in range(50):
            logging.log(Constants.DATA_LEVEL_LOGGING, embedding_model['model'].entity.weight.data[i])

        logging.log(Constants.DATA_LEVEL_LOGGING, embedding_model['model'].rel.weight.data)

    time_eval_stop = time.time()

    logging.info(f"-\\\tFinished {mode_str.lower()}ing the ensemble in "
                 f"{util.format_time(time_eval_start, time_eval_stop)}\t/-")

    return metrics


def calculate_and_combine_scores(embedding_models, examples, aggregation_method, attention, eval_mode=None,
                                 batch_size=500):
    """
        Calculate and combine scores from multiple embedding embedding_models for given examples.

        Args:
            embedding_models (list): List of embedding embedding_models with their parameters and data types.
            examples (Tensor): Tensor containing the examples for which scores need to be calculated.
            aggregation_method (tuple): Method used to aggregate scores (e.g., max, min, average).
            eval_mode (str, optional): Evaluation mode for the embedding_models. Defaults to None.
            batch_size (int, optional): Size of the batches for processing examples. Defaults to 500.

        Returns:
            dict: Aggregated scores for 'rhs' and 'lhs' directions.
            dict: Aggregated targets for 'rhs' and 'lhs' directions.
        """
    # Get the size of scores for rhs and lhs directions
    args = embedding_models[0]['args']
    dtype = embedding_models[0]['data_type']
    candidate_answers = len(embedding_models[0]['model'].get_rhs(examples, eval_mode=True)[0])

    size = {'score': (len(examples), candidate_answers),
            'target': (len(examples), 1)}

    # Initialize dictionary to store aggregated scores
    aggregated_scores = {'rhs': torch.zeros(size['score']).to(dtype),
                         'lhs': torch.zeros(size['score']).to(dtype)}
    aggregated_targets = {'rhs': torch.zeros(size['target']).to(dtype),
                          'lhs': torch.zeros(size['target']).to(dtype)}

    b_begin = 0
    steps = len(examples)
    progress_bar_testing = None
    active_models = []
    for index, embedding_model in enumerate(embedding_models):
        if not embedding_model['args'].model_dropout:
            active_models.append(index)
        else:
            logging.debug(f"Found inactive model")

    if not args.no_progress_bar:
        # Update progress bar
        progress_bar_testing = tqdm(total=len(examples),
                                    desc=f"Calculating and combining scores",
                                    unit=" queries", position=0, leave=True)

    while b_begin < steps:
        logging.debug(f"Calculating {eval_mode} scores for the ensemble for indices "
                      f"{b_begin} to {b_begin + batch_size}.")

        model_scores_lhs = []
        model_scores_rhs = []
        model_targets_lhs = []
        model_targets_rhs = []

        for embedding_model in embedding_models:
            args = embedding_model['args']
            if args.model_dropout:
                logging.debug(f"Skipping calculation of scores for {args.subgraph}, since the valid scores diverged "
                              f"to much (factor {args.model_dropout_factor}).")
                continue

            model = embedding_model['model']
            dtype = embedding_model['data_type']

            # Get the number of candidate answers
            candidate_answers = len(model.get_rhs(examples, eval_mode=True)[0])
            logging.debug(f"candidate_answers: {candidate_answers}\tExamples: {len(examples)}")

            scores_lhs, scores_rhs, targets_lhs, targets_rhs = calculate_scores(examples, model, dtype,
                                                                                candidate_answers, b_begin, batch_size)

            logging.debug(f"Combining scores of {args.subgraph} with {aggregation_method[1]} for indices "
                          f"{b_begin} to {b_begin + batch_size}.")

            # Collect scores
            model_scores_lhs += [scores_lhs]
            model_scores_rhs += [scores_rhs]
            model_targets_lhs += [targets_lhs]
            model_targets_rhs += [targets_rhs]

            # logging.debug(
            #     f"scores_lhs size: {((scores_lhs.element_size() * scores_lhs.nelement()) / 1024) / 1024} MB")
            # logging.debug(
            #     f"scores_rhs size: {((scores_rhs.element_size() * scores_rhs.nelement()) / 1024) / 1024} MB")
            # logging.debug(
            #     f"targets_lhs size: {((targets_lhs.element_size() * targets_lhs.nelement()) / 1024) / 1024} MB")
            # logging.debug(
            #     f"targets_rhs size: {((targets_rhs.element_size() * targets_rhs.nelement()) / 1024) / 1024} MB")

        (aggregated_scores['lhs'][b_begin:b_begin + batch_size],
         aggregated_scores['rhs'][b_begin:b_begin + batch_size],
         aggregated_targets['lhs'][b_begin:b_begin + batch_size],
         aggregated_targets['rhs'][b_begin:b_begin + batch_size]) = combine_scores(aggregation_method, model_scores_lhs,
                                                                                   model_scores_rhs, model_targets_lhs,
                                                                                   model_targets_rhs, attention,
                                                                                   examples[b_begin:
                                                                                            b_begin + batch_size],
                                                                                   active_models)

        b_begin += batch_size
        if not args.no_progress_bar:
            # Update progress bar
            progress_bar_testing.n = b_begin
            progress_bar_testing.refresh()

    del model_scores_lhs, model_scores_rhs, model_targets_lhs, model_targets_rhs
    return aggregated_scores, aggregated_targets


def calculate_scores(examples, model, dtype, candidate_answers, b_begin, batch_size):
    """
        Calculate scores and targets for given examples using a specified model.

        Args:
            examples (Tensor): Tensor containing the examples for which scores need to be calculated.
            model (Model): The embedding model used to calculate scores.
            dtype (torch.dtype): Data type of the scores and targets.
            candidate_answers (int): Number of candidate answers for each example.
            b_begin (int): Starting index for the batch of examples.
            batch_size (int): Size of the batch of examples.

        Returns:
            Tensor: Scores for 'lhs' direction.
            Tensor: Scores for 'rhs' direction.
            Tensor: Targets for 'lhs' direction.
            Tensor: Targets for 'rhs' direction.
        """
    # calculate scores, adapted from base.compute_metrics() and base.get_rankings()

    # Initialize tensors to store scores and targets
    scores_rhs = torch.zeros((batch_size, candidate_answers), dtype=dtype)
    scores_lhs = torch.zeros((batch_size, candidate_answers), dtype=dtype)
    targets_rhs = torch.zeros((batch_size, candidate_answers), dtype=dtype)
    targets_lhs = torch.zeros((batch_size, candidate_answers), dtype=dtype)

    for mode in ["rhs", "lhs"]:
        logging.debug(f"Current mode: {mode}")
        queries = examples[b_begin:b_begin + batch_size].clone()

        # Swap subject and object for lhs mode
        if mode == "lhs":
            tmp = torch.clone(queries[:, 0])
            queries[:, 0] = queries[:, 2]
            queries[:, 2] = tmp
            queries[:, 1] += model.sizes[1] // 2

        # Disable gradient computation for inference
        with torch.no_grad():
            candidates = model.get_rhs(queries, eval_mode=True)

            these_queries = queries.cuda()

            q = model.get_queries(these_queries)
            rhs = model.get_rhs(these_queries, eval_mode=False)

            if mode == "lhs":
                scores_lhs = model.score(q, candidates, eval_mode=True)
                targets_lhs = model.score(q, rhs, eval_mode=False)
            else:
                scores_rhs = model.score(q, candidates, eval_mode=True)
                targets_rhs = model.score(q, rhs, eval_mode=False)

        del queries

    return scores_lhs, scores_rhs, targets_lhs, targets_rhs


def combine_scores(aggregation_method, model_scores_lhs, model_scores_rhs, model_targets_lhs, model_targets_rhs,
                   attention, examples, active_models):
    """
    Combine scores from multiple embedding_models using the specified aggregation method.

    Args:
        aggregation_method (tuple): Method used to aggregate scores (e.g., max, min, average).
        model_scores_lhs (list): List of 'lhs' scores from different embedding_models.
        model_scores_rhs (list): List of 'rhs' scores from different embedding_models.
        model_targets_lhs (list): List of 'lhs' targets from different embedding_models.
        model_targets_rhs (list): List of 'rhs' targets from different embedding_models.

    Returns:
        Tensor: Aggregated scores for 'lhs' direction.
        Tensor: Aggregated scores for 'rhs' direction.
        Tensor: Aggregated targets for 'lhs' direction.
        Tensor: Aggregated targets for 'rhs' direction.
    """
    aggregated_scores_lhs, aggregated_scores_rhs = None, None
    aggregated_targets_lhs, aggregated_targets_rhs = None, None

    # Aggregate scores based on the specified method
    if aggregation_method[0] == Constants.MAX_SCORE_AGGREGATION[0]:
        # select maximum score between all embedding_models
        try:
            # Stack model scores, then select max scores
            stacked_tensor = torch.stack(model_scores_lhs, dim=1)
            aggregated_scores_lhs, _ = torch.max(stacked_tensor, dim=1)

            # Stack model scores, then select max scores
            stacked_tensor = torch.stack(model_scores_rhs, dim=1)
            aggregated_scores_rhs, _ = torch.max(stacked_tensor, dim=1)

            # Stack model targets, then select max target
            stacked_tensor = torch.stack(model_targets_lhs, dim=1)
            aggregated_targets_lhs, _ = torch.max(stacked_tensor, dim=1)

            # Stack model targets, then select max target
            stacked_tensor = torch.stack(model_targets_rhs, dim=1)
            aggregated_targets_rhs, _ = torch.max(stacked_tensor, dim=1)

        except Exception as e:
            logging.error(f"Error within {aggregation_method[0]}:\n{traceback.format_exception(e)}")
            return

    elif aggregation_method[0] == Constants.AVERAGE_SCORE_AGGREGATION[0]:
        # average the score across all embedding_models
        try:
            # Stack model scores, then calculate averaged scores
            stacked_tensor = torch.stack(model_scores_lhs, dim=1)
            aggregated_scores_lhs = torch.mean(stacked_tensor, dim=1)

            # Stack model scores, then calculate averaged scores
            stacked_tensor = torch.stack(model_scores_rhs, dim=1)
            aggregated_scores_rhs = torch.mean(stacked_tensor, dim=1)

            # Stack model targets, then calculate averaged target
            stacked_tensor = torch.stack(model_targets_lhs, dim=1)
            aggregated_targets_lhs = torch.mean(stacked_tensor, dim=1)

            # Stack model targets, then calculate averaged target
            stacked_tensor = torch.stack(model_targets_rhs, dim=1)
            aggregated_targets_rhs = torch.mean(stacked_tensor, dim=1)

        except Exception as e:
            logging.error(f"Error within {aggregation_method[0]}:\n{traceback.format_exception(e)}")
            return

    elif aggregation_method[0] == Constants.ATTENTION_SCORE_AGGREGATION[0]:
        # calculate attention between all embedding_models and average scores based on the attention of entities and relation names
        try:
            # Collect attention values for queries
            att_h = attention['ent'][examples[:, 0].unsqueeze(1)]
            att_r = attention['rel'][examples[:, 1].unsqueeze(1)]
            att_t = attention['ent'][examples[:, 2].unsqueeze(1)]

            # Combine Attention values for lhs and rhs, by summing across rank and using softmax
            att_lhs = (att_r * att_t).squeeze()
            att_rhs = (att_r * att_h).squeeze()

            # Filter out inactive embedding_models and do softmax
            activation = nn.Softmax(dim=-1)
            att_lhs = activation(att_lhs[:, active_models])
            att_rhs = activation(att_rhs[:, active_models])

            # Stack model scores, then calculate attention weighted scores
            stacked_tensor = torch.stack(model_scores_lhs, dim=1)
            aggregated_scores_lhs = torch.sum(stacked_tensor.cuda() * att_lhs.unsqueeze(-1).cuda(), dim=1)

            # Stack model scores, then calculate attention weighted scores
            stacked_tensor = torch.stack(model_scores_rhs, dim=1)
            aggregated_scores_rhs = torch.sum(stacked_tensor.cuda() * att_rhs.unsqueeze(-1).cuda(), dim=1)

            # Stack model targets, then calculate attention weighted target
            stacked_tensor = torch.stack(model_targets_lhs, dim=1)
            aggregated_targets_lhs = torch.sum(stacked_tensor.cuda() * att_lhs.unsqueeze(-1).cuda(), dim=1)

            # Stack model targets, then calculate attention weighted target
            stacked_tensor = torch.stack(model_targets_rhs, dim=1)
            aggregated_targets_rhs = torch.sum(stacked_tensor.cuda() * att_rhs.unsqueeze(-1).cuda(), dim=1)

        except Exception as e:
            logging.error(f"Error within {aggregation_method[0]}:\n{traceback.format_exception(e)}")
            return

    else:
        logging.error(f"Selected aggregation method '{aggregation_method}' does not exist!")

    return aggregated_scores_lhs, aggregated_scores_rhs, aggregated_targets_lhs, aggregated_targets_rhs


def calculate_valid_loss(embedding_models):
    valid_loss_dict = {}
    valid_loss = 0.0
    active_models = len(embedding_models)
    # Iterate over all embedding embedding_models
    for embedding_model in embedding_models:
        # Setup variables
        model = embedding_model["model"]
        optimizer = embedding_model["optimizer"]
        valid_examples = embedding_model["valid_examples"]
        subgraph = embedding_model["subgraph"]

        if embedding_model['args'].model_dropout:
            valid_loss_dict[subgraph] = "dropout"
            active_models -= 1
            continue

        # calculate single validation loss
        model.eval()
        # save individual valid losses for display
        valid_loss_sub = optimizer.calculate_valid_loss(valid_examples)
        # sum up valid losses
        valid_loss += valid_loss_sub
        valid_loss_dict[subgraph] = valid_loss_sub.item()

    # average valid loss over all "len(embedding_models)" embedding_models
    valid_loss /= active_models

    return valid_loss, valid_loss_dict


# --- unused functions ---


def calculate_scores_depreciated(embedding_models, examples, batch_size=500, eval_mode="test"):
    """
       Calculate scores for all queries and embedding_models provided.

       Args:
           embedding_models (list): A list of dictionaries, each containing an embedding model
               and its corresponding arguments.
           examples (torch.Tensor): Tensor containing query examples.
           batch_size (int, optional): Batch size for processing queries. Defaults to 500.

       Returns:
           - embedding_models (list): The updated list of embedding embedding_models, each containing
             the calculated scores.

    """

    # Initialize variables to store target scores
    progress_bar_testing = None
    args = None

    logging.info(f"Calculating {eval_mode} scores for the ensemble.")
    # Iterate over each embedding model
    for step, embedding_model in enumerate(embedding_models):
        model = embedding_model['model']
        args = embedding_model['args']
        dtype = embedding_model['data_type']

        if args.model_dropout:
            logging.debug(f"Skipping calculation of scores for {args.subgraph}, since the valid scores diverged "
                          f"to much (factor {args.model_dropout_factor}).")
            continue

        # Get the number of candidate answers
        candidate_answers = len(model.get_rhs(examples, eval_mode=True)[0])
        logging.debug(f"candidate_answers: {candidate_answers}\tExamples: {len(examples)}")

        # Initialize tensors to store scores and targets
        scores_rhs = torch.zeros((len(examples), candidate_answers), dtype=dtype)
        scores_lhs = torch.zeros((len(examples), candidate_answers), dtype=dtype)

        targets_rhs = torch.zeros((len(examples), candidate_answers), dtype=dtype)
        targets_lhs = torch.zeros((len(examples), candidate_answers), dtype=dtype)

        logging.debug(f"scores_rhs size: {((scores_rhs.element_size() * scores_rhs.nelement()) / 1024) / 1024} MB")
        logging.debug(f"scores_lhs size: {((scores_lhs.element_size() * scores_lhs.nelement()) / 1024) / 1024} MB")
        logging.debug(f"targets_rhs size: {((targets_rhs.element_size() * targets_rhs.nelement()) / 1024) / 1024} MB")
        logging.debug(f"targets_lhs size: {((targets_lhs.element_size() * targets_lhs.nelement()) / 1024) / 1024} MB")

        if not args.no_progress_bar:
            # Update progress bar
            progress_bar_testing = tqdm(total=len(examples) * 2,
                                        desc=f"Calculating {eval_mode} scores for {args.subgraph} "
                                             f"(step {step}/{len(embedding_models)})",
                                        unit=" queries", position=0, leave=True)

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
                    if not args.no_progress_bar:
                        # Update progress bar
                        if mode == 'lhs':
                            progress_bar_testing.n = b_begin + len(examples)
                        else:
                            progress_bar_testing.n = b_begin
                        progress_bar_testing.refresh()

                    these_queries = queries[b_begin:b_begin + batch_size].cuda()

                    q = model.get_queries(these_queries)
                    rhs = model.get_rhs(these_queries, eval_mode=False)

                    scores = model.score(q, candidates, eval_mode=True)
                    targets = model.score(q, rhs, eval_mode=False)

                    # Store scores and targets based on mode
                    if mode == "lhs":
                        scores_lhs[b_begin:b_begin + batch_size] = scores
                        targets_lhs[b_begin:b_begin + batch_size] = targets
                        # logging.log(Constants.DATA_LEVEL_LOGGING, f"From {b_begin} to {b_begin + batch_size} lhs:\n"
                        #                                           f"{scores_lhs[b_begin:b_begin + batch_size]}")
                    else:
                        scores_rhs[b_begin:b_begin + batch_size] = scores
                        targets_rhs[b_begin:b_begin + batch_size] = targets
                        # logging.log(Constants.DATA_LEVEL_LOGGING, f"From {b_begin} to {b_begin + batch_size} rhs:\n"
                        #                                           f"{scores_rhs[b_begin:b_begin + batch_size]}")

                    b_begin += batch_size

        if not args.no_progress_bar:
            progress_bar_testing.n = progress_bar_testing.total
            progress_bar_testing.close()
        # Update embedding model dictionary with calculated scores
        embedding_model['scores_lhs'] = scores_lhs
        embedding_model['scores_rhs'] = scores_rhs
        embedding_model['targets_lhs'] = targets_lhs
        embedding_model['targets_rhs'] = targets_rhs

    if not args.no_progress_bar:
        # Complete progress bar and close
        progress_bar_testing.n = progress_bar_testing.total
        progress_bar_testing.refresh()
        progress_bar_testing.close()

    return embedding_models


def combine_scores_depreciated(embedding_models, aggregation_method=Constants.MAX_SCORE_AGGREGATION, batch_size=500,
                               eval_mode="test"):
    """
        Combine scores from multiple embedding_models using the specified aggregation method.

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

    logging.info(f"Combining scores of all embedding_models with {aggregation_method[1]}.")

    # Get the size of scores for rhs and lhs directions
    # if embedding_models[0]['size_rhs'] is None or embedding_models[0]['size_lhs'] is None:
    embedding_models[0]['size_rhs'] = embedding_models[0]['scores_rhs'].size()
    embedding_models[0]['size_lhs'] = embedding_models[0]['scores_lhs'].size()

    size = {'rhs': embedding_models[0]['size_rhs'],
            'lhs': embedding_models[0]['size_lhs']}

    args = embedding_models[0]['args']
    dtype = embedding_models[0]['data_type']

    progress_bar_combination = None
    if not args.no_progress_bar:
        # Initialize progress bar for tracking progress
        progress_bar_combination = tqdm(total=size['rhs'][0] + size['lhs'][0], desc=f"Combine {eval_mode} scores",
                                        unit=" scores")
    # Initialize dictionary to store aggregated scores
    aggregated_scores = {'rhs': torch.zeros(size['rhs']).to(dtype),
                         'lhs': torch.zeros(size['lhs']).to(dtype)}
    aggregated_targets = {'rhs': torch.zeros(size['rhs']).to(dtype),
                          'lhs': torch.zeros(size['lhs']).to(dtype)}

    # Iterate over both directions (rhs and lhs)
    for mode in ["rhs", "lhs"]:
        logging.debug(f"Aggregating {mode}, with a size {size[mode]}")
        b_begin = 0

        # Iterate over each batch of queries
        while b_begin < size[mode][0]:
            if not args.no_progress_bar:
                # Update progress bar
                if mode == 'lhs':
                    progress_bar_combination.n = b_begin + size['rhs'][0]
                else:
                    progress_bar_combination.n = b_begin
                progress_bar_combination.refresh()

            # Collect scores from all embedding_models
            model_scores = []
            model_targets = []
            for embedding_model in embedding_models:
                if embedding_model['args'].model_dropout:
                    continue

                if mode == "lhs":
                    model_scores += [embedding_model['scores_lhs'][b_begin:b_begin + batch_size]]
                    model_targets += [embedding_model['targets_lhs'][b_begin:b_begin + batch_size]]
                else:
                    model_scores += [embedding_model['scores_rhs'][b_begin:b_begin + batch_size]]
                    model_targets += [embedding_model['targets_rhs'][b_begin:b_begin + batch_size]]

            # Aggregate scores based on the specified method
            if aggregation_method[0] == Constants.MAX_SCORE_AGGREGATION[0]:
                # select maximum score between all embedding_models
                try:
                    stacked_tensor = torch.stack(model_scores, dim=1)
                    aggregated_scores[mode][b_begin:b_begin + batch_size], _ = torch.max(stacked_tensor, dim=1)

                    stacked_tensor = torch.stack(model_targets, dim=1)
                    aggregated_targets[mode][b_begin:b_begin + batch_size], _ = torch.max(stacked_tensor, dim=1)

                except Exception as e:
                    for index, score in enumerate(model_scores):
                        logging.error(f"Aggregation method: {aggregation_method[1]}\tMode: {mode}\t"
                                      f"Size of score set {index}: {score.size()}\t"
                                      f"Size of target set {index}: {model_targets[index].size()}")
                    logging.error(traceback.format_exception(e))
                    return

            # Aggregate scores based on the specified method
            elif aggregation_method[0] == Constants.MIN_SCORE_AGGREGATION[0]:
                # select maximum score between all embedding_models
                try:
                    stacked_tensor = torch.stack(model_scores, dim=1)
                    aggregated_scores[mode][b_begin:b_begin + batch_size], _ = torch.min(stacked_tensor, dim=1)

                    stacked_tensor = torch.stack(model_targets, dim=1)
                    aggregated_targets[mode][b_begin:b_begin + batch_size], _ = torch.min(stacked_tensor, dim=1)

                except Exception as e:
                    for index, score in enumerate(model_scores):
                        logging.error(f"Aggregation method: {aggregation_method[1]}\tMode: {mode}\t"
                                      f"Size of score set {index}: {score.size()}\t"
                                      f"Size of target set {index}: {model_targets[index].size()}")
                    logging.error(traceback.format_exception(e))
                    return

            elif aggregation_method[0] == Constants.AVERAGE_SCORE_AGGREGATION[0]:
                # average the score across all embedding_models
                try:
                    stacked_tensor = torch.stack(model_scores, dim=1)
                    aggregated_scores[mode][b_begin:b_begin + batch_size] = torch.mean(stacked_tensor, dim=1)

                    stacked_tensor = torch.stack(model_targets, dim=1)
                    aggregated_targets[mode][b_begin:b_begin + batch_size] = torch.mean(stacked_tensor, dim=1)

                except Exception as e:
                    for index, score in enumerate(model_scores):
                        logging.error(f"Aggregation method: {aggregation_method[1]}\tMode: {mode}\t"
                                      f"Size of score set {index}: {score.size()}\t"
                                      f"Size of target set {index}: {model_targets[index].size()}")
                    logging.error(traceback.format_exception(e))
                    return

            elif aggregation_method[0] == Constants.ATTENTION_SCORE_AGGREGATION[0]:
                # calculate attention between all embedding_models and average scores based on this attention
                logging.error(f"Aggregation method {aggregation_method[1]} isn't implemented yet!")
                pass

            else:
                logging.error(f"Selected aggregation method '{aggregation_method}' does not exist!")

            b_begin += batch_size

    if not args.no_progress_bar:
        # Update progress bar and close
        progress_bar_combination.n = progress_bar_combination.total
        progress_bar_combination.refresh()
        progress_bar_combination.close()

    logging.info(f"Successfully aggregated all scores and targets with aggregation method {aggregation_method[1]}.")

    return aggregated_scores, aggregated_targets


def compute_metrics_from_ranks_depreciated(ranks_opt, ranks_pes, sizes):
    """
        Compute various evaluation metrics based on the computed ranks.

        Args:
            ranks_opt (dict): Dictionary containing computed optimistic ranks for both lhs and rhs directions.
            ranks_pes (dict): Dictionary containing computed pessimistic ranks for both lhs and rhs directions.

        Returns:
            tuple: A tuple containing:
                - mean_rank (dict): Mean rank for lhs and rhs directions.
                - mean_reciprocal_rank (dict): Mean reciprocal rank for lhs and rhs directions.
                - hits_at (dict): Dictionary containing hits at different positions for lhs and rhs directions.
                - amri (dict): Average Mean Rank Improvement for lhs and rhs directions.
                - rank_deviation (dict): Mean Rank Deviation for lhs and rhs directions.

    """

    # Initialize dictionaries for metrics
    mean_rank = {}
    mean_reciprocal_rank = {}
    hits_at = {}
    amri = {}
    rank_deviation = {}

    # Iterate over both directions (rhs and lhs)
    for mode in ["rhs", "lhs"]:
        optimistic_rank = ranks_opt[mode]
        pessimistic_rank = ranks_pes[mode]

        # Compute mean rank
        mean_rank[mode] = torch.mean(optimistic_rank).item()

        # Compute mean reciprocal rank
        mean_reciprocal_rank[mode] = torch.mean(1. / optimistic_rank).item()

        # Compute hits@1, hits@3 and hits@10
        hits_at[mode] = torch.FloatTensor((list(map(
            lambda x: torch.mean((optimistic_rank <= x).float()).item(), (1, 3, 10)
        ))))

        # Compute AMRI
        sum_ranks = torch.sum(ranks_opt[mode] - 1)
        sum_scores = ranks_opt[mode].size()[0] * sizes[0]
        amri[mode] = 1 - (2 * sum_ranks) / sum_scores

        # Compute rank_deviation
        rank_deviation[mode] = torch.sum(optimistic_rank - pessimistic_rank)

    return mean_rank, mean_reciprocal_rank, hits_at, amri, rank_deviation


def compute_ranks_depreciated(embedding_models, examples, filters, targets, aggregated_scores, batch_size=500):
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
    ranks_opt = {'rhs': torch.ones(len(queries)),
                 'lhs': torch.ones(len(queries))}

    ranks_pes = {'rhs': torch.ones(len(queries)),
                 'lhs': torch.ones(len(queries))}

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
                these_queries = queries[b_begin:b_begin + batch_size].cuda()

                # set filtered and true scores to -1e6 to be ignored
                for i, query in enumerate(these_queries):
                    filter_out = filters[mode][(query[0].item(), query[1].item())]
                    filter_out += [queries[b_begin + i, 2].item()]
                    aggregated_scores[mode][i, torch.LongTensor(filter_out)] = -1e6

                # Calculate optimistic rank
                ranks_opt[mode][b_begin:b_begin + batch_size] += torch.sum(
                    (aggregated_scores[mode][b_begin:b_begin + batch_size] >=
                     targets[mode][b_begin:b_begin + batch_size]).float(), dim=1).cpu()

                # Calculate pessimistic rank
                pessimistic_rank = torch.sum((aggregated_scores[mode][b_begin:b_begin + batch_size] >
                                              targets[mode][b_begin:b_begin + batch_size]).float(), dim=1)

                # Adjust for pessimistic rank (subtract 1 if target score is included)
                target_subtraction = torch.sum((aggregated_scores[mode][b_begin:b_begin + batch_size] ==
                                                targets[mode][b_begin:b_begin + batch_size]).float(), dim=1)

                ranks_pes[mode][b_begin:b_begin + batch_size] += (pessimistic_rank - target_subtraction).cpu()

                b_begin += batch_size

    return ranks_opt, ranks_pes
