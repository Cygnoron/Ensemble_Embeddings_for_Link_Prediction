import logging

import torch
from torch import nn

from ensemble import Constants
from models import hyperbolic


def calculate_self_attention(embedding_models, theta_calculation, batch_size=500):
    if theta_calculation[0] == Constants.NO_THETA[0]:
        # Do nothing, if no context vector was created
        pass

    else:
        logging.info(f"Calculating self-attention...")

        activation = nn.Softmax(dim=1)
        args = embedding_models[0]['args']

        cands_ent, cands_rel, theta_ent, theta_rel = get_cands(embedding_models, batch_size)

        # calculate attention
        att_weights_ent = None
        att_weights_rel = None

        if theta_calculation[0] == Constants.NO_THETA:
            return

        elif theta_calculation[0] == Constants.UNCHANGED_THETA[0]:
            # EXAMPLE WN18RR dimensions (40943 entities, 11*2 relation names, N subgraphs)
            # GOAL: att_weights_ent.size() = [40943, 32, N]     att_weights_ent.size() = [22, 32, N]
            # INPUT: theta_ent = [40943, 32, N]                 theta_rel = [22, 32, N]
            #        cands_ent = [40943, 32, N]                 cands_rel = [22, 32, N]
            logging.debug(f"BEFORE\t\ttheta_ent: {theta_ent.size()}\ncands_ent: {cands_ent.size()}")
            att_weights_ent = torch.sum(theta_ent * cands_ent, dim=-1, keepdim=True)
            att_weights_rel = torch.sum(theta_rel * cands_rel, dim=-1, keepdim=True)
            logging.debug(f"att_weights_ent size: {att_weights_ent.size()}")
            logging.debug(f"att_weights_rel size: {att_weights_rel.size()}")

        elif theta_calculation[0] == Constants.REVERSED_THETA[0]:
            # EXAMPLE WN18RR dimensions (40943 entities, 11*2 relation names, N subgraphs)
            # GOAL: att_weights_ent.size() = [40943, 32, N]     att_weights_ent.size() = [22, 32, N]
            # INPUT: theta_ent = [   22, 32, N]                 theta_rel = [40943, 32, N]
            #        cands_ent = [40943, 32, N]                 cands_rel = [   22, 32, N]

            theta_ent = theta_ent.unsqueeze(0)  # Shape: [1, 22, 32, N]
            theta_ent = theta_ent.expand(cands_ent.size(0), -1, -1, -1)  # Shape: [40943, 22, 32, N]
            att_weights_ent = torch.sum(theta_ent * cands_ent.unsqueeze(1), dim=1)  # Shape: [40943, 32, N]
            logging.debug(f"att_weights_ent size: {att_weights_ent.size()}")

            theta_rel = theta_rel.unsqueeze(0)  # Shape: [1, 40943, 32, N]
            theta_rel = theta_rel.expand(cands_rel.size(0), -1, -1, -1)  # Shape: [22, 40943, 32, N]
            att_weights_rel = torch.sum(theta_rel * cands_rel.unsqueeze(1), dim=1)  # Shape: [22, 32, N]
            logging.debug(f"att_weights_rel size: {att_weights_rel.size()}")

        elif theta_calculation[0] == Constants.RELATION_THETA[0]:
            # EXAMPLE WN18RR dimensions (40943 entities, 11*2 relation names, N subgraphs)
            # GOAL: att_weights_ent.size() = [40943, 32, N]     att_weights_ent.size() = [22, 32, N]
            # INPUT: theta_ent = [   22, 32, N]                 theta_rel = [22, 32, N]
            #        cands_ent = [40943, 32, N]                 cands_rel = [22, 32, N]

            theta_ent = theta_ent.unsqueeze(0)  # Shape: [1, 22, 32, N]
            theta_ent = theta_ent.expand(cands_ent.size(0), -1, -1, -1)  # Shape: [40943, 22, 32, N]
            att_weights_ent = torch.sum(theta_ent * cands_ent.unsqueeze(1), dim=1)  # Shape: [40943, 32, N]
            logging.debug(f"att_weights_ent size: {att_weights_ent.size()}")

            att_weights_rel = torch.sum(theta_rel * cands_rel, dim=-1, keepdim=True)
            logging.debug(f"att_weights_rel size: {att_weights_rel.size()}")

        elif theta_calculation[0] == Constants.MULTIPLIED_THETA[0]:
            raise ValueError(f"The method \"{args.theta_calculation[1]}\" has not been implemented yet!")
            # EXAMPLE WN18RR dimensions (40943 entities, 11*2 relation names, N subgraphs)
            # GOAL: att_weights_ent.size() = [40943, 32, N]     att_weights_ent.size() = [22, 32, N]
            # INPUT: theta_ent = [40943, 32, N]                 theta_rel = [22, 32, N]
            #        cands_ent = [40943, 32, N]                 cands_rel = [22, 32, N]

        # apply activation function
        att_weights_ent = activation(att_weights_ent)
        att_weights_rel = activation(att_weights_rel)

        logging.log(Constants.DATA_LEVEL_LOGGING, f"Attention entity (Size: {att_weights_ent.size()}):\n"
                                                  f"{att_weights_ent}")
        logging.log(Constants.DATA_LEVEL_LOGGING, f"Attention relation names (Size: {att_weights_rel.size()}):\n"
                                                  f"{att_weights_rel}")

        return {'cands_ent': cands_ent, 'cands_rel': cands_rel, 'att_weights_ent': att_weights_ent,
                'att_weights_rel': att_weights_rel}


def get_cands(embedding_models, batch_size):
    args = embedding_models[0]['args']
    steps = args.sizes[0]
    logging.debug(args.sizes)

    cands_ent = torch.zeros(args.sizes[0], args.rank, args.subgraph_amount)
    cands_rel = torch.zeros(args.sizes[1], args.rank, args.subgraph_amount)
    cands_ent.to('cuda')
    cands_rel.to('cuda')

    theta_ent = torch.zeros(args.sizes[0], args.rank, args.subgraph_amount)
    theta_rel = torch.zeros(args.sizes[1], args.rank, args.subgraph_amount)

    if args.theta_calculation[0] == Constants.UNCHANGED_THETA[0]:
        # theta_ent = theta_ent
        # theta_rel = theta_rel
        theta_ent = torch.zeros(args.sizes[0], args.rank, args.subgraph_amount)
        theta_rel = torch.zeros(args.sizes[1], args.rank, args.subgraph_amount)

    elif args.theta_calculation[0] == Constants.REVERSED_THETA[0]:
        # theta_ent = theta_rel
        # theta_rel = theta_ent
        theta_ent = torch.zeros(args.sizes[1], args.rank, args.subgraph_amount)
        theta_rel = torch.zeros(args.sizes[0], args.rank, args.subgraph_amount)

    elif args.theta_calculation[0] == Constants.RELATION_THETA[0]:
        # theta_ent = theta_rel
        # theta_rel = theta_rel
        theta_ent = torch.zeros(args.sizes[1], args.rank, args.subgraph_amount)
        theta_rel = torch.zeros(args.sizes[1], args.rank, args.subgraph_amount)

    elif args.theta_calculation[0] == Constants.MULTIPLIED_THETA[0]:
        # theta_ent = theta_ent * theta_rel
        # theta_rel = theta_rel
        theta_ent = torch.zeros(args.sizes[0], args.rank, args.subgraph_amount)
        theta_rel = torch.zeros(args.sizes[1], args.rank, args.subgraph_amount)

    theta_ent.to('cuda')
    theta_rel.to('cuda')

    b_begin = 0
    while b_begin < steps:

        cands_ent_temp = []
        theta_ent_temp = []
        cands_rel_temp = []
        theta_rel_temp = []

        for embedding_model in embedding_models:
            model = embedding_model["model"]
            cands_ent_temp.append(model.entity.weight.data[b_begin:b_begin + batch_size])
            theta_ent_temp.append(model.theta_ent.weight.data[b_begin:b_begin + batch_size])
            model = embedding_model["model"]
            if embedding_model["args"].model_name in hyperbolic.HYP_MODELS:
                logging.debug(f"HYPERBOLIC")
                cands_rel_temp.append(torch.chunk(model.rel.weight.data[b_begin:b_begin + batch_size], 2, dim=1))
            else:
                cands_rel_temp.append(model.rel.weight.data[b_begin:b_begin + batch_size])
            theta_rel_temp.append(model.theta_rel.weight.data[b_begin:b_begin + batch_size])

        cands_ent[b_begin:b_begin + batch_size] = torch.stack(cands_ent_temp, dim=-1)
        theta_ent[b_begin:b_begin + batch_size] = torch.stack(theta_ent_temp, dim=-1)
        cands_rel[b_begin:b_begin + batch_size] = torch.stack(cands_rel_temp, dim=-1)
        theta_rel[b_begin:b_begin + batch_size] = torch.stack(theta_rel_temp, dim=-1)

        b_begin += batch_size

    logging.debug(f"Entity sizes:\t-Cands: {cands_ent.size()}\t-Theta: {theta_ent.size()}")
    logging.debug(f"Relation name sizes:\t-Cands: {cands_rel.size()}\t-Theta: {theta_rel.size()}")

    return cands_ent, cands_rel, theta_ent, theta_rel


def calculate_and_apply_unified_embedding(general_embedding_ent, general_embedding_rel, embedding_models,
                                          cands_att_dict, theta_calculation, batch_size=500):
    if theta_calculation[0] == Constants.NO_THETA[0]:
        # Do nothing, if no context vector was created
        return

    else:
        # calculate unified embedding from alphas
        logging.info(f"Calculating unified embedding for entities and relation names.")
        logging.debug(f"Size of general entity embedding, before applying attention:\t"
                      f"{general_embedding_ent.weight.data.size()}")
        logging.debug(f"Size of general relation name embedding, before applying attention:\t"
                      f"{general_embedding_rel.weight.data.size()}")

        b_begin = 0
        while b_begin < general_embedding_ent.weight.data.size()[0]:
            general_embedding_ent.weight.data[b_begin:b_begin + batch_size] = torch.sum(
                cands_att_dict['att_weights_ent'][b_begin:b_begin + batch_size] *
                cands_att_dict['cands_ent'][b_begin:b_begin + batch_size], dim=-1)

            general_embedding_rel.weight.data[b_begin:b_begin + batch_size] = torch.sum(
                cands_att_dict['att_weights_rel'][b_begin:b_begin + batch_size] *
                cands_att_dict['cands_rel'][b_begin:b_begin + batch_size], dim=-1)

            b_begin += batch_size

        logging.info(f"Applying general embeddings to all models.")
        sizes = embedding_models[0]['args'].sizes
        # write new unified embedding into all models
        for embedding_model in embedding_models:
            if ('entities' not in embedding_model.keys()) or ('relation_names' not in embedding_model.keys()):
                entities, relation_names = embedding_model['dataset'].get_entities_relation_names(sizes,
                                                                                                  double_relations=True)
                embedding_model['entities'] = entities
                embedding_model['relation_names'] = relation_names

            # only set the embeddings, which are contained in the corresponding subgraph
            embedding_model['model'].entity.weight.data[embedding_model['entities']] = (
                general_embedding_ent.weight.data[embedding_model['entities']].to(torch.float))
            embedding_model['model'].rel.weight.data[embedding_model['relation_names']] = (
                general_embedding_rel.weight.data[embedding_model['relation_names']].to(torch.float))
