import logging

import torch
from torch import nn
from tqdm import tqdm

from ensemble import Constants


def calculate_self_attention(embedding_models, batch_size=500, entity_attention=True, relation_name_attention=True):
    logging.info(f"Calculating self-attention...")

    activation = nn.Softmax(dim=1)
    args = embedding_models[0]['args']
    steps = args.sizes[0]

    # Initialize progress bar for tracking progress
    progress_bar_attention = tqdm(total=steps, desc=f"Calculating attention scores", unit=" embeddings")
    b_begin = 0
    logging.debug(f"b_begin: {b_begin}\tSteps: {steps}")

    cands_ent, cands_rel, theta_ent, theta_rel = get_cands(embedding_models, batch_size, entity_attention,
                                                           relation_name_attention)

    # calculate attention
    att_weights_ent = {}
    att_weights_rel = {}

    b_begin = 0
    while b_begin < steps:
        progress_bar_attention.n = b_begin
        progress_bar_attention.refresh()

        if entity_attention:
            att_weights_ent = torch.sum(theta_ent * cands_ent, dim=-1, keepdim=True)
            att_weights_ent = activation(att_weights_ent)
            logging.log(Constants.DATA_LEVEL_LOGGING, f"Attention weights entity embeddings:\n{att_weights_ent}")

        if relation_name_attention:
            att_weights_rel = torch.sum(theta_rel * cands_rel, dim=-1, keepdim=True)
            att_weights_rel = activation(att_weights_rel)
            logging.log(Constants.DATA_LEVEL_LOGGING, f"Attention weights relation name embeddings:\n{att_weights_rel}")

        b_begin += batch_size

    if entity_attention:
        logging.log(Constants.DATA_LEVEL_LOGGING, f"Attention entity (Size: {att_weights_ent.size()}):\n"
                                                  f"{att_weights_ent}")
    if relation_name_attention:
        logging.log(Constants.DATA_LEVEL_LOGGING, f"Attention relation names (Size: {att_weights_rel.size()}):\n"
                                                  f"{att_weights_rel}")

    # Update progress bar and close
    progress_bar_attention.n = progress_bar_attention.total
    progress_bar_attention.refresh()
    progress_bar_attention.close()

    return {'cands_ent': cands_ent, 'cands_rel': cands_rel, 'att_weights_ent': att_weights_ent,
            'att_weights_rel': att_weights_rel}


def get_cands(embedding_models, batch_size, entity_attention, relation_name_attention):
    args = embedding_models[0]['args']
    steps = args.sizes[0]
    logging.debug(args.sizes)

    cands_ent = None
    theta_ent = None

    cands_rel = None
    theta_rel = None

    if entity_attention:
        cands_ent = torch.zeros(args.sizes[0], args.rank, args.subgraph_amount)
        theta_ent = torch.zeros(args.sizes[0], args.rank, args.subgraph_amount)
        cands_ent.to('cuda')
        theta_ent.to('cuda')

    if relation_name_attention:
        cands_rel = torch.zeros(args.sizes[1], args.rank, args.subgraph_amount)
        theta_rel = torch.zeros(args.sizes[1], args.rank, args.subgraph_amount)
        cands_rel.to('cuda')
        theta_rel.to('cuda')
    b_begin = 0
    while b_begin < steps:

        if entity_attention:
            cands_ent_temp = []
            theta_ent_temp = []

            for embedding_model in embedding_models:
                model = embedding_model["model"]
                cands_ent_temp.append(model.entity.weight.data[b_begin:b_begin + batch_size])
                theta_ent_temp.append(model.theta_ent.weight.data[b_begin:b_begin + batch_size])

            cands_ent[b_begin:b_begin + batch_size] = torch.stack(cands_ent_temp, dim=-1)
            theta_ent[b_begin:b_begin + batch_size] = torch.stack(theta_ent_temp, dim=-1)

        if relation_name_attention:
            cands_rel_temp = []
            theta_rel_temp = []

            for embedding_model in embedding_models:
                model = embedding_model["model"]
                cands_rel_temp.append(model.rel.weight.data[b_begin:b_begin + batch_size])
                theta_rel_temp.append(model.theta_rel.weight.data[b_begin:b_begin + batch_size])

            cands_rel[b_begin:b_begin + batch_size] = torch.stack(cands_rel_temp, dim=-1)
            theta_rel[b_begin:b_begin + batch_size] = torch.stack(theta_rel_temp, dim=-1)

        b_begin += batch_size

    if entity_attention:
        logging.debug(f"Entity sizes:\t-Cands: {cands_ent.size()}\t-Theta: {theta_ent.size()}")
    if relation_name_attention:
        logging.debug(f"Relation name sizes:\t-Cands: {cands_rel.size()}\t-Theta: {theta_rel.size()}")

    return cands_ent, cands_rel, theta_ent, theta_rel


def calculate_and_apply_unified_embedding(general_embedding_ent, general_embedding_rel, embedding_models,
                                          cands_att_dict, batch_size=500, entity_attention=True,
                                          relation_name_attention=True):
    # calculate unified embedding from alphas
    logging.info(f"Calculating unified embedding for entities and relation names.")
    logging.debug(f"Size of general entity embedding, before applying attention:\t"
                  f"{general_embedding_ent.weight.data.size()}")
    logging.debug(f"Size of general relation name embedding, before applying attention:\t"
                  f"{general_embedding_rel.weight.data.size()}")

    b_begin = 0
    while b_begin < general_embedding_ent.weight.data.size()[0]:
        if entity_attention:
            general_embedding_ent.weight.data[b_begin:b_begin + batch_size] = torch.sum(
                cands_att_dict['att_weights_ent'][b_begin:b_begin + batch_size] * cands_att_dict['cands_ent'][
                                                                                  b_begin:b_begin + batch_size], dim=-1)
        if relation_name_attention:
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
        if entity_attention:
            embedding_model['model'].entity.weight.data[embedding_model['entities']] = (
                general_embedding_ent.weight.data[embedding_model['entities']].to(torch.float))
        if relation_name_attention:
            embedding_model['model'].rel.weight.data[embedding_model['relation_names']] = (
                general_embedding_rel.weight.data[embedding_model['relation_names']].to(torch.float))
