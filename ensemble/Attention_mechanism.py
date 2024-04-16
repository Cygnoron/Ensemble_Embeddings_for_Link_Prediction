import logging

import torch
from torch import nn

from ensemble import Constants


def calculate_self_attention(embedding_models):
    activation = nn.Softmax(dim=0)

    for embedding_model in embedding_models:
        args = embedding_model["args"]
        model = embedding_model["model"]

        logging.info(f"Calculating self-attention for {args.subgraph} ({args.model_name})")

        scale = model.scale

        embedding_ent = model.entity.weight.data
        contex_vec_ent = model.theta_ent.weight.data

        embedding_rel = model.rel.weight.data
        contex_vec_rel = model.theta_rel.weight.data

        embedding_model["att_weights_ent"] = torch.sum(contex_vec_ent * embedding_ent * scale, dim=-1, keepdim=True)
        embedding_model["att_weights_rel"] = torch.sum(contex_vec_rel * embedding_rel * scale, dim=-1, keepdim=True)

        logging.debug(f"Sum entity before: {torch.sum(embedding_model['att_weights_ent'])}")
        logging.log(Constants.DATA_LEVEL, f"att_weights_ent before softmax:\n{embedding_model['att_weights_ent']}")

        embedding_model["att_weights_ent"] = activation(embedding_model["att_weights_ent"])

        logging.debug(f"Sum entity after: {torch.sum(embedding_model['att_weights_ent'])}")
        logging.log(Constants.DATA_LEVEL, f"att_weights_ent after softmax:\n{embedding_model['att_weights_ent']}")

        logging.debug(f"Sum rel before: {torch.sum(embedding_model['att_weights_rel'])}")
        logging.log(Constants.DATA_LEVEL, f"att_weights_rel before softmax:\n{embedding_model['att_weights_rel']}")

        embedding_model["att_weights_rel"] = activation(embedding_model["att_weights_rel"])

        logging.debug(f"Sum rel after: {torch.sum(embedding_model['att_weights_rel'])}")
        logging.log(Constants.DATA_LEVEL, f"att_weights_rel after softmax:\n{embedding_model['att_weights_rel']}")

    # collected_attention_ent = embedding_models[0]['att_weights_ent']
    # collected_attention_rel = embedding_models[0]['att_weights_rel']
    #
    # for index, embedding_model in enumerate(embedding_models):
    #     if index == 0:
    #         continue
    #     torch.cat((collected_attention_ent, embedding_model['att_weights_ent']),
    #               dim=0, out=collected_attention_ent)
    #     torch.cat((collected_attention_rel, embedding_model['att_weights_rel']),
    #               dim=0, out=collected_attention_rel)
    #
    #     collected_attention_ent = activation(collected_attention_ent)
    #     collected_attention_rel = activation(collected_attention_rel)

    return


def calculate_and_apply_unified_embedding(general_embedding_ent, general_embedding_rel, embedding_models):
    # calculate unified embedding from alphas
    logging.info(f"Calculating unified embedding for entities and relation names.")
    logging.debug(f"Size of general entity embedding, before applying attention:\t"
                  f"{general_embedding_ent.weight.data.size()}")
    logging.debug(f"Size of general relation name embedding, before applying attention:\t"
                  f"{general_embedding_rel.weight.data.size()}")

    general_embedding_ent.weight.data = (embedding_models[0]["att_weights_ent"] *
                                         embedding_models[0]["model"].entity.weight.data)
    general_embedding_rel.weight.data = (embedding_models[0]["att_weights_rel"] *
                                         embedding_models[0]["model"].rel.weight.data)

    for index, embedding_model in enumerate(embedding_models):
        if index == 0:
            continue
        torch.add(input=general_embedding_ent.weight.data, out=general_embedding_ent.weight.data,
                  other=embedding_model["att_weights_ent"] * embedding_model["model"].entity.weight.data)
        torch.add(input=general_embedding_rel.weight.data, out=general_embedding_rel.weight.data,
                  other=embedding_model["att_weights_rel"] * embedding_model["model"].rel.weight.data)

    logging.debug(f"Size of general entity embedding, after applying attention:\t"
                  f"{general_embedding_ent.weight.data.size()}")
    logging.debug(f"Size of general relation name embedding, after applying attention:\t"
                  f"{general_embedding_rel.weight.data.size()}")

    # write new unified embedding into all models
    for embedding_model in embedding_models:
        embedding_model['model'].entity.weight.data = general_embedding_ent.weight.data
        embedding_model['model'].rel.weight.data = general_embedding_rel.weight.data

    return general_embedding_ent, general_embedding_rel
