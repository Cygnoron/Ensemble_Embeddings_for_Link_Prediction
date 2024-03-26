import logging

import torch


def calculate_self_attention(embedding_model):
    args = embedding_model["args"]
    model = embedding_model["model"]

    logging.info(f"Calculating self-attention for {args.subgraph} ({args.model_name})")

    scale = model.scale
    activation = model.act

    embedding_ent = model.entity.weight.data
    contex_vec_ent = model.theta_ent.weight.data

    embedding_rel = model.rel.weight.data
    contex_vec_rel = model.theta_rel.weight.data

    embedding_model["att_weights_ent"] = torch.sum(contex_vec_ent * embedding_ent * scale, dim=-1, keepdim=True)
    embedding_model["att_weights_rel"] = torch.sum(contex_vec_rel * embedding_rel * scale, dim=-1, keepdim=True)

    # TODO Change softmax? -> Needs to be over all embedding models

    logging.critical(f"att_weights_ent before softmax:\n{embedding_model['att_weights_ent']}")
    embedding_model["att_weights_ent"] = activation(embedding_model["att_weights_ent"])
    logging.critical(f"att_weights_ent after softmax:\n{embedding_model['att_weights_ent']}")

    logging.critical(f"att_weights_rel before softmax:\n{embedding_model['att_weights_rel']}")
    embedding_model["att_weights_rel"] = activation(embedding_model["att_weights_rel"])
    logging.critical(f"att_weights_rel after softmax:\n{embedding_model['att_weights_rel']}")
    return


def calculate_unified_embedding(embedding_models):
    # calculate unified embedding from alphas
    logging.info(f"Calculating unified embedding for entities and relation names.")
    unified_embedding_ent = embedding_models[0]["att_weights_ent"] * embedding_models[0]["model"].entity.weight.data
    unified_embedding_rel = embedding_models[0]["att_weights_rel"] * embedding_models[0]["model"].rel.weight.data
    for index, embedding_model in enumerate(embedding_models):
        # skip first, since it was used for initialization
        if index == 0:
            continue
        unified_embedding_ent = embedding_model["att_weights_ent"] * embedding_model["model"].entity.weight.data
        unified_embedding_rel = embedding_model["att_weights_rel"] * embedding_model["model"].rel.weight.data

    return unified_embedding_ent, unified_embedding_rel
