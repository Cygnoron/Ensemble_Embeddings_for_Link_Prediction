# sampling methods
import logging

DEBUG_SAMPLING = [0, "Debug sampling", "Debug_sampling"]
ENTITY_SAMPLING = [1, "Entity sampling", "Entity_sampling"]
FEATURE_SAMPLING = [2, "Feature sampling", "Feature_sampling"]
SAMPLING_METHODS = [DEBUG_SAMPLING, ENTITY_SAMPLING, FEATURE_SAMPLING]

# score aggregation methods
MAX_SCORE_AGGREGATION = [10, "maximum score", "maximum_score", "max"]
AVERAGE_SCORE_AGGREGATION = [12, "average score", "average_score", "avg"]
ATTENTION_SCORE_AGGREGATION = [13, "attention score", "attention_score", "att"]
AGGREGATION_METHODS = [MAX_SCORE_AGGREGATION, AVERAGE_SCORE_AGGREGATION, ATTENTION_SCORE_AGGREGATION]

# embedding methods
COMPL_EX, ROTAT_E = "ComplEx", "RotatE"
TRANS_E, DIST_MULT, CP, MUR_E, ROT_E, REF_E, ATT_E = "TransE", "DistMult", "CP", "MurE", "RotE", "RefE", "AttE"
ROT_H, REF_H, ATT_H = "RotH", "RefH", "AttH"
EMBEDDING_METHODS = [COMPL_EX, ROTAT_E, TRANS_E, DIST_MULT, CP, MUR_E, ROT_E, REF_E, ATT_E, ROT_H, REF_H, ATT_H]

# logging levels
DATA_LEVEL_LOGGING = 5

# Wandb settings
LOG_WANDB = False
PROJECT_NAME = "Experiments"


def get_wandb(project_name):
    """
    If a project name is provided as a string, it will be used as the wandb project name.
    Otherwise, no wandb logging will take place.

    Args:
        project_name (str): The project name for the wandb project.
    """
    global LOG_WANDB, PROJECT_NAME

    if isinstance(project_name, str):
        if project_name.lower() == "false":
            LOG_WANDB = False
        else:
            LOG_WANDB = True
            PROJECT_NAME = project_name
