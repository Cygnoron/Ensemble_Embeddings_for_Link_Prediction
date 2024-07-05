# sampling methods
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

# theta calculation methods
NO_THETA = [20, "no theta", "no_theta"]
REGULAR_THETA = [21, "regular theta", "regular_theta"]
REVERSED_THETA = [22, "reversed theta", "reversed_theta"]
RELATION_THETA = [23, "relation theta", "relation_theta"]
MULTIPLIED_THETA = [24, "multiplied theta", "multiplied_theta"]
THETA_METHODS = [NO_THETA, REGULAR_THETA, REVERSED_THETA, RELATION_THETA, MULTIPLIED_THETA]

# Wandb settings
LOG_WANDB = False
PROJECT_NAME = "Experiments"


def get_wandb(project_name=False):
    global LOG_WANDB, PROJECT_NAME

    if isinstance(project_name, str) and project_name.lower() != "false":
        LOG_WANDB = True
        PROJECT_NAME = project_name
