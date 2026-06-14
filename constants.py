USE_CHECKPOINT_FOR_TRAINING = False
DEBUG = False

# Outputs
MODEL_PATH = "ppo_crawl"
REPLAY_OUTPUT_DIR = "output_replays"
CHECKPOINT_DIR = "checkpoints"
TENSORBOARD_LOG_DIR = "logs/tensorboard"
EVAL_REPLAY_DIR = "logs/eval_replays"

# Training
N_TRAINING_SUBPROC_ENVIRONMENTS = 20
EVAL_EVERY_N_STEPS = 100_000

# - - - Hyperparamaters  - - -

# When True, use Noah's wall-aware reward; when False, use the height/win-margin reward.
USE_NOAHS_REWARD_FUNC = False
N_RESIDUAL_BLOCKS = 4

# - - - Actions - - -
FACTORY_MAPPING = {
    0: "IDLE",
    1: "NORTH",
    2: "EAST",
    3: "SOUTH",
    4: "WEST",
    5: "BUILD_SCOUT",
    6: "BUILD_WORKER",
    7: "BUILD_MINER",
    8: "JUMP_NORTH",
    9: "JUMP_EAST",
    10: "JUMP_SOUTH",
    11: "JUMP_WEST",
}

SCOUT_MAPPING = {
    0: "IDLE",
    1: "NORTH",
    2: "EAST",
    3: "SOUTH",
    4: "WEST",
}

WORKER_MAPPING = {
    0: "IDLE",
    1: "NORTH",
    2: "EAST",
    3: "SOUTH",
    4: "WEST",
    5: "BUILD_NORTH",
    6: "BUILD_EAST",
    7: "BUILD_SOUTH",
    8: "BUILD_WEST",
    9: "REMOVE_NORTH",
    10: "REMOVE_EAST",
    11: "REMOVE_SOUTH",
    12: "REMOVE_WEST",
}

MINER_MAPPING = {
    0: "IDLE",
    1: "NORTH",
    2: "EAST",
    3: "SOUTH",
    4: "WEST",
    5: "TRANSFORM",
}
