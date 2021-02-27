from utils import NN, load_yaml
import matplotlib
import gym
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import copy
import numpy as np
import datetime
from torch.distributions import Categorical
from a2cAgent import *

if __name__ == "__main__":
    # config = load_yaml('./configs/config_random_gridworld.yaml')
    config = load_yaml("./configs/config_random_cartpole.yaml")
    # config = load_yaml('./configs/config_random_lunar.yaml')

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    print(config["env"])

    env = gym.make(config["env"])
    if hasattr(env, "setPlan"):
        env.setPlan(config["map"], config["rewards"])

    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    episode_count = config["nbEpisodes"]

    V_LR = 2e-4
    PI_LR = 2e-4

    pi_layers = [
        256,
    ]
    v_layers = [
        256,
    ]

    N_SAMPLES = 10

    writer_str = (
        "CARTPOLE/"
        "A2C" + "-"
        "TD_0"
        + "-"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        + "-"
        + str(PI_LR)
        + "-"
        + str(pi_layers)
        + "-"
        + str(V_LR)
        + "-"
        + str(v_layers)
        + "-"
        + str(N_SAMPLES)
    )
    writer = SummaryWriter(writer_str)
    GAMMA = 0.98
    agent = A2CAgent(env, config, GAMMA, V_LR, PI_LR, 4, pi_layers, v_layers)
    TEST_EVERY = 20
    CHANGE_EVERY = 500
    N_EPOCHS = 10000
    main(env, agent, N_EPOCHS, N_SAMPLES, TEST_EVERY, CHANGE_EVERY, writer)
