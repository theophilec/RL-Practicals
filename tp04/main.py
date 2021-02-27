import matplotlib
import copy
import datetime

import gym
import matplotlib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dqnAgent import DQNAgent
from utils import *

if __name__ == "__main__":
    config = load_yaml("./configs/config_dqn_cartpole.yaml")

    ENV_NAME = config["env"]

    env = gym.make(config["env"])
    if hasattr(env, "setPlan"):
        env.setPlan(config["map"], config["rewards"])

    tstart = str(time.time())
    tstart = tstart.replace(".", "_")

    # Random seed
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    CHANGE_TARGET = config["change_target"]
    EPISODE_COUNT = config["nbEpisodes"]
    TEST_EVERY = config.freqTest
    N_TEST = config.nbTest

    agent = DQNAgent(env, config)

    outdir = "./XP/" + config["env"] + "/" + agent.name + "_" + "-" + tstart
    print("Saving configuration in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    logger = LogMe(SummaryWriter(outdir))
    # loadTensorBoard(outdir)
    agent.writer = logger.writer

    rsum = 0
    mean = 0
    verbose = False
    itest = 0
    reward = 0
    done = False

    cumreward = 0
    agent.global_id = 0
    for episode in range(EPISODE_COUNT):
        agent.episode = episode

        agent.test = False
        if (episode % TEST_EVERY) == 1:
            agent.test = True
            test_r = []
            for test_episode in range(N_TEST):
                rsum = 0
                with torch.no_grad():
                    ob = env.reset()
                    while True:
                        if verbose:
                            env.render()
                        action = agent.act(ob, reward, done)
                        ob, reward, done, info = env.step(action)
                        rsum += agent.gamma * reward
                        if done:
                            break
                test_r.append(rsum)
            agent.writer.add_scalar("Reward/test", np.mean(test_r), episode)

        ob = env.reset()
        step = 0
        rsum = 0
        agent.test = False
        while True:
            old_ob = copy.deepcopy(ob)
            action = agent.act(ob, reward, done)
            agent.global_id += 1

            # Change target
            if (agent.global_id % CHANGE_TARGET) == 1 or (CHANGE_TARGET == 1):
                print(f"change: {agent.global_id}")
                agent.q_target = copy.deepcopy(agent.q)

            ob, reward, done, info = env.step(action)
            truncated = info.get("TimeLimit.truncated", False)

            # Store transition
            agent.store(
                old_ob, action, copy.deepcopy(ob), reward, done and (not truncated)
            )

            step += 1

            rsum += agent.gamma * reward

            if agent.memory.is_full() and not agent.test:
                agent.learn()
            if done:
                cumreward += rsum
                agent.writer.add_scalar("Reward/train", rsum, episode)
                agent.writer.add_scalar(
                    "CumulativeReward/train", cumreward / (episode + 1), episode
                )
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                ob = env.reset()
                break

    env.close()
