import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
import gym
import gridworld
import torch
from utils import *
from torch.utils.tensorboard import SummaryWriter

from behavior_clone import BehaviorClone
from gail import GAILAgent

if __name__ == '__main__':
    config = load_yaml('./configs/config_gail_lunar.yaml')

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]    

    env = gym.make(config["env"])
    if hasattr(env, 'setPlan'):
        env.setPlan(config["map"], config["rewards"])

    tstart = str(time.time())
    tstart = tstart.replace(".", "_")

    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    episode_count = config["nbEpisodes"]
    ob = env.reset()

    # agent = BehaviorClone(env, config)
    agent = GAILAgent(env, config)
    
    outdir = "./XP/" + config["env"] + "/" + agent.name + "_" + "-" + tstart
    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    logger = LogMe(SummaryWriter(outdir))
    loadTensorBoard(outdir)
    agent.writer = logger.writer

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    truncated = False
    for i in range(episode_count):
        verbose = False
        if i % freqTest == 0 and i >= freqTest:
            print("Test time! ")
            mean = 0
            agent.test = True
            verbose = True

        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        j = 0
        if verbose:
            env.render()

        while True:
            if verbose:
                env.render()

            action = agent.act(ob, reward, done, truncated)
            ob, reward, done, info = env.step(action)
            truncated = info.get("TimeLimit.truncated", False)
            j+=1

            rsum += reward
            if done:
                # Act one last time on the last reward
                agent.act(ob, reward, done, truncated)
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                # Complete reset for new episode
                rsum = 0
                ob = env.reset()
                reward = 0
                done = False
                break

    env.close()

