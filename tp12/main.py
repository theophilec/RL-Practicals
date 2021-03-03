import argparse
import sys
import matplotlib
matplotlib.use("TkAgg")
import gym
import gridworld
import torch
from utils import *
from torch.utils.tensorboard import SummaryWriter


from dqn import DoubleDQNAgent
from curriculum import HER, IGS

if __name__ == '__main__':
    # config = load_yaml('./configs/config_gridworld.yaml')
    config = load_yaml('./configs/config_plan3.yaml')
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    episode_length = config["episodeLength"]
    

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

    # agent = DoubleDQNAgent(env, config)
    # agent = HER(DoubleDQNAgent(env, config))
    agent = IGS(DoubleDQNAgent(env, config))

    outdir = "./XP/" + config["env"] + "/" + agent.name + "_" + "-" + tstart
    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    logger = LogMe(SummaryWriter(outdir))
    loadTensorBoard(outdir)
    agent.add_writer(logger.writer)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    truncated = False
    ob = agent.get_features(ob)
    for i in range(episode_count):
        verbose = False

        if i % freqTest == 0:
            print("Test time! ")
            mean = 0
            agent.test = True

        if i % freqTest == nbTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            # verbose = True
            agent.test = False

        j = 0
        if verbose:
            env.render()

        # Sample goal:
        goal, _ = env.sampleGoal()
        goal = agent.get_features(goal)

        while True:
            if verbose:
                env.render()

            action = agent.act(ob, reward, done, truncated, goal)
            ob, reward, done, info = env.step(action)
            ob = agent.get_features(ob)
            truncated = info.get("TimeLimit.truncated", False)
            j+=1
            if j == episode_length:
                truncated = True

            # Curriculum learning: overwrite done and reward
            done = (ob == goal).all()
            reward = 1.0 if done else -0.1
            done = done | truncated

            rsum += reward
            if done:
                # Last state reached:
                x, y = ob
                logger.direct_write("x", x, i)
                logger.direct_write("y", y, i)
                # Act one last time on the last reward
                agent.act(ob, reward, done, truncated, goal)
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                ob = env.reset()
                ob = agent.get_features(ob)

                # Complete reset for new episode
                reward = 0
                done = False
                break

    env.close()

