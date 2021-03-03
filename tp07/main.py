import argparse
import sys
import matplotlib
matplotlib.use("TkAgg")
import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from ddpg import DDPG
from utils import *


def run_experiment(env, agent, seed, logger, episode_count=1e6, freqTest=100, nbTest=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    ob = env.reset()
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
                mean += rsum
                
                # Reset for new episode
                rsum = 0
                ob = env.reset()
                reward = 0
                done = False
                break

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs experiment')
    parser.add_argument('env', type=str, choices = ["lunar", "mountaincar", "pendulum"],
                        help='Name of the environment.')
    parser.add_argument('--agent', type=str, help='Name of RL Agent', choices=["ddpg", "qprop"])
    parser.add_argument('--avg', default=5, type=int, help='Number of runs to compute the average on.')
    args = parser.parse_args()

    config = load_yaml(f'configs/config_{args.env}.yaml') 
    env = gym.make(config["env"])
    if hasattr(env, 'setPlan'):
        env.setPlan(config["map"], config["rewards"])
    agent = DDPG(env, config) if args.agent == "ddpg" else None

    for seed in np.arange(args.avg):
        tstart = str(time.time())
        tstart = tstart.replace(".", "_")
        outdir = "./XP/" + config["env"] + "/" + agent.name + "_seed" + str(seed) + "-" + tstart
        print("Saving in " + outdir)
        os.makedirs(outdir, exist_ok=True)
        save_src(os.path.abspath(outdir))
        write_yaml(os.path.join(outdir, 'info.yaml'), config)
        logger = LogMe(SummaryWriter(outdir))
        loadTensorBoard(outdir, 8008)
        agent.writer = logger.writer

        run_experiment(
            env=env,
            agent=agent,
            seed=seed,
            logger=logger,
            episode_count=config["nbEpisodes"],
            freqTest=config["freqTest"],
            nbTest=config["nbTest"]
        )

    