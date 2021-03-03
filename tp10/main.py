import copy
import os
import time
import matplotlib
import pickle

matplotlib.use("TkAgg")
import gym
import multiagent
import multiagent.scenarios
import multiagent.scenarios.simple_tag as simple_tag
import multiagent.scenarios.simple_tag as simple_spread
import multiagent.scenarios.simple_tag as simple_adversary
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from gym import wrappers, logger
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from maddpg import MADDPG
from utils import *

"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    world.dim_c = 0
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_space = False
    env.discrete_action_input = False
    scenario.reset_world(world)
    return env, scenario, world

if __name__ == '__main__':
    config = load_yaml('./configs/config_adversary.yaml')
    # config = load_yaml('./configs/config_cooperative.yaml')
    env, scenario, world = make_env(config["env"])

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]

    tstart = str(time.time())
    tstart = tstart.replace(".", "_")

    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    episode_count = config["nbEpisodes"]
    maxLengthTest = config["maxLengthTest"]
    maxLengthTrain = config["maxLengthTrain"] 

    # Init agent
    o = env.reset()
    agent = MADDPG(env, config)

    outdir = "./XP/" + config["env"] + "/" + agent.name + "_" + "-" + tstart
    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    logger = LogMe(SummaryWriter(outdir))
    loadTensorBoard(outdir, 8008)
    agent.writer = logger.writer

    rsum = [0, 0, 0]
    mean = [0, 0, 0]
    itest = 0
    r = [0, 0, 0]
    total_r = 0
    done = [False, False, False]
    truncated = False
    verbose = False
    for i in range(episode_count):

        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = [0, 0, 0]
            agent.test = True
            verbose = True

        if i % freqTest == nbTest and i > freqTest:
            mean = [m / nbTest for m in mean]
            print("End of test, mean reward=", mean)
            for k in range(env.n):
                itest += 1
                logger.direct_write(f"Agent_{k+1}/RewardTest", mean[k], itest)
                agent.test = False
                verbose = False
        
        if i % freqSave == 0:
            print("Saving agent")
            agent.save(f"checkpoints/{config['env']}_{agent.name}.pt")
        
        j = 0
        while True:
            a = agent.act(o, r, done, truncated)
            o, r, done, info = env.step(a)
            rsum = [rsum[k] + r[k] for k in range(env.n)]
            total_r += sum(r)
            truncated = info.get("TimeLimit.truncated", False)
            j += 1

            if verbose:
                env.render(mode="none")

            if agent.test:
                stop = j >= maxLengthTest
            else:
                stop = j >= maxLengthTrain

            if sum([int(d) for d in done]) == len(done) or (stop):
                # Act one last time on the last reward
                agent.act(o, r, done, truncated)
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                if not agent.test:
                    for k in range(env.n):
                        logger.direct_write(f"Agent_{k+1}/Reward", rsum[k], i)
                    logger.direct_write(f"All/Reward", total_r, i)
                mean = [mean[k] + rsum[k] for k in range(env.n)]
                # Complete reset for new episode
                rsum = [0, 0, 0]
                o = env.reset()
                r = [0, 0, 0]
                total_r = 0
                done = [False, False, False]
                break

    env.close()