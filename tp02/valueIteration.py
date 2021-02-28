import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class ValueIterationAgent(object):
    def __init__(self, action_space, statedic, mdp, gamma):
        self.action_space = action_space
        self.values = np.zeros(len(statedic))
        self.statedic = statedic
        self.mdp = mdp
        self.gamma = gamma
        # note: all states are not represented in the mdp

    def act(self, obs, reward, done):
        """Select action to maximize expected cumulative reward.

        Parameters:
            obs: str
            reward: float
            done: bool
        """
        return np.argmax(
            [
                sum(
                    (r + self.gamma * self.values[self.statedic[s_prime]]) * p
                    for p, s_prime, r, _ in self.mdp[obs][a]
                )
                for a in self.mdp[obs].keys()
            ]
        )


def value_iteration(mdp, statedic, values, gamma):
    """Update value estimation by applying the Bellman equation.

    For all states s, choose value of action with best expected value.
    """
    for s in mdp.keys():
        # for all states, represented as strings here
        values[statedic[s]] = max(
            [
                sum(
                    [
                        (r + gamma * values[statedic[s_prime]]) * p
                        for p, s_prime, r, _ in mdp[s][a]
                    ]
                )
                for a in mdp[s].keys()
            ]
        )
    return values


def train(agent, env, gamma=0.99, tol=1e-4, test_every=-1, n_test=1000, verbose_test=False, fps=0):
    """Learn state values by value iteration.

    While the infinity norm between successive estimates of
    the value function is less than tol, keep going.

    """
    it = 0
    cont = True

    convergence = []
    test_mean_hist = []
    test_std_hist = []
    test_it = []
    while cont:
        if test_every != -1 and it % test_every == 0 and it > 0:
            m, std = run(agent, 1.0, env, n_test, verbose_test, fps)
            test_mean_hist.append(m)
            test_std_hist.append(std)
            test_it.append(it)
        oldvalues = copy.deepcopy(agent.values)
        agent.values = value_iteration(agent.mdp, agent.statedic, agent.values, gamma)
        diff = np.max(np.abs(agent.values - oldvalues))
        cont = diff > tol
        convergence.append(diff)
        it += 1
    statistics = {
        "iterations": test_it,
        "convergence": np.array(convergence),
        "test_mean": np.array(test_mean_hist),
        "test_std": np.array(test_std_hist),
    }
    return agent, statistics


def run(agent, gamma, env, n_episodes, verbose, fps):
    """Run the agent on the env for n_episodes."""
    rsums = []
    for i in range(n_episodes):
        obs = env.reset()
        env.verbose = i % 100 == 0 and i > 0 and verbose
        if env.verbose:
            env.render(fps)
        j = 0
        rsum = 0
        reward = 0
        done = 0
        while True:
            obs = gridworld.gridworld_env.GridworldEnv.state2str(obs)
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            rsum += gamma ** j * reward
            j += 1
            if env.verbose:
                env.render(fps)
            if done:
                break
        rsums.append(rsum)
    return np.mean(rsums), np.std(rsums)


def plot_convergence(diff_array):
    plt.close()
    plt.plot(diff_array)
    plt.xlabel("Bellman iterations")
    plt.ylabel(r"$\Vert V_{k+1} -V_k \Vert_\infty$")
    plt.yscale("log")
    plt.title("Bellman Value iteration convergence")
    plt.show()


def plot_statistics(results):
    plt.close()
    plt.plot(results["iterations"], results["test_mean"])
    plt.plot(
        results["iterations"],
        results["test_mean"] + results["test_std"],
        linestyle=":",
        color="blue",
    )
    plt.plot(
        results["iterations"],
        results["test_mean"] - results["test_std"],
        linestyle=":",
        color="blue",
    )
    plt.show()


