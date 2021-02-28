import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
from typing import List

from valueIteration import plot_statistics, plot_convergence, ValueIterationAgent, value_iteration, train, run


def train_xp(map_n: int, tol: float, gamma: float, n_test: int, verbose: bool, seed: int):
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan" + str(int(map_n)) + ".txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    env.seed(seed)
    statedic, mdp = env.getMDP()

    FPS = 0.0001

    agent = ValueIterationAgent(env.action_space, statedic, mdp, gamma)
    agent, results = train(agent, env, gamma=gamma, tol=tol, test_every=2, n_test=100, verbose_test=verbose, fps=FPS)
    m, std = run(agent, 1.0, env, 1000, False,  FPS)
    if verbose:
        print("Final expected cumulative reward")
        print(m)
        print(std)
        plot_convergence(results["convergence"])
        plot_statistics(results)

    return agent, results, m, std

def gamma_effect(map_n: int, tol: float, gammas: List[float], n_test: int, verbose: bool, seed: int):
    cmap = plt.get_cmap("jet_r")
    results = []
    for gamma in gammas:
        agent, res, test_mean, test_std = train_xp(map_n, tol, gamma, n_test, verbose, seed)
        res["gamma"] = gamma
        results.append(res)
        print(fr"{gamma}: {test_mean} $\pm$ {test_std:.5f}")
    return results

def plot_gamma_effect(results: dict, qty: str):
    N = len(results)
    cmap = plt.get_cmap("viridis")
    for i, exp in enumerate(results):
        color = cmap(float(i) / N)
        if qty == "convergence":
            plt.plot(exp[qty], label=str(exp["gamma"]), c=color)
            plt.yscale("log")
            plt.xlabel("Iterations")
            plt.title("Value iteration" + qty)

        elif qty == "mean_std":
            plt.plot(exp["iterations"], exp["test_mean"], label=str(exp["gamma"]), c=color)
            plt.plot(exp["iterations"], exp["test_mean"] + exp["test_std"], linestyle=":", c=color)
            plt.plot(exp["iterations"], exp["test_mean"] - exp["test_std"], linestyle=":", c=color)
            plt.xlabel("Iterations")
            plt.title(r"Value iteration - Test mean $\pm$ std")
        else:
            plt.plot(exp["iterations"], exp[qty], label=str(exp["gamma"]), c=color)
    plt.legend(title=r"$\gamma$")
    plt.show()



if __name__ == "__main__":
    map_n = 5
    gamma = 0.99
    tol = 1e-2
    gamma = 0.99
    seed = 0
    verbose = True
    n_test = 1000
    """
    agent, results, m, std = train_xp(map_n, tol, gamma, n_test, verbose, seed)
    """

    """
    gammas = [0.5, 0.9, 0.99, 0.999]
    r = gamma_effect(map_n, tol, gammas, n_test, verbose, seed)
    plot_gamma_effect(r, "convergence")
    plot_gamma_effect(r, "mean_std")
    """

    for map_n in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        print(map_n)
        agent, exp, m, std = train_xp(map_n, tol, gamma, n_test, False, seed)
        plt.plot(exp["iterations"], exp["test_mean"], color="blue")
        plt.plot(exp["iterations"], exp["test_mean"] + exp["test_std"], linestyle=":", color="blue")
        plt.plot(exp["iterations"], exp["test_mean"] - exp["test_std"], linestyle=":", color="blue")
        plt.savefig(f"{map_n}.png", dpi=400)
        plt.close()
