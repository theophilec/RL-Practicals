import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
from scipy.special import softmax
import copy
from tqdm import tqdm
from collections import defaultdict


class BaseAgent(object):
    """Base for Agents."""

    def __init__(self, strategy, gamma, alpha):
        self.strategy = strategy
        self.qsa = defaultdict(lambda: np.zeros(4))
        self.alpha = alpha
        self.gamma = gamma
        # rp is useful for Dyna-Q
        # self.rp[s_t][a_t] = np.array([R(s_t, a_t, s_t+1), P(s_t+1|s_t, a_t)])

        self.rp = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: np.zeros(2)))
        )
        self.alpha_r = alpha

    def act(self, observation, reward, done):
        """
        observation = s_t
        reward = r_{t-1}
        done = s_t in final

        Step1 : choose action (with strategy)
        Step2 : learn
        """
        raise NotImplementedError


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose(self, qsa, obs):
        """Select action using Q-table from state obs.

        Parameters:
            qsa: defaultdict (qsa[s][a] is Q(s, a))
            s: state (ndarray)
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.arange(4))
        else:
            return np.argmax(qsa[obs])

    def update(self, s_m, s_p, action, reward):
        """Update strategy parameters.

        For EpsilonGreedy, do nothing.

        Parameters:
            s_m: previous state (ndarray)
            s_p: next state (ndarray)
            action: chosen action (int)
            reward: reward observed going from s_m to s_p with a (float)
        """
        pass


class BoltzmannSelection:
    def __init__(self, tau):
        self.tau = tau

    def choose(self, qsa, obs):
        """Select action using Q-table from state s.

        Choose action according to Boltzmann distribution with temperature tau.
        The distribution is defined as the softmax of qsa[s] / tau.

        Parameters:
            qsa: defaultdict (qsa[s][a] is Q(s, a))
            s: state (ndarray)
        """
        return np.random.choice(np.arange(4), p=softmax(qsa[obs] / self.tau))

    def update(self, s_m, s_p, action, reward):
        pass


class QLearningAgent(BaseAgent):
    """Off-policy TD control agent.

    Uses Q-derived strategy in self.strategy to choose actions but not sample for TD updates.
    """

    def act(self, observation, reward, done):
        # Choose action
        action = self.strategy.choose(self.qsa, observation)
        return action

    def learn(self, p_s, n_s, action, reward):
        """Update Q-table and strategy (e.g. for UCB1).

        prev_state = state of origin
        next_state = next state
        action = chosen action
        reward = received reward
        """
        self.qsa[p_s][action] = self.qsa[p_s][action] + self.alpha * (
            reward + self.gamma * np.max(self.qsa[n_s]) - self.qsa[p_s][action]
        )

        # Update metrics for strategy object (for UCB1)
        self.strategy.update(p_s, n_s, action, reward)

    def run(self, env, episode_count, learn=True):
        hist = np.zeros(episode_count)
        for i in tqdm(range(episode_count)):
            rsum = 0
            done = False
            t = 0
            obs = env.reset()
            obs = gridworld.gridworld_env.GridworldEnv.state2str(obs)
            while not done:
                action = self.act(obs, 0, done)  # TODO: replace 0
                obs_next, reward, done, _ = env.step(action)
                obs_next = gridworld.gridworld_env.GridworldEnv.state2str(obs_next)
                if learn:
                    self.learn(obs, obs_next, action, reward)
                obs = obs_next
                rsum += self.gamma ** t * reward
                t += 1
                if done:
                    break
            hist[i] = rsum
        return hist


class SARSAAgent(BaseAgent):
    """On-policy TD control agent.

    Uses Q-derived strategy in self.strategy to choose actions and sample for TD updates.
    """

    def act(self, observation, reward, done):
        # Choose action
        action = self.strategy.choose(self.qsa, observation)
        return action

    def learn(self, p_s, n_s, action, next_action, reward):
        """Update Q-table and strategy (e.g. for UCB1).

        prev_state = state of origin
        next_state = next state
        action = chosen action
        reward = received reward
        """
        self.qsa[p_s][action] = self.qsa[p_s][action] + self.alpha * (
            reward + self.gamma * self.qsa[n_s][next_action] - self.qsa[p_s][action]
        )

        # Update metrics for strategy object (for UCB1)
        self.strategy.update(p_s, n_s, action, reward)

    def run(self, env, episode_count, learn=True):
        hist = np.zeros(episode_count)
        for i in tqdm(range(episode_count)):
            rsum = 0
            done = False
            t = 0
            obs = env.reset()
            obs = gridworld.gridworld_env.GridworldEnv.state2str(obs)
            while not done:
                action = self.act(obs, 0, done)  # TODO: replace 0
                obs_next, reward, done, _ = env.step(action)
                obs_next = gridworld.gridworld_env.GridworldEnv.state2str(obs_next)
                if learn:
                    action_next = self.act(obs_next, 0, done)  # TODO: replace 0
                    self.learn(obs, obs_next, action, action_next, reward)
                obs = obs_next
                rsum += self.gamma ** t * reward
                t += 1
                if done:
                    break
            hist[i] = rsum
        return hist


class DynaQAgent(BaseAgent):
    """Off-policy TD control and planning agent.

    Uses Q-derived strategy in self.strategy to choose actions but not sample for TD updates.
    """

    def act(self, observation, reward, done):
        # Choose action
        action = self.strategy.choose(self.qsa, observation)
        return action

    def learn(self, p_s, n_s, action, reward):
        """Update Q-table and strategy (e.g. for UCB1).

        prev_state = state of origin
        next_state = next state
        action = chosen action
        reward = received reward
        """
        # update Q
        self.qsa[p_s][action] = self.qsa[p_s][action] + self.alpha * (
            reward + self.gamma * np.max(self.qsa[n_s]) - self.qsa[p_s][action]
        )
        # update R
        self.rp[p_s][action][n_s][0] = (1 - self.alpha_r) * self.rp[p_s][action][n_s][0] + self.alpha_r * reward
        # update P
        self.rp[p_s][action][n_s][1] = (1 - self.alpha_r) * self.rp[p_s][action][n_s][1] + self.alpha_r
        for k in self.rp[p_s][action].keys():
            if k != n_s:
                self.rp[p_s][action][k] = (1 - self.alpha_r) * self.rp[p_s][action][k]

        # Update metrics for strategy object (for UCB1)
        self.strategy.update(p_s, n_s, action, reward)

    def learn_from_sample(self, transitions):
        for p_s, action, n_s in transitions:
            w = sum(
                [
                    self.rp[p_s][action][n_s][1] * (self.rp[p_s][action][n_s][0] + self.gamma * float(np.max(self.qsa[n_s]))) for n_s in self.rp[p_s][action].keys()
                ]
            )
            if isinstance(w, np.ndarray):
                breakpoint()

            try:
                self.qsa[p_s][action] =  self.qsa[p_s][action] + self.alpha * (w - self.qsa[p_s][action])
            except ValueError:
                print(self.rp[p_s][action].keys())
                print(w)



    def sample(self, env, n_transitions):
        transitions = []
        while True:
            done = False
            obs = env.reset()
            obs = gridworld.gridworld_env.GridworldEnv.state2str(obs)
            while not done:
                action = self.act(obs, 0, done)  # TODO: replace 0
                obs_next, reward, done, _ = env.step(action)
                obs_next = gridworld.gridworld_env.GridworldEnv.state2str(obs_next)
                transitions.append((obs, action, obs_next))
                obs = obs_next
                if len(transitions) >= n_transitions:
                    return transitions
                if done:
                    break


    def run(self, env, episode_count, learn=True):
        hist = np.zeros(episode_count)
        for i in tqdm(range(episode_count)):
            rsum = 0
            done = False
            t = 0
            obs = env.reset()
            obs = gridworld.gridworld_env.GridworldEnv.state2str(obs)
            while not done:
                action = self.act(obs, 0, done)  # TODO: replace 0
                obs_next, reward, done, _ = env.step(action)
                obs_next = gridworld.gridworld_env.GridworldEnv.state2str(obs_next)
                if learn:
                    n_transitions = 10
                    self.learn(obs, obs_next, action, reward)
                    env_copy = copy.deepcopy(env)
                    transitions = self.sample(env_copy, n_transitions)
                    self.learn_from_sample(transitions)
                obs = obs_next
                rsum += self.gamma ** t * reward
                t += 1
                if done:
                    break
            hist[i] = rsum
        return hist


if __name__ == "__main__":

    verbose = 0
    env = gym.make("gridworld-v0")
    map_n = 1
    env.setPlan(
        "gridworldPlans/plan" + str(int(map_n)) + ".txt",
        {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1},
    )

    env.seed(5)  # Initialise le seed du pseudo-random

    FPS = 0.0001

    training_episodes = 600
    test_episodes = 400

    EPS = 0.1
    gamma = 0.99
    alpha = 0.01

    tau = 0.0001

    strategy = "boltzmann"
    if strategy == "eps":
        sarsa_strategy = EpsilonGreedy(EPS)
        ql_strategy = EpsilonGreedy(EPS)
        dyna_q_strategy = EpsilonGreedy(EPS)
    elif strategy == "boltzmann":
        sarsa_strategy = BoltzmannSelection(tau)
        ql_strategy = BoltzmannSelection(tau)
        dyna_q_strategy = BoltzmannSelection(tau)


    # Run SARSA
    sarsa_agent = SARSAAgent(sarsa_strategy, gamma, alpha)
    sarsa_h = sarsa_agent.run(env, training_episodes)
    sarsa_test = sarsa_agent.run(env, test_episodes, learn=False)

    # Run Q-Learning
    ql_agent = QLearningAgent(ql_strategy, gamma, alpha)
    ql_h = ql_agent.run(env, training_episodes)
    ql_test = ql_agent.run(env, test_episodes, learn=False)

    # Run Dyna-Q
    dyna_q_agent = DynaQAgent(dyna_q_strategy, gamma, alpha)
    dyna_q_h = dyna_q_agent.run(env, training_episodes)
    dyna_q_test = dyna_q_agent.run(env, test_episodes, learn=False)

    # Plot learning curves
    plt.plot(np.cumsum(sarsa_h) / np.arange(1, training_episodes + 1), label="SARSA")
    plt.plot(np.cumsum(ql_h) / np.arange(1, training_episodes + 1), label="Q-Learning")
    plt.plot(np.cumsum(dyna_q_h) / np.arange(1, training_episodes + 1), label="dyna-Q")
    print(f"SARSA test: {np.mean(sarsa_test)} +/- {np.std(sarsa_test)}")
    print(f"Q-Learning test: {np.mean(ql_test)} +/- {np.std(ql_test)}")
    print(f"DyanQ: {np.mean(dyna_q_test)} +/- {np.std(dyna_q_test)}")
    plt.legend()
    plt.savefig(f"figures/{map_n}-learning-{strategy}-{alpha}-{EPS}-{tau}.png", dpi=150)
    plt.show()

    plt.boxplot([sarsa_test, ql_test, dyna_q_test], labels=["SARSA", "Q-Learning", "Dyna-Q"])
    plt.savefig(f"figures/{map_n}-test-{strategy}-{alpha}-{EPS}-{tau}.png", dpi=150)
    plt.show()
