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
import torch.nn as nn

# matplotlib.use("Qt5agg")
# matplotlib.use("TkAgg")


class ClippedPPOAgent(object):
    def __init__(
        self, env, opt, gamma, v_lr, pi_lr, env_feature_size, pi_layers, v_layers
    ):
        self.name = "ClippedPPO"
        self.returns = "gae"
        self.lambd = 0.5
        self.eps = 1e-2
        self.opt = opt
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space

        self.gamma = gamma
        self.insize = env_feature_size

        self.outsize = self.action_space.n  # number of actions

        self.pi_lr = pi_lr
        self.pi_nn = NN(self.insize, self.outsize, pi_layers)
        self.pi_optim = torch.optim.Adam(params=self.pi_nn.parameters(), lr=self.pi_lr)

        self.V_lr = v_lr
        self.V_nn = NN(self.insize, 1, v_layers)
        self.V_optim = torch.optim.Adam(params=self.V_nn.parameters(), lr=self.V_lr)
        self.V_nn_target = copy.deepcopy(self.V_nn)

    def pi(self, s, single=False):
        if single:
            x = F.softmax(self.pi_nn.forward(s), dim=0)
        else:
            x = F.softmax(self.pi_nn.forward(s), dim=1)
        return x

    def V_nn_target(self, s):
        return self.V_nn_target.forward(s)

    def change_target(self):
        print("Value target changed")
        self.V_nn_target = copy.deepcopy(self.V_nn)

    def act(self, s, reward, done, compute_entropy=True, det=False):
        assert isinstance(s, np.ndarray)
        logits = self.pi_nn.forward(torch.from_numpy(s).float())
        dist = Categorical(logits=logits.detach())

        entropy_val = None
        if compute_entropy:
            entropy_val = dist.entropy()
        if det:
            return torch.argmax(dist.probs).item(), entropy_val
        else:
            try:
                return dist.sample().item(), entropy_val
            except RuntimeError:
                print(dist.probs)
                print(logits)

    def fit(
        self, states, actions, next_states, rewards, returns, dones, K, beta_0, delta
    ):
        old_pi = copy.deepcopy(self.pi_nn)
        with torch.no_grad():
            log_pi_old = torch.log(F.softmax(old_pi(states), dim=1))
            log_pi_old_a = log_pi_old.gather(1, actions.unsqueeze(-1))

        estimates = self.V_nn(states).squeeze(1)

        if self.returns == "td":
            s_values = self.V_nn_target(next_states).squeeze(1)
            target = rewards + self.gamma * s_values * dones
            assert estimates.shape == target.shape
        else:
            target = returns

        adv = target - self.V_nn_target(states).squeeze().detach()

        # Update policy with K steps of gradient ascent
        # on clipped objective
        actor_steps = []
        for step in range(K):

            log_pi = torch.log(self.pi(states))
            log_pi_a = log_pi.gather(1, actions.unsqueeze(-1))

            pi_ratio = torch.exp(log_pi_a - log_pi_old_a)
            L_actor = -torch.minimum(
                pi_ratio * adv, torch.clip(pi_ratio, 1 - self.eps, 1 + self.eps) * adv
            ).mean()

            self.pi_optim.zero_grad()
            L_actor.backward(retain_graph=True)
            self.pi_optim.step()
            actor_steps.append(L_actor.item())

            # TODO
            # entropy = (torch.log(probs + 1e-4) * probs).sum(-1)

        L_critic = nn.SmoothL1Loss()(target, estimates)

        self.V_optim.zero_grad()
        L_critic.backward()
        self.V_optim.step()
        self.pi_optim.zero_grad()
        return actor_steps, L_critic.item()


class KLPPOAgent(object):
    def __init__(
        self, env, opt, gamma, v_lr, pi_lr, env_feature_size, pi_layers, v_layers
    ):
        self.opt = opt
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space

        self.gamma = gamma
        self.insize = env_feature_size

        self.outsize = self.action_space.n  # number of actions

        self.pi_lr = pi_lr
        self.pi_nn = NN(self.insize, self.outsize, pi_layers)
        self.pi_optim = torch.optim.RMSprop(
            params=self.pi_nn.parameters(), lr=self.pi_lr
        )

        self.V_lr = v_lr
        self.V_nn = NN(self.insize, 1, v_layers)
        self.V_optim = torch.optim.RMSprop(params=self.V_nn.parameters(), lr=self.V_lr)
        self.V_nn_target = copy.deepcopy(self.V_nn)

    def pi(self, s, single=False):
        if single:
            x = F.softmax(self.pi_nn.forward(s), dim=0)
        else:
            x = F.softmax(self.pi_nn.forward(s), dim=1)
        return x

    def change_target(self):
        print("Value target changed")
        self.V_nn_target = copy.deepcopy(self.V_nn)

    def act(self, s, reward, done, compute_entropy=True, det=False):
        assert isinstance(s, np.ndarray)
        logits = self.pi_nn.forward(torch.from_numpy(s).float())
        dist = Categorical(logits=logits.detach())

        entropy_val = None
        if compute_entropy:
            entropy_val = dist.entropy()
        if det:
            return torch.argmax(dist.probs).item(), entropy_val
        else:
            try:
                return dist.sample().item(), entropy_val
            except RuntimeError:
                print(dist.probs)
                print(logits)

    def fit(
        self, states, actions, next_states, rewards, returns, dones, K, beta_0, delta
    ):
        old_pi = copy.deepcopy(self.pi_nn)
        with torch.no_grad():
            log_pi_old = torch.log(F.softmax(old_pi(s_batch), dim=1))
            log_pi_old_a = log_pi_old.gather(1, a_batch.unsqueeze(-1))
            pi_dist_old = Categorical(logits=log_pi_old)

        beta = beta_0

        for step in range(K):
            estimates = self.V_nn(s_batch).squeeze(1)

            td_0 = (
                r_batch
                + self.gamma * self.V_nn_target(s_p_batch).squeeze(1) * done_batch
            )
            assert estimates.shape == td_0.shape
            adv = -(estimates - td_0.detach())

            log_pi = torch.log(self.pi(s_batch))
            log_pi_a = log_pi.gather(1, a_batch.unsqueeze(-1))

            L = -torch.mean(torch.exp(log_pi_a - log_pi_old_a) * adv)

            # KL penalty
            pi_dist = Categorical(logits=log_pi)
            KLs = torch.distributions.kl_divergence(pi_dist, pi_dist_old)  # order ?
            # KLs = torch.distributions.kl_divergence(pi_dist_old, pi_dist)
            L += beta * KLs.mean()
            L.backward()
            # print(f"Step: {step} | {KLs.mean()}")

            self.pi_optim.step()
            self.V_optim.step()
            self.V_optim.zero_grad()
            self.pi_optim.zero_grad()

        if KLs.mean() >= 1.5 * delta:
            beta = 2 * beta
        elif KLs.mean() <= delta / 1.5:
            beta = 0.5 * beta
        return KLs.detach().mean()


def sample_steps(env, agent, n_steps):
    """Sample n_steps from from agent.act
    and finish started episode.

    Handles truncated for cartpole.

    Parameters:
        env: gym environment
        agent: agent with act method
        n_steps: number of steps to sample
    """
    batch = []
    i = 0
    while i < n_steps:
        done = False
        state = env.reset()
        reward = 0
        step_count = 0
        while not done:
            action, _ = agent.act(state, reward, done, compute_entropy=False)
            next_state, reward, done, info = env.step(action)
            truncated = info.get("TimeLimit.truncated", False)
            batch.append(
                [
                    state,
                    action,
                    next_state,
                    torch.tensor(reward),
                    0,
                    done and (not truncated),
                ]
            )
            # 0 is placeholder for returns
            state = next_state
            i += 1
            step_count += 1
        # rollout returns at the end of each episode
        # episode has step count transitions
        # print(step_count)
        if agent.returns == "td":
            # do nothing
            pass
        elif agent.returns == "rollout":
            returns = 0
            for backstep in range(1, step_count + 1):
                reward = batch[-backstep][3]
                batch[-backstep][4] = returns + reward
                returns = agent.gamma * (reward + returns)
                # print(f"{backstep}: {reward} {batch[-backstep][4]}")
            # breakpoint()
        elif agent.returns == "gae":
            returns = 0
            for backstep in range(1, step_count + 1):
                reward = batch[-backstep][3]
                state = batch[-backstep][0]
                next_state = batch[-backstep][2]
                done = batch[-backstep][5]

                td = (
                    reward
                    + (1 - done)
                    * agent.gamma
                    * agent.V_nn_target(torch.from_numpy(next_state).float())
                    .detach()
                    .squeeze()
                )
                lambda_returns = agent.lambd * agent.gamma * returns
                batch[-backstep][4] = td + lambda_returns
                returns = (
                    lambda_returns
                    + td
                    - agent.V_nn_target(torch.tensor(state).float()).detach().squeeze()
                )
                # print(f"{backstep}: {reward} {batch[-backstep][4]}")
            # breakpoint()

    return batch


def evaluate(env, agent, n_episodes):
    batch = []
    entropies = []
    with torch.no_grad():
        for episode in range(n_episodes):
            r_sum = 0
            done = False
            s = env.reset()
            r = 0
            while not done:
                a, step_entropy = agent.act(s, r, done, compute_entropy=True, det=True)
                s, r, done, info = env.step(a)
                r_sum += r
                entropies.append(step_entropy)
            batch.append(r_sum)
    return np.mean(batch), np.std(batch), np.mean(entropies)


def main(
    env, agent, n_epochs, n_samples, test_every, change_every, writer, beta_0, k, delta
):
    sample_id = 0
    for epoch in range(n_epochs):
        batch = sample_steps(env, agent, n_samples)
        s_t = torch.Tensor([transition[0] for transition in batch])
        action_t = torch.Tensor([transition[1] for transition in batch]).long()
        s_p_t = torch.Tensor([transition[2] for transition in batch])
        reward_t = torch.Tensor([transition[3] for transition in batch]).float()
        returns_t = torch.Tensor([transition[4] for transition in batch]).float()
        done_t = torch.logical_not(
            torch.Tensor(([transition[5] for transition in batch]))
        ).int()

        actor_steps, L_critic = agent.fit(
            s_t, action_t, s_p_t, reward_t, returns_t, done_t, K, BETA_0, DELTA
        )
        for i, l in enumerate(actor_steps):
            writer.add_scalar("Loss/Actor_step", l, i + sample_id)
        sample_id += len(actor_steps)
        writer.add_scalar("Loss/Actor", actor_steps[-1], epoch)
        writer.add_scalar("Loss/Critic", L_critic, epoch)
        if (epoch % test_every) == 0:
            r_mean, r_std, entrop = evaluate(env, agent, 100)
            print(f"Epoch: {epoch} | Mean reward: {r_mean}")
            writer.add_scalar("reward/test/mean", r_mean, epoch)
            writer.add_scalar("reward/test/std", r_std, epoch)
            writer.add_scalar("entropy/test", entrop, epoch)
        if (epoch % change_every) == 0:
            agent.change_target()


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

    #env.seed(config["seed"])
    #np.random.seed(config["seed"])
    env.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)

    episode_count = config["nbEpisodes"]


    V_LR = 1e-3
    PI_LR = 1e-4 #5e-5

    pi_layers = [
        24,
        24,
    ]
    v_layers = [
        24,
        24,
    ]

    N_SAMPLES = 64
    BETA_0 = 1.0 # not used
    K = 4
    DELTA = 1e-3 # not used
    TEST_EVERY = 20
    CHANGE_EVERY = 10
    N_EPOCHS = 10000
    GAMMA = 0.99

    agent = ClippedPPOAgent(env, config, GAMMA, V_LR, PI_LR, 4, pi_layers, v_layers)
    writer_str = (
        "CARTPOLE/"
        "PPO-" + "Clipped" + "-" +
        agent.returns
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
    main(
        env,
        agent,
        N_EPOCHS,
        N_SAMPLES,
        TEST_EVERY,
        CHANGE_EVERY,
        writer,
        BETA_0,
        K,
        DELTA,
    )
