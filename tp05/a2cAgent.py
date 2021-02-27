"""
Actor Critic implementation (TME 5).

Code by: Claire & Théophile.
"""
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


class A2CAgent(object):
    def __init__(
        self, env, opt, gamma, v_lr, pi_lr, env_feature_size, pi_layers, v_layers
    ):
        self.returns = "td"
        self.opt = opt
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space

        self.gamma = gamma
        self.lambd = 0.5
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

    def change_target(self):
        print("change")
        self.V_nn_target = copy.deepcopy(self.V_nn)

    def act(self, s, reward, done, compute_entropy=True):
        assert isinstance(s, np.ndarray)
        entropy_val = None
        action_probs = self.pi(torch.from_numpy(s).float(), single=True)
        if compute_entropy:
            entropy_val = entropy(action_probs)
        dist = Categorical(action_probs)
        return dist.sample().item(), entropy_val

    def fit(self, s_batch, a_batch, s_p_batch, r_batch, returns_batch, done_batch):
        # s_p = s_prime
        estimates = self.V_nn(s_batch).squeeze(1)

        if self.returns == "td":
            s_values = self.V_nn_target(s_p_batch).squeeze(1)
            target = r_batch + self.gamma * s_values * done_batch
        else:
            target = returns_batch

        assert estimates.shape == target.shape

        loss = F.smooth_l1_loss(estimates, target.detach())
        self.V_optim.zero_grad()
        loss.backward()

        critic_grad_norm = np.sqrt(
            sum([torch.norm(p.grad) ** 2 for p in list(self.V_nn.parameters())])
        )
        # TODO: log loss

        adv = target - self.V_nn_target(s_batch).squeeze()

        log_pi = torch.log(self.pi(s_batch)).gather(1, a_batch.unsqueeze(-1))

        J = -torch.mean(adv.detach() * log_pi)
        self.pi_optim.zero_grad()
        J.backward()
        self.V_optim.step()
        self.pi_optim.step()
        pi_grad_norm = np.sqrt(
            sum([torch.norm(p.grad) ** 2 for p in list(self.pi_nn.parameters())])
        )
        self.pi_optim.zero_grad()
        return loss.item(), J.item(), critic_grad_norm, pi_grad_norm


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
                a, step_entropy = agent.act(s, r, done, compute_entropy=True)
                s, r, done, _ = env.step(a)
                r_sum += r
                entropies.append(step_entropy)
            batch.append(r_sum)
    return np.mean(batch), np.std(batch), np.mean(entropies)


def main(env, agent, n_epochs, n_samples, test_every, change_every, writer):
    n_step = 0
    for epoch in range(n_epochs):
        batch = sample_steps(env, agent, n_samples)
        n_step += n_samples
        s_t = torch.Tensor([transition[0] for transition in batch])
        action_t = torch.Tensor([transition[1] for transition in batch]).long()
        s_p_t = torch.Tensor([transition[2] for transition in batch])
        reward_t = torch.Tensor([transition[3].item() for transition in batch]).float()
        returns_t = torch.Tensor([transition[4] for transition in batch]).float()
        done_t = torch.logical_not(
            torch.Tensor(([transition[5] for transition in batch]))
        ).int()

        v_loss, J, critic_norm, pg_norm = agent.fit(
            s_t, action_t, s_p_t, reward_t, returns_t, done_t
        )
        writer.add_scalar("Loss/critic", v_loss, epoch)
        writer.add_scalar("Loss/actor", J, epoch)
        writer.add_scalar("Gradients/critic", critic_norm, epoch)
        writer.add_scalar("Gradients/actor", pg_norm, epoch)
        if (epoch % test_every) == 0:
            r_mean, r_std, entrop = evaluate(env, agent, 100)
            print(f"Epoch: {epoch} | Mean reward: {r_mean}")
            writer.add_scalar("reward/test/mean", r_mean, epoch)
            writer.add_scalar("reward/test/std", r_std, epoch)
            writer.add_scalar("entropy/test", entrop, epoch)
        if (epoch % change_every) == 0:
            agent.change_target()


def entropy(probs):
    return -torch.sum(probs * torch.log(probs))


def discounted_cumsum(x, gamma):
    """Compute discounted cumulative sum of x.

    :param: x, torch.Tensor
    :param: gamma, float
    """
    assert len(x) == 1
    return torch.cumsum(x * torch.Tensor(gamma ** np.arange(len(x))))


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

    V_LR = 1e-3
    PI_LR = 1e-3

    pi_layers = [
        24,
        24,
    ]
    v_layers = [24, 24]

    N_SAMPLES = 10

    GAMMA = 0.999
    agent = A2CAgent(env, config, GAMMA, V_LR, PI_LR, 4, pi_layers, v_layers)
    writer_str = (
        "CARTPOLE/"
        "A2C"
        + "-"
        + agent.returns
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
    TEST_EVERY = 10
    CHANGE_EVERY = 50
    N_EPOCHS = 10000
    main(env, agent, N_EPOCHS, N_SAMPLES, TEST_EVERY, CHANGE_EVERY, writer)
