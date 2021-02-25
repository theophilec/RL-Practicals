import numpy as np

class Policy(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def chose(self, action_space):
        raise NotImplementedError()


class GreedyPolicy(Policy):
    def __init__(self, action_space, epsilon=0.01, decrease_rate=1, min_epsilon=0.001):
        super().__init__(action_space)
        self.epsilon = epsilon
        assert decrease_rate <= 1
        self.decrease_rate = decrease_rate
        self.min_epsilon = min_epsilon

    def chose(self, action_values):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decrease_rate
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return int(np.argmax(action_values))