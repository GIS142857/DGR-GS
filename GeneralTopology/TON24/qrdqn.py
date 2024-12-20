from torch import nn
import torch


class QRDQN():
    def __init__(self, num_actions, state_dim, agent_id, device, N=50):
        super(QRDQN, self).__init__()
        hidden_1 = 32
        hidden_2 = 64