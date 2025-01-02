from utils import *
from qrdqn import QRDQN
from replay_memory import RolloutBuffer


class Agent:
    def __init__(self, state_dim, action_dim, action_list, id, device, tau_N):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_list = action_list
        self.id = id
        self.tau_N = tau_N
        self.replay_memory = RolloutBuffer(3072, self.state_dim, device)  # buffer_size, state_shape, device
        self.model = QRDQN(self.action_dim, self.state_dim, self.id, device, tau_N)
        # self.model.load_parameters()
        # Log of the rewards obtained in each episode during calls to run()
        self.episode_rewards = []

    def save_model(self, subpath):
        self.model.save_parameters(subpath)

    def load_model(self, subpath):
        self.model.load_parameters(subpath)
        #self.online_net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
