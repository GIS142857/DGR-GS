import copy

import numpy as np
from Utils import *
from Net import NeuralNetwork as net
from qrdqn import QRDQN
import torch

class Agent:

    def __init__(self, state_dim, action_dim, action_list, id, device, episodes):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_list = action_list
        self.id = id
        # self.env = ENV(state_dim, action_dim)

        # Whether we are training (True) or testing (False).
        self.training = True

        # Epsilon-greedy policy for selecting an action from the Q-values.
        # During training the epsilon is decreased linearly over the given
        # number of iterations. During testing the fixed epsilon is used.
        num_iter = 3000
        self.epsilon_greedy = EpsilonGreedy(start_value=0.8,
                                            end_value=0.1,
                                            num_iterations=num_iter,
                                            num_actions=self.action_dim,
                                            epsilon_testing=0.01)

        if self.training:
            # The following control-signals are only used during training.

            # The learning-rate for the optimizer decreases linearly.
            self.learning_rate_control = LinearControlSignal(start_value=1e-2,
                                                             end_value=1e-3,
                                                             num_iterations=episodes)

            # The loss-limit is used to abort the optimization whenever the
            # mean batch-loss falls below this limit.
            self.loss_limit_control = LinearControlSignal(start_value=0.1,
                                                          end_value=0.015,
                                                          num_iterations=1e4)

            # The maximum number of epochs to perform during optimization.
            # This is increased from 5 to 10 epochs, because it was found for
            # the Breakout-game that too many epochs could be harmful early
            # in the training, as it might cause over-fitting.
            # Later in the training we would occasionally get rare events
            # and would therefore have to optimize for more iterations
            # because the learning-rate had been decreased.
            self.max_epochs_control = LinearControlSignal(start_value=5.0,
                                                          end_value=10.0,
                                                          num_iterations=1e4)

            # The fraction of the replay-memory to be used.
            # Early in the training, we want to optimize more frequently
            # so the Neural Network is trained faster and the Q-values
            # are learned and updated more often. Later in the training,
            # we need more samples in the replay-memory to have sufficient
            # diversity, otherwise the Neural Network will over-fit.
            self.replay_fraction = LinearControlSignal(start_value=0.1,
                                                       end_value=1.0,
                                                       num_iterations=1e4)
        else:
            # We set these objects to None when they will not be used.
            self.learning_rate_control = None
            self.loss_limit_control = None
            self.max_epochs_control = None
            self.replay_fraction = None

        if self.training:
            # We only create the replay-memory when we are training the agent,
            # because it requires a lot of RAM. The image-frames from the
            # game-environment are resized to 105 x 80 pixels gray-scale,
            # and each state has 2 channels (one for the recent image-frame
            # of the game-environment, and one for the motion-trace).
            # Each pixel is 1 byte, so this replay-memory needs more than
            # 3 GB RAM (105 x 80 x 2 x 200000 bytes).

            self.replay_memory = ReplayMemory(size=10000,
                                              state_dim=self.state_dim)  # size, num_actions, state_dim, discount_factor=0.97
            self.copy_memory = ReplayMemory(size=10000, state_dim=self.state_dim)
        else:
            self.replay_memory = None
            

        # Create the Neural Network used for estimating Q-values.
        #self.model = net(self.action_dim, self.state_dim, self.replay_memory, self.id, device)  # num_actions, state_dim, replay_memory, agent_id, load_checkpoint=False
        #self.model = self.model.net.to(device)
        self.model = QRDQN(self.action_dim, self.state_dim, self.id, device)
        # Log of the rewards obtained in each episode during calls to run()
        self.episode_rewards = []


    def get_copy_memory(self):
        self.copy_memory.states = copy.deepcopy(self.replay_memory.states)
        self.copy_memory.actions = copy.deepcopy(self.replay_memory.actions)
        self.copy_memory.action_indexs = copy.deepcopy(self.replay_memory.action_indexs)
        self.copy_memory.rewards = copy.deepcopy(self.replay_memory.rewards)
        self.copy_memory.next_state = copy.deepcopy(self.replay_memory.next_state)
        self.copy_memory.end_life = copy.deepcopy(self.replay_memory.end_life)
        self.copy_memory.num_used = self.replay_memory.num_used

    def reset_episode_rewards(self):
        """Reset the log of episode-rewards."""
        self.episode_rewards = []

    def save_parameters(self):
        torch.save(self.net.state_dict(), './params.pth')



