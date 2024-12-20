from torch import nn
import torch


class QRDQN():

    def __init__(self, num_actions, state_dim, agent_id, device, N=50):
        super(QRDQN, self).__init__()

        hidden_1 = 32
        hidden_2 = 64

        self.sharedlayer = nn.Sequential(
            nn.Linear(state_dim, hidden_1),
            nn.ReLU()
            # nn.Dropout()
        )
        self.tower1 = nn.Sequential(
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hidden_2, num_actions * N)
        )
        self.tower2 = nn.Sequential(
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hidden_2, num_actions * N)
        )
        self.tower3 = nn.Sequential(
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hidden_2, num_actions * N)
        )

        self.N = N
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.agent_id = agent_id

        self.lr = 0.001
        #self.opt = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)  # 这里的LR是定义的学习率
        self.opt1 = torch.optim.Adam([{'params': self.sharedlayer.parameters()}, {'params': self.tower1.parameters()}],
                                     lr=self.lr)  # 这里的LR是定义的学习率
        self.opt2 = torch.optim.Adam(self.tower2.parameters(), lr=self.lr)  # 这里的LR是定义的学习率
        self.opt3 = torch.optim.Adam(self.tower3.parameters(), lr=self.lr)  # 这里的LR是定义的学习率
        self.loss_func = torch.nn.MSELoss()
        self.loss_func = self.loss_func.to(device)

        self.sharedlayer = self.sharedlayer.to(device)
        self.tower1 = self.tower1.to(device)
        self.tower2 = self.tower2.to(device)
        self.tower3 = self.tower3.to(device)

        self.path = './' + 'params' + '/' + 'agent' + str(agent_id) + '_' + 'params.pth'

    def forward(self, batch_size, states, flow):
        assert states is not None
        #print('states', states, 'shape', states.shape)
        #states = torch.squeeze(states, 0)
        #batch_size = states.shape[0]
        #print('states', states, 'shape', states.shape)
        # print('batch_size', batch_size)

        h_shared = self.sharedlayer(states)
        if flow == 0:
            qu = self.tower1(h_shared)
        elif flow == 1:
            qu = self.tower2(h_shared)
        else:
            qu = self.tower3(h_shared)

        quantiles = qu.view(batch_size, self.N, self.num_actions)

        assert quantiles.shape == (batch_size, self.N, self.num_actions)

        #print('quant', quantiles)

        return quantiles

    def calculate_q(self, batch_size, states, flow):
        assert states is not None
        states = torch.unsqueeze(states, 0)
        #batch_size = states.shape[0]
        # print('batch_size', batch_size)
        # print('states', states)

        # Calculate quantiles.
        quantiles = self.forward(batch_size, states, flow)

        # Calculate expectations of value distributions.
        q = quantiles.mean(dim=1)
        # print('quantiles', quantiles)
        # print('q', q)
        # print('batch_size', batch_size, 'num_actions', self.num_actions)
        assert q.shape == (batch_size, self.num_actions)

        return q

    def save_parameters(self, flow):
        path1 = self.path + '_' + 'shared_net'
        path2 = self.path + '_' + 'tower1'
        path3 = self.path + '_' + 'tower2'
        path4 = self.path + '_' + 'tower3'
        if flow == 0:
            torch.save(self.sharedlayer.state_dict(), path1)
            torch.save(self.tower1.state_dict(), path2)
        elif flow == 1:
            torch.save(self.tower2.state_dict(), path3)
        else:
            torch.save(self.tower3.state_dict(), path4)

    def load_parametersforf1(self):
        # print('load')
        path1 = self.path + '_' + 'shared_net'
        path2 = self.path + '_' + 'tower1'
        path3 = self.path + '_' + 'tower2'
        print('path1', path1)
        self.sharedlayer.load_state_dict(torch.load(path1))
        self.tower1.load_state_dict(torch.load(path2))

    def load_parametersforf2(self):
        # print('load')
        path1 = self.path + '_' + 'shared_net'
        path3 = self.path + '_' + 'tower2'
        self.sharedlayer.load_state_dict(torch.load(path1))
        self.tower2.load_state_dict(torch.load(path3))

    def load_parametersforf3(self):
        # print('load')
        path1 = self.path + '_' + 'shared_net'
        path3 = self.path + '_' + 'tower3'
        self.sharedlayer.load_state_dict(torch.load(path1))
        self.tower3.load_state_dict(torch.load(path3))

    def load_parameters(self):
        # print('load')
        path1 = self.path + '_' + 'shared_net'
        path2 = self.path + '_' + 'tower1'
        path3 = self.path + '_' + 'tower2'
        path4 = self.path + '_' + 'tower3'

        self.sharedlayer.load_state_dict(torch.load(path1))
        self.tower1.load_state_dict(torch.load(path2))
        self.tower2.load_state_dict(torch.load(path3))
        self.tower3.load_state_dict(torch.load(path4))
        # print('parameters', self.params)
        # print('load2')
        # for p in self.sharedlayer.parameters():
        #     print('p', p)
