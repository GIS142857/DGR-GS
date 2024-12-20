#coding:utf-8

from torch import nn
import torch
import numpy as np
import copy
from torch import autograd
from torch.autograd import Variable as V


class NeuralNetwork:
    def __init__(self, num_actions, state_dim, replay_memory, agent_id, device):
        #super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            #nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        self.replay_memory = replay_memory
        self.lr = 0.0001

        # path for saving parameters  torch.save(model_object.state_dict(), './params.pth')
        self.path = './' + 'params' + '/' + 'agent' + str(agent_id) + '_' + 'params.pth'
        #print('path_name', self.path)

        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)  # 这里的LR是定义的学习率
        self.loss_func = torch.nn.MSELoss()
        self.loss_func = self.loss_func.to(device)
        self.net = self.net.to(device)
        self.sum_loss = 0
        #self.ewc_loss = 0

        self.last_t = 1

        # Init Matrix which will get Fisher Matrix
        self.Fisher = {}
        self.max_fisher_len = 10000

        # Self Params
        self.params = [param for param in self.net.parameters()]
        #self.load_parameters()

    def forward(self, x):
        return self.net(x)

    def get_values(self, states):
        return self.net(states)

    def set_lr(self, lr):
        self.lr = lr

    def set_loss(self, loss):
        avg_loss = (self.sum_loss + loss) / 2
        self.sum_loss = avg_loss

    # def estimate_fisher(self):
    #     # Get loglikelihoods from data
    #     self.F_accum = []
    #     for v, _ in enumerate(self.params):
    #         self.F_accum.append(np.zeros(list(self.params[v].size())))
    #
    #     loglikelihood = self.sum_loss
    #     print('loglikelihood', loglikelihood)
    #     print('paraeters', self.net.parameters())
    #     torch.autograd.set_detect_anomaly(True)
    #     loglikelihood_grads = autograd.grad(loglikelihood, self.net.parameters(), retain_graph=True)
    #     # print("FINISHED GRADING", len(loglikelihood_grads))
    #     for v in range(len(self.F_accum)):
    #         # print(len(self.F_accum))
    #         torch.add(torch.Tensor((self.F_accum[v])), torch.pow(loglikelihood_grads[v], 2).data)
    #
    #     parameter_names = [
    #         n.replace('.', '__') for n, p in self.net.named_parameters()
    #     ]
    #     # print("RETURNING", len(parameter_names))
    #     print('fisher', self.F_accum)
    #
    #     return {n: g for n, g in zip(parameter_names, self.F_accum)}

    def estimate_fisher(self, pred_q, tar_q):
        loss = self.loss_func(pred_q, tar_q)
        self.opt.zero_grad()
        loss.backward()

        self.fisher_list = []
        for param in self.net.parameters():
            self.fisher_list.append(param.grad ** 2)

        #print('fisher', self.fisher_list)
        self.store_variables()

        #return {n: g for n, g in zip(parameter_names, params)}

    def consolidate(self, fisher):
        for n, p in self.net.named_parameters():
            n = n.replace('.', '__')
            print('n', n, 'p', p)
            print('p_1', p[1])
            self.net.register_buffer('{}_estimated_mean'.format(n), p.data.clone())
            # print(dir(fisher[n].data))
            self.net.register_buffer('{}_estimated_fisher'
                                     .format(n), fisher[n].data)
        print('register', self.net.register_buffer)

    def store_variables(self):
        self.stored_variable_list = []
        for p in self.net.parameters():
            self.stored_variable_list.append(copy.deepcopy(p.detach()))
        #print('store_paramaters', self.stored_variable_list[0])
        # params = []
        # param_list = []
        # for param in self.net.parameters():
        #     params.append(param)
        #     param_list.append(param.detach())
        # print('------params', params)
        # print('parama', param_list)

    def update_ewc_loss(self, lamda, cuda=False):
        #print('update_ewc_loss')
        try:
            losses = []
            i = 0
            for n, p in self.net.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                # mean = getattr(self, '{}_estimated_mean'.format(n))
                # fisher = getattr(self, '{}_estimated_fisher'.format(n))
                # if i < 1:
                #     print('p_old', self.stored_variable_list[0])
                #     print('p_new', p)
                #     print('fisher', self.fisher_list[i])
                #print('p', p, 'size', len(p))
                # wrap mean and fisher in Vs.
                p_old = self.stored_variable_list[i]
                fisher = self.fisher_list[i]
                i += 1
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p - p_old) ** 2).sum())
            ewc_loss = (lamda / 2) * sum(losses)
            #print('ewc_loss', ewc_loss)
            #q.values
            return ewc_loss
        except AttributeError:
            return 0
        # try:
        #     losses = []
        #     for n, p in self.net.named_parameters():
        #         # retrieve the consolidated mean and fisher information.
        #         n = n.replace('.', '__')
        #         mean = getattr(self, '{}_estimated_mean'.format(n))
        #         fisher = getattr(self, '{}_estimated_fisher'.format(n))
        #         print('mean', mean)
        #         print('p', p)
        #         # wrap mean and fisher in Vs.
        #         mean = V(mean)
        #         fisher = V(fisher.data)
        #         # calculate a ewc loss. (assumes the parameter's prior as
        #         # gaussian distribution with the estimated mean and the
        #         # estimated cramer-rao lower bound variance, which is
        #         # equivalent to the inverse of fisher information)
        #         losses.append((fisher * (p - mean) ** 2).sum())
        #     self.ewc_loss = (lamda / 2) * sum(losses)
        #     print('ewc_loss', self.ewc_loss)
        # except AttributeError:
        #     print('bug')
        #     # ewc loss is 0 if there's no consolidated parameters.
        #     self.ewc_loss = (
        #         V(torch.zeros(1)).cuda() if cuda else
        #         V(torch.zeros(1))
        #     )

    def _is_on_cuda(self):
        return next(self.net.parameters()).is_cuda

    def save_parameters(self):
        torch.save(self.net.state_dict(), self.path)

    def load_parameters(self):
        # print('load')
        self.net.load_state_dict(torch.load(self.path))
        # print('parameters', self.params)
