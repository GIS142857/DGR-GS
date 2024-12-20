import math
import random
import numpy as np
from collections import defaultdict
import queue
from heapq import *


class TABLE:
    def __init__(self, g):
        self.inf = 1000000
        self.min_d = self.inf
        self.graph = g
        self.dmax = 20
        self.state_count_dictionary = defaultdict(dict)
        self.state_count_dictionary[self.graph.node_num - 1]['worst'] = [0]
        self.state_count_dictionary[self.graph.node_num - 1]['policy'] = []
        self.state_count_dictionary[self.graph.node_num - 1]['expect'] = [0]
        for i in range(self.graph.node_num-1):
            self.state_count_dictionary[i]['worst'] = []
            self.state_count_dictionary[i]['policy'] = []
            self.state_count_dictionary[i]['expect'] = []
        self.Q = queue.Queue(maxsize=10000)
        for i in range(self.graph.node_num):
            if self.graph.G.has_node(self.graph.node_num-1-i):
                self.Q.put(self.graph.node_num-1-i)
        #print('Q',self.Q.queue)
        #self.q = np.ones((self.graph.node_num,self.dmax,self.graph.node_num)) * self.inf
        self.q = defaultdict(dict) # key: node + D + child + k
        self.pass_weight = np.ones((g.node_num, g.node_num))

    def not_in_queue(self, node, q_list):
        for i in range(len(q_list)):
            if node == q_list[i]:
                return False
        return True


    def generate_tables(self):
        while not self.Q.empty():
            #print('b_len', len(self.Q.queue), 'queue', self.Q.queue)
            node = self.Q.get()
            # print('a_len', len(self.Q.queue))
            # print('change table node', node)
            nb_num, parent_list = self.graph.get_parent_node(node)

            #print('node', node, 'parent_num', nb_num)
            for i in range(nb_num):
                # print('i', i)
                # print('111')
                bl = self.relax(parent_list[i], node)
                if bl:
                    # print('222')
                    # print('parent_list', parent_list[i])
                    if self.not_in_queue(parent_list[i], self.Q.queue):
                        self.Q.put(parent_list[i])
        #             print('Q_len', len(self.Q.queue))
        #             print('333')
        # print('after_generate_graph')

    def get_bound_expected(self, d, node):
        # print('get_bound')
        # print('node_v',node,'d',d,'worst_list',self.state_count_dictionary[node]['worst'])
        for i in range(len(self.state_count_dictionary[node]['worst'])):
            #print('d', d, 'worst', self.state_count_dictionary[node]['worst'])
            if int(d * 100) >= int(100 * self.state_count_dictionary[node]['worst'][i]):
                #print('d',d,'worst_i',self.state_count_dictionary[node]['worst'][i])
                return self.state_count_dictionary[node]['expect'][i]
        return self.inf

    def get_expected(self, d, real_d_list, real_prob_list, node_v):
        sum = 0
        for i in range(len(real_d_list)):
            #print('d', d, 'real_d', real_d_list[i])
            e = self.get_bound_expected(d - real_d_list[i], node_v)
            #print('eeee',e)
            if e == self.inf:
                return sum, False

            sum += real_prob_list[i] * (real_d_list[i] + e)
        #print('sum', sum)
        return sum, True

    def relax(self, node_u, node_v):
        #print('relax', node_u, node_v)
        real_d_list = self.graph.G[node_u][node_v]['real_d']
        real_prob_list = self.graph.G[node_u][node_v]['real_prob']
        #print('real_d',real_d_list,'real_prob',real_prob_list)
        change_num = 0
        for d0 in self.state_count_dictionary[node_v]['worst']:
            for x0 in real_d_list:
                #print('d0', d0, 'x0', x0)
                d = d0 + x0
                #print('dd',d)
                e, bll = self.get_expected(d, real_d_list, real_prob_list, node_v)
                #print('bl', bll)
                if bll:
                    #print('dd', d, 'ee', e)
                    bl = self.merge(d, e, node_u, node_v)
                    #print('merge_bl', bl)
                    if bl:
                        change_num += 1
        if change_num > 0:
            return True
        else:
            return False

    def check_tuple(self, d, node):
        #print('node',node,'worst',self.state_count_dictionary[node]['worst'],'d',d)
        for i in range(len(self.state_count_dictionary[node]['worst'])):
            if self.state_count_dictionary[node]['worst'][i] == d:
                pi = self.state_count_dictionary[node]['policy'][i]
                e = self.state_count_dictionary[node]['expect'][i]
                return True, [d, pi, e], i
        return False, [], -1

    def check_index(self, index, list_index):
        for i in range(len(list_index)):
            if index == list_index[i]:
                return True
        return False

    def remove_tuple(self, list_index, node_u):
        temp_w_list = []
        temp_p_list = []
        temp_e_list = []
        k = 0
        # print('remove')
        # print('worst_list',self.state_count_dictionary[node_u]['worst'])
        for i in range(len(self.state_count_dictionary[node_u]['worst'])):
            if self.check_index(i, list_index):
                continue
            else:
                temp_w_list.append(self.state_count_dictionary[node_u]['worst'][i])
                temp_p_list.append(self.state_count_dictionary[node_u]['policy'][i])
                temp_e_list.append(self.state_count_dictionary[node_u]['expect'][i])
        self.state_count_dictionary[node_u]['worst'] = temp_w_list
        self.state_count_dictionary[node_u]['policy'] = temp_p_list
        self.state_count_dictionary[node_u]['expect'] = temp_e_list
        #print('after remove',self.state_count_dictionary[node_u]['worst'],self.state_count_dictionary[node_u]['policy'],self.state_count_dictionary[node_u]['expect'])

    def insert(self, d, node_v, e, node_u):
        # print('insert')
        # print('d',d,'policy',node_v,'e',e)
        l = len(self.state_count_dictionary[node_u]['worst'])
        if l == 0:
            self.state_count_dictionary[node_u]['worst'].append(d)
            self.state_count_dictionary[node_u]['policy'].append(node_v)
            self.state_count_dictionary[node_u]['expect'].append(e)
        else:
            k = 0
            for i in range(l):
                k += 1
                if e < self.state_count_dictionary[node_u]['expect'][i]:
                    self.state_count_dictionary[node_u]['worst'].append(
                        self.state_count_dictionary[node_u]['worst'][l - 1])
                    self.state_count_dictionary[node_u]['policy'].append(
                        self.state_count_dictionary[node_u]['policy'][l - 1])
                    self.state_count_dictionary[node_u]['expect'].append(
                        self.state_count_dictionary[node_u]['expect'][l - 1])
                    j = l - 1
                    while j >= i:
                        self.state_count_dictionary[node_u]['worst'][j + 1] = \
                        self.state_count_dictionary[node_u]['worst'][j]
                        self.state_count_dictionary[node_u]['policy'][j + 1] = \
                        self.state_count_dictionary[node_u]['policy'][j]
                        self.state_count_dictionary[node_u]['expect'][j + 1] = \
                        self.state_count_dictionary[node_u]['expect'][j]
                        j -= 1
                    self.state_count_dictionary[node_u]['worst'][i] = d
                    self.state_count_dictionary[node_u]['policy'][i] = node_v
                    self.state_count_dictionary[node_u]['expect'][i] = e
                    break
            if k == l:
                self.state_count_dictionary[node_u]['worst'].append(d)
                self.state_count_dictionary[node_u]['policy'].append(node_v)
                self.state_count_dictionary[node_u]['expect'].append(e)
            #print('node inster', node_u, 'policy', self.state_count_dictionary[node_u]['worst'], self.state_count_dictionary[node_u]['policy'],self.state_count_dictionary[node_u]['expect'])
            return

    def merge(self, d, e, node_u, node_v):
        #print('merge')
        bl, tuple, index = self.check_tuple(d, node_u)
        #print('check_result','tuple',tuple,'index',index)
        change = False
        #print('policy1', self.state_count_dictionary[node_u])
        if bl:
            #print('e', e, 'old_e', tuple[2])
            if e < tuple[2]:
                self.state_count_dictionary[node_u]['worst'][index] = d  ## replace
                self.state_count_dictionary[node_u]['policy'][index] = node_v
                self.state_count_dictionary[node_u]['expect'][index] = e
                change = True
        else:
            self.insert(d, node_v, e, node_u)
            # self.state_count_dictionary[node_u]['worst'].append(d)
            # self.state_count_dictionary[node_u]['policy'].append(node_v)
            # self.state_count_dictionary[node_u]['expect'].append(e)
            change = True
        #print('node',node_u,'worst_list',self.state_count_dictionary[node_u]['worst'],'policy',self.state_count_dictionary[node_u]['policy'],'expect',self.state_count_dictionary[node_u]['expect'])
        ### remove all 3-tules from table[u] with d'>d and e'>=e
        #print('policy2', self.state_count_dictionary[node_u])
        remove_index = []
        for i in range(len(self.state_count_dictionary[node_u]['worst'])):
            if self.state_count_dictionary[node_u]['worst'][i] > d and self.state_count_dictionary[node_u]['expect'][i] >= e:
                remove_index.append(i)
        if remove_index:
            self.remove_tuple(remove_index, node_u)
        #print('policy3', self.state_count_dictionary[node_u])
        return change

    def get_mind(self,node):
        dmin = 1000000
        #print('node',node,'worst',self.state_count_dictionary[node]['worst'])
        for d in self.state_count_dictionary[node]['worst']:
            if d < dmin:
                dmin = d
        return dmin

    def get_e(self,node_u,node_v,d):
        emin = 10000
        for i in range(len(self.state_count_dictionary[node_u]['worst'])):
            if self.state_count_dictionary[node_u]['policy'][i] == node_v:
                #print('node',node_u,'d',d,'worst_list',self.state_count_dictionary[node_u]['worst'],'expect',self.state_count_dictionary['expect'])
                if self.state_count_dictionary[node_u]['worst'][i] <= d and self.state_count_dictionary[node_u]['expect'][i] < emin:
                    emin = self.state_count_dictionary[node_u]['expect'][i]
        return emin

    def learn(self):
        #print('learn')
        for i in range(self.graph.node_num):
            dmin = self.get_mind(i)
            #print('u',i,'dmin',dmin)
            for d in range(math.floor(dmin),self.dmax+1):
                num, child_list = self.graph.get_nb_num(i)
                for k in range(num):
                    self.q[i][d-1][child_list[k]] = self.get_e(i,child_list[k],d)
                    # if i == 1:
                    #print('d',d,'u',i,'v',child_list[k],'q',self.q[i][d-1][child_list[k]])

    #def init_q(self):

    def check_action(self, action, action_list):
        for a in action_list:
            if a == action:
                return True
        return False

    def get_avg_d(self, u, v, h, f, adj_weight):
        if adj_weight[u][v] > 1:
            self.graph.get_delayBasedNC(u, v, f, adj_weight)
        key = str(v) + ',' + str(f)  # + ',' + str(h+1)
        real_d = self.graph.node_list[u].nb_delay[key]
        real_pro = self.graph.node_list[u].nb_delay_pro[key]
        avg_d = 0
        for i in range(len(real_d)):
            delay = real_d[i]
            avg_d += delay * real_pro[i]
        return avg_d

def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    #print('x',x)
    cumulative_probability = 0.0
    #print('some_list',some_list)
    #print('probabilities',probabilities)
    if len(some_list) == 1 and probabilities[0] != 0:
        return some_list[0]
    #u = -1
    l = len(some_list)
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        #print('item_probability',item_probability)
        if x < cumulative_probability:
            return item
    #print('uu', u)
    return some_list[l-1]
def get_reward(table, u, v, h, f, training):
    # print('get_reward')
    # print('u', u, 'v', v, 'f', f)
    # print('some_list',table.graph.G[u][v]['real_d'])
    # print('probabilities__',table.graph.G[u][v]['real_prob'])
    if v == -1:
        return 1000
    key = str(v) + ',' + str(f) #+ ',' + str(h)
    real_d = table.graph.node_list[u].nb_delay[key]
    real_pro = table.graph.node_list[u].nb_delay_pro[key]
    #print('real_d', real_d)
    if training:
        return random_pick(real_d, real_pro)
    else:
        return max(real_d)

def get_avg_reward(table, u, v, h):
    # print('u',u,'v',v,'G',table.graph.G)
    # print('some_list',table.graph.G[u][v]['real_d'])
    # print('probabilities__',table.graph.G[u][v]['real_prob'])
    if v == -1:
        return 1000
    key = str(v)# + ',' + str(0) #+ ',' + str(h)
    real_d = table.graph.node_list[u].nb_delay[key]
    real_pro = table.graph.node_list[u].nb_delay_pro[key]
    avg_d = 0
    for i in range(len(real_d)):
        avg_d += real_d[1]*real_pro[i]
    return avg_d


def get_minq(table, u, D, h, k, des):
    #print('get_mindddd', ' u', u, 'h', h)
    num, child_list = table.graph.get_nb_num(u)
    minq = 10000
    for i in range(num):
        #dmin = table.get_mind(child_list[i])
        nb = child_list[i]
        nb_num, _ = table.graph.get_nb_num(nb)
        if nb_num == 0 and nb < des:
            continue
        dmin = 0
        if nb < des:
            dmin = get_short_path(table.graph, nb, des, k)
        elif nb > des:
            dmin = 10000
        #print('key', key, 'dmin', dmin)
        key1 = str(nb) + ',' + str(k) #+ ',' + str(h+1)
        #print('u',u,'child',child_list[i],'worst',table.graph.G[u][child_list[i]]['worst'],'D',D)
        key2 = str(u) + ',' + str(D) + ',' + str(nb) + ',' + str(k)
        child_q = 0
        #print('u', u, 'nb', nb)
        if key2 in table.q.keys():
            child_q = table.q[key2]
        #print('key1', key1)
        max_d = max(table.graph.node_list[u].nb_delay[key1])
        #print('key1', key1, 'h', h, 'max', max_d)
        if (dmin + max_d) <= D and child_q < minq:
            minq = child_q
    # if minq == 10000:
    #     print('get_min', ' u', u, 'h', h, 'D', D)
    return minq

def update_q(table,u,D,v, reward, ahpa, h, k, des):
    #print('update','u',u,'v',v,'D',D,'d',D-reward, 'h', 'reward', reward)
    #print('q_value', table.q)
    D = int(D)
    key = str(u) + ',' + str(D) + ',' + str(v) + ',' + str(k)
    if v == des:
        if key in table.q.keys():
            table.q[key] = (1 - ahpa) * table.q[key] + ahpa * reward
        else:
            table.q[key] = reward
        #table.q[u][math.floor(D) - 1][v] = (1 - ahpa) * table.q[u][math.floor(D) - 1][v] + ahpa * reward
    else:
        if key in table.q.keys():
            table.q[key] = (1-ahpa) * table.q[key] + ahpa * reward + ahpa * get_minq(table, v, D-reward, h, k, des)
        else:
            table.q[key] = reward + ahpa * get_minq(table, v, D - reward, h, k, des)
    #print('a_q', table.q[u][D - 1][v])
    #print('after q_value', table.q)

def get_minq_nb(table, u,v,D, k):
    #key = str(u) + ',' + str(D) + ',' + str(v)
    child_q = 10000
    #print('table', table.q)
    for keys in list(table.q.keys()):
        #print('keys', keys)
        l = keys.split(',')
        #print('l', l)
        if int(l[0]) == u and int(l[2]) == v and int(l[1]) <= D and int(l[2] == k):
            #print('keys', keys)
            if table.q[keys] < child_q:
                child_q = table.q[keys]
    return child_q

def get_max_child(table,u,D, child_list, k):
    #print('get_max_child', u)
    D = int(D)
    qmin = 10000
    v = -1
    #print('node', u, 'q', table.q[u][math.floor(D)-1])
    for i in range(len(child_list)):
        nb = child_list[i]
        #D = D/1.0
        key = str(u) + ',' + str(D) + ',' + str(nb)
        #print('table_keys', table.q.keys())
        # print('key', key)
        # print('u',u,'D',D,'v',child_list[i],'q',table.q[key])
        # print('u', u, 'len', len(table.q[u]))
        # for j in range(len(table.q[u])):
        #     #print('q', table.q[u][j])
        #print('u',u,'v',child_list[i],'dmin',dmin,'worst',table.graph.G[u][child_list[i]]['worst'],'D',D,'q',table.q[u][math.floor(D)-1][child_list[i]])
        #if (dmin + table.graph.G[u][child_list[i]]['worst']) <= D:
        child_q = get_minq_nb(table, u, v, D, k)
        #print('child_q', child_q)

        if child_q <= qmin:
            #print('not in keys')
            qmin = child_q
            v = child_list[i]
            #print('qmin',qmin,'v',v)
    return v


# row = np.array([0, 0, 1, 1, 1, 3, 2])
# col = np.array([1, 4, 3, 4, 2, 4, 4])
# worst_d = np.array([10, 25, 15, 10, 10, 15, 10])
# # real_d = np.array([[4], [12], [1], [10], [3], [1], [3]])
# # real_prob = np.array([[1], [1], [1], [1], [1], [1], [1]])
# real_d = np.array([[4, 10],[12,25],[1,15],[10,6],[5,10],[1,15],[3,10]])
# real_prob = np.array([[0.4, 0.6],[0.3,0.7],[0.3,0.7],[0.6,0.4],[0.6,0.4],[0.2,0.8],[0.4,0.6]])
# # real_d = np.array([[6, 8, 10], [12, 15, 25], [1, 3, 15], [10, 9, 8], [5, 10, 3], [1, 6, 15], [10, 5, 7]])
# # real_prob = np.array([[0.2, 0.4, 0.4],[0.3,0.3,0.4],[0.3,0.6,0.1],[0.3,0.4,0.3],[0.4,0.4,0.2],[0.6,0.3,0.1],[0.4,0.3,0.3]])
# N = 5
# g = graph(row, col, worst_d, real_d, real_prob, N)
# table = TABLE(g)
# table.generate_tables()
# for i in range(table.graph.node_num):
#     print('node',i,'worst',table.state_count_dictionary[i]['worst'],'policy',table.state_count_dictionary[i]['policy'],'expect',table.state_count_dictionary[i]['expect'])
#
#
# ### learning in dynamic environment
# new_real_d = np.array([[6, 8, 10], [12, 15, 25], [1, 3, 15], [10, 9, 8], [5, 10, 3], [1, 6, 15], [10, 5, 7]])
# new_real_prob = np.array([[0.2, 0.4, 0.4],[0.3,0.3,0.4],[0.3,0.6,0.1],[0.3,0.4,0.3],[0.4,0.4,0.2],[0.6,0.3,0.1],[0.4,0.3,0.3]])
# table.graph.change_real_d(new_real_d,new_real_prob)
#
# print('----change real_d-------')
# for i in range(np.size(row)):
#     print('u',row[i],'v',col[i],'real_d',table.graph.G[row[i]][col[i]]['real_d'],'prob',table.graph.G[row[i]][col[i]]['real_prob'])
#
# table.learn()
# #print('q',table.q)
# Episodes = 1000
# pi = 0.1
# apha = 0.5
# for e in range(Episodes):
#     u = 0
#     D = 40
#     done = False
#     while not done:
#         A = []
#         num, child_list = table.graph.get_nb_num(u)
#         for i in range(num):
#             dmin = table.get_mind(child_list[i])
#             #print('u',u,'v',child_list[i],'dmin',dmin,'worst',table.graph.G[u][child_list[i]]['worst'],'D',D)
#             if (dmin + table.graph.G[u][child_list[i]]['worst']) <= D:
#                 A.append(child_list[i])
#         p = []
#         #print('A',A)
#         for i in range(num):
#             a = child_list[i]
#             if not table.check_action(child_list[i], A):
#                 p.append(0)
#             else:
#                 p.append(0.1/len(A))
#         w = -1
#         q = 10000
#         for i in range(len(A)):
#             #print('u',u,'v',A[i],'D',D,'q',table.q[u][D-1][A[i]])
#             if table.q[u][D-1][A[i]] < q:
#                 w = A[i]
#                 q = table.q[u][D-1][A[i]]
#         #print('q',q,'w',w)
#         for k in range(len(child_list)):
#             if child_list[k] == w:
#                 p[k] += 1-pi
#                 break
#
#         #print('p',p)
#         next_v = random_pick(child_list, p)
#         #print('u',u,'next_v',next_v)
#         reward = get_reward(table,u,next_v)
#         #print('reward',reward)
#         update_q(table,u,D,next_v,reward,apha)
#         D -= reward
#         u = next_v
#         if u == table.graph.node_num-1:
#             done = True
#
#     #test
#     if e >= 0:
#         #print('qq', table.q[0][39][1])
#         #print('test')
#         u = 0
#         D = 40
#         done = False
#         while not done:
#             A = []
#             num, child_list = table.graph.get_nb_num(u)
#             for i in range(num):
#                 dmin = table.get_mind(child_list[i])
#                 # print('u', u, 'v', child_list[i], 'dmin', dmin, 'worst', table.graph.G[u][child_list[i]]['worst'], 'D',
#                 #       D)
#                 if (dmin + table.graph.G[u][child_list[i]]['worst']) <= D:
#                     A.append(child_list[i])
#
#             next_v = get_max_child(table,u,D, child_list)
#             print('u', u, 'next_v', next_v)
#             reward = get_reward(table, u, next_v)
#             # update_q(table, u, D, next_v, D - reward, apha)
#             D -= reward
#             u = next_v
#             if u == table.graph.node_num - 1:
#                 done = True
def get_goal_result(g):
    table = TABLE(g)
    table.generate_tables()
    trip = []
    trip.append(0)
    id = 0
    while id < table.graph.node_num-1:
        child = table.state_count_dictionary[id]['policy']
        # print('policy', table.state_count_dictionary[id]['policy'])
        # print('id', id)
        # print('child', child)
        trip.append(child[0])
        print('node', id, 'worst', table.state_count_dictionary[id]['worst'], 'policy',
              table.state_count_dictionary[id]['policy'], 'expect', table.state_count_dictionary[id]['expect'])
        id = child[0]
    return trip, table.state_count_dictionary[0]['expect']

def online_rtss(table, deadline):
    print('online_rtss')
    u = 0
    done = False
    D = deadline
    trip = []
    trip.append(u)
    while not done:
        print('u',u, 'D', D)
        A = []
        num, child_list = table.graph.get_nb_num(u)
        for i in range(num):
            dmin = table.get_mind(child_list[i])
            #print('dmin', dmin, 'worst_to_des', table.graph.to_des_worst[child_list[i]])
            # print('u', u, 'v', child_list[i], 'dmin', dmin, 'worst', table.graph.G[u][child_list[i]]['worst'], 'D',
            #       D)
            if (dmin + table.graph.G[u][child_list[i]]['worst']) <= D:
                A.append(child_list[i])
        #print('A',A)
        next_v = get_max_child(table, u, D, A)
        if next_v == -1:
            print('no path')
            break
        trip.append(next_v)
        #reward = get_reward(table, u, next_v)
        reward = table.graph.G[u][next_v]['real_d'][0]
        #print('next_v', next_v, 'reward', reward)
        update_q(table, u, D, next_v, reward, 0.5) #table,u,D,v, reward, ahpa, h
        # update_q(table, u, D, next_v, D - reward, apha)
        D -= reward
        u = next_v
        if u == table.graph.node_num - 1:
            done = True

    ave_sum = 0
    #print('trip', trip)
    for i in range(len(trip) - 1):
        #print('node', trip[i], )
        # print('node', trip[i], 'n_node', trip[i+1], 'avg_d', table.graph.G[trip[i]][trip[i+1]]['ave_d'])
        ave_sum += table.graph.G[trip[i]][trip[i + 1]]['ave_d']
    # print('ave_dis',ave_sum)
    return ave_sum


def dijkstra_raw(edges, from_node, to_node):
        # print('from', from_node)
        # print('to', to_node)
        g = defaultdict(list)
        for l, r, c in edges:
            g[l].append((c, r))
        q, seen = [(0, from_node, ())], set()
        #print('q', q)
        while q:
            (cost, v1, path) = heappop(q)
            # print('qq', q)
            # print('v1', v1)
            # print('seen', seen)
            if v1 not in seen:
                seen.add(v1)
                path = (v1, path)
                if v1 == to_node:
                    # print('cost', cost)
                    # print('path', path)
                    return cost, path
                for c, v2 in g.get(v1, ()):
                    if v2 not in seen:
                        heappush(q, (cost + c, v2, path))
        return 10000, None

def dijkstra_(edges, from_node, to_node):
        # print('from_node',from_node)
        # print('to', to_node)
        len_shortest_path = -1
        ret_path = []
        # exist = nx.has_path(self.graph, from_node, to_node)
        # print('exist', exist)
        length, path_queue = dijkstra_raw(edges, from_node, to_node)
        if path_queue == None:
            return length, None
        if len(path_queue) > 0:
            len_shortest_path = length  ## 1. Get the length firstly;
            ## 2. Decompose the path_queue, to get the passing nodes in the shortest path.
            left = path_queue[0]
            ret_path.append(left)  ## 2.1 Record the destination node firstly;
            right = path_queue[1]
            while len(right) > 0:
                left = right[0]
                ret_path.append(left)  ## 2.2 Record other nodes, till the source-node.
                right = right[1]
            ret_path.reverse()  ## 3. Reverse the list finally, to make it be normal sequence.
        return len_shortest_path, ret_path

def pop_list(next_list):
    l = len(next_list)
    node = next_list[l - 1]
    temp = []
    for i in range(l - 1):
        temp.append(next_list[i])
    return node, temp

def push(child, next_list):
        for i in range(len(next_list)):
            if child == next_list[i]:
                return next_list
        next_list.append(child)
        return next_list

def get_short_path(graph, nb, des, k):
        # print('get_short_path', nb, h)
        M = 100000
        adj_weight = np.ones((graph.node_num, graph.node_num)) * M
        edges = []
        next_list = []
        pre_node = nb
        next_list.append(pre_node)
        while len(next_list) > 0:
            # print('node', node)
            node, next_list = pop_list(next_list)
            # print('node', node)
            if node == des:
                continue
            num, child_list = graph.get_nb_num(node)
            # print('childs', child_list)
            for i in range(num):
                child = child_list[i]
                # print('child', child)
                key = str(child) + ',' +str(k)
                # print('key', key)
                nb_delay = max(graph.node_list[node].nb_delay[key])
                adj_weight[node][child] = nb_delay
                next_list = push(child, next_list)
            # h += 1
        # print('pass_weight', self.graph.pass_matrix)
        # print('adj_weight', adj_weight)
        for i in range(len(adj_weight)):
            for j in range(len(adj_weight[0])):
                if i != j and adj_weight[i][j] != M:
                    edges.append((i, j, adj_weight[i][j]))

        #shrot_dis = self.dijstra(adj_weight, nb, self.graph.node_num - 1, self.graph.node_num)
        short_2, path = dijkstra_(edges, nb, des)
        #print('short_1', shrot_dis)
        #print('short_2', short_2, 'path', path)
        # Adjacent = [[0, 1, 12, Inf, Inf, Inf],
        #             [Inf, 0, 9, 3, Inf, Inf],
        #             [Inf, Inf, 0, Inf, 5, Inf],
        #             [Inf, Inf, 4, 0, 13, 15],
        #             [Inf, Inf, Inf, Inf, 0, 4],
        #             [Inf, Inf, Inf, Inf, Inf, 0]]
        # Src, Dst, N = 0, 5, 6
        # shrot_dis = self.dijstra(Adjacent, Src, Dst, N)
        return short_2

def get_best_path_rtss(g2, d_set, flow_num, sources, dess):
    print('rtss')
    print('d_set', d_set)
    print('sources', sources)
    print('des', dess)
    #g = graph(row, col, worst_d, real_d, real_prob, N)
    # table = TABLE(g1)
    # table.generate_tables()
    # for i in range(table.graph.node_num):
    #     print('node', i, 'worst', table.state_count_dictionary[i]['worst'], 'policy',
    #           table.state_count_dictionary[i]['policy'], 'expect', table.state_count_dictionary[i]['expect'])

    ### learning in dynamic environment
    table = TABLE(g2)
    #table.graph = g2
    # print('----change real_d-------')
    # for i in range(np.size(row)):
    #     print('u', row[i], 'v', col[i], 'real_d', table.graph.G[row[i]][col[i]]['real_d'], 'prob',
    #           table.graph.G[row[i]][col[i]]['real_prob'])

    #table.learn()
    # print('q',table.q)
    ### NA
    Episodes = 2000

    pi = 0.1
    apha = 0.5

    #d_min = 10000
    # print('11111')
    # print('flow_num', flow_num)
    for f in range(flow_num):
        source = sources[f]
        des = dess[f]
        #print('d_list', d_list)
        d_min = d_set[f]
        #print('d_min', d_min)

        old_D = d_min
        #print('old_D', old_D)
        while old_D >= d_min:
            #print('old_Dd', old_D)
            for e in range(Episodes):
                #print('episode', e)
                D = old_D
                u = source
                done = False
                h = 0
                while not done:
                    A = []
                    # na = []
                    # q = []
                    num, child_list = table.graph.get_nb_num(u)
                    if len(child_list) == 0:
                        break
                    # print('node', u)
                    # print('num', num)
                    # print('child_list', child_list)
                    for i in range(num):
                        nb = child_list[i]

                        nb_num, _ = table.graph.get_nb_num(nb)
                        if nb_num == 0 and nb < des:
                            continue
                        dmin = 0
                        if nb < des:
                            dmin = get_short_path(g2, nb, des, f)
                        elif nb > des:
                            dmin = 10000
                        # print('u',u,'v',child_list[i])
                        key1 = str(nb) + ',' + str(f) #+ ',' + str(h + 1)
                        # print('key1', key1)
                        # print('nb', nb)
                        # print('dmin', dmin, 'max', max(table.graph.node_list[u].nb_delay[key1]), 'D', D)
                        if (dmin + max(table.graph.node_list[u].nb_delay[key1])) <= D:
                            A.append(child_list[i])
                            # na.append(NA[u][child_list[i]])
                            # q.append(table.q[u][math.floor(D) - 1][child_list[i]])

                    p = []
                    #print('A',A)
                    for i in range(num):
                        a = child_list[i]
                        if not table.check_action(child_list[i], A):
                            p.append(0)
                        else:
                            p.append(0.1 / len(A))
                    w = -1
                    q = table.inf
                    for i in range(len(A)):
                        # print('u',u,'v',A[i],'D',D)
                        key = str(u) + ',' + str(D) + ',' + str(A[i]) + ',' + str(f)
                        child_q = 0
                        if key in table.q.keys():
                            child_q = table.q[key]
                        if child_q < q:
                            w = A[i]
                            q = child_q
                    # print('q',q,'w',w)
                    for k in range(len(child_list)):
                        if child_list[k] == w:
                            p[k] += 1 - pi
                            break

                    #print('child_list',child_list,'p',p)
                    #print('node', u, 'get_next_v')

                    next_v = random_pick(child_list, p)
                    #print('u', u, 'v', next_v)
                    if next_v == -1 or next_v > des:
                        break


                    # c = 2
                    # A_list = q + c * (np.sqrt(np.log(e) / (na)))
                    # arm_index = np.argmax(A_list)
                    # next_v = A[arm_index]
                    # #print('arm_index', arm_index)
                    # NA[u][next_v] += 1

                    # print('u',u,'next_v',next_v)
                    reward = get_reward(table, u, next_v, h + 1, f, training=True)
                    # print('reward', reward)
                    if reward == 1000:
                        done = True
                    # print('reward',reward)
                    h += 1
                    # print('u', u, 'v', next_v, 'h', h)
                    update_q(table, u, D, next_v, reward, apha, h, f, des)
                    D -= reward
                    u = next_v
                    if u == des:
                        done = True

            old_D -= 1
            # return table
            # print('q_value', table.q)
    #print('save_qTable')
    np.save('q_Table.npy', table.q)
    #print('After_save_qTable')

    #return get_opt_path_rtss(table, d_set, table.pass_weight, flow_num, sources, dess)


def get_opt_path_rtss(table, d_set, adj_weight, flow_num, sources, dess):
    cost_set = []
    trip_set = []
    #l = len(d_set)
    w = 0
    print('d_set', d_set)
    for k in range(flow_num):
        d = d_set[k]
        print('test')
        print('d', d)
        f = k
        # if w < l/2:
        #     f = 0
        # else:
        #     f = 1
        w += 1
        D = d
        u = sources[k]
        des = dess[k]
        done = False
        trip = []
        trip.append(u)
        h = 0
        while not done:
            # print('u', u, 'h', h)
            A = []
            num, child_list = table.graph.get_nb_num(u)
            for i in range(num):
                nb = child_list[i]
                nb_num, _ = table.graph.get_nb_num(nb)
                if nb_num == 0 and nb < des:
                    continue
                dmin = 0
                if nb < des:
                    dmin = get_short_path(table.graph, nb, des, k)
                elif nb > des:
                    dmin = 10000
                # print('u', u, 'v', child_list[i], 'dmin', dmin, 'D',
                #       D)

                key1 = str(nb) + ',' + str(f)  # + ',' + str(h + 1)
                # print('key1', key1)
                # print('nb', nb, 'dmin', dmin, 'max_d', max(table.graph.node_list[u].nb_delay[key1]),'D', D)
                if (dmin + max(table.graph.node_list[u].nb_delay[key1])) <= D:
                    A.append(child_list[i])
            # print('A',A)
            next_v = get_max_child(table, u, D, child_list, f)
            if next_v == -1:
                print('no path')
                break
            trip.append(next_v)
            #print('u', u, 'next_v', next_v)
            reward = get_reward(table, u, next_v, h + 1, f, training=False)
            # print('reward', reward)
            # update_q(table, u, D, next_v, D - reward, apha)
            D -= reward
            u = next_v
            h += 1
            if u == des:
                done = True

        ave_sum = 0
        for i in range(len(trip) - 1):
            # print('node', trip[i], 'n_node', trip[i+1], 'avg_d', table.graph.G[trip[i]][trip[i+1]]['ave_d'])
            # ave_sum += table.graph.G[trip[i]][trip[i+1]]['ave_d']
            ave_sum += table.get_avg_d(trip[i], trip[i + 1], i, f, adj_weight)
        # print('ave_dis', ave_sum, 'trip', trip)
        cost_set.append(ave_sum)
        trip_set.append(trip)
        for i in range(len(trip) - 1):
            adj_weight[trip[i]][trip[i + 1]] += 1
        #table.graph.update_arrivalInterval(trip, k)

    return cost_set, trip_set







