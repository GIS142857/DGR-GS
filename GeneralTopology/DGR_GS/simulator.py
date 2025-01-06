import numpy as np
import simpy

from GeneralTopology.DGR_GS.exp_buffer import RouteExpBuffer
from node import Node
from GeneralTopology.util.utils import *
from simpy.util import start_delayed
from config import *
from route import *


class Simulator:
    def __init__(self, sim_time, nodes_num, src, des, adj_T, nodes_pos, frame_slot, slot_duration, arrival_rate):
        self.env = simpy.Environment()
        self.start_time = self.env.now
        self.sim_time = sim_time
        self.nodes_num = nodes_num
        self.nodes = []
        self.src = src
        self.des = des
        self.adj_T = adj_T
        self.nodes_pos = nodes_pos
        self.frame_slot = frame_slot
        self.slot_duration = slot_duration
        self.arrival_rate = arrival_rate
        self.episode = 0
        self.can_not_dg = 0
        self.loss_cnt = {}
        self.route_exp_buffer = RouteExpBuffer(MAX_EXPERIENCE_SIZE)
        self.default_e2ed_delay = {}  # {'0-15': 4499.55, ...}
        self.e2ed_delay = {}  # {'0-15':[5954.83, 4594.83, 5554.83, ...]}
        self.paths_delay = {}  # {'0-3-5-7-9':[5954.83, 4594.83, 5554.83, ...]}
        self.e2ed_delay_plus = {}  # {'0-15':[5954.83, 4594.83, 5554.83, ...]}
        self.e2ed_delay_minus = {}  # {'0-15':[5954.83, 4594.83, 5554.83, ...]}
        self.queue_delay = {}  # {'flow1': [1978.01, ...], 'flow2': [1978.01, ...], 'flow3': [1978.01, ...]}
        self.update_done = False
        self.init()
        self.run()

    def init(self):
        for node_id in range(self.nodes_num):
            self.nodes.append(
                Node(node_id, self, self.adj_T[node_id], self.nodes_pos[node_id], self.frame_slot[node_id]))

    def run(self):
        self.env.run(until=self.env.now + self.sim_time)
        for node_id in self.src:
            self.nodes[node_id].src_run()

    def reset(self):
        while True:
            if self.update_done or self.episode == 0:
                for node in self.nodes:
                    node.mac.queues = []
                    node.send_cnt = 0
                    node.recv_cnt = 0
                    node.recv_for_me = []
                    node.has_recv = []
                for key in self.e2ed_delay_plus:
                    self.default_e2ed_delay[key] = np.mean(self.e2ed_delay_plus[key])
                for key in self.e2ed_delay_minus:
                    if self.default_e2ed_delay.get(key):
                        continue
                    self.default_e2ed_delay[key] = np.mean(self.e2ed_delay_minus[key])
                self.can_not_dg = 0
                self.loss_cnt = {}
                self.e2ed_delay = {}
                self.paths_delay = {}  # {'0-3-5-7-9':[5954.83, 4594.83, 5554.83]}
                self.e2ed_delay_plus = {}
                self.e2ed_delay_minus = {}
                self.queue_delay = {}
                self.update_done = False
                print('reset done')
                break

    def delayed_exec(self, delay, func, *args, **kwargs):
        func = ensure_generator(self.env, func, *args, **kwargs)
        start_delayed(self.env, func, delay=delay)

    def update(self):
        for node in self.nodes:
            if node.node_id in [10, 11, 14, 15, 16, 17]:
                continue

            src_node = node.node_id
            for des_node in node.route_tb.route_vector.keys():
                if node.route_tb.route_vector[des_node] == {}:
                    continue
                e2ed_key = str(src_node) + '-' + str(des_node)
                # print(e2ed_key)
                if self.e2ed_delay_plus.get(e2ed_key):
                    hat_D_plus = np.mean(self.e2ed_delay_plus[e2ed_key]) / dim_reduction_factor
                else:
                    if self.default_e2ed_delay.get(e2ed_key):
                        hat_D_plus = self.default_e2ed_delay[e2ed_key] / dim_reduction_factor
                    else:
                        # print("1", e2ed_key)
                        hat_D_plus = DEFAULT_MAX_DELAY / dim_reduction_factor
                if self.e2ed_delay_minus.get(e2ed_key):
                    hat_D_minus = np.mean(self.e2ed_delay_minus[e2ed_key]) / dim_reduction_factor
                else:
                    if self.default_e2ed_delay.get(e2ed_key):
                        hat_D_minus = self.default_e2ed_delay[e2ed_key] / dim_reduction_factor
                    else:
                        # print("2", e2ed_key)
                        hat_D_minus = DEFAULT_MAX_DELAY / dim_reduction_factor
                u = node.route_tb.vector_u[des_node]
                # print("hat_D_plus-hat_D_minus:", hat_D_plus - hat_D_minus)
                for key in u:
                    u[key] = round(len(u) * (hat_D_plus - hat_D_minus) * u[key] / (2 * delta), 2)
                for key in node.route_tb.route_vector[des_node]:
                    node.route_tb.route_vector[des_node][key] = node.route_tb.route_vector[des_node][key] - eta * u[key]

                # vector normalization
                length = len(node.route_tb.route_vector[des_node])
                min_value = min(node.route_tb.route_vector[des_node].values())
                sum_value = sum(node.route_tb.route_vector[des_node].values())
                for key in node.route_tb.route_vector[des_node]:
                    node.route_tb.route_vector[des_node][key] = (node.route_tb.route_vector[des_node][
                                                                     key] - min_value) / (
                                                                        sum_value + length * (-min_value))
                node.route_tb.route_vector[des_node] = clip_and_normal_pr(node.route_tb.route_vector[des_node], epsilon)

                # update all_cdf [k, loc, scale]
                for path in find_all_paths(self.adj_T, node.node_id, des_node):
                    path_str = '-'.join(map(str, path))
                    # print(path_str, len(self.route_exp_buffer.get_experiences(path_str)))
                    try:
                        node.route_tb.all_paths_cdf[path_str] = data_to_cdf(self.route_exp_buffer.get_experiences(path_str))
                    except:
                        pass
            # print(node.route_tb.route_vector, "\n")
            # print(node.route_tb.all_paths_cdf)
            node.route_tb.sample_vector_u()
        self.update_done = True
        # print("sim.update() end!")

    def get_avg_queue_delay(self, queue_delay):
        print("\nCalculating Avg queue delay...")
        delay = {}
        for key in queue_delay:
            delay[key] = round(np.mean(queue_delay[key])/1000, 2)
        print(delay)

    def get_avg_e2ed_delay(self, e2ed_delay):
        print("\nCalculating Avg E2ED Delay...")
        avg_e2ed_delay = {}
        for key in e2ed_delay:
            # print(key, round(np.mean(e2ed_delay[key]), 2))
            avg_e2ed_delay[key] = round(np.mean(e2ed_delay[key])/1000, 2)
            if self.episode == 50:
                print(key, e2ed_delay[key])
        print(avg_e2ed_delay)

    def get_worst_case_e2ed_delay(self, e2ed_delay):
        print("\nCalculating Worst Case E2ED Delay...")
        worst_case_e2ed_delay = {}
        for key in e2ed_delay:
            if key[0] in ['0', '1', '2']:
                # print(key, round(np.max(e2ed_delay[key]), 2))
                worst_case_e2ed_delay[key] = round(np.max(e2ed_delay[key])/1000, 2)
        print(max(worst_case_e2ed_delay.items(), key=lambda item: item[1]))
