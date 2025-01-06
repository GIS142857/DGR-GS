import random
import time

from route import RouteTb
from packet import Packet
from GeneralTopology.util.utils import *
from GeneralTopology.NET.MacLayer import MacLayer
from GeneralTopology.NET.PhyLayer import PhyLayer
from GeneralTopology.NET.Pdu import PDU
from config import *


class Node:
    def __init__(self, node_id, sim, neighbors_id_list, pos, time_slot):
        self.node_id = node_id
        self.sim = sim
        self.neighbors_id_list = neighbors_id_list
        self.pos = pos
        self.time_slot = time_slot
        self.node_start_time = self.sim.env.now
        self.isSrc = (True if self.node_id in self.sim.src else False)
        self.isDes = (True if self.node_id in self.sim.des else False)
        self.route_tb = RouteTb(self)
        self.mac = MacLayer(self)
        self.phy = PhyLayer(self)
        self.send_cnt = 0
        self.recv_cnt = 0
        self.recv_for_me = []
        self.has_recv = []
        if self.node_id not in self.sim.des:
            print("route_vector: ", self.node_id, self.route_tb.route_vector)
        if self.node_id in [1, 2, 3, 4, 5, 6]:
            print("vector_u: ", self.node_id, self.route_tb.vector_u)

    @property
    def now(self):
        return self.sim.env.now

    def delayed_exec(self, delay, func, *args, **kwargs):
        return self.sim.delayed_exec(delay, func, *args, **kwargs)

    def src_run(self):

        self.node_start_time = self.now
        self.sim.env.process(self.src_generate_packets())

    def src_generate_packets(self):
        duration = 2.5*UNIT
        while self.sim.env.now < self.node_start_time + duration:

            if self.sim.env.now < self.node_start_time + duration / 2:
                stage = 'r_i+delta_u'
            else:
                stage = 'r_i-delta_u'

            if self.node_id == 0:
                des_node_id = FLOW_DICT[0]
                flow_type = 'flow1'
                priority = 1
                interval = self.sim.arrival_rate[flow_type]
            elif self.node_id == 1:
                des_node_id = FLOW_DICT[1]
                flow_type = 'flow2'
                priority = 1
                # interval = random.expovariate(1 / self.sim.arrival_rate)
                interval = self.sim.arrival_rate[flow_type]
            else:
                des_node_id = FLOW_DICT[2]
                flow_type = 'flow3'
                priority = 1
                # interval = 2 * self.sim.arrival_rate if random.random() < 3 / 8 else 2 / 5 * self.sim.arrival_rate
                interval = self.sim.arrival_rate[flow_type]
            yield self.sim.env.timeout(interval)
            self.send_cnt += 1
            packet_id = str(self.node_id) + '_' + str(self.send_cnt) + '_' + str(des_node_id) + '_' + str(self.sim.episode)
            packet = Packet(self.node_id, des_node_id, packet_id, 'data', flow_type, stage, priority, PACKET_LENGTH,
                            self.now)
            packet.arrival_time[self.node_id] = round(self.now, 2)
            packet.in_queue_time[self.node_id] = round(self.now, 2)
            next_node_id = self.get_next_node(packet, stage)
            mac_pdu = PDU(self.mac.layer_name,
                          packet.length + MAC_HEADER_LENGTH,
                          'data',
                          self.node_id,
                          next_node_id,
                          packet)
            self.mac.addPdu(mac_pdu, flow_type)

            if len(self.mac.flow1_queue) + len(self.mac.flow2_queue) + len(self.mac.flow3_queue) == 1:
                self.sim.env.process(self.mac.process_mac_queue())

    def get_next_node(self, packet, stage):
        # start_time = time.time()
        # print("start:", start_time)
        if packet.des_node_id in self.neighbors_id_list:
            return packet.des_node_id
        route_vector = self.route_tb.route_vector[packet.des_node_id]
        vector_u = self.route_tb.vector_u[packet.des_node_id]
        route = {}
        if stage == 'r_i+delta_u':
            for key, val in route_vector.items():
                route[key] = val + delta * vector_u[key]
        else:
            for key, val in route_vector.items():
                route[key] = val - delta * vector_u[key]
        if self.sim.episode > 0:
            remain_time = DEADLINE[packet.flow_type] - packet.arrival_time[self.node_id] + packet.create_time
            if remain_time <= 0:
                return -1
            r_j = self.get_r_j(remain_time, self.route_tb.all_paths_cdf, packet)
            if sum(r_j.values()) == 0:
                self.sim.can_not_dg += 1  # can't guarantee
            else:
                for key in route:
                    route[key] = route[key] * r_j[key]
        paths = list(route.keys())
        probabilities = list(route.values())
        selected_path = random.choices(paths, weights=probabilities, k=1)[0]
        paths_nodes = selected_path.split('-')
        next_node_id = int(paths_nodes[1])
        # print("end:", time.time())
        return next_node_id

    def on_receive_packet(self, packet):

        packet.arrival_time[self.node_id] = round(self.now, 2)
        packet.in_queue_time[self.node_id] = round(self.now, 2)
        if packet.id in self.has_recv:
            return
        self.has_recv.append(packet.id)
        packet.trans_path.append(self.node_id)

        if packet.des_node_id != self.node_id:
            next_node_id = self.get_next_node(packet, packet.stage)
            if next_node_id != -1:
                mac_pdu = PDU(self.mac.layer_name,
                              packet.length + MAC_HEADER_LENGTH,
                              'data',
                              self.node_id,
                              next_node_id,
                              packet)
                self.mac.addPdu(mac_pdu, mac_pdu.payload.flow_type)

                if len(self.mac.flow1_queue) + len(self.mac.flow2_queue) + len(self.mac.flow3_queue) == 1:
                    self.sim.env.process(self.mac.process_mac_queue())
            else:
                if self.sim.loss_cnt.get(packet.flow_type):
                    self.sim.loss_cnt[packet.flow_type] += 1
                else:
                    self.sim.loss_cnt[packet.flow_type] = 1

                des_node_id = packet.des_node_id
                deadline = DEADLINE[packet.flow_type]
                for i in range(len(packet.trans_path)):
                    src_node_id = packet.trans_path[i]
                    path_key = '-'.join(map(str, packet.trans_path[i:]))
                    e2ed_key = str(src_node_id) + '-' + str(des_node_id)
                    consume_time = packet.arrival_time[src_node_id] - packet.create_time
                    e2ed_delay = round(deadline - consume_time, 2)

                    if self.sim.paths_delay.get(path_key):
                        self.sim.paths_delay[path_key].append(e2ed_delay)
                    else:
                        self.sim.paths_delay[path_key] = [e2ed_delay]

                    if packet.stage == 'r_i+delta_u':
                        if self.sim.e2ed_delay_plus.get(e2ed_key):
                            self.sim.e2ed_delay_plus[e2ed_key].append(e2ed_delay)
                        else:
                            self.sim.e2ed_delay_plus[e2ed_key] = [e2ed_delay]
                    else:
                        if self.sim.e2ed_delay_minus.get(e2ed_key):
                            self.sim.e2ed_delay_minus[e2ed_key].append(e2ed_delay)
                        else:
                            self.sim.e2ed_delay_minus[e2ed_key] = [e2ed_delay]

                    if e2ed_key in ['0-15', '1-16', '2-17']:
                        if self.sim.e2ed_delay.get(e2ed_key):
                            self.sim.e2ed_delay[e2ed_key].append(e2ed_delay)
                        else:
                            self.sim.e2ed_delay[e2ed_key] = [e2ed_delay]

        else:
            self.recv_for_me.append(packet.id)
            self.recv_cnt += 1
            des_node_id = packet.trans_path[-1]
            packet.out_queue_time[self.node_id] = round(self.now, 2)
            # print(packet.flow_type, packet.in_queue_time, packet.out_queue_time)
            queue_delay = []
            for key in packet.in_queue_time:
                queue_delay.append(packet.out_queue_time[key]-packet.in_queue_time[key])
            if self.sim.queue_delay.get(packet.flow_type):
                self.sim.queue_delay[packet.flow_type].append(np.mean(queue_delay))
            else:
                self.sim.queue_delay[packet.flow_type] = [np.mean(queue_delay)]
            # print(packet.arrival_time[des_node_id]-packet.arrival_time[packet.trans_path[0]])
            for i in range(len(packet.trans_path) - 1):
                src_node_id = packet.trans_path[i]
                path_key = '-'.join(map(str, packet.trans_path[i:]))
                e2ed_key = str(src_node_id) + '-' + str(des_node_id)
                e2ed_delay = round(packet.arrival_time[des_node_id] - packet.arrival_time[src_node_id], 2)

                if self.sim.paths_delay.get(path_key):
                    self.sim.paths_delay[path_key].append(e2ed_delay)
                else:
                    self.sim.paths_delay[path_key] = [e2ed_delay]

                if packet.stage == 'r_i+delta_u':
                    if self.sim.e2ed_delay_plus.get(e2ed_key):
                        self.sim.e2ed_delay_plus[e2ed_key].append(e2ed_delay)
                    else:
                        self.sim.e2ed_delay_plus[e2ed_key] = [e2ed_delay]
                else:
                    if self.sim.e2ed_delay_minus.get(e2ed_key):
                        self.sim.e2ed_delay_minus[e2ed_key].append(e2ed_delay)
                    else:
                        self.sim.e2ed_delay_minus[e2ed_key] = [e2ed_delay]

                if e2ed_key in ['0-15', '1-16', '2-17']:
                    if self.sim.e2ed_delay.get(e2ed_key):
                        self.sim.e2ed_delay[e2ed_key].append(e2ed_delay)
                    else:
                        self.sim.e2ed_delay[e2ed_key] = [e2ed_delay]

    def get_r_j(self, remain_time, all_paths_cdf, packet):
        r_j = {}
        des_node_id = packet.des_node_id
        des_key = '-' + str(des_node_id)
        for key, val in all_paths_cdf.items():
            if des_key in key:
                k, loc, scale = val[0], val[1], val[2]
                worst_delay = gamma.ppf(1 - 1e-5, k, loc=loc, scale=scale)
                # print(worst_delay)
                r_j[key] = 1 if worst_delay < remain_time else 0
        return r_j
