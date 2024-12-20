from collections import defaultdict
from dqnEWC_agent import Agent
import numpy as np
import copy

class NODE:
    def __init__(self, id, action_dim, action_list, device, episodes, sourPackets, sources):
        #print('node_id', id, 'action_dim', action_dim)
        self.hop = []
        self.id = id
        if id == 0:
            self.hop.append(0)
        self.worst_to_des = 1000000  # key: k
        self.min_path = defaultdict(dict)  # key: k
        self.all_worst_to_des = []  # key: k + h
        self.nb_delay = defaultdict(dict)  # key: nb + k
        self.nb_delay_pro = defaultdict(dict)  # key: nb + k
        self.nb_worst_delay = defaultdict(dict)  # key: nb + k
        self.pdr = 0
        self.state_dim = 4 + action_dim #### D, k, src, des, numPacketNb
        self.action_dim = action_dim
        #self.D_list = []
        self.action_list = action_list
        self.device = device
        self.episodes = episodes
        self.num = 1
        self.packets = []
        # self.ddls = []
        # self.arrivalDDLs = []
        # self.paths = []
        # self.slots = []
        self.sourPackets = sourPackets
        self.sources = sources
        self.minBandwidth = 50
        self.maxBandwidth = 100
        self.bandwidth = np.random.randint(self.minBandwidth, self.maxBandwidth) * 10000

        self.agent = Agent(self.state_dim, self.action_dim, self.action_list, self.id, device, episodes)   ### state_dim, action_dim, action_list

    def clean(self):
        self.packets = []
        # self.slots = []
        # self.paths = []
        # self.ddls = []
        # self.arrivalDDLs = []

    def set_pdr(self, pdr):
        self.pdr = pdr

    # def set_d_list(self, D_list):
    #     self.D_list = D_list

    # def add_worst_to_des(self, worst_to_des):
    #     for d in self.D_list:
    #         if worst_to_des == d:
    #             return
    #     self.D_list.append(worst_to_des)

    def add_hop(self, hop):
        for i in range(len(self.hop)):
            if self.hop[i] == hop:
                return
        self.hop.append(hop)

    def addPacketFirst(self, newPacket):
        #print('addpacket', self.id)
        #print('bf_add_packets', self.packets)
        packet = copy.deepcopy(newPacket)
        old_len = len(self.packets)
        #packet.setPath(path)
        if len(self.packets) == 0:
            self.packets.append(packet)
            # self.ddls.append(D)
            # self.arrivalDDLs.append(D)
            # self.paths.append(path)
            # self.slots.append(slot)
        else:
            temp = []
            ddl_temp = []
            Addl_temp = []
            path_temp = []
            slot_temp = []
            for i in range(len(self.packets)):
                if self.packets[i].type > packet.type:
                    temp.append(packet)
                    # ddl_temp.append(D)
                    # Addl_temp.append(D)
                    # path_temp.append(path)
                    # slot_temp.append(slot)
                    for j in range(i, len(self.packets)):
                        temp.append(self.packets[j])
                        # ddl_temp.append(self.ddls[j])
                        # Addl_temp.append(self.arrivalDDLs[j])
                        # path_temp.append(self.paths[j])
                        # slot_temp.append(self.slots[j])
                    self.packets = temp
                    # self.ddls = ddl_temp
                    # self.arrivalDDLs = Addl_temp
                    # self.paths = path_temp
                    # self.slots = slot_temp
                    return
                else:
                    temp.append(self.packets[i])
                    # ddl_temp.append(self.ddls[i])
                    # Addl_temp.append(self.arrivalDDLs[i])
                    # path_temp.append(self.paths[i])
                    # slot_temp.append(self.slots[i])
            if len(self.packets) == old_len:
                self.packets.append(packet)
                #self.packets.append(flow)
                # self.ddls.append(D)
                # self.arrivalDDLs.append(D)
                # self.paths.append(path)
                # self.slots.append(slot)

        #print('packets', self.packets)
        #print('ddls', self.ddls)
        #print('paths', self.paths)
        #print('slots', self.slots)

    def addPacket(self, newPacket, D, slot, dess):
        #print('addpacket', self.id)
        #print('bf_add_packets', self.packets)
        old_len = len(self.packets)
        packet = copy.deepcopy(newPacket)
        packet.flag = 0
        packet.setParatemeters(D, self.id, slot)
        #rint('packet_len', self.getPacketlen())
        if self.getPacketlen(dess) == 0:
            self.packets.append(packet)
            # self.packets.append(flow)
            # self.ddls.append(D)
            # self.arrivalDDLs.append(D)
            # self.paths.append(path)
            # self.slots.append(slot)
        else:
            headPacket = self.getHeadPacket() #self.packets[0]
            #print('headPacket', headPacket)
            #headSlot = self.packets[0].slot
            temp = []
            # ddl_temp = []
            # Addl_temp = []
            # path_temp = []
            # slot_temp = []
            if packet.type < headPacket.type:
                if headPacket.type == self.sourPackets[self.id]:
                    if headPacket.slot < slot:
                        temp.append(packet)
                        # ddl_temp.append(D)
                        # Addl_temp.append(D)
                        # path_temp.append(path)
                        # slot_temp.append(slot)
                        for j in range(len(self.packets)):
                            temp.append(self.packets[j])
                            # ddl_temp.append(self.ddls[j])
                            # Addl_temp.append(self.arrivalDDLs[j])
                            # path_temp.append(self.paths[j])
                            # slot_temp.append(self.slots[j])
                        self.packets = temp
                        # self.ddls = ddl_temp
                        # self.arrivalDDLs = Addl_temp
                        # self.paths = path_temp
                        # self.slots = slot_temp
                        return
                else:
                    if headPacket.slot == slot:
                        temp.append(packet)
                        # temp.append(flow)
                        # ddl_temp.append(D)
                        # Addl_temp.append(D)
                        # path_temp.append(path)
                        # slot_temp.append(slot)
                        for j in range(len(self.packets)):
                            temp.append(self.packets[j])
                            # ddl_temp.append(self.ddls[j])
                            # Addl_temp.append(self.arrivalDDLs[j])
                            # path_temp.append(self.paths[j])
                            # slot_temp.append(self.slots[j])
                        self.packets = temp
                        # self.ddls = ddl_temp
                        # self.arrivalDDLs = Addl_temp
                        # self.paths = path_temp
                        # self.slots = slot_temp
                        #print('af_add_packets', self.packets)
                        return
            #print()
            temp.append(self.packets[0])
            # ddl_temp.append(self.ddls[0])
            # Addl_temp.append(self.arrivalDDLs[0])
            # path_temp.append(self.paths[0])
            # slot_temp.append(self.slots[0])
            for i in range(1, len(self.packets)):
                if self.packets[i].type > packet.type:
                    temp.append(packet)
                    # temp.append(flow)
                    # ddl_temp.append(D)
                    # Addl_temp.append(D)
                    # path_temp.append(path)
                    # slot_temp.append(slot)
                    for j in range(i, len(self.packets)):
                        temp.append(self.packets[j])
                        # ddl_temp.append(self.ddls[j])
                        # Addl_temp.append(self.arrivalDDLs[j])
                        # path_temp.append(self.paths[j])
                        # slot_temp.append(self.slots[j])
                    self.packets = temp
                    # self.ddls = ddl_temp
                    # self.arrivalDDLs = Addl_temp
                    # self.paths = path_temp
                    # self.slots = slot_temp
                    return
                else:
                    temp.append(self.packets[i])
                    # ddl_temp.append(self.ddls[i])
                    # Addl_temp.append(self.arrivalDDLs[i])
                    # path_temp.append(self.paths[i])
                    # slot_temp.append(self.slots[i])
            if len(self.packets) == old_len:
                self.packets.append(packet)
                # self.ddls.append(D)
                # self.arrivalDDLs.append(D)
                # self.paths.append(path)
                # self.slots.append(slot)

    def getPacket(self, current_slot, dess):
        # print('getPacket')
        # print('current_slot', current_slot)
        for i in range(0, len(self.packets)):
            #print('id', self.packets[i].id, 'packet_slot', self.packets[i].slot, 'flag', self.packets[i].flag)
            if self.packets[i].flag == 0 and self.packets[i].slot <= current_slot and dess[self.packets[i].type] != self.id:
                #print('packet_id', self.packets[i].id)
                return self.packets[i]
        return None

    def getPacketlen(self, dess):
        sum = 0
        for i in range(len(self.packets)):
            if self.packets[i].flag != 1 and dess[self.packets[i].type] != self.id:
                sum += 1
        return sum

    def getHeadPacket(self):
        #print('getHeadPacket', 'len', len(self.packets))
        for i in range(len(self.packets)):
            if self.packets[i].flag == 0:
               # print('type', self.packets[i].type)
                return self.packets[i]

    def addSample(self, packet, queue_delay, normal):
        #print('node addSample', self.id, 'packet_id', packet.id)
        for i in range(len(self.packets)):
            if self.packets[i].id == packet.id:
                state = self.packets[i].state
                action = self.packets[i].action
                action_index = self.packets[i].action_index
                reward = self.packets[i].reward + queue_delay
                #print('sample_queue_delay', queue_delay)
                # print('state', state)
                # print('action', action)
                next_state = self.packets[i].next_state
                #print('next_state', next_state)
                next_state[0] = (next_state[0] * normal[0] - queue_delay) / normal[0]
                done = self.packets[i].done
                sample = (state, action, action_index, reward, next_state, done)
                #print('sample', sample)
                key = str(state) + ',' + str(action)
                self.agent.replay_memory.add(key, sample)
                self.popPacket(packet)
                #print('after sample', self.id)
               # self.printPacket()
                return

    def popPacket(self, packet):
        temp = []
        for i in range(0, len(self.packets)):
            if self.packets[i].id == packet.id:
                for j in range(i+1, len(self.packets)):
                    temp.append(self.packets[j])
                break
            else:
                temp.append(self.packets[i])
        self.packets = temp

    def setPacketFlag(self, packet):
        for i in range(len(self.packets)):
            if self.packets[i].id == packet.id:
                self.packets[i].flag = 1

    def setddl(self, delay, current_slot, dess):
        # print('node', self.id, 'setddl')
        # print('slots', self.slots)
        # print('ddls', self.ddls)
        for i in range(len(self.packets)):
            if self.packets[i].flag == 0 and dess[self.packets[i].type] != self.id:
                self.packets[i].reduceDDL(delay, current_slot)
        # for i in range(len(self.ddls)):
        #     #self.ddls[i] -= delay
        #     #sour_packet = self.getSourPacket()
        #     if self.slots[i] <= current_slot:
        #         self.ddls[i] -= delay
            # elif self.slots[i] == current_slot and self.sources[self.packets[i]] == self.id:
            #     self.ddls[i] -= delay
        #print('after_ddl', self.ddls)


    def setslot(self, slots):
        for i in range(len(self.packets)):
            self.packets[i].reduceSlot(slots)


    def setChilds(self, num, child_list):
        #print('setChilds')
        self.action_dim = num
        self.action_list = child_list
        #print('num', num)
        state_dim = 4 + num
        self.state_dim = state_dim
        self.agent = Agent(self.state_dim, self.action_dim, self.action_list, self.id, self.device, self.episodes)


    def getPacketsLen(self, packet_flow, current_slot, dess):
        sum = 0
        for i in range(len(self.packets)):
            if self.packets[i].type <= packet_flow and self.packets[i].slot <= current_slot and self.packets[i].flag == 0 and dess[self.packets[i].type] != self.id:
                sum += 1
        return sum

    def get_ete(self, flow, D):
        sum_delay = 0
        numPacket = 0
        for i in range(len(self.packets)):
            if self.packets[i].type == flow:
                numPacket += 1
                sum_delay += D - self.packets[i].reddl
        #print('numPacket ', numPacket )
        if numPacket == 0:
            return 0
        else:
            return sum_delay / numPacket

    def printPacket(self):
        for i in range(len(self.packets)):
            print('id', self.packets[i].id, 'type', self.packets[i].type, self.packets[i].reddl, self.packets[i].arrivalD, self.packets[i].path, self.packets[i].slot, 'flag', self.packets[i].flag)
            print(self.packets[i].state, self.packets[i].action, self.packets[i].reward, self.packets[i].next_state, self.packets[i].done)

class Packet:
    def __init__(self, id, type, arrivalD, arrivalSlot, start_node):
        self.id = id
        self.type = type
        self.arrivalD = arrivalD
        self.reddl = arrivalD
        self.slot = arrivalSlot
        self.state = []
        self.action = -1
        self.action_index = -1
        self.reward = 0
        self.next_state = []
        self.done = 0
        self.flag = 0
        self.path = []
        self.path.append(start_node)

    def setParatemeters(self, D, next_node, slot):
        self.reddl = D
        self.arrivalD = D
        self.path.append(next_node)
        self.slot = slot

    def reduceDDL(self, delay, current_slot):
        #print('packet_id', self.id, 'slot', self.slot, 'current_slot', current_slot)
        if self.slot <= current_slot:
            self.reddl -= delay
        #self.reddl -= delay

    def reduceSlot(self, slots):
        self.slot += slots

    def setState(self, state, action, action_index, reward, next_state, done):
        # print('set', self.id)
        # print('state', state)
        self.state = state
        self.action = action
        self.action_index = action_index
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def setNextState(self, state):
        self.next_state = state