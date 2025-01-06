from pdu import PDU
from config import *


class TonMacLayer:
    def __init__(self, node):
        self.layer = 'mac'
        self.node = node
        self.queues = {'flow1': [], 'flow2': [], 'flow3': []}
        self.ack_event = None
        self.total_tx_broadcast = 0
        self.total_tx_unicast = 0
        self.total_rx_broadcast = 0
        self.total_rx_unicast = 0
        self.total_retransmit = 0
        self.total_ack = 0
        self.backoff = 0
        self.retries = 0
        self.has_reces = []

    def reset(self):
        self.queues = {'flow1': [], 'flow2': [], 'flow3': []}
        self.total_tx_broadcast = 0
        self.total_tx_unicast = 0
        self.total_rx_broadcast = 0
        self.total_rx_unicast = 0
        self.total_retransmit = 0
        self.total_ack = 0
        self.backoff = 0
        self.retries = 0
        self.has_reces = []

    def add_pdu(self, mac_pdu, flow_type):
        mac_pdu.payload.in_queue_time[self.node.id] = round(self.node.now, 2)
        self.queues[flow_type].append(mac_pdu)

    def process_queue(self):
        # Send the packets in the queue in sequence
        while len(self.queues['flow1']) + len(self.queues['flow2']) + len(self.queues['flow3']) > 0:
            send_slot = self.node.time_slot
            wait_time = (self.node.sim.env.now - self.node.sim.start_time) % (self.node.sim.slot_duration * send_slot)
            interval = self.node.sim.slot_duration * send_slot - wait_time
            yield self.node.sim.env.timeout(interval)
            # 从多个优先级队列中选择发送的 PDU
            opt_pdu = []
            if len(self.queues['flow1']) > 0:
                opt_pdu.append(self.queues['flow1'][0])
            if len(self.queues['flow2']) > 0:
                opt_pdu.append(self.queues['flow2'][0])
            if len(self.queues['flow3']) > 0:
                opt_pdu.append(self.queues['flow3'][0])
            max_time = 100 * DEADLINE['flow3']
            if len(opt_pdu) > 0:
                trans_pdu = opt_pdu[0]
                for pdu in opt_pdu:
                    # queue_length = max(len(self.queues[pdu.payload.flow_type]), 1)
                    remain_time = DEADLINE[pdu.payload.flow_type] - self.node.now + pdu.payload.create_time
                    # remain_time = remain_time/queue_length
                    if remain_time < max_time:
                        max_time = remain_time
                        trans_pdu = pdu
                trans_pdu.payload.out_queue_time[self.node.id] = self.node.now
                state = [trans_pdu.payload.des_node_id]
                for node_id in self.node.neighbors:
                    state.append(self.node.nb_pri_queues[node_id][trans_pdu.payload.flow_type])
                while len(state) < 4:
                    state.append(0)
                trans_pdu.payload.state_map[self.node.id] = state
                self.node.phy.send_pdu(trans_pdu)
                yield self.node.sim.env.timeout(self.node.sim.slot_duration)

    def send_pdu(self, packet, pdu_dst):
        # Send a packet to the next node
        packet.arrival_time[self.node.id] = round(self.node.now, 2)
        mac_pdu = PDU(self.layer,
                      packet.length + MAC_HEADER_LENGTH,
                      'data',
                      self.node.id,
                      pdu_dst,
                      packet)
        self.add_pdu(mac_pdu, packet.flow_type)
        if len(self.queues['flow1']) + len(self.queues['flow2']) + len(self.queues['flow3']) == 1:
            self.node.sim.env.process(self.process_queue())

    def on_receive_pdu(self, pdu):
        if pdu.pdu_dst == self.node.id:
            if pdu.payload.id in self.has_reces:
                return
            self.has_reces.append(pdu.payload.id)
            self.node.on_receive_pdu(pdu.payload)
            if len(self.node.sim.nodes[pdu.pdu_src].mac.queues[pdu.payload.flow_type]) > 0:
                self.node.sim.nodes[pdu.pdu_src].mac.queues[pdu.payload.flow_type].pop(0)
                # 模拟广播队列信息
                for node in self.node.sim.nodes:
                    if self.node.id in node.neighbors:
                        node.nb_pri_queues[self.node.id][pdu.payload.flow_type] = len(self.queues[pdu.payload.flow_type])
