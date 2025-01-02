from GeneralTopology.NET.Pdu import PDU
from GeneralTopology.util.model_config import *

class MacLayer:
    def __init__(self, node):
        self.layer_name = 'mac'
        self.node = node
        self.queue = []
        self.total_tx_broadcast = 0
        self.total_tx_unicast = 0
        self.total_rx_broadcast = 0
        self.total_rx_unicast = 0
        self.total_retransmit = 0
        self.total_ack = 0
        self.has_reces = []

    def addPdu(self, mac_pdu, flow_type):
        self.queue.append(mac_pdu)

    def process_mac_queue(self):
        while len(self.queue) > 0:
            send_slot = self.node.time_slot
            wait_time = (self.node.sim.env.now - self.node.sim.start_time) % (self.node.sim.slot_duration * send_slot)
            interval = self.node.sim.slot_duration * send_slot - wait_time
            yield self.node.sim.env.timeout(interval)
            if len(self.queue) > 0:
                trans_pdu = self.queue[0]
                trans_pdu.payload.out_queue_time[self.node.node_id] = self.node.now
                self.node.phy.send_pdu(trans_pdu)
                yield self.node.sim.env.timeout(self.node.sim.slot_duration)

    def on_receive_pdu(self, pdu):
        if pdu.pdu_dst == self.node.node_id:
            if pdu.payload.id in self.has_reces:
                return
            self.has_reces.append(pdu.payload.id)
            self.node.on_receive_packet(pdu.payload)
            last_node = pdu.pdu_src
            if len(self.node.sim.nodes[last_node].mac.queue) > 0:
                self.node.sim.nodes[last_node].mac.queue.pop(0)

