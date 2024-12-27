class TonPhyLayer:
    '''
        Setting up the physical layer
    '''
    LAYER_NAME = 'phy'
    def __init__(self, node, bitrate=4e6):
        self.node = node
        self.bitrate = bitrate
        self._current_rx_count = 0
        self._channel_busy_start = 0
        self.total_tx = 0
        self.total_rx = 0
        self.total_collision = 0
        self.total_error = 0
        self.total_bits_tx = 0
        self.total_bits_rx = 0
        self.total_channel_busy = 0
        self.total_channel_tx = 0

    def send_pdu(self, pdu):
        # transmit a pdu
        tx_time = pdu.nbits / self.bitrate
        next_node = self.node.sim.nodes[pdu.pdu_dst]
        self.node.delayed_exec(
            1e-8, next_node.phy.on_rx_start, pdu)
        self.node.delayed_exec(
            1e-8 + tx_time, next_node.phy.on_rx_end, pdu)

    def on_tx_start(self, pdu):
        pass

    def on_tx_end(self, pdu):
        pass

    def on_rx_start(self, pdu):
        pass

    def on_rx_end(self, pdu):
        # receive a pdu
        source = pdu.pdu_src
        src_node = self.node.sim.nodes[source]
        per = 0.2
        if self.node.sim.random.random() > per:
            # print('phy transmit sucesslly to', pdu.dst)
            self.node.mac.on_receive_pdu(pdu)
            self.total_rx += 1
            self.total_bits_rx += pdu.nbits
        else:
            self.total_error += 1

    def on_collision(self, pdu):
        pass

    def cca(self):
        """Return True if the channel is clear"""
        return self._current_rx_count == 0