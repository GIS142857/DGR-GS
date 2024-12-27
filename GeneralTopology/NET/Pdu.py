class PDU:
    def __init__(self, layer, nbits, packet_type, pdu_src, pdu_dst, payload):
        self.layer = layer
        self.nbits = nbits
        self.type = packet_type
        self.pdu_src = pdu_src
        self.pdu_dst = pdu_dst
        self.payload = payload