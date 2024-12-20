class PDU:
    def __init__(self, layer, nbits, **fields):
        self.layer = layer
        self.nbits = nbits
        for f in fields:
            setattr(self, f, fields[f])