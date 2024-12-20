import random
from collections import deque
import simpy
from logger import Stat
from pdu import PDU
from configure import *


class DefaultMacLayer:
    LAYER_NAME = 'DefaultMacLayer'
    HEADER_BITS = 12 * 8
    '''
    - 初始化类的属性，如配置字典、节点、传输队列、ack事件、统计信息等。
    - 队列长度随机设置在 `MAC_QUEUE_MIN` 和 `MAC_QUEUE_MAX` 之间。
    - 各种统计信息初始化为 0，如广播和单播的发送和接收数量等。
    '''
    def __init__(self, node, configuration_dict):
        self.configuration_dict =configuration_dict
        self.node = node
        self.tx_queue = deque(maxlen=MAC_QUEUE_MAX_LENGTH)  # 一个使用 deque 实现的有限长度传输队列。存储待发送的数据包，支持先进先出操作，并且有最大长度限制以避免过度占用资源。
        self.ack_event = None  # 用来存储对于已发送数据包的确认事件。在需要等待确认的通信中，管理ACK消息的接收。
        self.stat = Stat()
        self.queue_len = random.randrange(MAC_QUEUE_MIN, MAC_QUEUE_MAX) # 动态调整队列长度可以帮助适应不同的网络负载和优化性能。
        self.stat.total_tx_broadcast = 0  # 广播发送总次数。
        self.stat.total_tx_unicast = 0  # 单播发送总次数。
        self.stat.total_rx_broadcast = 0  # 广播接收总次数。
        self.stat.total_rx_unicast = 0  # 单播接收总次数。
        self.stat.total_retransmit = 0  # 总重传次数。
        self.stat.total_ack = 0  # 接收到的总确认数。
        self.stat.total_tx_loss = 0  # 数据传输丢失的总次数。

    '''
    - 方法 `process_queue` 用于处理MAC层的发送队列。
    - 使用循环从队列中取出帧，并根据帧的目的地址和重试次数进行处理。
    - 如果目标地址不是广播且重试次数为零，计算延时。
    - 基于退避算法计算等待时间。
    - 使用物理层的 `send_pdu` 方法发送帧。
    - 如果目的地址不是广播，处理ack事件，包括设置超时和等待ack。
    - 根据帧的目的地址和是否收到ack调整重试次数或结束发送。
    '''
    def process_queue(self):
        retries = 0
        while self.tx_queue:
            frame = self.tx_queue[0]
            if retries == 0 and frame.dst != BROADCAST_ADDR:
                yield self.node.timeout(self.queue_len * UNIT * 0.001)
            yield self.node.timeout(DIFS_DURATION)
            wait_time = random.randint(0, min(pow(2, retries) * CW_MIN, CW_MAX) - 1)
            while wait_time > 0:
                yield self.node.timeout(SLOT_DURATION)
                if self.node.phy.cca():
                    wait_time -= 1
            self.node.phy.send_pdu(frame)
            if frame.dst != BROADCAST_ADDR and retries < RETRIES_NUM:
                frame.payload.payload.kwargs['retries_sum'] += 1
                self.ack_event = self.node.create_event()
                self.ack_event.wait_for = frame
                yield simpy.AnyOf(self.node.sim.env, [
                    self.node.timeout(self.configuration_dict["ACK_TIMEOUT"]),
                    self.ack_event,
                ])
                if self.ack_event.triggered:
                    retries = 0
                    self.tx_queue.popleft()
                else:
                    retries += 1
            elif frame.dst != BROADCAST_ADDR and retries >= RETRIES_NUM:
                retries = 0
                self.tx_queue.popleft()
            else:
                retries = 0
                self.tx_queue.popleft()
            self.ack_event = None

    '''
    - `send_pdu` 方法用于外部调用，将生成一个MAC层PDU并加入发送队列。
    - 如果队列之前为空，则开始处理队列过程。
    '''
    def send_pdu(self, dst, pdu):
        mac_pdu = PDU(self.LAYER_NAME, pdu.nbits + self.HEADER_BITS,
                      type='data',
                      src=self.node.id,
                      dst=dst,
                      payload=pdu)
        self.tx_queue.append(mac_pdu)
        if len(self.tx_queue) == 1:
            self.node.start_process(self.node.create_process(
                self.process_queue))

    '''
    - `on_receive_pdu` 方法处理接收到的PDU。
    - 如果PDU类型是数据，并且目标地址是广播或节点ID，处理数据接收。
    - 如果是单播数据，发送ack。
    - 更新统计信息。
    '''
    def on_receive_pdu(self, pdu):
        if pdu.type is 'data':
            if pdu.dst == BROADCAST_ADDR or pdu.dst == self.node.id:
                # ack if this is a unicast frame
                # self.node.net.on_receive_pdu(pdu.src, pdu.payload)
                if pdu.dst != BROADCAST_ADDR:
                    yield self.node.timeout(SIFS_DURATION)
                    ack = PDU(self.LAYER_NAME, nbits=self.HEADER_BITS,
                              type='ack',
                              for_frame=pdu, src=self.node.id)
                    self.node.phy.send_pdu(ack)
                    self.stat.total_ack += 1
                    self.stat.total_rx_unicast += 1
                else:
                    self.stat.total_rx_broadcast += 1
                    self.node.net.on_receive_pdu(pdu.src, pdu.payload)
        elif pdu.type == 'ack' and self.ack_event is not None:  # 如果接收到ack，调用 `succeed` 方法来处理成功接收的情况。
            if pdu.for_frame == self.ack_event.wait_for:
                self.ack_event.succeed()
                self.node.sim.nodes[pdu.for_frame.dst].net.on_receive_pdu(pdu.for_frame.src, pdu.for_frame.payload)
