import random
import numpy as np

# replay memory 用于存储训练数据
class ReplayMemory:
    def __init__(self, capacity):
        # capacity为buffer的容量
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    # 将数据存入buffer
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        # 这里的done表示是否是终止状态
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    # 从buffer中随机采样batch_size个数据
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
