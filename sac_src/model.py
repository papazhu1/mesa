import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# Q网络结构是两个三层的全连接网络，输入是状态和动作，输出是Q值
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


# SAC支持两个网络，一个是确定性策略网络，一个是高斯策略网络
# 高斯策略通过神经网络生成动作的均值和标准差，接着从这个分布中采样动作，并对其进行缩放和约束。
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        # linear1: 这是输入层，用于将状态映射到隐藏维度。

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear4 = nn.Linear(hidden_dim, hidden_dim)

        # mean_linear: 用于输出动作均值的线性层。
        # log_std_linear: 用于输出动作的标准差（以log形式）的线性层。

        # 标准差必须是正数，但神经网络可以输出任何实数值。
        # 如果直接输出标准差，可能会出现负值，违反标准差的定义。
        # 通过输出对数标准差 log_std，然后通过 exp(log_std) 将其转换为标准差
        # 确保标准差始终为正数（因为指数函数的输出范围是正数）。
        # 这是处理标准差约束的一种简单、有效的方法。
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        # 如果 action_space 存在，action_scale 和 action_bias 会根据 action_space 的上下限进行计算
        # 用来后续将网络输出的动作值缩放到合适范围。否则，将默认缩放值设为 1，偏移值设为 0。
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    # 调用 forward 方法获取动作均值和对数标准差。
    def forward(self, state):
        x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        # x = F.relu(self.linear4(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        # clamp将对数标准差限制在一个范围内，防止过大或过小
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    # 采样函数，用于从策略网络中采样动作
    # 调用 forward 方法获取动作均值和对数标准差。
    # 通过对数标准差的指数函数获取标准差 std。
    # 使用均值 mean 和标准差 std 定义一个正态分布 normal
    # 并从中进行再参数化采样 rsample（即从标准正态分布采样并乘以标准差，再加上均值，这就是重参数化技巧）。
    # 通过 tanh 函数约束采样结果 x_t，确保动作的输出在 (-1, 1) 的范围内。
    # 通过 action_scale 和 action_bias 对动作进行缩放和偏移，生成最终动作。
    # 计算采样的对数概率 log_prob，并进行一些修正以确保动作在边界范围内。
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        # 使用均值 mean 和标准差 std 定义一个正态分布 normal
        normal = Normal(mean, std)

        # 用这个正态分布来采样，是和网络无关的采样过程
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))

        # 生成一个（-1，1）范围内的动作
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
