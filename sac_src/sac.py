import os
import torch
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from sac_src.utils import soft_update, hard_update
from sac_src.model import GaussianPolicy, QNetwork, DeterministicPolicy

# 软演员-评论家算法
# AC（Actor-Critic）通常使用确定性策略，即 Actor 网络直接输出一个确定性的动作。
# 这在连续动作空间中意味着每个状态都有一个固定的动作输出。
# 虽然也有随机版本的 Actor-Critic 算法（例如 A3C），但经典的 AC 算法多为确定性策略。
# SAC（Soft Actor-Critic）：
# 使用随机策略，即 Actor 网络输出一个动作分布（通常是高斯分布），并从中采样动作。
# SAC 强调最大化策略的熵，这意味着它鼓励智能体保持一定的随机性和探索性，以避免陷入局部最优解。
class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        # gamma：折扣因子，用于调整未来奖励的折扣。
        # tau：软更新的步长，控制目标网络的更新速度。
        # alpha：熵系数，控制探索的强度。
        # action_space：动作空间，用于确定策略网络输出的维度。
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.action_space = action_space
        self.learning_rate = args.lr

        # SAC中的actor是一个策略网络，用于生成动作。有两种策略网络，一种是确定性策略网络，另一种是高斯策略网络。
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu") 

        # SAC中的critic是Q网络，用于评估状态-动作对的价值。
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        self.policy_scheduler = lr_scheduler.StepLR(self.critic_optim, step_size=args.lr_decay_steps, gamma=args.lr_decay_gamma)

    def learning_rate_decay(self, decay_ratio=0.5):
        self.learning_rate = self.learning_rate * decay_ratio
        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.learning_rate)

    # 根据给定状态从策略网络中选择一个动作，动作应该就是mu
    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        # 在训练过程中，SAC 鼓励智能体进行探索，以便更好地学习环境的动态。
        # 因此，策略网络会基于当前状态生成一个随机的动作
        # 该动作是通过策略网络输出的均值和标准差生成的正态分布中采样得到的。
        if eval == False:
            # 策略网络返回的依次是动作、对数概率和均值
            action, _, _ = self.policy.sample(state)

        # 在评估阶段，通常不再需要探索，而是希望智能体执行确定性的动作，即选择当前策略网络认为最优的动作。
        # 这种确定性动作是通过策略网络输出的均值直接得到的，不带有随机性。
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        # 从经验池中采样一个批次
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        # 将经验池中存储的数据转换为 PyTorch 张量
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # 计算目标 Q 值
        with torch.no_grad():
            # 从策略网络中为下一个状态 next_state_batch 采样动作 next_state_action
            # 并获取该动作的 log 概率 next_state_log_pi。
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        self.policy_scheduler.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, actor_path, critic_path):
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
    
    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

