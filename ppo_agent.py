import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class PPOAgent(nn.Module):
    def __init__(self, state_size, action_size, device='cpu'):
        super(PPOAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # Define the policy network (simple MLP example, can be optimized)
        self.policy_net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )

        # Define the value network
        self.value_net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    # def act(self, state):
    #     """
    #     Generate an action based on the current state.
    #     """
    #     print(f"State data type before conversion: {type(state)}")
    #     state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    #     action_probs = self.policy_net(state)
    #     dist = Categorical(action_probs)
    #     # if np.random.rand() < self.epsilon:  # 这里的self.epsilon初始设为较高值，如0.5，后面逐渐降低
    #     if np.random.rand() <0.5:
    #         action = dist.sample()
    #     else:
    #         action = torch.argmax(action_probs, dim=1)
    #     return action.item(), dist.log_prob(action).item()

    def act(self, state):
            """
            Generate an action based on the current state.
            """
            print(f"State data type before conversion: {type(state)}")
            
            # 确保 state 的形状为 [1, state_size]
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            elif isinstance(state, torch.Tensor):
                state = state.squeeze(0).to(self.device)  # 去除多余的 batch 维度
            
            print(f"State shape after conversion: {state.shape}")
            
            # 获取动作概率分布
            action_probs = self.policy_net(state).squeeze(0)
            print(f"Action probabilities shape: {action_probs.shape}")
            
            dist = Categorical(action_probs)  # 去除 batch 维度
            
            epsilon = 0.1  # 随着训练逐渐减少epsilon
            if np.random.rand() < epsilon:
                print(f"Sampled action shape: {action.shape}")
            else:
                action = torch.argmax(action_probs, dim=0) # 选择最大概率的动作，并去除 batch 维度
                print(f"Max action shape: {action.shape}")
            
            # 确保 action 是标量
            action = action.item()
            log_prob = dist.log_prob(torch.tensor(action).to(self.device)).item()
            
            return action, log_prob

    def evaluate(self, states, actions):
        """
        Evaluate state values and action log probabilities, along with entropy.
        """
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)

        action_probs = self.policy_net(states)
        dist = Categorical(action_probs)

        action_log_probs = dist.log_prob(actions.squeeze(1))
        entropy = dist.entropy()

        values = self.value_net(states).squeeze(1)

        return action_log_probs, values, entropy.mean()

    def update(self, states, actions, old_action_log_probs, advantages, clip_param=0.2):
        """
        Update the policy and value networks using the PPO algorithm.
        """
        # Convert advantages to a torch tensor if it's a list or NumPy array
        if isinstance(advantages, (np.ndarray, list)):
            advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        elif isinstance(advantages, torch.Tensor):
            advantages = advantages.to(self.device)
        else:
            raise TypeError("Advantages must be a list, NumPy array, or torch.Tensor")

        # Ensure old_action_log_probs is a list, NumPy array, or torch.Tensor
        if isinstance(old_action_log_probs, (np.ndarray, list)):
            old_action_log_probs = torch.tensor(old_action_log_probs, dtype=torch.float32, device=self.device)
        elif isinstance(old_action_log_probs, torch.Tensor):
            old_action_log_probs = old_action_log_probs.to(self.device)
        else:
            raise TypeError("old_action_log_probs must be a list, NumPy array, or torch.Tensor")

        # Evaluate current policy
        action_log_probs, values, entropy = self.evaluate(states, actions)

        # Calculate the ratio (pi_theta / pi_theta_old)
        ratios = torch.exp(action_log_probs - old_action_log_probs)

        # Calculate surrogate losses
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * advantages

        # Calculate actor and critic losses
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(values, advantages.detach())

        # Total loss
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.optimizer.step()

    def save_model(self, file_path):
        """
        Save the policy and value networks' state_dict to a file.
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, file_path)

    def load_model(self, file_path):
        """
        Load the policy and value networks' state_dict and optimizer state_dict from a file.
        """
        checkpoint = torch.load(file_path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])