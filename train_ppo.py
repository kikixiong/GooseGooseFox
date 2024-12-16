import gym
from fox_and_goose_env import FoxAndGooseEnv
from ppo_agent import PPOAgent
import numpy as np
import time
import os

def train_ppo(env, fox_agent, goose_agent, num_episodes=1000, max_steps_per_episode=1000, gamma=0.99, win_threshold=0.6, save_interval=100):
    save_dir = "saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 用于记录狐狸和鹅分别与随机对手对战的胜率
    fox_win_rate_vs_random = 0
    goose_win_rate_vs_random = 0

    for episode in range(num_episodes):
        # 决定对手是随机策略还是训练好的智能体
        fox_opponent = 'random' if fox_win_rate_vs_random < win_threshold else goose_agent
        goose_opponent = 'random' if goose_win_rate_vs_random < win_threshold else fox_agent

        # Reset environment for each episode
        state = env.reset()
        done = False
        step = 0

        states = []
        actions = []
        old_action_log_probs = []
        rewards = []

        # Track which agent performed better (fox or goose)
        fox_reward = 0
        goose_reward = 0

        while not done and step < max_steps_per_episode:
            # Fox takes action
            if fox_opponent == 'random':
                fox_action = env.action_space.sample()  # 随机动作
            else:
                fox_action, fox_action_log_prob = fox_agent.act(state)
            next_state, fox_reward, done, _ = env.step(fox_action)
            states.append(state)
            if fox_opponent!= 'random':
                actions.append(fox_action)
                old_action_log_probs.append(fox_action_log_prob)
            rewards.append(fox_reward)
            state = next_state
            step += 1

            # Goose takes action
            if goose_opponent == 'random':
                goose_action = env.action_space.sample()  # 随机动作
            else:
                goose_action, goose_action_log_prob = goose_agent.act(state)
            next_state, goose_reward, done, _ = env.step(goose_action)
            states.append(state)
            if goose_opponent!= 'random':
                actions.append(goose_action)
                old_action_log_probs.append(goose_action_log_prob)
            rewards.append(goose_reward)
            state = next_state
            step += 1

        # Calculate discounted rewards and advantages
        discounted_rewards = []
        Gt = 0
        for r in reversed(rewards):
            Gt = r + gamma * Gt
            discounted_rewards.insert(0, Gt)

        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-10)

        # 根据对手类型决定是否更新智能体
        if fox_opponent!= 'random':
            fox_agent.update(states, actions, old_action_log_probs, discounted_rewards)
        if goose_opponent!= 'random':
            goose_agent.update(states, actions, old_action_log_probs, discounted_rewards)

        # 统计本回合狐狸和鹅的胜负情况，更新胜率
        if fox_reward > goose_reward:
            if fox_opponent == goose_agent:
                fox_win_rate_vs_random += 1
            else:
                fox_win_rate_vs_random = 0 if fox_win_rate_vs_random == 0 else fox_win_rate_vs_random - 1
            if goose_opponent == fox_agent:
                goose_win_rate_vs_random = 0 if goose_win_rate_vs_random == 0 else goose_win_rate_vs_random - 1
            else:
                goose_win_rate_vs_random += 1
        elif goose_reward > fox_reward:
            if goose_opponent == fox_agent:
                goose_win_rate_vs_random += 1
            else:
                goose_win_rate_vs_random = 0 if goose_win_rate_vs_random == 0 else goose_win_rate_vs_random - 1
            if fox_opponent == goose_agent:
                fox_win_rate_vs_random = 0 if fox_win_rate_vs_random == 0 else fox_win_rate_vs_random - 1
            else:
                fox_win_rate_vs_random += 1
        else:
            # 平局情况，适当微调胜率，避免长时间平局导致无法切换对手
            if fox_opponent == goose_agent:
                fox_win_rate_vs_random = max(0, fox_win_rate_vs_random - 0.1)
                goose_win_rate_vs_random = max(0, goose_win_rate_vs_random - 0.1)
            else:
                fox_win_rate_vs_random = min(1, fox_win_rate_vs_random + 0.1)
                goose_win_rate_vs_random = min(1, goose_win_rate_vs_random + 0.1)

        # 标准化胜率，使其在 0 - 1 之间
        fox_win_rate_vs_random = min(1, max(0, fox_win_rate_vs_random / (episode + 1)))
        goose_win_rate_vs_random = min(1, max(0, goose_win_rate_vs_random / (episode + 1)))

        # Print the episode result every 100 episodes
        if episode % 100 == 0:
            print(f'Episode {episode}: Fox Reward = {fox_reward}, Goose Reward = {goose_reward}, '
                  f'Fox Win Rate vs Opponent = {fox_win_rate_vs_random}, '
                  f'Goose Win Rate vs Opponent = {goose_win_rate_vs_random}')

        # 定期保存模型
        if episode % save_interval == 0 and episode > 0:
            fox_agent.save_model(os.path.join(save_dir, f'fox_agent_episode_{episode}.pt'))
            goose_agent.save_model(os.path.join(save_dir, f'goose_agent_episode_{episode}.pt'))

if __name__ == "__main__":
    env = FoxAndGooseEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create both the Fox and Goose agents
    fox_agent = PPOAgent(state_size, action_size)
    goose_agent = PPOAgent(state_size, action_size)

    # Train the agents in a multi-episode loop
    train_ppo(env, fox_agent, goose_agent)