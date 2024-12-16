import matplotlib.pyplot as plt
import torch

def plot_rewards(rewards):
    """
    绘制奖励曲线。
    """
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.show()

def save_model(model, path):
    """
    保存模型到指定路径。
    """
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """
    加载模型参数。
    """
    model.load_state_dict(torch.load(path))
    return model