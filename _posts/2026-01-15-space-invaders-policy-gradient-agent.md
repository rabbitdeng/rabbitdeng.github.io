---
layout: post
title: "训练了一个会躲在掩体后面peek射击的太空侵略者智能体"
date: 2026-01-15 14:00:00
author: "BunnyRabbit"
tags: [强化学习, 策略梯度, 深度学习, PyTorch, 游戏AI, Atari]
---

## 笑死！我训练了个"怕死"的游戏AI

最近闲着没事，用强化学习的策略梯度算法训练了一个玩《太空侵略者》的AI。结果训练到第5000多回合时，我发现了个特别有趣的现象——这AI居然学会了"战术"！它特别怕死，一进游戏就先找掩体躲着，然后只敢在安全的时候偷偷探出头开一枪，立刻又缩回去，活像个在CSGO里躲在墙角peek的新手。

虽然看起来有点怂，但不得不承认，这AI多少是"开窍"了！它知道活着才能拿更多分数，所以把生存放在第一位。今天就来跟大家唠唠这个"怕死AI"的诞生过程，以及我在训练中遇到的趣事。

## 技术栈选择

作为一个AI爱好者，我选择了最经典的组合：

- **强化学习算法**：策略梯度（REINFORCE）——简单易懂，适合入门
- **深度学习框架**：PyTorch——API友好，调试方便
- **游戏环境**：OpenAI Gym的ALE/SpaceInvaders-v5——经典游戏，环境稳定
- **网络架构**：深度卷积神经网络——专门处理图像输入

## AI的"大脑"设计

AI的核心是一个深度卷积神经网络，负责从游戏画面中提取特征，然后决定该做什么动作。我设计的网络结构不算复杂：

- 5层卷积层，层层递进提取画面特征
- 1层自适应平均池化，压缩数据维度
- 1层全连接层，输出6个动作的概率分布

这个结构能让AI"看懂"游戏画面，知道哪里是敌人，哪里是掩体，哪里是自己的飞船。

## 核心代码分享

### 1. 策略网络实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, output_size):
        super(Policy, self).__init__()
        # 卷积层：提取图像特征
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # 输入3通道（RGB），输出32通道
            nn.MaxPool2d(3, 2),          # 池化层，降低尺寸
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(3, 2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(3, 2),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.MaxPool2d(3, 2),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1, 0),  # 1x1卷积，调整通道数
            nn.ReLU(inplace=True)
        )
        
        # 全局平均池化：把特征图变成向量
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 输出层：6个动作的概率
        self.linear = nn.Linear(256, output_size)
    
    def forward(self, x):
        # 前向传播：输入图像 → 特征提取 → 动作概率
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 展平
        # 用softmax输出概率分布
        return F.softmax(self.linear(x), dim=1)
```

### 2. 策略梯度实现

```python
from torch.distributions import Categorical
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim

class ReinforceAgent:
    def __init__(self, output_size, lr=1e-3):
        self.net = Policy(output_size).to("cuda")  # 模型放GPU
        self.optim = optim.Adam(self.net.parameters(), lr=lr)  # 优化器
        self.scaler = GradScaler()  # 混合精度训练，加速
    
    def select_action(self, state):
        # 预处理：调整图像维度，放到GPU
        state = state.permute(2, 1, 0).unsqueeze(0).to("cuda")
        
        # 混合精度推理
        with autocast(device_type='cuda', dtype=torch.float16):
            probs = self.net(state)
        probs = probs.float()
        
        # 根据概率采样动作
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
    
    def update(self, rewards, log_probs, gamma=0.99):
        # 计算折扣回报：越晚获得的奖励，当前价值越低
        discounted_reward = 0
        loss = 0
        
        # 从后往前计算，更高效
        for i in reversed(range(len(rewards))):
            discounted_reward = rewards[i] + gamma * discounted_reward
            loss += -(log_probs[i] * discounted_reward)
        
        # 反向传播更新网络
        with autocast(device_type='cuda', dtype=torch.float16):
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        self.optim.zero_grad()  # 清空梯度
```

### 3. 训练主循环

```python
import gym
import numpy as np
from transformers import ViTImageProcessor

# 初始化游戏环境
env = gym.make('ALE/SpaceInvaders-v5')
# 使用预训练的图像处理器，把游戏画面缩放到模型需要的尺寸
transform = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# 创建智能体
agent = ReinforceAgent(env.action_space.n)

# 训练参数
num_episodes = 10000  # 训练1万回合
gamma = 0.99          # 折扣因子
best_score = 0        # 最佳得分记录

for episode in range(1, num_episodes + 1):
    state = env.reset()  # 重置环境，开始新回合
    log_probs = []       # 保存每一步的动作概率
    rewards = []         # 保存每一步的奖励
    
    while True:
        # 预处理游戏画面
        state_tensor = torch.tensor(transform(image=state)['image'])
        # 选择动作
        action, log_prob = agent.select_action(state_tensor)
        # 执行动作，获取下一个状态
        next_state, reward, done, _ = env.step(action)
        
        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state
        
        if done:  # 回合结束
            episode_reward = np.sum(rewards)  # 计算总得分
            
            # 保存最佳模型
            if episode_reward > best_score:
                best_score = episode_reward
                torch.save(agent.net.state_dict(), "best_space_invader.pth")
                print(f"🎉 新纪录！得分: {best_score}")
            
            # 更新策略网络
            agent.update(rewards, log_probs, gamma)
            break
    
    # 每100回合打印一次进度
    if episode % 100 == 0:
        print(f"回合: {episode}, 得分: {episode_reward}, 最佳: {best_score}")
```

## AI的成长历程

看着AI从啥都不会到学会"战术"，就像养了个电子宠物一样有趣。它的成长大概可以分为三个阶段：

### 1. 懵懂探索期（0-1000回合）
一开始AI完全是个"愣头青"，动作全是随机的，要么在那发呆不动，要么疯狂按开火键，经常还没碰到敌人就死了，平均得分只有20-30分。

### 2. 逐渐适应期（1000-5000回合）
训练到1000回合后，AI开始慢慢"懂事"了。它学会了移动飞船躲避子弹，偶尔还能击中几个敌人，得分能稳定在100分左右。

### 3. 战术形成期（5000+回合）
到了5000回合之后，神奇的事情发生了！AI突然学会了找掩体，开始采用"peek-shoot"策略——躲在障碍物后面，等安全了再露头开火。得分也飙升到了300+，最高甚至到过500分！

## 为什么AI会变成"怕死鬼"？

后来我仔细想了想，AI之所以形成这种"怕死"的策略，主要是因为游戏的奖励机制：

游戏每帧都会给AI一个很小的正奖励，生存时间越长，累积的奖励就越多。而被击中后会扣血，死亡则直接结束游戏。所以AI通过学习意识到：**活着比什么都重要**，只有生存下来才能获得更多奖励。

这种行为其实很像人类玩家的本能——玩射击游戏时，我们也会不自觉地寻找掩体，避免不必要的伤害。这说明只要奖励设计合理，AI是能学习到类似人类的"智能行为"的。

## 改进方向

这个AI虽然表现不错，但还有很多提升空间：

1. **换个更高效的算法**：比如PPO，样本效率更高，训练更稳定
2. **增加经验回放**：让AI能从过去的经验中学习，加速成长
3. **加入熵正则化**：鼓励AI多探索，避免过早"固化"在一种策略里
4. **多步回报**：考虑未来多步的奖励，让AI的决策更长远

## 写在最后

通过这个小项目，我再次感受到了强化学习的魅力——不需要告诉AI"怎么玩"，它自己就能通过不断试错，学习到复杂的行为策略。这个"怕死AI"虽然看起来有点搞笑，但它真实地展现了AI从"一无所知"到"逐渐成长"的过程。

如果你也对AI感兴趣，建议从这种简单的游戏环境开始入手。看着自己训练的AI一点点进步，那种成就感真的是无法言喻的！

最后，完整代码已经上传到我的GitHub仓库，感兴趣的朋友可以去看看，说不定你能训练出一个更"勇敢"的AI呢？

---

*喜欢这篇文章的话，记得点个赞哦！有什么想法也可以在评论区交流~*