import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import copy

np.random.seed(1)
torch.manual_seed(1)

'''使用卷积层'''

# define the network architecture 定义网络结构

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        self.n=n_feature
        super(Net, self).__init__()
        # self.el = nn.Linear(n_feature, n_hidden)
        # self.q = nn.Linear(n_hidden, n_output)
        self.avgpool = nn.AvgPool2d(2)
        self.maxpool=nn.MaxPool2d(2)
        self.pool=self.maxpool

        #三层卷积（3×3卷积核），三层最大池化，每经过一层卷积n×n变n-2,经过一层池化变为n/2向上取整
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3,padding=1)
       # self.conv3=nn.Conv2d(in_channels=6,out_channels=12,kernel_size=3)
        n=n_feature
        for i in range(2):
            n = n - 2
            n = n // 2 + 1
        n=n*n*6

        self.fc1 = nn.Linear(54, 128)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization 初始化网络参数
        # self.fc2 = nn.Linear(128, 64)
        # self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(128, n_feature)
        self.out.weight.data.normal_(0, 0.1)   # initialization


    def forward(self, x):
        # n = int(math.sqrt(n) + 1)
        # x=x+np.array()
        # x=np.array(x+np.array([0]*(n*n-N_ACTIONS)).reshape(1,n*n-N_ACTIONS))
        # print(x)

        # m = int(math.sqrt(self.n) + 1) #m=边长
        # x = x.resize_(m, m) #将x形状变为m*m m-1²<x<m²
        # print(x)
        # x = x[np.newaxis, np.newaxis, :]

        x=self.transto3D(x)
        x=x[:,np.newaxis, :,:]
        # x=x.permute(1,2,3,0)
        x = self.pool(self.conv1(x))
        x=F.relu(x)
        x = self.pool(self.conv2(x))
        x=F.relu(x)
        # x = self.pool(self.conv3(x))
        # x=F.relu(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        actions_value = self.out(x)
        return actions_value

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def transto3D(self,twoDimensionList):
        num = twoDimensionList.shape[0]
        n = len(twoDimensionList[0])
        edge = int(math.sqrt(n) + 1)
        threeDList = torch.zeros(num, edge, edge)
        for numcount in range(num):
            i = 0
            for j in range(edge):
                if i >= n: break
                for k in range(edge):
                    if i >= n: break
                    threeDList[numcount, j, k] = twoDimensionList[numcount][i]
                    i += 1
        result = threeDList.clone().detach()
        return result

class Net_pool(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        self.n=n_feature
        super(Net_pool, self).__init__()
        # self.el = nn.Linear(n_feature, n_hidden)
        # self.q = nn.Linear(n_hidden, n_output)
        self.avgpool = nn.AvgPool2d(3,stride=2)
        self.maxpool=nn.MaxPool2d(2)
        self.pool=self.avgpool

       #  #三层卷积（3×3卷积核），三层最大池化，每经过一层卷积n×n变n-2,经过一层池化变为n/2向上取整
       #  self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3,padding=1)
       #  self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3,padding=1)
       # # self.conv3=nn.Conv2d(in_channels=6,out_channels=12,kernel_size=3)
       #  n=n_feature
       #  for i in range(2):
       #      n = n - 2
       #      n = n // 2 + 1
       #  n=n*n*6

        self.fc1 = nn.Linear(36, 128)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization 初始化网络参数
        # self.fc2 = nn.Linear(128, 64)
        # self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(128, n_feature)
        self.out.weight.data.normal_(0, 0.1)   # initialization


    def forward(self, x):
        # n = int(math.sqrt(n) + 1)
        # x=x+np.array()
        # x=np.array(x+np.array([0]*(n*n-N_ACTIONS)).reshape(1,n*n-N_ACTIONS))
        # print(x)

        # m = int(math.sqrt(self.n) + 1) #m=边长
        # x = x.resize_(m, m) #将x形状变为m*m m-1²<x<m²
        # print(x)
        # x = x[np.newaxis, np.newaxis, :]

        x=self.transto3D(x)
        x=x[:,np.newaxis, :,:]
        # x=x.permute(1,2,3,0)
        x = self.pool(x)
        x=F.relu(x)
        # x = self.pool(self.conv3(x))
        # x=F.relu(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        actions_value = self.out(x)
        return actions_value

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def transto3D(self,twoDimensionList):
        num = twoDimensionList.shape[0]
        n = len(twoDimensionList[0])
        edge = int(math.sqrt(n) + 1)
        threeDList = torch.zeros(num, edge, edge)
        for numcount in range(num):
            i = 0
            for j in range(edge):
                if i >= n: break
                for k in range(edge):
                    if i >= n: break
                    threeDList[numcount, j, k] = twoDimensionList[numcount][i]
                    i += 1
        result = threeDList.clone().detach()
        return result

class DeepQNetwork2():
    def __init__(self, n_actions, n_features, n_hidden=20, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=500, batch_size=32, e_greedy_increment=None,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max


        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.loss_func = nn.MSELoss()
        self.cost_his = []

        self._build_net()

    def method(self):
        return 'CNN-DQN'

    def _build_net(self):
        self.q_eval = Net_pool(self.n_features, self.n_hidden, self.n_actions)
        self.q_target = Net_pool(self.n_features, self.n_hidden, self.n_actions)
        self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=self.lr)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = torch.Tensor(observation[np.newaxis, :])
        if np.random.uniform() < self.epsilon:
            actions_value = self.q_eval(observation)

            action = np.argmax(actions_value.data.numpy())
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters 如果已经学习了n（200）次，则将target网络参数更新为eval网络参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())
            # print("\ntarget params replaced\n")

        # sample batch memory from all memory 随机选择一个batch的经验
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        # print(batch_memory,end='\nbatch_memory\n')

        # q_next is used for getting which action would be choosed by target network in state s_(t+1)  计算Q（target）和Q估计 （eval）
        #Q估计是使用eval（一直更新）在s1计算得到的Q值（最大值），同时得到对应动作a，利用s1和a得到R和s2，然后利用target计算S2的Q值（最大值）,Q现实=R+gama*Q（s2）
        q_next, q_eval = self.q_target(torch.Tensor(batch_memory[:, -self.n_features:])), self.q_eval(
            torch.Tensor(batch_memory[:, :self.n_features])) #q_next使用s2，q_eval使用s1
        # print("qqqqqqqqqqqqqqqqqqq")
        # print(q_next,end='\nqnext\n')
        # print(q_eval,end='\nqeval\n')

        # used for calculating y, we need to copy for q_eval because this operation could keep the Q_value that has not been selected unchanged,
        # so when we do q_target - q_eval, these Q_value become zero and wouldn't affect the calculation of the loss
        q_target = torch.Tensor(q_eval.data.numpy().copy())

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # print('batchindex:')
        # print(batch_index)
        eval_act_index = batch_memory[:, self.n_features].astype(int) #动作a
        # print(eval_act_index)
        reward = torch.Tensor(batch_memory[:, self.n_features + 1]) #奖励r
        # print(reward)
        # print(q_target,end='qtar')
        # print(torch.max(q_next, 1)[0])
        q_target[batch_index, eval_act_index] = reward + self.gamma * torch.max(q_next, 1)[0] #计算Q现实 torch.max(input, dim) 0为列/1为行
        pass
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # increase epsilon
        self.cost_his.append(loss)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()