
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
N_ACTIONS = 0
N_STATES = 0
ENV_A_SHAPE = 0


class Net(nn.Module):
    def __init__(self,n):
        super(Net, self).__init__()
        global N_STATES,N_ACTIONS
        N_ACTIONS=n
        N_STATES=n
        m = int(math.sqrt(n) + 1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3) #3x3卷积核 六个卷积核 步长为1 输出图为6x（n-2）x（n-2）
        self.avgpool = nn.AvgPool2d(2)
        self.maxpool=nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)#5x5卷积核 16个卷积核 步长为1 输出图为16x（n-2-4）x（n-2-4）
        self.fc1 = nn.Linear((m-4)*(m-4), 1024)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(1024, 512)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(512, n)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
       #  # x=x.squeeze()
       #  # x = F.relu(self.conv1(x))
       #  # x = F.relu(self.conv2(x))
       #  n=int(math.sqrt(N_ACTIONS)+1)
       #  # x=x+np.array()
       #  #x=np.array(x+np.array([0]*(n*n-N_ACTIONS)).reshape(1,n*n-N_ACTIONS))
       #  # print(x)
       #  x=x.resize_(n,n)
       #  print(x)
       #  x=x[np.newaxis,np.newaxis,:]
       #  print(x)
       # # x=x.permute(1,2,3,0)
        x=self.conv1(x)
        x=self.conv2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions_value = self.out(x)
        return actions_value

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class TDQN(object):
    def __init__(self,n):
        self.eval_net, self.target_net = Net(n), Net(n)

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS-1)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()