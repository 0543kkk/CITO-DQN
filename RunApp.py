import pandas as pd
import tkinter as tk
import time
import math
from Environments.Environment import Environment
from DQN.torchCITOAgent import DQN
from DQN.doubleDQN import DoubleDQN
from DQN.duelingDQN import DuelingDQN
from DQN.Prioritized_Replay_DQN import DQNPrioritizedReplay
from DQN.CNNCITO import DeepQNetwork2
from DQN.CITOAgent import DeepQNetwork
import copy
from util import Util
n_train=10

# 创建进度条主窗口
window = tk.Tk()
window.title('进度条')
window.geometry('630x150')
# 设置运行进度条
tk.Label(window, text='运行进度:', ).place(x=50, y=60)
canvas = tk.Canvas(window, width=500, height=22, bg="white")
canvas.place(x=110, y=60)


class RunDQN:
    def __init__(self,filename,algorithm):
        #self.filename=input("please input the filename:\n")
        #self.filename='test'
        # self.filename='daisy'
        #self.filename='JHotDraw'
        self.tool=Util()
        self.algorithm=algorithm
        self.filename=filename
        self.env=Environment(self.filename)
        self.class_n=self.env.class_n
        self.order=[]
        self.mincost = 10000.0
        self.bestOrder = []
        self.minMethoddeps=10000.0
        self.minAttrdeps=10000.0
        self.n_sqrt = int(math.sqrt(self.class_n) + 1)  # 边长

    def reset(self):
        '''
        重新建立一个对象，以初始化所有参数
        ps:在环境类内部写一个reset函数更好！
        '''
        self.env = Environment(self.filename)
        self.class_n = self.env.class_n
        self.order = []

    def runDQN(self,textout):
        step = 0
        # RL_Conv1D=DqnConv1d(
        #     self.class_n, self.class_n,
        #     learning_rate=0.01,
        #     reward_decay=0.9,
        #     e_greedy=0.9,
        #     replace_target_iter=20,
        #     memory_size=20000,
        #     # output_graph=True
        # )
        if self.algorithm=="PrioritizedReplayDQN":
            print("PrioritizedReplayDQN")
            RL=DQNPrioritizedReplay(
                self.class_n, self.class_n,
                learning_rate=0.01,
                reward_decay=0.9,
                e_greedy=0.9,
                replace_target_iter=20,
                memory_size=20000,
                # output_graph=True
            )
        if self.algorithm=="DQN":
            print("DQN")
            # RL= DQN(self.class_n,self.class_n)
            RL= DeepQNetwork(self.class_n, self.class_n,
                                            learning_rate=0.01,
                                            reward_decay=0.9,
                                            e_greedy=0.9,
                                            replace_target_iter=20,
                                            memory_size=20000,
                                            # output_graph=True
                                            )
        if self.algorithm=="DQN-CNN":
            print("DQN-CNN")
            RL= DeepQNetwork2(self.class_n, self.class_n,
                          learning_rate=0.01,
                          reward_decay=0.9,
                          e_greedy=0.9,
                          replace_target_iter=20,
                          memory_size=20000,
                          # output_graph=True
                          )
        if self.algorithm=="duelingDQN":
            print("duelingDQN")
            RL=DuelingDQN(self.class_n, self.class_n,
                          learning_rate=0.01,
                          reward_decay=0.9,
                          e_greedy=0.9,
                          replace_target_iter=20,
                          memory_size=20000,
                          # output_graph=True
         )
        if self.algorithm=="doubleDQN":
            RL=DoubleDQN(self.class_n, self.class_n,
                          learning_rate=0.01,
                          reward_decay=0.9,
                          e_greedy=0.9,
                          replace_target_iter=20,
                          memory_size=20000,
                          # output_graph=True
         )

        start = time.time()

        for episode in range(n_train):

            #屏幕显示进度条
            # x=n_train/500
            # fill_line = canvas.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="green")
            # canvas.coords(fill_line, (0, 0, episode/x, 60))
            # window.update()
            # s='第'+str(episode)+'轮:'
            # canvas.create_text(50,40,text=s)

            #print(s)
            self.reset()
            # RL = RL_select
            method = RL.method()
            #observation=self.env.select   observation成为一个指向select的指针,select改变,observation改变 只能用a=b.copy()
            observation=self.env.select.copy()
            #observation = observation.resize(self.n_sqrt, self.n_sqrt)  # 将x形状变为m*m m-1²<x<m²
            #print('ob:')
            #print(observation)
            #print('--------')
            while True:
                action = RL.choose_action(observation)
                # print('act:')
                # print(action)
                observation_,reward,done=self.env.step(action)
                #observation_ = observation_.resize(self.n_sqrt, self.n_sqrt)
                # print(observation)
                # print(action)
                # print(reward)
                # print(observation_)
                # obs=np.zeros(25)
                # obs_,r,d=self.env.step(5)
                # print('test:')
                # print(obs)
                # print('5')
                # print(r)
                # print(obs_)
                # print('------------')
                RL.store_transition(observation,action,reward,observation_)

                if(step > 50) and (step % 5 == 0):
                    RL.learn()

                observation=observation_.copy()

                if done:
                    order=self.env.getOrder()
                    cost=self.env.cost
                    Attrdeps=self.env.getNumberOfAttrdeps()
                    Methoddpes=self.env.getNumberOfMethoddeps()
                    if cost<=self.mincost:
                        self.mincost=cost
                        self.bestOrder=order.copy()   #若果直接赋值，会导致两个变量名同时指向同一片内存空间
                        self.minAttrdeps=Attrdeps
                        self.minMethoddeps=Methoddpes
                    # print('order:')
                    # print(order)
                    # print(self.env.cost)
                    break
                step+=1
            #RL.plot_cost()
            a=episode
            if self.mincost<0.8:
                break
            print('第{}轮训练，整体测试装复杂度为{}。'.format(episode,cost))
            textout('第{}轮训练，整体测试装复杂度为{}。\n'.format(episode,cost))
        f=open('./experiment','a')

        print('您选择的系统是:'+self.filename)
        f.write('您选择的系统是:'+self.filename+'\n')
        textout('您选择的系统是:'+self.filename+'\n')


        s='该系统共有'+str(self.class_n)+'个类'
        print(s)
        f.write(s+'\n')
        textout(s+'\n')

        s='使用的方法是:'+method
        print(s)
        f.write(s+'\n')
        textout(s+'\n')

        print('训练轮数为:'+str(n_train))
        f.write('训练轮数为:'+str(n_train)+'\n')
        textout('训练轮数为:'+str(n_train)+'\n')

        print('最佳测试序列:')
        print(self.bestOrder)
        f.write('最佳测试序列:'+str(self.bestOrder)+'\n')

        print('整体测试桩复杂度:')
        print(self.mincost)
        f.write('整体测试桩复杂度:'+str(self.mincost)+'\n')


        b=time.time()
        print('所用时间:')
        print(str(b-start)+'s')
        f.write('所用时间:'+str(b-start)+'s'+'\n\n\n')
        f.close()

        numOfGS=self.env.getNumOfGS()
        GS=self.env.getGeneticStub()
        # book = load_workbook('.\DataOfExperiment\experiment.xlsx')
        # writer = pd.ExcelWriter('.\DataOfExperiment\experiment.xlsx',engine='openpyxl')
        # writer.book = book
        # #experimentDataFile=pd.read_excel('./experimentData.xlsx',sheet_name=filename)
        # Df_expData=pd.DataFrame()
        # Df_expData=Df_expData.append([{'系统名':self.filename,'类个数':self.class_n,'算法':method,'时间':b-start,
        #                             '属性复杂度':self.minAttrdeps,'方法复杂度':self.minMethoddeps,
        #                             '训练轮数':n_train,'最佳测试序列':self.bestOrder,'整体测试桩复杂度':self.mincost,
        #                             '通用测试桩个数':numOfGS,'通用测试桩序列':GS}])
        # print(Df_expData)
        # Df_expData.to_excel(excel_writer=writer)
        # writer.save()
        experimentDataFile = pd.read_excel('.\DataOfExperiment\experiment.xlsx')
        Df_expData = pd.DataFrame(experimentDataFile)
        Df_expData=Df_expData.append([{'系统名':self.filename,'类个数':self.class_n,'算法':method,'时间':b-start,
                                    '属性复杂度':self.minAttrdeps,'方法复杂度':self.minMethoddeps,
                                    '训练轮数':n_train,'最佳测试序列':self.bestOrder,'整体测试桩复杂度':self.mincost,
                                    '通用测试桩个数':numOfGS,'通用测试桩序列':GS}])
        Df_expData.to_excel('.\DataOfExperiment\experiment.xlsx')
        result={'系统名':self.filename,'类个数':self.class_n,'算法':method,'时间':b-start,
                                    '属性复杂度':self.minAttrdeps,'方法复杂度':self.minMethoddeps,
                                    '训练轮数':n_train,'最佳测试序列':self.bestOrder,'整体测试桩复杂度':self.mincost,
                                    '通用测试桩个数':numOfGS,'通用测试桩序列':GS}

        dtype=self.env.getPath()
        couple=self.env.getCouple()
        class_name=self.env.getClassName()
        class_name_result={}
        dtype_result=[]
        couple_result=[]
        for i in range(self.class_n):
            class_name_result.update({str(class_name.index[i]): str(class_name.iloc[i].Name)})
            for j in range(self.class_n):
                if dtype[i][j]!=0:
                    x=self.tool.num_to_dtype(dtype[i][j])
                    dtype_result.append((i,j,x))
                    couple_result.append((i,j,couple[i][j]))
        print("classname\n{}".format(class_name))
        print('tab3:{}'.format(couple_result))
        return result.copy(),dtype_result.copy(),couple_result.copy(),class_name_result.copy()

