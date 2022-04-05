import numpy as np
import pandas as pd
import sys
'''以通用测试桩或特殊测试桩个数作为评价指标'''
class GsEnvironment():
    '''from to表中，from表示动作类，to表示被依赖类'''
    def __init__(self, filename):
        ##super(filename).__init__()
        AM_deps = 'infodata\\' + filename + '\\Attr_Method_deps.csv'  ##属性依赖(类) 方法依赖(类) Id Name Set_of_Attrdeps属性依赖的类 Set_of_Methoddeps方法依赖的类
        Cid_im = 'infodata\\' + filename + '\\CId_importance.csv'  ## Id Importance
        Cid_name = 'infodata\\' + filename + '\\CId_Name.csv'  ## Id Name
        Couple_List = 'infodata\\' + filename + '\\Couple_List.csv'  ## From To Couple_value(耦合度)
        deps_type = 'infodata\\' + filename + '\\deps_type.csv'  ##From To Type(代码值) Stype(类型名)
        df_AM = pd.read_csv(AM_deps)
        df_Im = pd.read_csv(Cid_im)
        df_Name = pd.read_csv(Cid_name)
        self.df_Cl = pd.read_csv(Couple_List)
        self.df_Dtype = pd.read_csv(deps_type)

        self.df_AM = pd.DataFrame(df_AM.iloc[:, 1:], index=df_AM.Id, columns=df_AM.columns.delete(0))
        self.df_AM=self.df_AM.sort_index()

        # '''测试'''
        # sum=0
        # print(self.df_AM.iloc[3])
        # print(self.df_AM.iloc[3][1])
        # sourcename=self.df_AM.iloc[2][0]
        # print(sourcename)
        # str=self.df_AM.iloc[3][2]
        # if sourcename in str:
        #     print(sourcename in str)
        # strlist=str.split(',')
        # n=len(strlist)
        # for i in range(0,n-1):
        #     if sourcename in strlist[i]:
        #         strlist[i]=strlist[i][::-1]
        #         sum=sum+int(strlist[i][0])
        #         break
        # if sourcename in strlist[n-1]:
        #     strlist[n - 1]=strlist[n-1][::-1]
        #     sum += int(strlist[n-1][2])
        # print(sum)



        self.df_Im = pd.DataFrame(df_Im.iloc[:, 1:], index=df_Im.Id, columns=df_Im.columns.delete(0))
        self.df_Name = pd.DataFrame(df_Name.iloc[:, 1:], index=df_Name.Id, columns=df_Name.columns.delete(0))
        self.class_n = len(df_Im.index)
        self.order = []  # order.append(x)
        self.testOrder = []
        self.minCost = 10000000
        self.minGS = 1000
        self.minSS = 1000
        self.minDeps = 1000
        self.c = 100
        self.Max = 1000
        self.MIN = -1000.0
        self.select = np.zeros(self.class_n)
        self.GS = np.zeros(self.class_n)  #是否已经构造过该测试桩
        self.deps = np.zeros(self.class_n) #
        self.geneticStubs=0
        self.specificStubs=0
        self.MethodDeps=0
        self.AttrDeps=0


        index = 0
        # print(self.select)
        self.path = np.zeros((self.class_n, self.class_n))
        self.Cplx = np.zeros((self.class_n, self.class_n))
        for i in range(len(self.df_Dtype.index)):
            self.path[self.df_Dtype.iloc[i, 0], self.df_Dtype.iloc[i, 1]] = self.df_Dtype.iloc[i, 2] #path[action,pre]
        for i in range(len(self.df_Cl.index)):
            self.Cplx[self.df_Cl.iloc[i, 0], self.df_Cl.iloc[i, 1]] = self.df_Cl.iloc[i, 2]  #Cplx[action,pre]
        self.costMatrix = np.zeros((self.class_n, self.class_n))
        for i in range(self.class_n):  ##直接复制Cplx再修改某些值 or 只选择path>0的
            for j in range(self.class_n):
                if self.path[i, j] > 0:   #i依赖于j
                    self.deps[i] -= 1
                    self.costMatrix[i][j] = self.Cplx[i][j]
                elif self.path[j, i] > 0: #j依赖于i
                    self.deps[i] += 1
                if self.path[i, j] == 2 or self.path[i, j] == 3:
                    self.costMatrix[i][j] = 5 * self.Cplx[i][j]
            # if (self.deps[i]>self.deps[index] or self.deps[i]==self.deps[index]) and ()
        self.cost = 0.0
        # print(self.df_Dtype)
        # print(self.df_Cl)
        # print('path:')
        # print(self.path)
        # print(self.Cplx)
        # print('--------------')
        # print(self.costMatrix)
        # sys.exit()


    def calculateNumOfAttrdeps(self,action,pre):
        '''计算属性复杂度'''
        sum=0
        # print(self.df_AM.iloc[3])
        # print(self.df_AM.iloc[3][1])
        prename=self.df_AM.iloc[pre][0]
        str=self.df_AM.iloc[action][1]
        if not isinstance(str,float):
            if prename in str:
                strlist=str.split(',')
                n=len(strlist)
                for i in range(0,n-1):
                    if prename in strlist[i]:
                        strlist[i]=strlist[i][::-1]
                        sum+=int(strlist[i][0])
                        break
                if prename in strlist[n-1]:
                    strlist[n-1]=strlist[n-1][::-1]
                    sum += int(strlist[n-1][2])
        return sum

    def calculateNumOfMethoddeps(self,action,pre):
        '''计算方法复杂度'''
        sum=0
        # print(self.df_AM.iloc[3])
        # print(self.df_AM.iloc[3][1])
        prename=self.df_AM.iloc[pre][0]
        str=self.df_AM.iloc[action][2]
        if not isinstance(str,float):
            if prename in str:
                strlist=str.split(',')
                n=len(strlist)
                for i in range(0,n-1):
                    if prename in strlist[i]:
                        strlist[i]=strlist[i][::-1]
                        sum+=int(strlist[i][0])
                        break
                if prename in strlist[n-1]:
                    strlist[n-1]=strlist[n-1][::-1]
                    sum += int(strlist[n-1][2])
        return sum

    def getGeneticStubs(self):
        return self.minGS

    def getSpecificStubs(self):
        return self.minSS

    def getMinDeps(self):
        return self.minDeps

    def getCost(self):
        return self.cost

    def getMinCost(self):
        return self.minCost

    def getNumOfDeps(self):
        return self.getNumOfDeps

    def getNumOfGS(self):
        '''获取通用测试桩个数'''
        return self.geneticStubs

    def getGeneticStub(self):
        '''获取通用测试桩序列'''
        gsList=[]
        for i in range(0,self.class_n):
            if self.GS[i]==1:
                gsList.append(i)
        result=gsList.copy()
        return result

    def getNumOfSS(self):
        '''获取特殊测试桩个数'''
        return self.spcialStubs

    def getSpcialStub(self):
        '''获取特殊测试桩序列'''

    def getNumberOfMethoddeps(self):
        return self.MethodDeps

    def getNumberOfAttrdeps(self):
        return self.AttrDeps


    def calculateSpecificStub(self, action):
        '''
        寻找action的所有依赖类，检查他们是否已经测试过或者已经构建通用测试桩，若没有则构建通用测试桩。
        cost为需要构建的测试桩个数，profit为当前action被后续类依赖次数
        '''
        a_GS = 0.0   #为action构建的测试桩
        a_PGS=0.0   #action为后续带来的便利
        a_profit = 0.0 #profit of action
        a_cost=0.0 # cost of action
        # print(self.path)
        for i in range(self.class_n):
            if (self.path[action, i] > 0):
                if (self.select[i] == 1):
                    continue
                a_cost += self.costMatrix[action, i]
                if self.GS[i]==0: #通用测试桩尚未构建
                    self.GS[i]=1
                    a_GS+=1
                    self.MethodDeps+=self.calculateNumOfMethoddeps(action,i)
                    self.AttrDeps+=self.calculateNumOfAttrdeps(action,i)
            if (self.path[i, action] > 0 and self.select[i] == 0):
                a_profit += self.costMatrix[i, action]
                a_PGS+=1
        self.geneticStubs += a_GS
        self.cost += a_cost
        return a_PGS-a_GS

    def calculateReward(self, action):
        for i in self.order:
            if i == action:
                return self.MIN                      #选择了重复类，返回最小奖励
        profit = self.calculateSpecificStub(action)
        reward = self.c * profit
        if len(self.order) == self.class_n - 1 and self.cost <= self.minCost:
            self.minCost = self.cost
            self.testOrder = self.order.copy()  #
            self.testOrder.append(action)
            return 10 * self.Max
        else:
            return reward

    def getOrder(self):
        return self.order.copy()

    def getTestOrder(self):
        return self.testOrder

    def step(self, action):
        done = False
        reward = self.calculateReward(action)
        if not self.order.__contains__(action):
            self.order.append(action)
        if len(self.order) == self.class_n:  # or reward==self.MIN:
            done = True
        self.select[action] = 1
        return self.select, reward, done

if __name__ == '__main__':
    e = GsEnvironment('test')
    dfx=pd.DataFrame()
    dfx=dfx.append([{'算法':10086,'系统':30086},{'系统':20086}])
    print(dfx)
    print(e.df_AM)
    # print(e.calculateNumOfMethoddeps(1,0))
    # print(e.calculateNumOfMethoddeps(20,0))
    # print(e.calculateNumOfMethoddeps(13,0))
    # print(e.calculateNumOfMethoddeps(0, 1))
    # print(e.calculateNumOfAttrdeps(13,0))
    # print(e.calculateNumOfMethoddeps(0,13))
    # e.calculateNumOfMethoddeps(0, 20)
    # e.calculateNumOfMethoddeps(0, 1)


