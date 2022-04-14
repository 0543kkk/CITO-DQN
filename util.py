
import torch

class Util:
    def num_to_dtype(self,num):
        numbers={
            1:'As',
            2:'Ag',
            3:'I',
            4:'Dy'
        }
        return numbers.get(num)

    def save_net(self,program,state_dict,method,epoch):
        torch.save({'epoch':epoch,'state_dict':state_dict},'model/'+program+'/'+method+'/model.pth')