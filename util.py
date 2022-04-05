
class Util:
    def num_to_dtype(self,num):
        numbers={
            1:'As',
            2:'Ag',
            3:'I',
            4:'Dy'
        }
        return numbers.get(num)