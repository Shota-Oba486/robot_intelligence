import numpy as np

class Sigmoid:
    def __init__(self):
        self.update_params = False
        self.name = "sigmoid"
    def forward(self,x):
        self.x = x
        self.y = 1/(1+np.exp(-x))
        return self.y
    def backward(self,dout):
        self.dout = dout
        grad = dout * self.y * (1 - self.y)
        self.grad = grad
        return grad
    
class Relu:
    def __init__(self):
        self.y = None
        self.update_params = False
        self.name = "relu"
    def forward(self,x):
        self.x = x
        self.y = np.maximum(0,x)
        return self.y
    def backward(self, dout):
        self.dout = dout
        self.grad = dout * (self.x > 0)
        return self.grad

class Affine:
    def __init__(self,in_num,out_num):
        ave = 0
        var = 1.0
        # self.weight = 0.01 * np.random.normal(ave,var,(in_num,out_num))
        self.weight = np.random.randn(in_num, out_num) * np.sqrt(1.0 / in_num)
        self.bias = 0.01 * np.random.rand(out_num)
        self.update_params = True
        self.name = "affine"
    def forward(self,x):
        self.x = x
        self.y = np.dot(x,self.weight) + self.bias
        return self.y
    def backward(self,delta):
        dout = np.dot(delta,self.weight.T)
        self.dw = np.dot(self.x.T, delta)
        self.db = np.dot(np.ones(len(self.x)), delta)
        return dout

class Softmax:
    def __init__(self):
        self.x = None
        self.update_params = False
        
    def forward(self,x):
        self.x = x
        exp_x = np.exp(x)
        self.y = exp_x / np.sum(exp_x,axis = 1,keepdims=True)
        return self.y

class MLP:
    def __init__(self,layers):
        self.layers = layers
    
    def forward(self,x,t):
        self.y = x
        self.t = t
        for layer in self.layers:
            self.y = layer.forward(self.y)
        self.loss = - np.sum(self.t * np.log(self.y +1e-7)) / len(x)
        return self.loss

    def backward(self):
        dout = (self.y - self.t) / len(self.layers[-1].x)
        for layer in self.layers[-2::-1]:
            dout = layer.backward(dout)