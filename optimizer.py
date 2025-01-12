class SGD:
    def __init__(self,lr):
        self.lr = lr

    def set(self,network):
        self.network = network
        
    def update(self):
        for layer in self.network.layers:
            if layer.update_params:
                layer.weight -= self.lr * layer.dw
                layer.bias -= self.lr * layer.db