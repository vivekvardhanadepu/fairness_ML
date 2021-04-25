class loss_wrapper:
    def __init__(self, function):
        self.f = function # actual objective function
        self.num_calls = 0 # how many times f has been called
        self.values = {'c':[], 'losses':[]} # storing c's and losses
    
    def simulate(self, x, *args):
        self.values['c'].append(x)
        loss = self.f(x, *args)
        # self.values['c'].append(x)
        self.values['losses'].append(loss)
        self.num_calls += 1
        return loss