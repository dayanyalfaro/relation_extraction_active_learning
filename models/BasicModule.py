import os
import time
import torch
import torch.nn as nn


class BasicModule(nn.Module):
    '''
    Encapsulate nn.Module, provide save and load methods
    '''
    def __init__(self):
        #First find the parent class nn.Module of BasicModule, and convert the current object into a parent class object, calling the __init__() method of the parent class
        super(BasicModule, self).__init__()


    def load(self, path, device):
        '''
        Load the model of the specified path
        '''
        self.load_state_dict(torch.load(path, map_location=device))


    def save(self, epoch=0, cfg=None):
        '''
        Save the model, use "model name + time" as the file name by default
        '''
        time_prefix = time.strftime('%Y-%m-%d_%H-%M-%S')
        prefix = os.path.join(cfg.cwd, 'weights',time_prefix)
        os.makedirs(prefix, exist_ok=True)
        name = os.path.join(prefix, cfg.model_name + '_' + f'epoch{epoch}' + '.pth')

        torch.save(self.state_dict(), name)
        return name


