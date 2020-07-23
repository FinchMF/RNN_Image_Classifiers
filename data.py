import torch
import torchvision.transforms as transforms
import torchvision.datasets as data 

import nn_config as config

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#########################
# DATA PIPELINE - MNIST #
#########################

train_set = data.MNIST(root='./data',
                             download=True,
                             train=True,
                             transform=transforms.ToTensor())
                             
                             
test_set = data.MNIST(root='./data',
                            download=True,
                            train=False,
                            transform=transforms.ToTensor())

params = config.set_data_params(len(train_set))
print(params['batch_size'])

train_loader = torch.utils.data.DataLoader(
                                            dataset=train_set,
                                            batch_size=params['batch_size'],
                                            shuffle=True
                                            )
test_loader = torch.utils.data.DataLoader(
                                            dataset=test_set,
                                            batch_size=params['batch_size'],
                                            shuffle=False
                                            )


