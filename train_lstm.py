import torch
import torch.nn as nn

from torch.autograd import Variable
from rnn_models import LSTMmodel

import nn_config as config
import data as d


#####################################
# INITIALIZE NETWORK and PARAMETERS #
#####################################


net_params = config.set_network_params()

LSTM_Classifier = LSTMmodel(net_params['input_dim'],
                            net_params['hidden_dim'],
                            net_params['n_layer'],
                            net_params['output_dim'])

#move network to GPU

if torch.cuda.is_available():
    LSTM_Classifier.cuda()

##########################################
# SET LOSS // LEARNING RATE // OPTIMIZER #
##########################################

criterion = nn.CrossEntropyLoss()

lr = net_params['learning_rate']

optimizer = torch.optim.SGD(LSTM_Classifier.parameters(), lr=lr)


#################
# TRAINING LOOP #
#################

loss_list = []
iter = 0

for epoch in range(d.params['num_epochs']):
    for i, (images, labels) in enumerate(d.train_loader):

        if torch.cuda.is_available():
            images = Variable(images.view(-1, net_params['sequence_dim'], net_params['input_dim']).cuda())
            labels = Variable(labels.cuda())
        else: 
            images = Variable(images.view(-1, net_params['sequence_dim'], net_params['input_dim']))
            labels = Variable(labels)
        
        optimizer.zero_grad()

        outputs = LSTM_Classifier(images)

        loss = criterion(outputs, labels)

        if torch.cuda.is_available():
            loss.cuda()
        loss.backward()

        optimizer.step()

        loss_list.append(loss.item())
        iter += 1

        if iter % 500 == 0:

            correct = 0
            total = 0

            for images, labels in d.test_loader:
                if torch.cuda.is_available():
                    images =  Variable(images.view(-1, net_params['sequence_dim'], net_params['input_dim']).cuda())

                else:
                    images = Variable(images.view(-1, net_params['sequence_dim'], net_params['input_dim']))


                outputs = LSTM_Classifier(images)

                _, pred = torch.max(outputs.data, 1)

                total += labels.size(0)

                if torch.cuda.is_available():
                    correct += (pred.cpu() == labels.cpu()).sum()
                
                else:
                    correct += (pred == labels).sum()
            
            accuracy = 100 * correct // total

            print(f'Iteration: {iter}.. | Loss: {loss.item()}.. | Accuracy: {accuracy}%..')



##################################################
# RECORD LOSS FOR EVALUATING NETWORK PERFORMANCE #
##################################################

with open('network_loss_lstm.txt', 'w') as f:
    for loss in loss_list:
        f.write(f'{loss} \n')
    f.close()


