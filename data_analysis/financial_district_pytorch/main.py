import argparse
import random
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from fastai import torch_core

from dataset import trafficDataLoader
from model import GRNN
from utils import Log

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

timStart = datetime.datetime.now()

# generate random value as GRNN id to keep track of model
grnn_id = random.randint(0, 100000)

parser = argparse.ArgumentParser()
parser.add_argument('--grnnID', type=int, default=grnn_id, help='GRNN model id')
parser.add_argument('--taskID', type=int, default=1, help='traffic prediction task id')
parser.add_argument('--finterval', type=int, default=10, help='interval of data')
parser.add_argument('--alpha', type=float, default=0.1, help='traffic prediction task id') # regularization
parser.add_argument('--batchSize', type=int, default=1, help='input batch size') # batch size per step
parser.add_argument('--dimHidden', type=int, default=32, help='GRNN hidden state size')
parser.add_argument('--truncate', type=int, default=144, help='BPTT length for GRNN') # interval; step size
parser.add_argument('--nIter', type=int, default=2, help='number of epochs to train') # epoch
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--showNum', type=int, default=None, help='prediction plot. None: no plot')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--verbal', action='store_true', help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed') # random seed is used for reproducing results
parser.add_argument('--test', type=int, default=None, help='for several-node prediction testing')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

def main(opt):
    dataLoader = trafficDataLoader(opt.taskID, opt.finterval)

    opt.nNode = dataLoader.nNode
    opt.dimFeature = dataLoader.dimFeature
    data = dataLoader.data          # [n, T]

    # scale data using MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(data)
    scaler_data = scaler.transform(data)
    
    # scale data using MaxAbsScaler
    # scaler = MaxAbsScaler()
    # scaler.fit(data)
    # scaler_data = scaler.transform(data)

    
    data = np.transpose(scaler_data)  # [T, n]
    data = data[np.newaxis, :, :, np.newaxis]

    # load A and set 1 on the diagonal and zeros elsewhere of each node
    A = dataLoader.A
    A = opt.alpha * A + np.eye(opt.nNode)


    if opt.test is not None:
        opt.nNode = opt.test
        data = data[:, :, :opt.nNode, :]
        A = np.eye(opt.nNode)

    # convert data and A to tensor and randomize weights in hState
    data = torch.from_numpy(data)                                           # [b, T, n, d]
    A = torch.from_numpy(A[np.newaxis, :, :])                               # [b, n, n]
    hState = torch.randn(opt.batchSize, opt.dimHidden, opt.nNode).double()  # [b, D, n]

    opt.interval = data.size(1)
    yLastPred = 0

    # set model configuration
    log = Log(opt, timStart)
    net = GRNN(opt)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    # optimizer = torch_core.AdamW(net.parameters(), lr=opt.lr, weight_decay=1e-1)
    
    net.double()
    print(net)

    if opt.cuda:
        net.cuda()
        criterion.cuda()
        data = data.cuda()
        A = A.cuda()
        hState = hState.cuda()

    # set plot details
    if opt.showNum != None:
        plt.style.use('ggplot')
        plt.figure(1, figsize=(10, 5))
        plt.xlabel(f'Iteration (data on {opt.finterval}-min interval)')    # iteration = lenBptt + (# propogations - 1)
        plt.ylabel('Scaled Speed')
        plt.title(f'GRNN{opt.grnnID} + Adam: Edge {opt.showNum}')
        # plt.ion # turn interactive mode on
    
        # create static legend
        data_color = '#484D6D'
        learning_color = '#DA5552'
        patchA = mpatches.Patch(color=data_color, label='data')
        patchB = mpatches.Patch(color=learning_color, label='learning curve')
        plt.legend(handles=[patchA, patchB])
    
    # use cuDNN along with cuda to train faster
    torch.backends.cudnn.benchmark = True 
    
    # begin training
    for t in range(opt.interval - opt.truncate):
        x = data[:, t:(t + opt.truncate), :, :] # batch, interval, node, feature
        y = data[:, (t + 1):(t + opt.truncate + 1), :, :]

        for i in range(opt.nIter):
            O, _ = net(x, hState, A) # data, hState, A
            hState = hState.data
            
            loss = criterion(O, y)  #criterion(y_true, y_pred)
            MSE = criterion(O[:, -1, :, :], y[:, -1, :, :])
            optimizer.zero_grad()
            loss.backward() # compute gradients
            optimizer.step() # update parameters

            # get explained variance score
            variance = explained_variance_score(O[-1, -1, :, -1].data, y[-1, -1, :, -1].data)

            # log to tensor in matlab file
            # TODO: unscale the results before saving, show timestamp
            log.prediction[:, t, :, :] = O[:, -1, :, :].data
            log.mseLoss[t] = MSE.data
            log.varianceScore[t] = variance

        # show and save log training details
        log.showIterState(t)

        _, hState = net.propogator(x[:, 0, :, :], hState, A)
        hState = hState.data

        # print(O.shape)

        # update plot of training model at each iteration
        if opt.showNum != None:
            if t == 0:
                # x = scaler.inverse_transform(x[-1, :, opt.showNum, -1].cpu().data.numpy())
                # O = scaler.inverse_transform(O[-1, :, opt.showNum, -1].cpu().data.numpy())

                if opt.cuda:
                    transform_x_temp = np.zeros((opt.truncate, data.size(1)))
                    transform_x_temp[:, 0] = np.reshape(x[0, :, opt.showNum, 0].cpu().data.numpy().flatten(), (opt.truncate, 1)).flatten()
                    x = scaler.inverse_transform(transform_x_temp)[:, [0]].flatten()

                    transform_O_temp = np.zeros((opt.truncate, data.size(1)))
                    transform_O_temp[:, 0] = np.reshape(O[0, :, opt.showNum, 0].cpu().data.numpy().flatten(), (opt.truncate, 1)).flatten()
                    O = scaler.inverse_transform(transform_O_temp)[:, [0]].flatten()

                    plt.plot([v for v in range(opt.truncate)], x, color=data_color, linestyle='-', linewidth=1.5)
                    plt.plot([v + 1 for v in range(opt.truncate)], O, color=learning_color, linestyle='-')

                else:

                    transform_x_temp = np.zeros((opt.truncate, data.size(1)))
                    transform_x_temp[:, 0] = np.reshape(x[0, :, opt.showNum, 0].data.numpy().flatten(), (opt.truncate, 1)).flatten()
                    x = scaler.inverse_transform(transform_x_temp)[:, [0]].flatten()

                    transform_O_temp = np.zeros((opt.truncate, data.size(1)))
                    transform_O_temp[:, 0] = np.reshape(O[0, :, opt.showNum, 0].data.numpy().flatten(), (opt.truncate, 1)).flatten()
                    O = scaler.inverse_transform(transform_O_temp)[:, [0]].flatten()

                    plt.plot([v for v in range(opt.truncate)], x, color=data_color, linestyle='-', linewidth=1.5)
                    plt.plot([v + 1 for v in range(opt.truncate)], O, color=learning_color, linestyle='-')

            else:
                # x = scaler.inverse_transform(x[-1, -2:, opt.showNum, -1].cpu().data.numpy())
                # O = scaler.inverse_transform(O[-1, -1, opt.showNum, -1])

                # these blocks are going to explode
                if opt.cuda:
                    transform_x_temp = np.zeros((2, data.size(1)))
                    transform_x_temp[:, 0] = np.reshape(x[0, -2:, opt.showNum, 0].cpu().data.numpy().flatten(), (2, 1)).flatten()
                    x = scaler.inverse_transform(transform_x_temp)[:, [0]]

                    transform_O_temp = np.zeros((1, data.size(1)))
                    transform_O_temp[:, 0] = np.reshape(O[0, -1, opt.showNum, 0].cpu().data.numpy().flatten(), (1, 1)).flatten()
                    O = scaler.inverse_transform(transform_O_temp)[:, [0]]

                    plt.plot([t + opt.truncate - 2, t + opt.truncate - 1], x, color=data_color, linestyle='-', linewidth=1.5)
                    plt.plot([t + opt.truncate - 1, t + opt.truncate], [yLastPred, O], color=learning_color, linestyle='-')

                else:
                    transform_x_temp = np.zeros((2, data.size(1)))
                    transform_x_temp[:, 0] = np.reshape(x[0, -2:, opt.showNum, 0].data.numpy().flatten(), (2, 1)).flatten()
                    x = scaler.inverse_transform(transform_x_temp)[:, [0]]

                    transform_O_temp = np.zeros((1, data.size(1)))
                    transform_O_temp[:, 0] = np.reshape(O[0, -1, opt.showNum, 0].data.numpy().flatten(), (1, 1)).flatten()
                    O = scaler.inverse_transform(transform_O_temp)[:, [0]]

                    plt.plot([t + opt.truncate - 2, t + opt.truncate - 1], x, color=data_color, linestyle='-', linewidth=1.5)
                    plt.plotkind([t + opt.truncate - 1, t + opt.truncate], [yLastPred, O], color=learning_color, linestyle='-')
            
            plt.savefig(f'result/experiment/grnn{opt.grnnID}-{opt.finterval}int-{opt.taskID}tid-{opt.alpha}a-{opt.truncate}T-{opt.dimHidden}D-{opt.nIter}i-{opt.lr}lr-{opt.manualSeed}ms-{opt.batchSize}b-{opt.showNum}sn.jpg')
           
            # updates the graph at some interval
            plt.draw()
            plt.pause(0.7)
            yLastPred = O[-1]

        log.saveResult(t)

        # save trained model as pt file
        torch.save({
            'epoch': opt.nIter,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion
        }, f'result/experiment/grnn{opt.grnnID}-{opt.finterval}int-{opt.taskID}tid-{opt.alpha}a-{opt.truncate}T-{opt.dimHidden}D-{opt.nIter}i-{opt.lr}lr-{opt.manualSeed}ms-{opt.batchSize}b-{opt.showNum}sn.pt')


    if opt.showNum != None:
        plt.savefig(f'result/experiment/grnn{opt.grnnID}-{opt.finterval}int-{opt.interval}int-{opt.taskID}tid-{opt.alpha}a-{opt.truncate}T-{opt.dimHidden}D-{opt.nIter}i-{opt.lr}lr-{opt.manualSeed}ms-{opt.batchSize}b-{opt.showNum}sn.jpg')
        # plt.ioff()  # interactive mode off
        plt.show()

    # terminate app
    exit(1)


if __name__ == "__main__":
    main(opt)

