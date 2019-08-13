import datetime as dt
import torch
import os
import scipy.io as spio

def getTime(begin, end=None):
    if end is None:
        end = dt.datetime.now()
    timeDelta = end - begin
    return '%d h %d m %d.%ds' % (timeDelta.seconds // 3600, (timeDelta.seconds%3600) // 60, timeDelta.seconds % 60, timeDelta.microseconds)

def ms2f(ms):
    ms = float(ms)
    while ms >= 1:
        ms /= 10
    return ms

class Log(object):
    def __init__(self, opt, startTime):
        self.opt = opt
        self.resLength = opt.interval - opt.truncate
        self.startTime = startTime

        self.prediction = torch.zeros(opt.batchSize, self.resLength, opt.nNode, opt.dimFeature)
        self.mseLoss = torch.zeros(self.resLength)
        self.varianceScore = torch.zeros(self.resLength)


    def printLR(self, optimizer):   # LR = learning rate
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
    

    def showIterState(self, t):
        date = dt.date.today()
        datestring = dt.datetime.now().strftime('%a %b %d %Y %H:%M:%S')
        
        # save results to csv for plotting
        with open(f'log/grnn{self.opt.grnnID}-{self.opt.finterval}int-{self.opt.taskID}tid-{self.opt.alpha}a-{self.opt.truncate}T-{self.opt.dimHidden}D-{self.opt.nIter}i-{self.opt.lr}lr-{self.opt.manualSeed}ms-{self.opt.batchSize}b-{self.opt.showNum}sn.csv', 'a') as g:
            # datetime, iteration, variance score, mseloss, duration
            g.write('%s, %.4f, %.4f, %s\n' % (datestring, self.varianceScore[t], self.mseLoss[t], getTime(self.startTime)))
            
            g.close()

        # print training details in cli
        print('[Log] %d iteration. Variance Score: %.4f, MSELoss: %.4f. Training Duration: %s.' % (
            t + 1, self.varianceScore[t], self.mseLoss[t], getTime(self.startTime)))

    
    def saveResult(self, t):
        if not os.path.exists('result'):
            os.mkdir('result')
        
        if (t + 1) % 100 == 0 or t == self.resLength - 1 and self.opt.verbal:
            if t == self.resLength - 1:
                print('[Log] Train finished. All results saved! Total duration: %s.' % getTime(self.startTime))
            else:
                print('[Log] Results saved.')
            
            duration = dt.datetime.now() - self.startTime
            timeStamp = '%d%02d%02d_%02d%02d' % (self.startTime.year, self.startTime.month, self.startTime.day,
                    self.startTime.hour, self.startTime.minute)
            
            spio.savemat(f'result/experiment/grnn{self.opt.grnnID}-{self.opt.finterval}int-{self.opt.taskID}tid-{self.opt.alpha}a-{self.opt.truncate}T-{self.opt.dimHidden}D-{self.opt.nIter}i-{self.opt.lr}lr-{self.opt.manualSeed}ms-{self.opt.batchSize}b-{self.opt.showNum}sn.mat', {
                        'prediction': self.prediction.data.numpy(),
                        'mseLoss': self.mseLoss.data.numpy(),
                        'iter': t + 1,
                        'startTime': timeStamp,
                        'totalTime': duration.seconds + ms2f(duration.microseconds),
                        'batchSize': self.opt.batchSize,
                        'cuda': self.opt.cuda,
                        'manualSeed': self.opt.manualSeed})
