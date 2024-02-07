# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from DCANet import DCANet
import torch.nn as nn
from torch.backends import cudnn
from logger import get_logger
import utils
import os

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=0.0005, metavar='wd',
                    help='weight_decay (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')

# cuda related
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    args.gpu = None

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

valid_path = "UERD0.4/QF75/val/"
test_path = "UERD0.4/QF75/test/"
train_cover_path = "BOSS/train/QF75/cover/"
train_stego_path = "BOSS/train/QF75/UERD0.4/"

print('torch ', torch.__version__)
print('train_path = ', train_cover_path)
print('valid_path = ', valid_path)
print('test1_path = ', test_path)
print('train_batch_size = ', args.batch_size)
print('test_batch_size = ', args.test_batch_size)

train_transform = transforms.Compose([utils.AugData(), utils.ToTensor()])  # utils.AugData(),
train_data = utils.DatasetPair(train_cover_path, train_stego_path, train_transform)
# valid_data= utils.DatasetPair(valid_cover_path,valid_stego_path,train_transform)
valid_data = datasets.ImageFolder(valid_path,
                                  transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]))
test1_data = datasets.ImageFolder(test_path,
                                  transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)
test1_loader = torch.utils.data.DataLoader(test1_data, batch_size=args.test_batch_size, shuffle=True, **kwargs)

model = DCANet()
# print(model)
# model.load_state_dict(torch.load('C:\\Users\\User\\Desktop\\39.pkl'))


if args.cuda:
    model.cuda()
cudnn.benchmark = True

pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
for k, v in model.named_modules():
    if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
        pg2.append(v.bias)  # biases
    if isinstance(v, nn.BatchNorm2d):
        pg0.append(v.weight)  # no decay
    elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
        pg1.append(v.weight)  # apply decay

if optim.Adamax:
    optimizer = optim.Adamax(pg0, lr=0.001, betas=(0.9, 0.999))  # adjust beta1 to momentum

optimizer.add_param_group({'params': pg1, 'weight_decay': 0.0002})  # add pg1 with weight_decay
optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
del pg0, pg1, pg2
# optimizer = optim.Adamax(model.parameters(), lr=0.001, betas=(0.9, 0.999),weight_decay=0.0002)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""#  https://blog.csdn.net/flyfish1986/article/details/104846368
    if 0 <= epoch <= 150:
        lr = 0.001
    if epoch > 150:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch):
    total_loss = 0
    lr_train = (optimizer.state_dict()['param_groups'][0]['lr'])
    print(lr_train)
    model.train()

    for batch_idx, data in enumerate(train_loader):

        if args.cuda:
            data, label = data['images'].cuda(), data['labels'].cuda()
        data, label = Variable(data), Variable(label)

        if batch_idx == len(train_loader) - 1:
            last_batch_size = len(os.listdir(train_cover_path)) - args.batch_size * (len(train_loader) - 1)
            datas = data.view(last_batch_size * 2, 1, 256, 256)
            labels = label.view(last_batch_size * 2)
        else:
            datas = data.view(args.batch_size * 2, 1, 256, 256)
            labels = label.view(args.batch_size * 2)
        optimizer.zero_grad()
        output = model(datas)
        # print('output = ',output)
        output1 = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output1, labels)
        total_loss = total_loss + loss.item()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % args.log_interval == 0:
            b_pred = output.max(1, keepdim=True)[1]
            b_correct = b_pred.eq(labels.view_as(b_pred)).sum().item()

            b_accu = b_correct / (labels.size(0))
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\ttrain_accuracy: {:.6f}\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), b_accu, loss.item()))
    logger.info('train Epoch: {}\tavgLoss: {:.6f}'.format(epoch, total_loss / len(train_loader)))
    scheduler.step()
    # writer.add_scalar('Train_loss', loss ,epoch)


def test():
    model.eval()
    test1_loss = 0
    correct = 0.
    with torch.no_grad():
        for data, target in test1_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test1_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            # print(pred,target)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test1_loss /= len(test1_loader.dataset)
    logger.info('Test1 set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        test1_loss, correct, len(test1_loader.dataset),
        100. * correct / len(test1_loader.dataset)))
    accu = float(correct) / len(test1_loader.dataset)
    return accu, test1_loss


def valid():
    model.eval()
    valid_loss = 0
    correct = 0.
    with torch.no_grad():
        for data, target in valid_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            valid_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            # print(pred,target)
            correct += pred.eq(target.view_as(pred)).sum().item()

    valid_loss /= len(valid_loader.dataset)
    logger.info('valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    accu = float(correct) / len(valid_loader.dataset)
    return accu, valid_loss



logger = get_logger('UERD0.4/QF75/dcanet.log')
logger.info('start training!')
#model.load_state_dict(torch.load('J-UNI0.4/QF75/At/162.pkl'))
for epoch in range(1, args.epochs + 1):
    #adjust_learning_rate(optimizer, epoch)
    train(epoch)
    torch.save(model.state_dict(), 'UERD0.4/QF75/dcanet/' + str(epoch) + '.pkl', _use_new_zipfile_serialization=False)
    valid()






