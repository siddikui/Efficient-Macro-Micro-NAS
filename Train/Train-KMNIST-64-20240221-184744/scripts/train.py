#python train.py --dataset 'KMNIST' --save 'KMNIST' --gpu 3 --layers 8 --channels 32 --kernels 3 3 3 3 3 3 3 3 --ops 1 1 1 1 1 1 1 1


import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from NetworkMix import NetworkMix


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--dataset', type=str, default='KMNIST', help='location of the data corpus')
parser.add_argument('--datapath', type=str, default='/home/sdki/Neural_architecture_search/data', help='location of the data corpus')

parser.add_argument('--valid_size', type=float, default=0.0, help='validation data size')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs') 

parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='KMNIST', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--channels', type=int, default=96, help='maximum number of init channels in search')
parser.add_argument('--layers', type=int, default=10, help='minimum number of init layers in search')
parser.add_argument('--ops', nargs='+', type=int, help='opration in 0 or 1 ')
parser.add_argument('--kernels', nargs='+', type=int, help='kernels in lis like 3 3 5 5 7 7 ')

args = parser.parse_args()

args.save = 'Train-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'Train.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
  if not torch.cuda.is_available():
    logging.info('GPU not available.')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('GPU Device = %d' % args.gpu)
  logging.info("Arguments = %s", args)

  
  # Fetch the desired sub dataset.
  train_queue, valid_queue, test_queue, classes, grayscale = get_desired_dataset(args)

  CIFAR_CLASSES = len(classes) 

  logging.info('Building model...')
  
  model = NetworkMix(args.channels, CIFAR_CLASSES, args.layers, args.ops, args.kernels, grayscale)
  model = model.cuda()
  
  logging.info('MODEL DETAILS')
  logging.info("Model Depth %s Model Width %s", args.layers, args.channels)
  logging.info("Model Layers %s Model Kernels %s", args.ops, args.kernels)
  logging.info('Training epochs %f', int(args.epochs))
  logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
  logging.info('Training Model...')
  curr_arch_train_acc, curr_arch_test_acc = train_test(args, classes, model, train_queue, valid_queue, test_queue)
  logging.info("Baseline Train Acc %f Baseline Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)


def get_desired_dataset(args):

  train_transform, valid_transform, classes, grayscale = utils.data_transforms(args)
  if args.dataset == 'CIFAR10':
    train_data = dset.CIFAR10(root=args.datapath, train=True, download=True, transform=train_transform)
    test_data = dset.CIFAR10(root=args.datapath, train=False, download=True, transform=valid_transform)
  elif args.dataset == 'CIFAR100':
    train_data = dset.CIFAR100(root=args.datapath, train=True, download=True, transform=train_transform)
    test_data = dset.CIFAR100(root=args.datapath, train=False, download=True, transform=valid_transform)
  elif args.dataset == 'EMNIST':
    train_data = dset.EMNIST(root=args.datapath, train=True,
                            download=True, split='balanced',
                            transform=train_transform)
    test_data = dset.EMNIST(root=args.datapath, train=False,
                            download=True, split='balanced',
                            transform=valid_transform)
  elif args.dataset == 'FashionMNIST':
    train_data = dset.FashionMNIST(root=args.datapath, train=True,
                            download=True, transform=train_transform)
    test_data = dset.FashionMNIST(root=args.datapath, train=False,
                            download=True, transform=valid_transform)
  elif args.dataset == 'KMNIST':
    train_data = dset.KMNIST(root=args.datapath, train=True,
                            download=True, transform=train_transform)
    test_data = dset.KMNIST(root=args.datapath, train=False,
                            download=True, transform=valid_transform)

  # obtain training indices that will be used for validation
  valid_size = args.valid_size
  num_train = len(train_data)
  indices = list(range(num_train))
  np.random.shuffle(indices)
  split = int(np.floor(valid_size * num_train))
  train_idx, valid_idx = indices[split:], indices[:split]

  # define samplers for obtaining training and validation batches
  train_sampler = SubsetRandomSampler(train_idx)
  valid_sampler = SubsetRandomSampler(valid_idx)

  train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,sampler=train_sampler, num_workers=2)
  valid_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,sampler=valid_sampler, num_workers=2)
  print()

  test_queue = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=2)

  return train_queue, valid_queue, test_queue,  classes, grayscale

def train_test(args, classes, model, train_queue, valid_queue, test_queue):

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  best_train_acc = 0.0
  best_test_acc = 0.0

  for epoch in range(args.epochs):
    scheduler.step()    

    start_time = time.time()
    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    #logging.info('train_acc %f', train_acc)

    if args.valid_size == 0:
      valid_acc, valid_obj = infer(test_queue, model, criterion)
    else:
      valid_acc, valid_obj = infer(valid_queue, model, criterion)

    if epoch % args.report_freq == 0:
      logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])	
      logging.info('train_acc %f', train_acc)  
      logging.info('valid_acc %f', valid_acc)
       

    end_time = time.time()
    duration = end_time - start_time
    print('Epoch time: %ds.' %duration)
    print('Train acc: %f ' %train_acc)
    print('Valid_acc: %f ' %valid_acc)

    if train_acc > best_train_acc:
      best_train_acc = train_acc

    if valid_acc > best_test_acc:
      best_test_acc = valid_acc
      utils.save(model, os.path.join(args.save, 'weights.pt'))

    #if best_train_acc == 100:
    #	break
    	

  logging.info('Best Training Accuracy %f', best_train_acc)
  logging.info('Best Validation Accuracy %f', best_test_acc)
  utils.load(model, os.path.join(args.save, 'weights.pt'))
  classwisetest(model, classes, test_queue, criterion)

  return best_train_acc, best_test_acc

def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  
  model.train()

  for step, (input, target) in enumerate(train_queue):

    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1 = utils.accuracy(logits, target, topk=(1,))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    #top1.update(prec1.data.item(), n)
    top1.update(prec1, n)

    #if step % args.report_freq == 0:
    #  logging.info('train %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()

  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    with torch.no_grad():
      logits = model(input)
      loss = criterion(logits, target)

    prec1 = utils.accuracy(logits, target, topk=(1,))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    #top1.update(prec1.data.item(), n)
    top1.update(prec1, n)
    #if step % args.report_freq == 0:
    #  logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg

def classwisetest(model, classes, test_queue, criterion):

    
    num_classes = len(classes)
    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    model.eval()
    # iterate over test data
    for data, target in test_queue:
        # move tensors to GPU if CUDA is available        
        data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        #correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        
        # calculate test accuracy for each object class
        for i in range(len(target)):
            #print(i)
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss/len(test_queue.dataset)
    
    logging.info('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(num_classes):
        if class_total[i] > 0:
            logging.info('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            logging.info('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    logging.info('\nTest Accuracy (Overall): %2f%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))    


if __name__ == '__main__':
  start_time = time.time()
  main() 
  end_time = time.time()
  duration = end_time - start_time
  logging.info('Total Search Time: %ds', duration)
