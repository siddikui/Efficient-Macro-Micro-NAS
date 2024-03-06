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
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs') 


parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='KMNIST', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--add_epochs', type=int, default=1, help='num of training epochs to increase for depth') 
parser.add_argument('--add_epochs_w', type=int, default=2, help='num of training epochs to increase for width') 
parser.add_argument('--target_acc', type=float, default=100.00, help='desired target accuracy')
parser.add_argument('--target_acc_tolerance', type=float, default=0.10, help='tolerance for desired target accuracy')

parser.add_argument('--ch_drop_tolerance', type=float, default=0.05, help='tolerance when dropping channels')
parser.add_argument('--dp_break_tolerance', type=int, default=2, help='tolerance when terminating depth search')
parser.add_argument('--ch_break_tolerance', type=int, default=2, help='tolerance when terminating channel search')
parser.add_argument('--dp_add_tolerance', type=float, default=0.10, help='tolerance when increasing depth')


parser.add_argument('--min_width', type=int, default=16, help='minimum number of init channels in search')
parser.add_argument('--max_width', type=int, default=256, help='maximum number of init channels in search')
parser.add_argument('--width_resolution', type=int, default=2, help='resolution for number of channels search')
parser.add_argument('--min_depth', type=int, default=10, help='minimum number of init layers in search')
parser.add_argument('--max_depth', type=int, default=100, help='maximum number of init layers in search')

args = parser.parse_args()

args.save = 'Search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'Searchlog.txt'))
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

  # Run search on the fetched sub dataset.
  curr_arch_ops, curr_arch_kernel, f_epochs, f_channels, f_layers, curr_arch_train_acc, curr_arch_test_acc = search_depth_and_width(args,   
  	                                                                                                                      classes,
                                                                                                                          grayscale,        
                                                                                                                          train_queue,
                                                                                                                          valid_queue,                                                                                                                          
                                                                                                                          test_queue)
  
  d_w_model_info = {'curr_arch_ops': curr_arch_ops,
                    'curr_arch_kernel': curr_arch_kernel,
                    'curr_arch_train_acc': curr_arch_train_acc,
                    'curr_arch_test_acc': curr_arch_test_acc,
                    'f_channels': f_channels,
                    'f_layers': f_layers}

  
  curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc, channels_ks = search_operations_and_kernels(args, 
                                                                                                    classes,
                                                                                                    grayscale,
                                                                                                    train_queue,
                                                                                                    valid_queue,
                                                                                                    test_queue, 
                                                                                                    d_w_model_info)

  
  logging.info('END OF SEARCH...')
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('')

  model = NetworkMix(channels_ks, len(classes), f_layers, curr_arch_ops, curr_arch_kernel, grayscale)
  model = model.cuda()                                                                                                                           

  logging.info('FINAL DISCOVERED ARCHITECTURE DETAILS:')
  logging.info("Model Depth %s Model Width %s", f_layers, channels_ks)
  logging.info('Discovered Final Epochs %s', f_epochs)
  logging.info("Model Layers %s Model Kernels %s", curr_arch_ops, curr_arch_kernel)
  logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
  logging.info("Training Accuracy %f Validation Accuracy %f", curr_arch_train_acc, curr_arch_test_acc)

def search_depth_and_width(args, classes, grayscale, train_queue, valid_queue, test_queue):

  logging.info('#############################################################################')
  logging.info('INITIALIZING DEPTH AND WIDTH SEARCH...')

  CIFAR_CLASSES = len(classes) 
  target_acc=args.target_acc
  min_width=args.min_width
  max_width=args.max_width
  width_resolution=args.width_resolution
  min_depth=args.min_depth
  max_depth=args.max_depth
  ch_drop_tolerance = args.ch_drop_tolerance
  target_acc_tolerance = args.target_acc_tolerance
  # We start with max width but with min depth.
  channels = max_width 
  layers = min_depth

  # Initialize
  curr_arch_ops = next_arch_ops = np.zeros((layers,), dtype=int)
  curr_arch_kernel = next_arch_kernel = 3*np.ones((layers,), dtype=int)
  curr_arch_train_acc = next_arch_train_acc = 0.0
  curr_arch_test_acc = next_arch_test_acc = 0.0

  logging.info('RUNNING DEPTH SEARCH FIRST...')

  model = NetworkMix(channels, CIFAR_CLASSES, layers, curr_arch_ops, curr_arch_kernel, grayscale)
  model = model.cuda()
  
  logging.info('MODEL DETAILS')
  logging.info("Model Depth %s Model Width %s", layers, channels)
  logging.info("Model Layers %s Model Kernels %s", curr_arch_ops, curr_arch_kernel)
  logging.info('Training epochs %f', int(args.epochs))
  logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
  logging.info('Training Model...')
  curr_arch_train_acc, curr_arch_test_acc = train_test(args, classes, model, train_queue, valid_queue, test_queue)
  logging.info("Baseline Train Acc %f Baseline Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)

  # Search depth
  depth_fail_count = 0

  while ((curr_arch_test_acc < (target_acc - target_acc_tolerance)) and (layers != max_depth)):
    
    # The possibility exists if trained for too long.
    if (curr_arch_train_acc > 99.5):
      break;  
      
    else:
      # prepare next candidate architecture.  
      args.epochs = args.epochs + args.add_epochs
      layers += 1
      next_arch_ops = np.zeros((layers,), dtype=int)
      next_arch_kernel = 3*np.ones((layers,), dtype=int)
      model = NetworkMix(channels, CIFAR_CLASSES, layers, next_arch_ops, next_arch_kernel, grayscale)
      model = model.cuda()
      
      logging.info('#############################################################################')
      logging.info('Moving to Next Candidate Architecture...')
      logging.info('MODEL DETAILS')
      logging.info("Model Depth %s Model Width %s", layers, channels)
      logging.info("Model Layers %s Model Kernels %s", next_arch_ops, next_arch_kernel)
      logging.info('Total number of epochs %f', args.epochs)
      logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
      logging.info("Depth Fail Count %s", depth_fail_count)
      logging.info('Training Model...')
      next_arch_train_acc, next_arch_test_acc = train_test(args, classes, model, train_queue, valid_queue, test_queue)
      logging.info("Candidate Train Acc %f Candidate Val Acc %f", next_arch_train_acc, next_arch_test_acc)
     
      # As long as we get significant improvement by increasing depth.
      
      if (next_arch_test_acc >= curr_arch_test_acc + args.dp_add_tolerance):
        # update current architecture.
        depth_fail_count = 0
        curr_arch_ops = next_arch_ops
        curr_arch_kernel = next_arch_kernel
        logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
        curr_arch_train_acc = next_arch_train_acc
        curr_arch_test_acc = next_arch_test_acc
        logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)
      # But we still keep trying deeper candidates.
      elif (next_arch_test_acc >= curr_arch_test_acc - 0.05):
        logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)
        logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
        continue
      elif((next_arch_test_acc < curr_arch_test_acc - 0.05) and ((depth_fail_count != args.dp_break_tolerance))):
        depth_fail_count += 1
        logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)
        logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
        continue
      else:
        logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)
        logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
        break
  # Search width
  # During width search lenght of curr_arch_ops and curr_arch_kernel shall not change but only channels.

  f_layers = len(curr_arch_ops) # discovered final number of layers
  f_channels = max_width # discovered final number of channels
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('Discovered Final Depth %s', f_layers)
  
  logging.info('END OF DEPTH SEARCH...')
  best_arch_test_acc = curr_arch_test_acc
  best_arch_train_acc = curr_arch_train_acc
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('')
  logging.info('RUNNING WIDTH SEARCH NOW...') 
  f_epochs = args.epochs
  width_fail_count = 0
  while (channels > min_width):
    # prepare next candidate architecture.
    channels = channels - width_resolution
    # Although these do not change.
    model = NetworkMix(channels, CIFAR_CLASSES, f_layers, curr_arch_ops, curr_arch_kernel, grayscale)
    model = model.cuda()
    args.epochs = args.epochs + args.add_epochs_w

    logging.info('Moving to Next Candidate Architecture...')
    logging.info('MODEL DETAILS')
    logging.info("Model Depth %s Model Width %s", f_layers, channels)
    logging.info("Model Layers %s Model Kernels %s", curr_arch_ops, curr_arch_kernel)
    logging.info('Total number of epochs %f', args.epochs)
    logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
    logging.info('Training Model...')
    logging.info("Width Fail Count %s", width_fail_count)
    # train and test candidate architecture.
    next_arch_train_acc, next_arch_test_acc = train_test(args, classes, model, train_queue, valid_queue, test_queue)
    logging.info("Candidate Train Acc %f Candidate Val Acc %f", next_arch_train_acc, next_arch_test_acc)

    if (next_arch_test_acc >= (curr_arch_test_acc - 0.0)):

      logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
      curr_arch_train_acc = next_arch_train_acc
      curr_arch_test_acc = next_arch_test_acc
      #best_arch_train_acc = curr_arch_train_acc
      #best_arch_test_acc = curr_arch_test_acc
      f_channels = channels 
      f_epochs = args.epochs
      logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)
      width_fail_count = 0
    elif (width_fail_count != args.ch_break_tolerance):
      width_fail_count += 1
      logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
      logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)

      continue
    else:
      logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
      logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)

      break; 
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('Discovered Final Width %s', f_channels)
  logging.info('Discovered Final Epochs %s', f_epochs)
  logging.info('END OF WIDTH SEARCH...')  
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('')  
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('#############################################################################')

  logging.info('END OF MACRO SEARCH...')
  logging.info('MODEL DETAILS')
  logging.info("Model Depth %s Model Width %s", f_layers, f_channels)
  logging.info("Model Layers %s Model Kernels %s", curr_arch_ops, curr_arch_kernel)
  logging.info('Total number of epochs %f', args.epochs)
  model = NetworkMix(f_channels, CIFAR_CLASSES, f_layers, curr_arch_ops, curr_arch_kernel, grayscale)
  logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
  logging.info('END OF MACRO SEARCH...')
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('#############################################################################')

  return curr_arch_ops, curr_arch_kernel, f_epochs, f_channels, f_layers, curr_arch_train_acc, curr_arch_test_acc

def search_operations_and_kernels(args, classes, grayscale, train_queue, valid_queue, test_queue, model_info):

  logging.info('STARTING OF MICRO SEARCH...')
  logging.info('#############################################################################')
  logging.info('#############################################################################')

  curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc, channels_ops = search_operations(args, 
                                                                                               classes, 
                                                                                               grayscale, 
                                                                                               train_queue, 
                                                                                               valid_queue, 
                                                                                               test_queue, 
                                                                                               model_info)
  
  model_info['curr_arch_ops'] = curr_arch_ops
  model_info['curr_arch_kernel'] = curr_arch_kernel
  model_info['curr_arch_train_acc'] = curr_arch_train_acc
  model_info['curr_arch_test_acc'] = curr_arch_test_acc
  model_info['channels_ops'] = channels_ops

  logging.info('#############################################################################')
  logging.info('#############################################################################')
  
  curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc, channels_ks = search_kernels(args, 
                                                                                            classes, 
                                                                                            grayscale,
                                                                                            train_queue, 
                                                                                            valid_queue, 
                                                                                            test_queue, 
                                                                                            model_info)

  return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc, channels_ks

def search_operations(args, classes, grayscale, train_queue, valid_queue, test_queue, model_info):

  logging.info('RUNNING OPERATION SEARCH...')

  CIFAR_CLASSES = len(classes) 
  curr_arch_ops = model_info['curr_arch_ops']
  curr_arch_kernel = model_info['curr_arch_kernel']
  channels = model_info['f_channels']
  layers = model_info['f_layers']

  channels_ops = model_info['f_channels']

  curr_arch_train_acc = 0.0
  curr_arch_test_acc = 0.0
  next_arch_train_acc = 0.0
  next_arch_test_acc = 0.0
  
  next_arch_ops = curr_arch_ops
  next_arch_kernel = curr_arch_kernel
  args.epochs = 2
  
  model = NetworkMix(channels, CIFAR_CLASSES, layers, curr_arch_ops, curr_arch_kernel, grayscale)
  baseline_arch_params = utils.count_parameters_in_MB(model)

  model = model.cuda()
    
  logging.info('NEXT MODEL DETAILS')
  logging.info("Model Depth %s Model Width %s", layers, channels)
  logging.info("Model Layers %s Model Kernels %s", curr_arch_ops, curr_arch_kernel)
  logging.info('Total number of epochs %f', args.epochs)
  logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
  
  logging.info('Training Model...')
  curr_arch_train_acc, curr_arch_test_acc = train_test(args, classes, model, train_queue, valid_queue, test_queue)
  logging.info("Baseline Training Accuracy %f Baseline Validation Accuracy %f", curr_arch_train_acc, curr_arch_test_acc)  
  logging.info('#############################################################################')


  for i in range(layers):
    channels = model_info['f_channels']

    #if i < 2*layers//3:
    if i < layers+1:  
   
      next_arch_ops[i] = 1

      model = NetworkMix(channels, CIFAR_CLASSES, layers, next_arch_ops, next_arch_kernel, grayscale)

      candidate_arch_params = utils.count_parameters_in_MB(model)

      while(candidate_arch_params > baseline_arch_params + 0.01):

        channels -= 1
        model = NetworkMix(channels, CIFAR_CLASSES, layers, next_arch_ops, next_arch_kernel, grayscale)
        candidate_arch_params = utils.count_parameters_in_MB(model)
        


      model = model.cuda()
    
      logging.info('NEXT MODEL DETAILS')
      logging.info("Model Depth %s Model Width %s", layers, channels)
      logging.info("Model Layers %s Model Kernels %s", next_arch_ops, next_arch_kernel)
      logging.info('Total number of epochs %f', args.epochs)
      logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
      logging.info('Training Model...')
      next_arch_train_acc, next_arch_test_acc = train_test(args, classes, model, train_queue, valid_queue, test_queue)
      logging.info("Best Training Accuracy %f Best Validation Accuracy %f", next_arch_train_acc, next_arch_test_acc)

      #if next_arch_test_acc > curr_arch_test_acc + 0.25: ######### Add arg
      if next_arch_test_acc > curr_arch_test_acc - 0.0: ######### Add arg      
        logging.info("Highest Train Acc %f Highest Val Acc %f", next_arch_train_acc, next_arch_test_acc)
        logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
        curr_arch_ops = next_arch_ops
        curr_arch_kernel = next_arch_kernel
        curr_arch_train_acc = next_arch_train_acc
        curr_arch_test_acc = next_arch_test_acc
        channels_ops = channels

      else:
        next_arch_ops[i] = 0
        logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)
        logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)

      logging.info('#############################################################################')
 
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('RETURNING Model...')
  logging.info("Model Depth %s Model Width %s", layers, channels_ops)
  logging.info("Model Layers %s Model Kernels %s", curr_arch_ops, curr_arch_kernel)
  logging.info("Train Acc %f Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)
  logging.info('Total number of epochs %f', args.epochs)
  logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc, channels_ops  


def search_kernels(args, classes, grayscale, train_queue, valid_queue, test_queue, model_info):

  logging.info('RUNNING KERNEL SEARCH...')

  CIFAR_CLASSES = len(classes) 
  curr_arch_ops = model_info['curr_arch_ops']
  curr_arch_kernel = model_info['curr_arch_kernel']
  curr_arch_train_acc = model_info['curr_arch_train_acc']
  curr_arch_test_acc = model_info['curr_arch_test_acc']
  #channels = model_info['f_channels']
  channels = model_info['channels_ops']
  channels_ks = model_info['channels_ops']
  layers = model_info['f_layers']
  grayscale = grayscale
  #args.epochs = 5

  model = NetworkMix(channels, CIFAR_CLASSES, layers, curr_arch_ops, curr_arch_kernel, grayscale)
  baseline_arch_params = utils.count_parameters_in_MB(model)

  next_arch_ops = curr_arch_ops
  next_arch_kernel = curr_arch_kernel

  kernels = [5]
  #kernels = [5, 7]

  for i in range(layers): 
    channels = model_info['channels_ops']
  
    #if i < 2*layers//3:
    if i < layers+1:
      best_k = 3 
      for k in kernels:

        next_arch_kernel[i] = k
   
        model = NetworkMix(channels, CIFAR_CLASSES, layers, next_arch_ops, next_arch_kernel, grayscale)

        candidate_arch_params = utils.count_parameters_in_MB(model)

        while(candidate_arch_params > baseline_arch_params + 0.01):

          channels -= 1
          model = NetworkMix(channels, CIFAR_CLASSES, layers, next_arch_ops, next_arch_kernel, grayscale)
          candidate_arch_params = utils.count_parameters_in_MB(model)

        model = model.cuda()
    
        logging.info('MODEL DETAILS')
        logging.info("Model Depth %s Model Width %s", layers, channels)
        logging.info("Model Layers %s Model Kernels %s", next_arch_ops, next_arch_kernel)
        logging.info('Total number of epochs %f', args.epochs)
        logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
        logging.info('Training Model...')
        next_arch_train_acc, next_arch_test_acc = train_test(args, classes, model, train_queue, valid_queue, test_queue)
        logging.info("Best Training Accuracy %f Best Validation Accuracy %f", next_arch_train_acc, next_arch_test_acc)


        # Bigger kernel comes at a cost therefore possibility of a search hyper parameter exists.
        #if (next_arch_test_acc > curr_arch_test_acc + 0.25): # Add args
        if next_arch_test_acc > curr_arch_test_acc - 0.0: ######### Add arg
          best_k = k
          logging.info("Highest Train Acc %f Highest Val Acc %f", next_arch_test_acc, next_arch_test_acc)
          logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
          curr_arch_ops = next_arch_ops
          curr_arch_kernel[i] = k
          curr_arch_train_acc = next_arch_train_acc
          curr_arch_test_acc = next_arch_test_acc
          channels_ks = channels

        else:
          next_arch_kernel[i] = best_k
          logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)
          logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)

        logging.info('#############################################################################')
  
  logging.info('#############################################################################')
  logging.info('END OF MICRO SEARCH')
  logging.info("Model Depth %s Model Width %s", layers, channels_ks)
  logging.info("Model Layers %s Model Kernels %s", curr_arch_ops, curr_arch_kernel)
  logging.info("Train Acc %f Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)
  logging.info('Total number of epochs %f', args.epochs)
  logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
        
  return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc, channels_ks




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

def search_ops_and_ks_simultaneous(args, classes, grayscale, train_queue, valid_queue, test_queue, model_info):

  logging.info('RUNNING OPERATIONS AND KERNELS SEARCH SIMULTANEOUSLY...')

  CIFAR_CLASSES = len(classes) 
  curr_arch_ops = model_info['curr_arch_ops']
  curr_arch_kernel = model_info['curr_arch_kernel']
  curr_arch_train_acc = model_info['curr_arch_train_acc']
  curr_arch_test_acc = model_info['curr_arch_test_acc']
  channels = model_info['f_channels']
  layers = model_info['f_layers']

  kernels = [3, 5, 7]
  operations = [0, 1]

  next_arch_ops = curr_arch_ops
  next_arch_kernel = curr_arch_kernel
  # Can be navigated from the last layers instead of first ones.
  for i in range(layers):  
    for k in kernels:
      for o in operations:

        args.epochs = args.epochs + args.add_epochs

        next_arch_ops[i] = o
        next_arch_kernel[i] = k
 
        model = NetworkMix(channels, CIFAR_CLASSES, layers, next_arch_ops, next_arch_kernel, grayscale)
        model = model.cuda()
  
        logging.info('MODEL DETAILS')
        logging.info("Model Depth %s Model Width %s", layers, channels)
        logging.info("Model Layers %s Model Kernels %s", next_arch_ops, next_arch_kernel)
        logging.info('Total number of epochs %f', args.epochs)
        logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
        logging.info('Training Model...')
        next_arch_train_acc, next_arch_test_acc = train_test(args, classes, model, train_queue, valid_queue, test_queue)
        logging.info("Best Training Accuracy %f Best Validation Accuracy %f", next_arch_train_acc, next_arch_test_acc)

        # Bigger kernel comes at a cost therefore possibility of a search hyper parameter exists.
        if (next_arch_test_acc > curr_arch_test_acc):
          curr_arch_ops = next_arch_ops
          curr_arch_kernel = next_arch_kernel
          curr_arch_train_acc = next_arch_train_acc
          curr_arch_test_acc = next_arch_test_acc

  return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc    

def search_kernels_and_operations(args, classes, grayscale, train_queue, valid_queue, test_queue, model_info):

  logging.info('SEARCHING FOR KERNELS FIRST...')

  curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc = search_kernels(args, classes, grayscale, train_queue, valid_queue, test_queue, model_info)
  
  model_info['curr_arch_ops'] = curr_arch_ops
  model_info['curr_arch_kernel'] = curr_arch_kernel
  model_info['curr_arch_train_acc'] = curr_arch_train_acc
  model_info['curr_arch_test_acc'] = curr_arch_test_acc

  logging.info('SEARCHING FOR OPERATIONS...')

  curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc = search_operations(args, classes, grayscale, train_queue, valid_queue, test_queue, model_info)

  return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc


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
