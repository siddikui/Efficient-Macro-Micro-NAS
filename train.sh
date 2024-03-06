python train.py --dataset 'CIFAR10' --save 'CIFAR10-T' --gpu 2 --layers 26 --channels 46 --kernels 3 3 3 3 3 3 3 3 3 3 3 3 3 5 3 3 3 5 3 3 3 3 3 3 3 3 --ops 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 

python train.py --dataset 'CIFAR10' --save 'CIFAR10-M' --gpu 1 --layers 21 --channels 102 --kernels 3 3 3 3 3 3 3 3 3 3 5 3 3 3 3 3 3 3 3 3 3 --ops 1 1 0 0 0 1 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 

python train.py --dataset 'CIFAR100' --save 'C100-T' --gpu 2 --layers 21 --channels 46 --kernels 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 --ops 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

python train.py --dataset 'CIFAR100' --save 'C100-M' --gpu 2 --layers 21 --channels 118 --kernels 3 3 3 3 3 3 3 5 3 3 3 3 3 3 3 3 3 3 3 3 3 --ops 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 

python train.py --dataset 'KMNIST' --save 'KMNIST-T' --gpu 3 --layers 18 --channels 50 --kernels 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 5 --ops 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0

python train.py --dataset 'KMNIST' --save 'KMNIST-M' --gpu 3 --layers 24 --channels 120 --kernels 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 --ops 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

python train.py --dataset 'FashionMNIST' --save 'FashionMNIST-T' --gpu 1 --layers 18 --channels 54 --kernels 3 3 3 3 3 3 3 3 5 3 3 3 3 3 3 3 3 3 --ops 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0

python train.py --dataset 'FashionMNIST' --save 'FashionMNIST-M' --gpu 1 --layers 16 --channels 120 --kernels 3 3 3 3 3 3 5 3 3 3 3 3 5 3 3 3 --ops 0 1 1 0 1 0 0 0 1 0 0 1 0 0 0 1

python train.py --dataset 'EMNIST' --save 'EMNIST-T' --gpu 1 --layers 17 --channels 50 --kernels 3 3 3 3 3 3 3 3 3 3 3 5 3 5 3 3 3 --ops 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0

python train.py --dataset 'EMNIST' --save 'EMNIST-M' --gpu 1 --layers 21 --channels 106 --kernels 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 --ops 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
###############################################################################################################################
###############################################################################################################################
