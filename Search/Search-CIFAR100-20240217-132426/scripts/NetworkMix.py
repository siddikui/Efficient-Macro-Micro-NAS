import torch
import torch.nn as nn

class Conv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(Conv, self).__init__()
    self.op = nn.Sequential(

      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),      
      nn.BatchNorm2d(C_out, affine=affine)
      )

  def forward(self, x):
    return self.op(x)

class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class NetworkMix(nn.Module):

  def __init__(self, C, num_classes, layers, mixnet_code, k_size, grayscale):
    super(NetworkMix, self).__init__()
    self._layers = layers
    if grayscale:
        start_layer = 1
    else:
        start_layer = 3
    stem_multiplier = 2
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(start_layer, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    
    C_prev, C_curr = C_curr, C
    
    self.mixlayers = nn.ModuleList()
    reduction_prev = False
    
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
        
      stride = 2 if reduction else 1
      
      if k_size[i]==3:
        pad=1
      elif k_size[i]==5: 
        pad=2
      else:
        pad=3

      
      if mixnet_code[i] == 0:
        mixlayer = SepConv(C_prev, C_curr, kernel_size=k_size[i], stride=stride, padding=pad, affine=True)
      else:
        mixlayer = Conv(C_prev, C_curr, kernel_size=k_size[i], stride=stride, padding=pad, affine=True)

      reduction_prev = reduction
        
      self.mixlayers += [mixlayer]
      C_prev = C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, x):
    
    x = self.stem(x)
    
    for i, mixlayer in enumerate(self.mixlayers):
      x = mixlayer(x)
        
    out = self.global_pooling(x)
    logits = self.classifier(out.view(out.size(0),-1))
    
    return logits
