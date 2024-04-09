# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 08:56:20 2023

@author: Anish Hilary
"""

import torch

result_dir = r'C:\Users\Anish Hilary\RESNET\normal_cifar_100\results\15_12-21_24_150\resnet_34/latest_model.pth'

result_dict = torch.load(result_dir)

# print(f"The total epochs : {result_dict['epoch']}")
print(f"all_accuracy : {result_dict['valid_epoch_accuracy']}")
# print(f"best_accuracy : {result_dict['best_accuracy']}")
# print(f"Parameters : {result_dict['learnable_params']}")
print(f"lr : {result_dict['al_lr']}")