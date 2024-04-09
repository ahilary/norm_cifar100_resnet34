# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:46:53 2023

@author: Anish Hilary
"""


import os

from utils_atten import device, checkpoints_results, start_timer, stop_timer
from torchvision.models import resnet34
from torchvision.models import ResNet34_Weights
from data_creation import create_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from train_valid import train, test
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Load the config.yaml file
import yaml

with open('config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)


# Access the variables
cnf_epoch = config_data.get('epochs')
cnf_lr = config_data.get('lr')
cnf_lr_decay_step = config_data.get('lr_decay_step')
cnf_dataset_name = config_data.get('dataset_name')
num_classes = config_data.get('num_classes')


print('Model is running in {}'.format(device))

# start training from intermediate point
inter_dir = None

# init to store results and checkpoints
model_saver = checkpoints_results('resnet_34', cnf_epoch, inter_dir)

# dataset loaders
train_loader, test_loader = create_dataset()

num_classes = 100

model = resnet34(weights=ResNet34_Weights.DEFAULT)    
#model = resnet34()   

# best_mod = r'C:\Users\Anish Hilary\RESNET\normal_cifar_100\results\18_12-16_48_200\resnet_34/best_model.pth'
# best_mod = torch.load(best_mod)

# model.load_state_dict(best_mod['model_state_dict'])

if inter_dir and os.path.isfile(model_saver.save_latest()):
    
    checkpoint = torch.load(model_saver.save_latest())
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    start_epoch = checkpoint['epoch']+1
        

if inter_dir is None:
    start_epoch = 1


model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

#optimizer = optim.SGD(model.parameters(), lr=cnf_lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(model.parameters(), lr=cnf_lr)
criterion = nn.CrossEntropyLoss()
#scheduler = MultiStepLR(optimizer, milestones=[40,80,120,160], gamma=0.2)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience= 5, factor= 0.5, min_lr= 1e-6, verbose=1)

train_epoch_loss = []
valid_epoch_loss = []

train_epoch_time = []
valid_epoch_time = []

valid_epoch_accuracy = []
    
all_lr = []
    
# normal resnet model parameters
learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The learnable parameters before pruning: {learnable_params}')


for epoch in range(start_epoch, cnf_epoch + 1):
    if not os.path.isfile(model_saver.save_latest()):
        pass
        
    else:
        checkpoint = torch.load(model_saver.save_latest())
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']+1
        train_epoch_time = checkpoint['train_epoch_time']
        train_epoch_loss = checkpoint['train_epoch_loss']
        valid_epoch_loss = checkpoint['valid_epoch_loss']
        valid_epoch_time = checkpoint['valid_epoch_time']
        valid_epoch_accuracy = checkpoint['valid_epoch_accuracy']
        best_accuracy = checkpoint['best_accuracy']
    
# Training
    start_timer('Training')  
    train_loss, model = train(train_loader, model, optimizer, criterion, epoch)
    train_epoch_time.append(stop_timer('Training'))
    
    train_epoch_loss.append(train_loss)

# Validation
    start_timer('Validation')  
    valid_loss, top1_accu = test(test_loader, model, criterion, epoch)
    valid_epoch_time.append(stop_timer('Validation'))
    
    valid_epoch_loss.append(valid_loss)
    valid_epoch_accuracy.append(top1_accu)

    scheduler.step(valid_loss)
    #scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']
    all_lr.append(current_lr)
    
# save best model
    if top1_accu > model_saver.best_model_accuracy:
        model_saver.best_model_accuracy = top1_accu
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_epoch_time': train_epoch_time,
                'train_epoch_loss': train_epoch_loss,
                'valid_epoch_loss': valid_epoch_loss,
                'valid_epoch_time': valid_epoch_time,
                'valid_epoch_accuracy': valid_epoch_accuracy,
                'learnable_params': learnable_params,
                'best_accuracy': model_saver.best_model_accuracy,
                'lr': current_lr
                }, model_saver.save_best())
        
        
# save checkpoint
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_epoch_time': train_epoch_time,
            'train_epoch_loss': train_epoch_loss,
            'valid_epoch_loss': valid_epoch_loss,
            'valid_epoch_time': valid_epoch_time,
            'valid_epoch_accuracy': valid_epoch_accuracy,
            'learnable_params': learnable_params,
            'best_accuracy': model_saver.best_model_accuracy,
            'al_lr': all_lr
            }, model_saver.save_latest())
            
    



# Model plots
model_saver.epoch_plot(cnf_dataset_name, (train_epoch_loss, valid_epoch_loss),
                       (valid_epoch_accuracy),
                       (train_epoch_time),
                       (valid_epoch_time),
                       ('train_loss','valid_loss'),
                       ('valid_accuracy'),
                       ('train_epoch_time'),
                       ('valid_epoch_time'))


