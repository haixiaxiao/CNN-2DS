from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model.cnn_2ds import CNN_2DS
from torch.utils.tensorboard import SummaryWriter
import time
import os
import random
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='cnn_2ds', type=str,
                    choices=['resnet', 'wideresnet', 'efficientnet', 'mobile_large_ca', 'mobile_small_ca', 'densenet', 
                    'densenetcg','senet','ViT', 'CvT','NesT', "vgg19"], 
                    help='the name of model')
parser.add_argument('--gpu', default=1, type=int, help='the name of gpus')
parser.add_argument('--seed', default=1234, type=int, help='seed for initializing training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight-decay')
parser.add_argument('--batch-size', default=16, type=int, help='size of batch_size')
parser.add_argument('--num-epochs', default=100, type=int, help='number of epochs size')
parser.add_argument('--num-classes', default=8, type=int, help='number of classes')
parser.add_argument('--image-size', default=224, type=int, help='image size')
parser.add_argument('--patch-size', default=32, type=int, help='patch size')
parser.add_argument('--depth', default=28, type=float, help='deep layer')
parser.add_argument('--widen-factor', default=2, type=float, help='widen_factor of layer')
parser.add_argument('--student-dropout', default=0.2, type=float, help='dropout on last dense layer')
parser.add_argument('--data-dir', default='/home/xiaohx/icecrystal/data/', type=str, help='the file directory')
parser.add_argument('--save-dir', default='/home/xiaohx/icecrystal/omodels/savemodel/', type=str, help='the save directory')

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def save_checkpoint(checkpoint, epoch_label, acc_label):
    save_filename = '%d_%s_%f.pth.tar' % (epoch_label, args.name, acc_label)
    save_path = os.path.join(args.save_dir, save_filename)
    torch.save(checkpoint, save_path)

def train_model(args, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    obloss = 2

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step(obloss)
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0
            for data in dataloaders[phase]:
                inputs, labels = data
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]
            obloss = epoch_loss
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                args.writer.add_scalar("train/loss", epoch_loss, epoch)
                args.writer.add_scalar("train/acc", epoch_acc, epoch)
            else:
                args.writer.add_scalar("val/loss", epoch_loss, epoch)
                args.writer.add_scalar("val/acc", epoch_acc, epoch)
                # fd = open('./result/valacc'+args.name+'.txt', 'a+')
                # fd.write(str(epoch_acc) + '\n')
                # fd.close()

        if phase == 'valid' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        save_checkpoint(model.state_dict(), epoch, epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    args = parser.parse_args()
    args.device = torch.device('cuda', args.gpu)
    args.writer = SummaryWriter(f"results/{args.name}")

    if args.seed is not None:
        set_seed(args)

    if args.name == 'cnn_2ds':
        model_ft = CNN_2DS()
        pretrained_dict = model_ft.state_dict()
        model_dict = model_ft.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model_ft.load_state_dict(model_dict)
    else:
        raise NameError("the model is not in our model zoo!")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'valid']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

    model_ft.to(args.device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)  

    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=5, verbose=False)

    model_ft = train_model(args, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=args.num_epochs)

    torch.save(model_ft.state_dict(), "./savemodel/" + args.name + '.pkl')

