import torch
from torch import nn
import numpy as np
import time
import argparse
from torchvision import datasets, transforms
from sklearn.metrics import classification_report,confusion_matrix
from model.cnn_2ds import CNN_2DS
from utils import AverageMeter, accuracy, save_pred_images

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='cnn_2ds', type=str,
                    choices=['resnet', 'wideresnet', 'efficientnet', 'mobile_large_ca', 'mobile_small_ca', 'densenet', 
                    'densenetcg','senet','ViT', 'CvT','NesT', "vgg19"], 
                    help='the name of model')
parser.add_argument('--gpu', default=0, type=int, help='the name of gpus')
parser.add_argument('--batch-size', default=1, type=int, help='size of batch_size')
parser.add_argument('--num-classes', default=8, type=int, help='number of classes')
parser.add_argument('--image-size', default=224, type=int, help='image size')
parser.add_argument('--patch-size', default=32, type=int, help='patch size')
parser.add_argument('--depth', default=28, type=float, help='deep layer')
parser.add_argument('--widen-factor', default=2, type=float, help='widen_factor of layer')
parser.add_argument('--student-dropout', default=0.2, type=float, help='dropout on last dense layer')
parser.add_argument('--test_path', default='/home/xiaohx/icecrystal/data/test', type=str, help='the file directory')
parser.add_argument('--model_path', default='/home/xiaohx/icecrystal/savemodel/ CNN-2DS.pkl', type=str, help='the models directory')
parser.add_argument('--imagesave_path', default='/home/xiaohx/icecrystal/testimages/', type=str, help='the images save directory')
parser.add_argument('--target_names', nargs='+', type=str, default=['aggregate', 'dendrite', 'donut', 
    'graupel', 'irregular', 'linear', 'plate', 'sphere'], help='the names of icecrystal')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())/1e6

def prediect(args, model, data_test_img):
    target_pred = []
    target_true = []
    top1 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader_test_img):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            target_pred += preds.data.tolist()
            target_true += labels.data.tolist()

            batch_size = inputs.shape[0]
            acc = accuracy(outputs, labels, (1, ))
            top1.update(acc[0], batch_size)
            
            save_pred_images(preds, labels, args.target_names, data_test_img[i], args.imagesave_path) 

        report = classification_report(target_true, target_pred, target_names=args.target_names)
        confusion_matrixs = confusion_matrix(target_true, target_pred)

    print('Accuracy: {:.2f}'.format(top1.avg))
    print(report)
    print(confusion_matrixs)

if __name__ == '__main__':
    args = parser.parse_args()

    args.device = torch.device('cuda', args.gpu)

    if args.name == 'cnn_2ds':
        model_ft = CNN_2DS()
        pretrained_dict = model_ft.state_dict()
        model_dict = model_ft.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model_ft.load_state_dict(model_dict)
    else:
        raise NameError("the model is not in our model zoo!")


    ##### Load Data
    transformation = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    data_test_img = datasets.ImageFolder(root=args.test_path, transform=transformation)
    data_loader_test_img = torch.utils.data.DataLoader(dataset=data_test_img,
                                                   batch_size=args.batch_size)

    print("Params: {:.2f}M".format(count_parameters(model_ft))) 

    since = time.time()
    model_ft.load_state_dict(torch.load(args.model_path))
    model_ft =model_ft.to(args.device)
    prediect(args, model_ft, data_test_img.imgs)

    time_elapsed = time.time() - since
    print('Predicting complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
