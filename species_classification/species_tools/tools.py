import os, argparse, time, json, sys
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
#from torch.utils.data import DataLoader, Subset,ConcatDataset
from torch.cuda.amp import autocast, GradScaler

from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as trn
import torchvision.datasets as dset
from torchvision.models import resnet50

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,Subset, Dataset,ConcatDataset,WeightedRandomSampler
from torchvision.datasets.folder import default_loader

import timm
from collections import defaultdict


import logging

import shutil
from PIL import Image

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report,precision_recall_fscore_support
from datetime import datetime
import math
import copy
import gc

import timm.optim
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics.pairwise import cosine_similarity

#import geopandas
from torch.utils.data import Sampler

def save_checkpoint(state, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    #if is_best:
    #    shutil.copyfile(filepath, os.path.join(checkpoint,
    #                                           'model_best.pth.tar'))
    #else:
    torch.save(state, filepath)
def save_checkpoint_by_epoch(args,state, epoch, filename='checkpoint.pth.tar'):
    save_epoch_list, saveFolder = args.save_epoch_list, args.save_epoch_folder
    if epoch in save_epoch_list:
        outFolder=f'{saveFolder}'
        outFile=f'{outFolder}/{filename}'
        if not os.path.isdir(saveFolder):
            os.mkdir(saveFolder)
        if not os.path.isdir(outFolder):
            os.mkdir(outFolder)
        torch.save(state, outFile)

def make_dataset_nolist(image_list, class_index_dict=None):
    # when input is josn (AL labels), class index dict must given!
    label_list = []
    img_path_list = []
    if image_list.endswith('txt'):
        with open(image_list) as f:
            for x in f.readlines():
                label = int(x.split(' ')[1].strip())
                imgpath=x.split(' ')[0]
                label_list.append(label)
                img_path_list.append(imgpath)
    elif image_list.endswith('json'):
        with open(image_list,"r") as oj:
            data=json.load(oj)
            for imn in data.values():
                al_label=imn['AL_label']
                if al_label!='' and imn['imgPath'] not in img_path_list: # avoid duplcate
                    label_list.append(class_index_dict[al_label]) #Classnum : index (0,1...)
                    img_path_list.append(imn['imgPath']) 
    else:
        classes=os.listdir(image_list)
        for index, clsFolder in enumerate(classes):
            imgRootPath=f"{image_list}/{clsFolder}"
            for imgfile in os.listdir(imgRootPath):
                imgPath=f"{imgRootPath}/{imgfile}"
                img_path_list.append(imgPath)
                label_list.append(index)
    image_index = np.array(img_path_list)
    labels = np.array(label_list)
    return image_index, labels

def default_loader(path):
    return Image.open(path).convert('RGB')
    
#class ImageFolder(Dataset):
#    """
#    A generic data loader where the images are arranged in this way: ::
#        root/dog/xxx.png
#        root/dog/xxy.png
#        root/dog/xxz.png
#        root/cat/123.png
#        root/cat/nsdf3.png
#        root/cat/asd932_.png
#
#    Args:
#        image_list (list): List of image paths.
#        transform (callable, optional): A function/transform that takes in a PIL image
#            and returns a transformed version.
#        target_transform (callable, optional): A function/transform that takes in the
#            target and transforms it.
#        loader (callable, optional): A function to load an image given its path.
#        allowed_classes (list, optional): List of allowed class names or class indices to filter.
#        class_index_dict (dict, optional): Dictionary mapping class names to class indices.
#
#    Attributes:
#        classes (list): List of the class names.
#        class_to_idx (dict): Dict with items (class_name, class_index).
#        imgs (list): List of (image path, class_index) tuples.
#    """
#    def __init__(self, img_folder, transform=None, target_transform=None, return_paths=False,
#                 loader=default_loader, train=False, return_id=False, class_index_dict=None, include_non_allowed=False,other_class_id=7,filter_by_files=None):
#        # Filter by allowed_classes if provided
#        self.include_non_allowed = include_non_allowed
#        self.class_index_dict = class_index_dict
#        self.other_class_id=other_class_id
#        self.filter_by_files=filter_by_files #list of image name with .jpg extension
#        # Create dataset with optional class filtering
#        imgs, labels = self.make_dataset(img_folder)
#        
#        self.imgs = imgs
#        self.labels = labels
#        self.transform = transform
#        self.target_transform = target_transform
#        self.loader = loader
#        self.return_paths = return_paths
#        self.return_id = return_id
#        self.train = train
#    
#    def loop_image_folder(self,img_folder,):
#        imgs = []
#        labels = []
#        for class_name in os.listdir(img_folder):
#            #class_name = path.split('/')[-2]  # Assuming folder structure root/class_name/filename
#            if self.class_index_dict is not None:
#                try:
#                    class_idx = self.class_index_dict[class_name]
#                except:
#                    class_idx = None
#                if class_idx is not None:
#                    for img in os.listdir(f"{img_folder}/{class_name}"):
#                        imgs.append(f"{img_folder}/{class_name}/{img}")
#                        labels.append(class_idx)
#                elif self.include_non_allowed :
#                    for img in os.listdir(f"{img_folder}/{class_name}"):
#                        imgs.append(f"{img_folder}/{class_name}/{img}")
#                        labels.append(self.other_class_id)
#                else:
#                    print("Error : cannot find the class in class dict and incldued_non_allowed is False")
#            else:
#                print("Error : cannot find class dict, should be Class123:0, 0:Class123")
#                #class_idx=0
#                #for img in os.listdir(f"{img_folder}/{class_name}"):
#                #    imgs.append(f"{img_folder}/{class_name}/{img}")
#                #    labels.append(class_idx)
#        return imgs,labels
#    
#    def make_dataset(self, img_folder):
#        """
#        Creates a list of (image path, class_index) tuples.
#        
#        Args:
#            image_list (list): List of image paths.
#
#        Returns:
#            tuple: (imgs, labels) where imgs is a list of image paths and labels are corresponding class indices.
#        """
#        imgs = []
#        labels = []
#        if isinstance(img_folder,list):
#            for folder in img_folder:
#                subimgs,sublabels=self.loop_image_folder(folder)
#                imgs+=subimgs
#                labels+=sublabels
#        else:
#            imgs,labels=self.loop_image_folder(img_folder)
#
#        if self.filter_by_files is not None:
#            filtered_imgs=[]
#            filtered_labels=[]
#            for index,img in enumerate(imgs):
#                imgbasename=os.path.basename(img)
#                if imgbasename not in self.filter_by_files:
#                    filtered_imgs.append(img)
#                    filtered_labels.append(labels[index])
#            return filtered_imgs,filtered_labels
#        else:
#            return imgs, labels
#
#    def __getitem__(self, index):
#        """
#        Args:
#            index (int): Index
#        Returns:
#            tuple: (image, target) where target is class_index of the target class.
#        """
#        path = self.imgs[index]
#        target = self.labels[index]
#        img = self.loader(path)
#        if self.transform is not None:
#            img = self.transform(img)
#
#        if self.target_transform is not None:
#            target = self.target_transform(target)
#        
#        if self.return_paths:
#            return img, target, path
#        elif self.return_id:
#            return img, target, index
#        else:
#            return img, target
#
#    def __len__(self):
#        return len(self.imgs)


class ImageFolder(Dataset):
    """
    A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        image_list (list): List of image paths.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        allowed_classes (list, optional): List of allowed class names or class indices to filter.
        class_index_dict (dict, optional): Dictionary mapping class names to class indices.
        shuffle (bool, optional): If True, shuffles the dataset after loading.

    Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples.
    """
    def __init__(self, img_folder, transform=None, target_transform=None, return_paths=False,
                 loader=default_loader, train=False, return_id=False, class_index_dict=None, 
                 include_non_allowed=False, other_class_id=7, filter_by_files=None, shuffle=False):
        # Filter by allowed_classes if provided
        self.include_non_allowed = include_non_allowed
        self.class_index_dict = class_index_dict
        self.other_class_id = other_class_id
        self.filter_by_files = filter_by_files  # list of image name with .jpg extension
        
        # Create dataset with optional class filtering
        imgs, labels = self.make_dataset(img_folder)
        
        # Shuffle the dataset if requested
        if shuffle:
            indices = list(range(len(imgs)))
            random.shuffle(indices)
            imgs = [imgs[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.return_paths = return_paths
        self.return_id = return_id
        self.train = train
    
    def loop_image_folder(self, img_folder):
        imgs = []
        labels = []
        for class_name in os.listdir(img_folder):
            if self.class_index_dict is not None:
                try:
                    class_idx = self.class_index_dict[class_name]
                except:
                    class_idx = None
                if class_idx is not None:
                    for img in os.listdir(f"{img_folder}/{class_name}"):
                        imgs.append(f"{img_folder}/{class_name}/{img}")
                        labels.append(class_idx)
                elif self.include_non_allowed:
                    for img in os.listdir(f"{img_folder}/{class_name}"):
                        imgs.append(f"{img_folder}/{class_name}/{img}")
                        labels.append(self.other_class_id)
                else:
                    print("Error : cannot find the class in class dict and incldued_non_allowed is False")
            else:
                print("Error : cannot find class dict, should be Class123:0, 0:Class123")
        return imgs, labels
    
    def make_dataset(self, img_folder):
        """
        Creates a list of (image path, class_index) tuples.
        
        Args:
            image_list (list): List of image paths.

        Returns:
            tuple: (imgs, labels) where imgs is a list of image paths and labels are corresponding class indices.
        """
        imgs = []
        labels = []
        if isinstance(img_folder, list):
            for folder in img_folder:
                subimgs, sublabels = self.loop_image_folder(folder)
                imgs += subimgs
                labels += sublabels
        else:
            imgs, labels = self.loop_image_folder(img_folder)

        if self.filter_by_files is not None:
            filtered_imgs = []
            filtered_labels = []
            for index, img in enumerate(imgs):
                imgbasename = os.path.basename(img)
                if imgbasename not in self.filter_by_files:
                    filtered_imgs.append(img)
                    filtered_labels.append(labels[index])
            return filtered_imgs, filtered_labels
        else:
            return imgs, labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.imgs[index]
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.return_paths:
            return img, target, path
        elif self.return_id:
            return img, target, index
        else:
            return img, target

    def __len__(self):
        return len(self.imgs)

class custom_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices, all_labels,transform,return_paths=True):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.labels = [all_labels[i] for i in indices]
        self.return_paths=return_paths
        self.transform=transform
        #self.return_paths=return_paths
    def __getitem__(self, idx):
        img = self.dataset[idx][0]
        target = self.labels[idx]
        path = self.dataset[idx][-1]
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, target, path
        else:
            return img, target

    def __len__(self):
        return len(self.labels)

def sum_datasets(train_dataset,test_dataset,val_dataset,class_index_dict):
    clscount = {}
    def update_clscount(dataset, split_name):
        for l in dataset.labels:
            label=class_index_dict[l]
            if label not in clscount:
                clscount[label] = {split_name: 1}
            else:
                clscount[label][split_name] = clscount[label].get(split_name, 0) + 1

    update_clscount(train_dataset, 'train')
    update_clscount(test_dataset, 'test')
    update_clscount(val_dataset, 'val')

    # Convert clscount to a pandas DataFrame
    df_clscount = pd.DataFrame.from_dict(clscount, orient='index').fillna(0).astype(int)

    # Reset index for a cleaner look
    df_clscount.reset_index(inplace=True)
    df_clscount.rename(columns={'index': 'label'}, inplace=True)
    # Add a sum row
    total_row = df_clscount[['train', 'test', 'val']].sum()
    total_row['label'] = 'Total'
    df_clscount = pd.concat([df_clscount, pd.DataFrame([total_row])], ignore_index=True)
    print(df_clscount)
    return(df_clscount)
    #logger = logging.getLogger(__name__)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets,_ in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def accuracy(predictions, targets):
    """
    Computes accuracy between predicted class and true class.
    
    Args:
    predictions (torch.Tensor): The predicted labels (batch_size, num_classes).
    targets (torch.Tensor): The true labels (batch_size).
    
    Returns:
    float: The accuracy as a percentage (between 0 and 100).
    """
    # Get the index of the maximum predicted score (final prediction)
    _, predicted_labels = predictions.max(1)
    
    # Compare predictions with the true labels
    correct = predicted_labels.eq(targets).sum().item()
    
    # Calculate accuracy as percentage
    accuracy = correct / targets.size(0) * 100.0
    
    return accuracy

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

logger = logging.getLogger(__name__)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, args=None):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.args=args

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        #device = (torch.device('cuda')
        #          if features.is_cuda
        #          else torch.device('cpu'))
        device=self.args.device
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



        # test function

def test(model,val_dataset,valloader,print_acc=False,class_index_dict=None,species_dict=None,device='cpu',return_pred_stats=False):
    pred_stats_dict={}
    model.eval()
    if type(val_dataset) is torch.utils.data.dataset.ConcatDataset:
        gt_labels=[]
        for d in val_dataset.datasets:
            gt_labels+=d.labels
    else:
        gt_labels=val_dataset.labels
    preds=[]
    with torch.no_grad():
        for x, y, imgs in valloader:
            x, y = x.cuda(device), y.cuda(device)
            try:
                output = model(x)
                probabilities = torch.softmax(output, dim=1)
            except:
                #for feature model
                output,_ = model(x)
                probabilities = torch.softmax(output, dim=1)
            # Get maximum logits and the corresponding predicted class
            # Get the predicted probabilities
            #probabilities = torch.softmax(output, dim=1)
            # Get the max probability and predicted class
            max_probs, predicted_classes = torch.max(probabilities, dim=1)
            # Classify based on the threshold for OOD detection
            for i in range(x.size(0)):
                preds.append(predicted_classes[i].item())  # Use predicted class for ID
                imgpath=imgs[i]
                pred=int(predicted_classes[i].item())
                prob=float(max_probs[i].detach().cpu())
                pselb=str(probabilities[i].detach().cpu().numpy().tolist())
                pred_stats_dict[imgpath]={'pred':pred,
                                          'prob':prob,
                                          'pselb':pselb,
                                          'gt':gt_labels[i]}
    if class_index_dict is not None and species_dict is not None:
        gt_labels=[species_dict[class_index_dict[i]] for i in gt_labels]
        preds=[species_dict[class_index_dict[i]] for i in preds]
    accDict=classification_report(gt_labels[:len(preds)],preds,digits=4,output_dict=True)
    if print_acc:
        print(classification_report(gt_labels[:len(preds)],preds,digits=4,output_dict=False))
    if return_pred_stats:
        return accDict,pred_stats_dict
    else:
        return accDict

def test_maxlogit(model,val_dataset,valloader,threshold=-6.5,ood_class=6,class_index_dict=None,species_dict=None,device='cpu'):
    model.eval()
    gt_labels=val_dataset.labels
    preds=[]
    with torch.no_grad():
        for x, y, _ in valloader:
            x, y = x.cuda(device), y.cuda(device)
            try:
                output = model(x)
            except:
                #for feature model
                output,_ = model(x)
            # Get maximum logits and the corresponding predicted class
            # Get the predicted probabilities
            probabilities = torch.softmax(output, dim=1)
            # Get the max probability and predicted class
            max_probs, predicted_classes = torch.max(probabilities, dim=1)
            maxlogits=torch.min(output, dim=1)
        
            # Classify based on the threshold for OOD detection
            for i in range(x.size(0)):
                if maxlogits.values[i] < threshold:
                    preds.append(ood_class)  # Classify as OOD
                else:
                    preds.append(predicted_classes[i].item())  # Use predicted class for ID
                # Track accuracy for ID samples (only if the ground truth is not OOD)
                if y[i].item() != ood_class:
                    correct = (predicted_classes[i].item() == y[i].item())

    if class_index_dict is not None and species_dict is not None:
        gt_labels=[species_dict[class_index_dict[i]] for i in gt_labels]
        preds=[species_dict[class_index_dict[i]] for i in preds]
    accDict=classification_report(gt_labels,preds,digits=4,output_dict=True)
    print(classification_report(gt_labels,preds,digits=4,output_dict=False))
    return accDict

def train_supervised(
        model=None,
        labeled_trainloader=None,
        val_dataset=None,
        valloader=None,
        test_dataset=None,
        testloader=None,
        optimizer=None,
        scheduler=None,
        args=None,
        class_index_dict=None,
        criterion=nn.CrossEntropyLoss(),
        save_best=True,
        save_epoch=False,
        return_model=False,
        **kwargs):
    # supervised train with TensorBoard writer
    log_dir = os.path.join(f'{args.out}/tensorboard', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    trainLogDict={
        'epoch':[],
        'loss':[],
        'lr':[],
        'val_f1':[],
        'val_acc_dict':[],
        'test_f1':[],
        'test_acc_dict':[]
    }
    #args.epochs=48
    #args.lr=0.01                                        
    #optimizer=optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False)
    #optimizer=optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9)
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    #supervised train
    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    model.to(args.device)
    #criterion = nn.CrossEntropyLoss()
    epoch_iter = tqdm(list(range(1, args.epochs+1)), total=args.epochs, desc='Epoch',leave=True, position=1)
    #num_classes=args.num_classes
    iternumber=len(labeled_trainloader)  
    best_acc1 = 0
    bset_acc1_test=0
    best_test_acc_epoch=0
    best_acc1_epoch=0
    for epoch in epoch_iter:
        model.train()  # enter train mode
        current_lr = scheduler.get_last_lr()[0]
        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('batch_time', ':6.2f')
        data_time = AverageMeter('data_time', ':6.2f')
        p_bar = tqdm(range(iternumber))
        # Log learning rate
        writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
        for batch_idx, (x, y, _) in enumerate(labeled_trainloader):
            #if args.mixup:
            #    mx, my,_ =mixup_data(x,y)
            #    x, y = x.cuda(args.device), y.cuda(args.device).long()
            #    mx, my = mx.cuda(args.device), my.cuda(args.device).long()
            #    x = torch.cat([x, mx], dim=0)
            #    y = torch.cat([y, my], dim=0)
            #else:
            #    x, y = x.cuda(args.device), y.cuda(args.device).long()
            x, y = x.cuda(args.device), y.cuda(args.device).long()
            logits = model(x)
            id_loss = criterion(logits, y)
            loss = id_loss     
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()     
            losses.update(loss.item(), x.size(0))
            #id_losses.update(id_loss.item(), x.size(0))     
            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                epoch=epoch,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=iternumber,
                lr=optimizer.param_groups[0]["lr"],
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg))
            p_bar.update()
        
        # End of epoch training metrics
        writer.add_scalar('Training/Batch_Total_Loss', loss.item(), epoch * iternumber + batch_idx)
        #writer.add_scalar('Training/Batch_ID_Loss', id_loss.item(), epoch * iternumber + batch_idx) 
        scheduler.step()
        p_bar.close()
        # Evaluation phase
        accDict = test(model=model, val_dataset=val_dataset, valloader=valloader,device=args.device)
        acc1 = accDict['macro avg']['f1-score']
        testAccDict=test(model=model, val_dataset=test_dataset, valloader=testloader,device=args.device)
        testAcc1=testAccDict['macro avg']['f1-score']

        # Log averaged metrics
        writer.add_scalar('Validation/Accuracy', accDict['accuracy'], epoch)
        writer.add_scalar('Validation/Macro_F1', accDict['macro avg']['f1-score'], epoch)
        writer.add_scalar('Validation/Weighted_F1', accDict['weighted avg']['f1-score'], epoch)
        #writer.add_scalar('Validation/Averaged_Metrics/macro_precision', accDict['macro avg']['precision'], epoch)
        #writer.add_scalar('Validation/Averaged_Metrics/macro_recall', accDict['macro avg']['recall'], epoch)
        #writer.add_scalar('Validation/Averaged_Metrics/macro_f1', accDict['macro avg']['f1-score'], epoch)
        #writer.add_scalar('Validation/Averaged_Metrics/weighted_precision', accDict['weighted avg']['precision'], epoch)
        #writer.add_scalar('Validation/Averaged_Metrics/weighted_recall', accDict['weighted avg']['recall'], epoch)
        #writer.add_scalar('Validation/Averaged_Metrics/weighted_f1', accDict['weighted avg']['f1-score'], epoch)
        writer.add_scalar('Test/Accuracy', accDict['accuracy'], epoch)
        writer.add_scalar('Test/Macro_F1', accDict['macro avg']['f1-score'], epoch)
        writer.add_scalar('Test/Weighted_F1', accDict['weighted avg']['f1-score'], epoch)

        #writer.add_scalar('Validation/Averaged_Metrics', {
        #    'macro_precision': accDict['macro avg']['precision'],
        #    'macro_recall': accDict['macro avg']['recall'],
        #    'macro_f1': accDict['macro avg']['f1-score'],
        #    'weighted_precision': accDict['weighted avg']['precision'],
        #    'weighted_recall': accDict['weighted avg']['recall'],
        #    'weighted_f1': accDict['weighted avg']['f1-score']
        #}, epoch)
        trainLogDict['epoch'].append(epoch)
        trainLogDict['loss'].append(loss.detach().cpu())
        trainLogDict['lr'].append(current_lr)
        trainLogDict['val_f1'].append(accDict['macro avg']['f1-score'])
        trainLogDict['val_acc_dict'].append(accDict)
        trainLogDict['test_f1'].append(testAccDict['macro avg']['f1-score'])
        trainLogDict['test_acc_dict'].append(testAccDict)

        if epoch in args.save_epoch_list and save_epoch:
            model_to_save = model.module if hasattr(model, "module") else best_model
            state={
                'epoch': epoch,
                'state_dict': model_to_save.state_dict(),
                #'acc_in': acc1,
                #'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            save_checkpoint(state,f"{args.out}\epochs",filename=rf"epoch_{epoch}.pth.tar")
        if testAcc1>bset_acc1_test:
            bset_acc1_test=testAcc1
            best_test_acc_epoch=epoch
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_model = copy.deepcopy(model)
            val_test_acc=testAcc1
            best_acc1_epoch=epoch
            if save_best:
                model_to_save = best_model.module if hasattr(model, "module") else best_model
                state={
                    'epoch': epoch,
                    'state_dict': best_model.state_dict(),
                    #'acc_in': acc1,
                    #'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                save_checkpoint(state,args.out,filename=rf"model_best.pth.tar")
            # Log best model metrics
            writer.add_scalar('Validation/Best_Macro_F1', best_acc1, epoch) 
        print(f"current tesdt/val acc: {testAcc1}/{acc1}, best val acc macro f1 : {best_acc1} @epoch{best_acc1_epoch}(test acc:{val_test_acc}),best test acc:{bset_acc1_test}@epoch{best_test_acc_epoch}")
    # Close TensorBoard writer when training is complete
    writer.close()
    if return_model:
        return trainLogDict,model,best_model
    else:
        return trainLogDict

def train_supervised_consistency_loss(
        model=None,
        labeled_trainloader=None,
        val_dataset=None,
        valloader=None,
        test_dataset=None,
        testloader=None,
        optimizer=None,
        scheduler=None,
        args=None,
        class_index_dict=None,
        criterion=nn.CrossEntropyLoss(),
        save_best=True,
        save_epoch=False,
        return_model=False,
        **kwargs):
    # supervised train with TensorBoard writer
    log_dir = os.path.join(f'{args.out}/tensorboard', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    trainLogDict={
        'epoch':[],
        'loss':[],
        'lr':[],
        'val_f1':[],
        'val_acc_dict':[],
        'test_f1':[],
        'test_acc_dict':[]
    }
    #args.epochs=48
    #args.lr=0.01                                        
    #optimizer=optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False)
    #optimizer=optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9)
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    #supervised train
    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    model.to(args.device)
    #criterion = nn.CrossEntropyLoss()
    epoch_iter = tqdm(list(range(1, args.epochs+1)), total=args.epochs, desc='Epoch',leave=True, position=1)
    #num_classes=args.num_classes
    iternumber=len(labeled_trainloader)  
    best_acc1 = 0
    bset_acc1_test=0
    best_test_acc_epoch=0
    best_acc1_epoch=0
    for epoch in epoch_iter:
        model.train()  # enter train mode
        current_lr = scheduler.get_last_lr()[0]
        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('batch_time', ':6.2f')
        data_time = AverageMeter('data_time', ':6.2f')
        p_bar = tqdm(range(iternumber))
        # Log learning rate
        writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
        for batch_idx, ((x,strong_x), y, _) in enumerate(labeled_trainloader):
            x, strong_x, y = x.to(device), strong_x.to(device), y.to(device)
            logits = model(x)
            probabilities = torch.softmax(logits, dim=1)
            strong_output,_=model(strong_x, return_features=True)
            strong_probabilities=torch.softmax(strong_output, dim=1)

            conf_consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()

            conf_class = probabilities.clone()
            conf_class_flip = strong_probabilities.clone()

            consistency_conf_loss_a = conf_consistency_criterion(conf_class.log(),
                                                                conf_class_flip.detach()).sum(-1)
            consistency_conf_loss_b = conf_consistency_criterion(conf_class_flip.log(),
                                                                conf_class.detach()).sum(-1)
            consistency_loss= (consistency_conf_loss_a + consistency_conf_loss_b).sum() / 2
            id_loss = criterion(logits, y)

            loss = id_loss + consistency_loss     
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()     
            losses.update(loss.item(), x.size(0))
            #id_losses.update(id_loss.item(), x.size(0))     
            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                epoch=epoch,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=iternumber,
                lr=optimizer.param_groups[0]["lr"],
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg))
            p_bar.update()
        
        # End of epoch training metrics
        writer.add_scalar('Training/Batch_Total_Loss', loss.item(), epoch * iternumber + batch_idx)
        #writer.add_scalar('Training/Batch_ID_Loss', id_loss.item(), epoch * iternumber + batch_idx) 
        scheduler.step()
        p_bar.close()
        # Evaluation phase
        accDict = test(model=model, val_dataset=val_dataset, valloader=valloader,device=args.device)
        acc1 = accDict['macro avg']['f1-score']
        testAccDict=test(model=model, val_dataset=test_dataset, valloader=testloader,device=args.device)
        testAcc1=testAccDict['macro avg']['f1-score']
        # Log validation metrics
        
        # Log per-class metrics
        #psplog={}
        #scalarname=f'Per_Class_Metrics'
        #for class_idx in range(args.num_classes):
        #    if str(class_idx) in accDict:  # Only log if class exists in results       
        #        psplog[class_index_dict[class_idx]]= accDict[str(class_idx)]['f1-score']
       # writer.add_scalars(scalarname,psplog,epoch)
                #writer.add_scalars(scalarname, 
                #           {class_index_dict[class_idx]: accDict[str(class_idx)]['f1-score']}, 
                #           epoch)
                #writer.add_scalar(f'Validation/{class_index_dict[class_idx]}_Metrics/precision', accDict[str(class_idx)]['precision']
                #writer.add_scalar(f'Validation/{class_index_dict[class_idx]}_Metrics/recall', accDict[str(class_idx)]['recall']
                #    'recall': accDict[str(class_idx)]['recall'],
                #    'f1_score': accDict[str(class_idx)]['f1-score']
                #}, epoch)
        # Log averaged metrics
        writer.add_scalar('Validation/Accuracy', accDict['accuracy'], epoch)
        writer.add_scalar('Validation/Macro_F1', accDict['macro avg']['f1-score'], epoch)
        writer.add_scalar('Validation/Weighted_F1', accDict['weighted avg']['f1-score'], epoch)
        #writer.add_scalar('Validation/Averaged_Metrics/macro_precision', accDict['macro avg']['precision'], epoch)
        #writer.add_scalar('Validation/Averaged_Metrics/macro_recall', accDict['macro avg']['recall'], epoch)
        #writer.add_scalar('Validation/Averaged_Metrics/macro_f1', accDict['macro avg']['f1-score'], epoch)
        #writer.add_scalar('Validation/Averaged_Metrics/weighted_precision', accDict['weighted avg']['precision'], epoch)
        #writer.add_scalar('Validation/Averaged_Metrics/weighted_recall', accDict['weighted avg']['recall'], epoch)
        #writer.add_scalar('Validation/Averaged_Metrics/weighted_f1', accDict['weighted avg']['f1-score'], epoch)
        writer.add_scalar('Test/Accuracy', accDict['accuracy'], epoch)
        writer.add_scalar('Test/Macro_F1', accDict['macro avg']['f1-score'], epoch)
        writer.add_scalar('Test/Weighted_F1', accDict['weighted avg']['f1-score'], epoch)

        #writer.add_scalar('Validation/Averaged_Metrics', {
        #    'macro_precision': accDict['macro avg']['precision'],
        #    'macro_recall': accDict['macro avg']['recall'],
        #    'macro_f1': accDict['macro avg']['f1-score'],
        #    'weighted_precision': accDict['weighted avg']['precision'],
        #    'weighted_recall': accDict['weighted avg']['recall'],
        #    'weighted_f1': accDict['weighted avg']['f1-score']
        #}, epoch)
        trainLogDict['epoch'].append(epoch)
        trainLogDict['loss'].append(loss.detach().cpu())
        trainLogDict['lr'].append(current_lr)
        trainLogDict['val_f1'].append(accDict['macro avg']['f1-score'])
        trainLogDict['val_acc_dict'].append(accDict)
        trainLogDict['test_f1'].append(testAccDict['macro avg']['f1-score'])
        trainLogDict['test_acc_dict'].append(testAccDict)

        if epoch in args.save_epoch_list and save_epoch:
            model_to_save = model.module if hasattr(model, "module") else best_model
            state={
                'epoch': epoch,
                'state_dict': model_to_save.state_dict(),
                #'acc_in': acc1,
                #'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            save_checkpoint(state,f"{args.out}\epochs",filename=rf"epoch_{epoch}.pth.tar")
        if testAcc1>bset_acc1_test:
            bset_acc1_test=testAcc1
            best_test_acc_epoch=epoch
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_model = copy.deepcopy(model)
            val_test_acc=testAcc1
            best_acc1_epoch=epoch
            if save_best:
                model_to_save = best_model.module if hasattr(model, "module") else best_model
                state={
                    'epoch': epoch,
                    'state_dict': best_model.state_dict(),
                    #'acc_in': acc1,
                    #'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                save_checkpoint(state,args.out,filename=rf"model_best.pth.tar")
            # Log best model metrics
            writer.add_scalar('Validation/Best_Macro_F1', best_acc1, epoch) 
        print(f"current tesdt/val acc: {testAcc1}/{acc1}, best val acc macro f1 : {best_acc1} @epoch{best_acc1_epoch}(test acc:{val_test_acc}),best test acc:{bset_acc1_test}@epoch{best_test_acc_epoch}")
    # Close TensorBoard writer when training is complete
    writer.close()
    if return_model:
        return trainLogDict,model,best_model
    else:
        return trainLogDict

def train_supervised_triplet(
        model=None,
        labeled_trainloader=None,
        val_dataset=None,
        valloader=None,
        test_dataset=None,
        testloader=None,
        optimizer=None,
        scheduler=None,
        args=None,
        class_index_dict=None,
        loss_func=None,
        miner=None,
        save_best=True,
        save_epoch=False,
        return_model=False,
        **kwargs):
    # supervised train with TensorBoard writer
    log_dir = os.path.join(f'{args.out}/tensorboard', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    trainLogDict={
        'epoch':[],
        'loss':[],
        'lr':[],
        'val_f1':[],
        'val_acc_dict':[],
        'test_f1':[],
        'test_acc_dict':[]
    }
    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    model.to(args.device)
    #criterion = nn.CrossEntropyLoss()
    epoch_iter = tqdm(list(range(1, args.epochs+1)), total=args.epochs, desc='Epoch',leave=True, position=1)
    #num_classes=args.num_classes
    iternumber=len(labeled_trainloader)  
    best_acc1 = 0
    bset_acc1_test=0
    best_test_acc_epoch=0
    best_acc1_epoch=0
    for epoch in epoch_iter:
        model.train()  # enter train mode
        current_lr = scheduler.get_last_lr()[0]
        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('batch_time', ':6.2f')
        data_time = AverageMeter('data_time', ':6.2f')
        p_bar = tqdm(range(iternumber))
        # Log learning rate
        writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
        for batch_idx, (x, y, _) in enumerate(labeled_trainloader):
            x, y = x.cuda(args.device), y.cuda(args.device).long()
            logits,embeddings = model(x,return_features=True)
            #embeddings.to(args.device)
            hard_pairs = miner(embeddings, y)
            loss = loss_func(embeddings, y, hard_pairs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()     
            losses.update(loss.item(), x.size(0))
            #id_losses.update(id_loss.item(), x.size(0))     
            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                epoch=epoch,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=iternumber,
                lr=optimizer.param_groups[0]["lr"],
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg))
            p_bar.update()
        
        # End of epoch training metrics
        writer.add_scalar('Training/Batch_Total_Loss', loss.item(), epoch * iternumber + batch_idx)
        #writer.add_scalar('Training/Batch_ID_Loss', id_loss.item(), epoch * iternumber + batch_idx) 
        scheduler.step()
        p_bar.close()
        # Evaluation phase
        accDict = test(model=model, val_dataset=val_dataset, valloader=valloader,device=args.device)
        acc1 = accDict['macro avg']['f1-score']
        testAccDict=test(model=model, val_dataset=test_dataset, valloader=testloader,device=args.device)
        testAcc1=testAccDict['macro avg']['f1-score']
  
        # Log averaged metrics
        writer.add_scalar('Validation/Accuracy', accDict['accuracy'], epoch)
        writer.add_scalar('Validation/Macro_F1', accDict['macro avg']['f1-score'], epoch)
        writer.add_scalar('Validation/Weighted_F1', accDict['weighted avg']['f1-score'], epoch)
    
        writer.add_scalar('Test/Accuracy', accDict['accuracy'], epoch)
        writer.add_scalar('Test/Macro_F1', accDict['macro avg']['f1-score'], epoch)
        writer.add_scalar('Test/Weighted_F1', accDict['weighted avg']['f1-score'], epoch)
        trainLogDict['epoch'].append(epoch)
        trainLogDict['loss'].append(loss.detach().cpu())
        trainLogDict['lr'].append(current_lr)
        trainLogDict['val_f1'].append(accDict['macro avg']['f1-score'])
        trainLogDict['val_acc_dict'].append(accDict)
        trainLogDict['test_f1'].append(testAccDict['macro avg']['f1-score'])
        trainLogDict['test_acc_dict'].append(testAccDict)

        if epoch in args.save_epoch_list and save_epoch:
            model_to_save = model.module if hasattr(model, "module") else best_model
            state={
                'epoch': epoch,
                'state_dict': model_to_save.state_dict(),
                #'acc_in': acc1,
                #'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            save_checkpoint(state,f"{args.out}\epochs",filename=rf"epoch_{epoch}.pth.tar")
        if testAcc1>bset_acc1_test:
            bset_acc1_test=testAcc1
            best_test_acc_epoch=epoch
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_model = copy.deepcopy(model)
            val_test_acc=testAcc1
            best_acc1_epoch=epoch
            if save_best:
                model_to_save = best_model.module if hasattr(model, "module") else best_model
                state={
                    'epoch': epoch,
                    'state_dict': best_model.state_dict(),
                    #'acc_in': acc1,
                    #'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                save_checkpoint(state,args.out,filename=rf"model_best.pth.tar")
            # Log best model metrics
            writer.add_scalar('Validation/Best_Macro_F1', best_acc1, epoch) 
        print(f"current tesdt/val acc: {testAcc1}/{acc1}, best val acc macro f1 : {best_acc1} @epoch{best_acc1_epoch}(test acc:{val_test_acc}),best test acc:{bset_acc1_test}@epoch{best_test_acc_epoch}")
    # Close TensorBoard writer when training is complete
    writer.close()
    if return_model:
        return trainLogDict,model,best_model
    else:
        return trainLogDict

def clear_memory():
    torch.cuda.empty_cache()

def train_supcon_backbone(
        mode='SupCon',
        model=None,
        labeled_trainloader=None,
        optimizer=None,
        scheduler=None,
        args=None,
        #class_index_dict=None,
        loss_func=None,
        #save_best=True,
        save_model=False,
        return_model=False,
        **kwargs):
    
    lowest_loss=0
    model.to(args.device)
    epoch_iter = tqdm(list(range(1, args.epochs+1)), total=args.epochs, desc='Epoch',
        leave=True, position=1)
    #num_classes=args.num_classes
    iternumber=len(labeled_trainloader)     
    scaler = GradScaler()
    accumulation_steps = 4
    for epoch in epoch_iter:
        model.train()
        current_lr = scheduler.get_last_lr()[0]
        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('batch_time', ':6.2f')
        data_time = AverageMeter('data_time', ':6.2f')
        p_bar = tqdm(range(iternumber))
        for batch_idx, (images, y, _) in enumerate(labeled_trainloader):
            x = torch.cat([images[0], images[1]], dim=0)
            bsz = y.shape[0]
            #if args.mixup:
            #    mx, y, _ = mixup_data(images, y)
            #    x = torch.cat([mx[0], mx[1]], dim=0)
            x, y = x.cuda(args.device), y.cuda(args.device).long() 
            with autocast():
                pretrain_logits, _ = model(x)
                f1, f2 = torch.split(pretrain_logits, [bsz, bsz], dim=0)
                pretrain_logits_con = torch.stack([f1, f2], dim=1)
                if mode=='SupCon':
                    loss = loss_func(pretrain_logits_con, y) / accumulation_steps
                elif mode=='SimCLR':
                    loss = loss_func(pretrain_logits_con) / accumulation_steps      
            scaler.scale(loss).backward()
            #if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            losses.update(loss.item() * accumulation_steps, bsz)
            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                            epoch=epoch,
                            epochs=args.epochs,
                            batch=batch_idx + 1,
                            iter=iternumber,
                            lr=current_lr,
                            data=data_time.avg,
                            bt=batch_time.avg,
                            loss=losses.avg))
                            #loss_x=losses_x.avg))
            p_bar.update()
            #writer.add_scalar('Training/Batch_Total_Loss', loss.item(), epoch * iternumber + batch_idx)
        p_bar.close()
        scheduler.step()
        clear_memory()
        if epoch == 1 or losses.avg < lowest_loss:
            lowest_model=model
            #print(f'update lowest model with loss of {lowest_loss}')
            lowest_loss = losses.avg
            #best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f'update lowest model with loss of {lowest_loss}')
        outfolder=rf"{args.out}\feature_extractor"
    if save_model:
        if os.path.exists(outfolder):
            pass
        else:
            os.makedirs(outfolder)
        model_to_save = model.module if hasattr(model, "module") else model
        state={
            'epoch': epoch,
            'state_dict': model_to_save.state_dict(),
            #'acc_in': acc1,
            #'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        #print('Best top-1 acc: {:.2f}'.format(best_acc))
        save_checkpoint(state, outfolder,filename="last_feature_extractor.pth.tar")
        model_to_save = model.module if hasattr(model, "module") else lowest_model
        state={
            'epoch': epoch,
            'state_dict': model_to_save.state_dict(),
            #'acc_in': acc1,
            #'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        #print('Best top-1 acc: {:.2f}'.format(best_acc))
        save_checkpoint(state, outfolder,filename="lowest_loss_feature_extractor.pth.tar")
    if return_model:
        return model,lowest_model

def train_supcon_head(
        model=None,
        labeled_trainloader=None,
        val_dataset=None,
        valloader=None,
        test_dataset=None,
        testloader=None,
        optimizer=None,
        scheduler=None,
        args=None,
        class_index_dict=None,
        criterion=nn.CrossEntropyLoss(),
        save_best=True,
        save_epoch=False,
        return_model=False,
        **kwargs):

    #model.to(args.device)
    # Freeze the backbone (i.e., all layers except the final classification layer)
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    # supervised train with TensorBoard writer
    log_dir = os.path.join(f'{args.out}/tensorboard', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    trainLogDict={
        'epoch':[],
        'loss':[],
        'lr':[],
        'val_f1':[],
        'val_acc_dict':[],
        'test_f1':[],
        'test_acc_dict':[]
    }
    #supervised train
    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    model.to(args.device)
    #criterion = nn.CrossEntropyLoss()
    epoch_iter = tqdm(list(range(1, args.epochs+1)), total=args.epochs, desc='Epoch',leave=True, position=1)
    #num_classes=args.num_classes
    iternumber=len(labeled_trainloader)  
    best_acc1 = 0
    bset_acc1_test=0
    best_test_acc_epoch=0
    best_acc1_epoch=0
    for epoch in epoch_iter:
        model.train()  # enter train mode
        current_lr = scheduler.get_last_lr()[0]
        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('batch_time', ':6.2f')
        data_time = AverageMeter('data_time', ':6.2f')
        p_bar = tqdm(range(iternumber))
        # Log learning rate
        writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
        for batch_idx, (x, y, _) in enumerate(labeled_trainloader):
            x, y = x.cuda(args.device), y.cuda(args.device).long()
            logits,_ = model(x)
            loss = criterion(logits, y)   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()     
            losses.update(loss.item(), x.size(0))
            #id_losses.update(id_loss.item(), x.size(0))     
            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                epoch=epoch,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=iternumber,
                lr=optimizer.param_groups[0]["lr"],
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg))
            p_bar.update()
        
        # End of epoch training metrics
        writer.add_scalar('Training/Batch_Total_Loss', loss.item(), epoch * iternumber + batch_idx)
        #writer.add_scalar('Training/Batch_ID_Loss', id_loss.item(), epoch * iternumber + batch_idx) 
        scheduler.step()
        p_bar.close()
        # Evaluation phase
        accDict = test(model=model, val_dataset=val_dataset, valloader=valloader,device=args.device)
        acc1 = accDict['macro avg']['f1-score']
        testAccDict=test(model=model, val_dataset=test_dataset, valloader=testloader,device=args.device)
        testAcc1=testAccDict['macro avg']['f1-score']

        # Log averaged metrics
        writer.add_scalar('Validation/Accuracy', accDict['accuracy'], epoch)
        writer.add_scalar('Validation/Macro_F1', accDict['macro avg']['f1-score'], epoch)
        writer.add_scalar('Validation/Weighted_F1', accDict['weighted avg']['f1-score'], epoch)
        writer.add_scalar('Test/Accuracy', accDict['accuracy'], epoch)
        writer.add_scalar('Test/Macro_F1', accDict['macro avg']['f1-score'], epoch)
        writer.add_scalar('Test/Weighted_F1', accDict['weighted avg']['f1-score'], epoch)

        trainLogDict['epoch'].append(epoch)
        trainLogDict['loss'].append(loss.detach().cpu())
        trainLogDict['lr'].append(current_lr)
        trainLogDict['val_f1'].append(accDict['macro avg']['f1-score'])
        trainLogDict['val_acc_dict'].append(accDict)
        trainLogDict['test_f1'].append(testAccDict['macro avg']['f1-score'])
        trainLogDict['test_acc_dict'].append(testAccDict)

        if epoch in args.save_epoch_list and save_epoch:
            model_to_save = model.module if hasattr(model, "module") else best_model
            state={
                'epoch': epoch,
                'state_dict': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            save_checkpoint(state,f"{args.out}\epochs",filename=rf"epoch_{epoch}.pth.tar")
        if testAcc1>bset_acc1_test:
            bset_acc1_test=testAcc1
            best_test_acc_epoch=epoch
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_model = copy.deepcopy(model)
            val_test_acc=testAcc1
            best_acc1_epoch=epoch
            if save_best:
                model_to_save = best_model.module if hasattr(model, "module") else best_model
                state={
                    'epoch': epoch,
                    'state_dict': best_model.state_dict(),
                    #'acc_in': acc1,
                    #'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                save_checkpoint(state,args.out,filename=rf"model_best.pth.tar")
            # Log best model metrics
            writer.add_scalar('Validation/Best_Macro_F1', best_acc1, epoch) 
        print(f"current tesdt/val acc: {testAcc1}/{acc1}, best val acc macro f1 : {best_acc1} @epoch{best_acc1_epoch}(test acc:{val_test_acc}),best test acc:{bset_acc1_test}@epoch{best_test_acc_epoch}")
    # Close TensorBoard writer when training is complete
    writer.close()
    if return_model:
        return trainLogDict,model,best_model
    else:
        return trainLogDict
    
#fixmatch
class ModelEMA(object):
    def __init__(self, args, model, decay):
        self.ema = copy.deepcopy(model)
        self.ema.to(args.device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])

class RepeatBatchSampler(Sampler):
    def __init__(self, labeled_dataset, unlabeled_dataset, batch_size, mu):
        """
        Args:
            labeled_dataset: The labeled dataset
            unlabeled_dataset: The unlabeled dataset
            batch_size: Batch size for labeled data
            mu: Ratio between unlabeled and labeled samples in a batch
        """
        self.labeled_size = len(labeled_dataset)
        self.unlabeled_size = len(unlabeled_dataset)
        self.batch_size = batch_size
        self.mu = mu
        
        # Calculate how many times we need to repeat labeled data
        # to match unlabeled data considering batch ratio
        self.num_labeled_in_batch = batch_size
        self.num_unlabeled_in_batch = batch_size * mu
        
        # Number of batches is determined by unlabeled dataset
        self.num_batches = self.unlabeled_size // (self.batch_size * mu)
        
    def __iter__(self):
        # Create indices for labeled dataset that will be repeated
        labeled_indices = list(range(self.labeled_size))
        # Shuffle the indices
        np.random.shuffle(labeled_indices)
        # Repeat labeled indices enough times to cover all batches
        num_labeled_needed = self.num_batches * self.batch_size
        repeats_needed = math.ceil(num_labeled_needed / self.labeled_size)
        labeled_indices = labeled_indices * repeats_needed
        labeled_indices = labeled_indices[:num_labeled_needed]
        
      
        return iter(labeled_indices)
    
    def __len__(self):
        return self.num_batches * self.batch_size
    
def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def calculate_pseudo_label_accuracy(targets_u, ulb_labels, mask):
    # Get total number of samples
    targets_u=targets_u.to('cpu')
    mask=mask.to('cpu')
    ulb_labels=ulb_labels.to('cpu')
    total_samples = targets_u.size(0)
    # Get number of samples above threshold
    selected_samples = mask.sum().item()
    # Calculate percentage of samples above threshold
    percent_selected = (selected_samples / total_samples) * 100
    
    # If no samples above threshold, return 0 accuracy
    if selected_samples == 0:
        return 0.0, 0.0
    # Get correct predictions (only for samples above threshold)
    correct_predictions = (targets_u == ulb_labels) * mask
    # Calculate accuracy for selected samples
    accuracy = (correct_predictions.sum().item() / selected_samples) * 100
    return accuracy, percent_selected

def train_fixmatch(
        model=None,
        labeled_trainloader=None,
        unlabeled_trainloader=None,
        val_dataset=None,
        valloader=None,
        test_dataset=None,
        testloader=None,
        optimizer=None,
        scheduler=None,
        args=None,
        #class_index_dict=None,
        #criterion=nn.CrossEntropyLoss(),
        save_best=True,
        save_epoch=False,
        return_model=False,
        **kwargs):
    
    # supervised train with TensorBoard writer
    log_dir = os.path.join(f'{args.out}/tensorboard', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    trainLogDict={
        'epoch':[],
        'loss':[],
        'lr':[],
        'val_f1':[],
        'val_acc_dict':[],
        'test_f1':[],
        'test_acc_dict':[]
    }

    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    model.to(args.device)
    #criterion = nn.CrossEntropyLoss()
    epoch_iter = tqdm(list(range(1, args.epochs+1)), total=args.epochs, desc='Epoch',leave=True, position=1)
    #num_classes=args.num_classes
    iternumber=len(labeled_trainloader)  
    best_acc1 = 0
    bset_acc1_test=0
    best_test_acc_epoch=0
    best_acc1_epoch=0
    for epoch in epoch_iter:
        model.train()  # enter train mode
        current_lr = scheduler.get_last_lr()[0]
        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('batch_time', ':6.2f')
        data_time = AverageMeter('data_time', ':6.2f')
        p_bar = tqdm(range(iternumber))
        losses_u = AverageMeter('Loss_u', ':.4e')
        losses_x = AverageMeter('Loss_x', ':.4e')
        # Log learning rate
        unlabeled_iter = iter(unlabeled_trainloader)
        writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
        for batch_idx, (x, y, _) in enumerate(labeled_trainloader):
            x, y = x.to(args.device), y.to(args.device).long()

            try:
                (inputs_u_w, inputs_u_s), ulb_labels, ulb_imgs = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                
                (inputs_u_w, inputs_u_s), ulb_labels, ulb_imgs = next(unlabeled_iter)
            
            inputs_u_w = inputs_u_w.to(args.device)
            inputs_u_s = inputs_u_s.to(args.device)
            batch_size = x.shape[0]
            inputs=torch.cat((x, inputs_u_w, inputs_u_s)).to(args.device)
            
            
            all_logits = model(inputs)
            logits_x = all_logits[:batch_size]
            logits_u_w, logits_u_s = all_logits[batch_size:].chunk(2)

            del inputs
                #logits_u_w = model(inputs_u_w)
                #pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
                ##maxlogits=torch.max(logits_u_w, dim=1).values
                #max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                #mask_threshold=args.threshold
                #mask = max_probs.ge(mask_threshold).float()
            #inputs = torch.cat((x,inputs_u_w, inputs_u_s))
            
            

            # Calculate losses
            Lx = F.cross_entropy(logits_x, y, reduction='mean')
            with torch.no_grad():
                pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()

            Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
            
            loss = Lx + args.lambda_u * Lu

            # Update meters

            losses_x.update(Lx.item())
            losses_u.update(Lu.item())

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            losses.update(loss.item(), x.size(0))

            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. losses_u: {losses_u:.4f}. losses_x: {losses_x:.4f}.".format(
                epoch=epoch,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=iternumber,
                lr=optimizer.param_groups[0]["lr"],
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                losses_u=losses_u.avg,
                losses_x=losses_x.avg))
            p_bar.update()
        
        # End of epoch training metrics
        writer.add_scalar('Training/Batch_Total_Loss', loss.item(), epoch * iternumber + batch_idx)
        #writer.add_scalar('Training/Batch_ID_Loss', id_loss.item(), epoch * iternumber + batch_idx) 
        scheduler.step()
        p_bar.close()
        # Evaluation phase
        accDict = test(model=model, val_dataset=val_dataset, valloader=valloader,device=args.device)
        acc1 = accDict['macro avg']['f1-score']
        testAccDict=test(model=model, val_dataset=test_dataset, valloader=testloader,device=args.device)
        testAcc1=testAccDict['macro avg']['f1-score']

        # Log averaged metrics
        writer.add_scalar('Validation/Accuracy', accDict['accuracy'], epoch)
        writer.add_scalar('Validation/Macro_F1', accDict['macro avg']['f1-score'], epoch)
        writer.add_scalar('Validation/Weighted_F1', accDict['weighted avg']['f1-score'], epoch)
        writer.add_scalar('Test/Accuracy', accDict['accuracy'], epoch)
        writer.add_scalar('Test/Macro_F1', accDict['macro avg']['f1-score'], epoch)
        writer.add_scalar('Test/Weighted_F1', accDict['weighted avg']['f1-score'], epoch)

        trainLogDict['epoch'].append(epoch)
        trainLogDict['loss'].append(loss.detach().cpu())
        trainLogDict['lr'].append(current_lr)
        trainLogDict['val_f1'].append(accDict['macro avg']['f1-score'])
        trainLogDict['val_acc_dict'].append(accDict)
        trainLogDict['test_f1'].append(testAccDict['macro avg']['f1-score'])
        trainLogDict['test_acc_dict'].append(testAccDict)

        if epoch in args.save_epoch_list and save_epoch:
            model_to_save = model.module if hasattr(model, "module") else best_model
            state={
                'epoch': epoch,
                'state_dict': model_to_save.state_dict(),
                #'acc_in': acc1,
                #'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            save_checkpoint(state,f"{args.out}\epochs",filename=rf"epoch_{epoch}.pth.tar")
        if testAcc1>bset_acc1_test:
            bset_acc1_test=testAcc1
            best_test_acc_epoch=epoch
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_model = copy.deepcopy(model)
            val_test_acc=testAcc1
            best_acc1_epoch=epoch
            if save_best:
                model_to_save = best_model.module if hasattr(model, "module") else best_model
                state={
                    'epoch': epoch,
                    'state_dict': best_model.state_dict(),
                    #'acc_in': acc1,
                    #'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                save_checkpoint(state,args.out,filename=rf"model_best.pth.tar")
            # Log best model metrics
            writer.add_scalar('Validation/Best_Macro_F1', best_acc1, epoch) 
        print(f"current tesdt/val acc: {testAcc1}/{acc1}, best val acc macro f1 : {best_acc1} @epoch{best_acc1_epoch}(test acc:{val_test_acc}),best test acc:{bset_acc1_test}@epoch{best_test_acc_epoch}")
    # Close TensorBoard writer when training is complete
    writer.close()
    if return_model:
        return trainLogDict,model,best_model
    else:
        return trainLogDict

def train_fixmatch_with_pseudo_label_stats(
        model=None,
        labeled_trainloader=None,
        unlabeled_trainloader=None,
        val_dataset=None,
        valloader=None,
        test_dataset=None,
        testloader=None,
        optimizer=None,
        scheduler=None,
        args=None,
        #class_index_dict=None,
        #criterion=nn.CrossEntropyLoss(),
        save_best=True,
        save_epoch=False,
        return_model=False,
        **kwargs):
    pseudo_label_stats_dict={}
    # supervised train with TensorBoard writer
    log_dir = os.path.join(f'{args.out}/tensorboard', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    trainLogDict={
        'epoch':[],
        'loss':[],
        'lr':[],
        'val_f1':[],
        'val_acc_dict':[],
        'test_f1':[],
        'test_acc_dict':[]
    }

    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    model.to(args.device)
    #criterion = nn.CrossEntropyLoss()
    epoch_iter = tqdm(list(range(1, args.epochs+1)), total=args.epochs, desc='Epoch',leave=True, position=1)
    #num_classes=args.num_classes
    iternumber=len(labeled_trainloader)  
    best_acc1 = 0
    bset_acc1_test=0
    best_test_acc_epoch=0
    best_acc1_epoch=0
    for epoch in epoch_iter:
        pseudo_label_stats_dict[epoch]={}
        model.train()  # enter train mode
        current_lr = scheduler.get_last_lr()[0]
        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('batch_time', ':6.2f')
        data_time = AverageMeter('data_time', ':6.2f')
        p_bar = tqdm(range(iternumber))
        losses_u = AverageMeter('Loss_u', ':.4e')
        losses_x = AverageMeter('Loss_x', ':.4e')
        # Log learning rate
        unlabeled_iter = iter(unlabeled_trainloader)
        writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
        for batch_idx, (x, y, _) in enumerate(labeled_trainloader):
            x, y = x.to(args.device), y.to(args.device).long()

            try:
                (inputs_u_w, inputs_u_s), ulb_labels, ulb_imgs = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                
                (inputs_u_w, inputs_u_s), ulb_labels, ulb_imgs = next(unlabeled_iter)
            
            inputs_u_w = inputs_u_w.to(args.device)
            inputs_u_s = inputs_u_s.to(args.device)
            batch_size = x.shape[0]
            inputs=torch.cat((x, inputs_u_w, inputs_u_s)).to(args.device)
            
            
            all_logits = model(inputs)
            logits_x = all_logits[:batch_size]
            logits_u_w, logits_u_s = all_logits[batch_size:].chunk(2)

            del inputs
                #logits_u_w = model(inputs_u_w)
                #pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
                ##maxlogits=torch.max(logits_u_w, dim=1).values
                #max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                #mask_threshold=args.threshold
                #mask = max_probs.ge(mask_threshold).float()
            #inputs = torch.cat((x,inputs_u_w, inputs_u_s))
            
            

            # Calculate losses
            Lx = F.cross_entropy(logits_x, y, reduction='mean')
            with torch.no_grad():
                pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()
            for index,p in enumerate(pseudo_label):
                pred=int(targets_u[index].detach().cpu())
                prob=float(max_probs[index].detach().cpu())
                pselb=str(p.detach().cpu().numpy().tolist())
                imgpath=ulb_imgs[index]
                pseudo_label_stats_dict[epoch][imgpath]={'pred':pred,
                                                         'prob':prob,
                                                         'pselb':pselb,
                                                         'gt':None}
            Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
            
            loss = Lx + args.lambda_u * Lu

            # Update meters

            losses_x.update(Lx.item())
            losses_u.update(Lu.item())

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            losses.update(loss.item(), x.size(0))

            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. losses_u: {losses_u:.4f}. losses_x: {losses_x:.4f}.".format(
                epoch=epoch,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=iternumber,
                lr=optimizer.param_groups[0]["lr"],
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                losses_u=losses_u.avg,
                losses_x=losses_x.avg))
            p_bar.update()
        
        # End of epoch training metrics
        writer.add_scalar('Training/Batch_Total_Loss', loss.item(), epoch * iternumber + batch_idx)
        #writer.add_scalar('Training/Batch_ID_Loss', id_loss.item(), epoch * iternumber + batch_idx) 
        scheduler.step()
        p_bar.close()
        # Evaluation phase
        accDict,val_pred_stats = test(model=model, val_dataset=val_dataset, valloader=valloader,device=args.device,return_pred_stats=True)
        acc1 = accDict['macro avg']['f1-score']
        testAccDict, test_pred_stats=test(model=model, val_dataset=test_dataset, valloader=testloader,device=args.device,return_pred_stats=True)
        testAcc1=testAccDict['macro avg']['f1-score']
        for testimgpath,stats in test_pred_stats.items():
            pseudo_label_stats_dict[epoch][testimgpath]=stats
        # Log averaged metrics
        writer.add_scalar('Validation/Accuracy', accDict['accuracy'], epoch)
        writer.add_scalar('Validation/Macro_F1', accDict['macro avg']['f1-score'], epoch)
        writer.add_scalar('Validation/Weighted_F1', accDict['weighted avg']['f1-score'], epoch)
        writer.add_scalar('Test/Accuracy', accDict['accuracy'], epoch)
        writer.add_scalar('Test/Macro_F1', accDict['macro avg']['f1-score'], epoch)
        writer.add_scalar('Test/Weighted_F1', accDict['weighted avg']['f1-score'], epoch)

        trainLogDict['epoch'].append(epoch)
        trainLogDict['loss'].append(loss.detach().cpu())
        trainLogDict['lr'].append(current_lr)
        trainLogDict['val_f1'].append(accDict['macro avg']['f1-score'])
        trainLogDict['val_acc_dict'].append(accDict)
        trainLogDict['test_f1'].append(testAccDict['macro avg']['f1-score'])
        trainLogDict['test_acc_dict'].append(testAccDict)

        if epoch in args.save_epoch_list and save_epoch:
            model_to_save = model.module if hasattr(model, "module") else best_model
            state={
                'epoch': epoch,
                'state_dict': model_to_save.state_dict(),
                #'acc_in': acc1,
                #'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            save_checkpoint(state,f"{args.out}\epochs",filename=rf"epoch_{epoch}.pth.tar")
        if testAcc1>bset_acc1_test:
            bset_acc1_test=testAcc1
            best_test_acc_epoch=epoch
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_model = copy.deepcopy(model)
            val_test_acc=testAcc1
            best_acc1_epoch=epoch
            if save_best:
                model_to_save = best_model.module if hasattr(model, "module") else best_model
                state={
                    'epoch': epoch,
                    'state_dict': best_model.state_dict(),
                    #'acc_in': acc1,
                    #'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                save_checkpoint(state,args.out,filename=rf"model_best.pth.tar")
            # Log best model metrics
            writer.add_scalar('Validation/Best_Macro_F1', best_acc1, epoch) 
        print(f"current tesdt/val acc: {testAcc1}/{acc1}, best val acc macro f1 : {best_acc1} @epoch{best_acc1_epoch}(test acc:{val_test_acc}),best test acc:{bset_acc1_test}@epoch{best_test_acc_epoch}")
    # Close TensorBoard writer when training is complete
    writer.close()
    if return_model:
        return trainLogDict,model,best_model,pseudo_label_stats_dict
    else:
        return trainLogDict,pseudo_label_stats_dict