import os
import time
import datetime
import random
import torch
import psutil
from my_parser import parse_args_adni2_pet
from data import build_data_adni2_paired_data
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchsummary import summary
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import utils
import argparse
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
import math
import pandas as pd
import numpy as np
from torchsampler import ImbalancedDatasetSampler
from models import ViM
from models import vim_model
import gc

parser = argparse.ArgumentParser()
### Dataset
parser.add_argument("--path_dataset", type=str, default="/data/krdu/dataset")

### Model S/L
parser.add_argument("--load_path", type=str, default="save_models")
parser.add_argument("--load_model", action='store_true', default=True)
parser.add_argument("--load_name", type=str, default="multi_B_mae_1600")
parser.add_argument("--save_path", type=str, default="saved_models")
parser.add_argument("--model_name", type=str, default="vit_B")
parser.add_argument("--model_type", type=str, default="multi_B", help="Which model to use (vit_L, vit_B, etc.)")

### Training
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--input_D", type=int, default=128)
parser.add_argument("--input_H", type=int, default=128)
parser.add_argument("--input_W", type=int, default=128)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--optimizer", type=str, default='AdamW')
parser.add_argument("--scheduler", type=str, default='cosLR')
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--weight_decay", type=float, default=0.02)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--workers", type=int, default="12")
parser.add_argument("--data_parallel", action='store_true', default=False)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import psutil
def use_cpus(gpus: list):
    cpus = []
    for gpu in gpus:
        cpus.extend(list(range(gpu*12, (gpu+1)*12)))
    p = psutil.Process()
    p.cpu_affinity(cpus)
    print("A total {} CPUs are used, making sure that num_worker is small than the number of CPUs".format(len(cpus)))
use_cpus(gpus=[int(args.gpu)])


class focal_loss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma
        #
        # print('Focal Loss:')
        # print('    Alpha = {}'.format(self.alpha))
        # print('    Gamma = {}'.format(self.gamma))

    def forward(self, preds, labels):
        """
            focal_loss损失计算
            :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
            :param labels:  实际类别. size:[B,N] or [B]
            :return:
        """

        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

def lock_random_seed(seed) : 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def data_augmentation(x:np.ndarray) :
    rotate_max_deg = 10
    x_list = []
    RotMat = cv2.getRotationMatrix2D((x.shape[3] / 2, x.shape[2] / 2), rotate_max_deg * (random.random() * 2 - 1), 1.0)
    for i in range(x.shape[1]) :
        x_list.append(cv2.warpAffine(x[0, i], RotMat, (x.shape[3], x.shape[2])))
    x = np.stack(x_list, axis=0)
    x = np.expand_dims(x, axis=0)

    ### random rotate-nod
    x_list = []
    RotMat = cv2.getRotationMatrix2D((x.shape[2] / 2, x.shape[1] / 2), rotate_max_deg * (random.random() * 2 - 1), 1.0)
    for i in range(x.shape[3]) :
        x_list.append(cv2.warpAffine(x[0, ..., i], RotMat, (x.shape[2], x.shape[1])))
    x = np.stack(x_list, axis=2)
    x = np.expand_dims(x, axis=0)

    # if random.random() > 0.5 : ### upside down
    #     x = np.flip(x, axis=2)
    if random.random() > 0.5 : ### mirrorten
        x = np.flip(x, axis=3)
    # cv2.imwrite('./tmp.png', (x[0, x.shape[1] // 2] + 0.5) * 255)


    return x.copy()


def main():
    train_loader, val_loader, test_loader = build_data_adni2_paired_data(args.batch_size, args.input_D, args.input_H, args.input_W, workers=args.workers)
    for seed in [600,800,100]:
        utils.lock_random_seed(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('{} is available'.format(device))
        print('==> Preparing data..')
        root_dir = '/data/krdu/dataset/ADNI/processed/'
        

        net = vim_model(args).cuda()
        if args.load_model :
            print('loading weights form '+ os.path.join(args.load_path, args.load_name + '.pt'))
            net.load_state_dict(torch.load(os.path.join(args.load_path, args.load_name + '.pt')), strict=False)
        if not os.path.exists('logs'):
            os.makedirs('logs')
        if not os.path.exists('save_results'):
            os.makedirs('save_results')
        num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('The number of parameters of model is', num_params)
        weights = torch.Tensor([1,1,1]).to(device)
        #criterion = torch.nn.CrossEntropyLoss()s
        criterion = focal_loss(num_classes=2, gamma=3)
        optimizer = torch.optim.AdamW(params=net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=1e-7)
        max_epoch = args.epochs
        log_dir = 'logs/log_dir_4_8_'+args.model_type+'_'+str(args.epochs)+'_seed'+str(seed)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
        train(net, criterion, optimizer, train_loader, val_loader, test_loader, device, max_epoch, scheduler, writer, seed)
        torch.cuda.empty_cache()
        gc.collect()

def train(net, criterion, optimizer, train_loader, val_loader, test_loader, device, max_epoch, scheduler, writer, seed):
    best_test_labels = []
    best_test_y_scores = []
    save_predictions_path = 'save_results/predictions_'+str(seed)+'.xlsx'
    save_results_path = 'save_results/results_'+str(seed)+'.xlsx'
    best_acc = 0
    best_f1_score = 0
    best_train_acc = 0
    best_train_f1_score = 0
    best_val_auc = 0
    best_train_auc = 0

    best_test_acc = 0
    best_test_f1_score = 0
    best_test_auc = 0
    total_train_loss = []
    total_val_loss = []
    total_train_acc = []
    total_val_acc = []
    learning_rates = []
    total_val_f1_score = []
    total_train_f1_socre = []

    for epoch in range(max_epoch):
        curr_learning_rate = optimizer.param_groups[0]['lr']
        print('Epoch: {} | Learning rate: {:.6f} '.format(epoch, curr_learning_rate))
        train_loss = 0
        epoch_start = time.time()
        predictions = []
        labels = []
        y_scores = []
        net.train()
        print('Training---------------->')
        i  = 0
        for data in tqdm(train_loader):
            inputs_mri, inputs_pet, targets = data
            inputs_mri = inputs_mri.to(device)
            inputs_pet = inputs_pet.to(device)
            targets = targets.to(device)
            outputs = net(inputs_mri,inputs_pet)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            labels.extend(targets.cpu().numpy())
            softmax_outputs = nn.functional.softmax(outputs, dim=1)
            y_scores.extend(softmax_outputs.cpu().detach().numpy()[:,1])

        trainloss = train_loss/len(train_loader)
        train_acc = accuracy_score(labels, predictions)
        train_f1_score = f1_score(labels, predictions, average='binary')
        train_auc = roc_auc_score(labels, y_scores)
        writer.add_scalar('train/loss', trainloss, epoch)
        writer.add_scalar('train/Acc', train_acc, epoch)
        writer.add_scalar('train/F1 score', train_f1_score, epoch)
        writer.add_scalar('train/AUC', train_auc, epoch)
        print('Validation---------------->')

        with torch.no_grad():
            net.eval()
            predictions = []
            labels = []
            val_loss = 0
            y_scores = []
            for data in tqdm(val_loader):
                inputs_mri, inputs_pet, targets = data
                inputs_mri = inputs_mri.to(device)
                inputs_pet = inputs_pet.to(device)
                targets = targets.to(device)
                outputs = net(inputs_mri, inputs_pet)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                softmax_outputs = nn.functional.softmax(outputs, dim=1)
                y_scores.extend(softmax_outputs.cpu().detach().numpy()[:,1])
                labels.extend(targets.cpu().numpy())
            
            test_predictions = []
            test_labels = []
            test_y_scores = []

            for data in tqdm(test_loader):
                inputs_mri, inputs_pet, targets = data
                inputs_mri = inputs_mri.to(device)
                inputs_pet = inputs_pet.to(device)
                targets = targets.to(device)
                outputs = net(inputs_mri, inputs_pet)
                test_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                softmax_outputs = nn.functional.softmax(outputs, dim=1)
                test_y_scores.extend(softmax_outputs.cpu().detach().numpy()[:,1])
                test_labels.extend(targets.cpu().numpy())

        valloss = val_loss/len(val_loader)
        val_acc = accuracy_score(labels, predictions)
        val_f1_score = f1_score(labels,predictions, average='binary')
        val_auc = roc_auc_score(labels, y_scores)
        writer.add_scalar('val/loss', valloss, epoch)
        writer.add_scalar('val/Acc', val_acc, epoch)
        writer.add_scalar('val/F1 score', val_f1_score, epoch)
        writer.add_scalar('val/AUC', val_auc, epoch)

    
        test_acc = accuracy_score(test_labels, test_predictions)
        test_f1_score = f1_score(test_labels,test_predictions, average='binary')
        test_auc = roc_auc_score(test_labels, test_y_scores)
        writer.add_scalar('test/Acc', test_acc, epoch)
        writer.add_scalar('test/F1 score', test_f1_score, epoch)
        writer.add_scalar('test/AUC', test_auc, epoch)

        writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)
        if val_acc > best_acc:
            best_test_labels = test_labels
            best_test_y_scores = test_y_scores
            best_acc = val_acc
            best_f1_score = val_f1_score
            best_train_acc = train_acc
            best_train_f1_score = train_f1_score
            best_train_auc = train_auc
            best_val_auc = val_auc

            best_test_acc = test_acc
            best_test_f1_score = test_f1_score
            best_test_auc = test_auc

        scheduler.step()
        print('Training loss: {} | Training accuracy: {:.3f} | training f1_socre: {:.3f} | training AUC: {:.3f}'.format(trainloss, train_acc, train_f1_score, train_auc))
        print('Validation loss: {} | validation accuracy: {:.3f} | validation f1_socre: {:.3f} | testing AUC: {:.3f}'.format(valloss, val_acc, val_f1_score, val_auc))
    print('Best training metrics....................................')
    print('The best training accuracy is: {}'.format(best_train_acc))
    print('The best training auc is: {}'.format(best_train_auc))
    print('The best training f1 score is: {}'.format(best_train_f1_score))

    print('Best validation metrics....................................')
    print('The best validation accuracy is: {}'.format(best_acc))
    print('The best validation auc is: {}'.format(best_val_auc))
    print('The best validation f1 score is: {}'.format(best_f1_score))

    print('Best testing metrics....................................')
    print('The best testing accuracy is: {}'.format(best_test_acc))
    print('The best testing auc is: {}'.format(best_test_auc))
    print('The best testing f1 score is: {}'.format(best_test_f1_score))

    print('Final testing metrics....................................')
    print('The final testing accuracy is: {}'.format(test_acc))
    print('The final testing auc is: {}'.format(test_auc))
    print('The final testing f1 score is: {}'.format(test_f1_score))

    save_prediction = pd.DataFrame({'label':best_test_labels,'pred':best_test_y_scores})
    save_prediction.to_excel(save_predictions_path)

    results = pd.DataFrame({'train_acc': [best_train_acc], 'train_auc': [best_train_auc], 'train_f1_score': [best_train_f1_score],
                            'best_test_acc': [best_test_acc], 'best_test_auc': [best_test_auc], 'best_test_f1_score': [best_test_f1_score],
                            'final_test_acc': [test_acc], 'final_test_auc': [test_auc], 'final_test_f1_score': [test_f1_score]})
    results.to_excel(save_results_path)

if __name__=='__main__':
    
    main()