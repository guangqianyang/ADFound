import os
import time
import datetime
import random
import torch
import psutil
from my_parser import parse_args_adni2_pet
from data import build_data_adni2_paired_data_v2
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn

import time
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
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,confusion_matrix
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
parser.add_argument("--load_path", type=str, default="weights")
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
        cpus.extend(list(range(gpu*24, (gpu+1)*24)))
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
    train_loader, val_loader, test_loader = build_data_adni2_paired_data_v2(args.batch_size, args.input_D, args.input_H, args.input_W, workers=args.workers)
    ACC_list = []
    AUC_list = []
    SPE_list = []
    SEN_list = []
    F1_list = []
    for seed in [600, 800, 1000]:
        utils.lock_random_seed(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('{} is available'.format(device))
        print('==> Preparing data..')
        root_dir = '/data/krdu/dataset/ADNI/processed/'
        

        net = vim_model(args).cuda()
        #if args.load_model :
        #    print('loading weights form '+ os.path.join(args.load_path, 'multi_B_seed'+str(seed) + '.pt'))
        #    net.load_state_dict(torch.load(os.path.join(args.load_path, 'multi_B_seed'+str(seed) + '.pt')), strict=False)

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
        acc, auc, spe, sen, f1 = test(net, criterion, optimizer, train_loader, val_loader, test_loader, device, max_epoch, scheduler, writer, seed)
        torch.cuda.empty_cache()
        gc.collect()
        ACC_list.append(acc)
        AUC_list.append(auc)
        SPE_list.append(spe)
        SEN_list.append(sen)
        F1_list.append(f1)

    ACC_list = np.array(ACC_list)
    AUC_list = np.array(AUC_list)
    SPE_list = np.array(SPE_list)
    SEN_list = np.array(SEN_list)
    F1_list = np.array(F1_list)

    print('Average ACC: ', np.mean(ACC_list))
    print('std ACC: ', np.std(ACC_list))
    print('Average AUC: ', np.mean(AUC_list))
    print('std AUC: ', np.std(AUC_list))
    print('Average SPE: ', np.mean(SPE_list))
    print('std SPE: ', np.std(SPE_list))
    print('Average SEN: ', np.mean(SEN_list))
    print('std SEN: ', np.std(SEN_list))
    print('Average F1: ', np.mean(F1_list))
    print('std F1: ', np.std(F1_list))

def test(net, criterion, optimizer, train_loader, val_loader, test_loader, device, max_epoch, scheduler, writer, seed):

    save_predictions_path = 'save_results/predictions_'+str(seed)+'.xlsx'
    time_list = []
    with torch.no_grad():
        net.eval()

        test_predictions = []
        test_labels = []
        test_y_scores = []

        for data in tqdm(test_loader):
            inputs_mri, inputs_pet, targets = data
            inputs_mri = inputs_mri.to(device)
            sample_size = inputs_mri.size(0)
            inputs_pet = inputs_pet.to(device)
            targets = targets.to(device)
            start_time = time.perf_counter()
            outputs = net(inputs_mri, inputs_pet)
            end_time = time.perf_counter()
            time_list.append((end_time - start_time)/sample_size)
            test_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            softmax_outputs = nn.functional.softmax(outputs, dim=1)
            test_y_scores.extend(softmax_outputs.cpu().detach().numpy())
            test_labels.extend(targets.cpu().numpy())

        acc = accuracy_score(test_labels, test_predictions)
        f1 = f1_score(test_labels,test_predictions,average='macro')

        auc = roc_auc_score(test_labels, test_y_scores, multi_class='ovr')
        sen = recall_score(test_labels, test_predictions, average="macro")


        labels = np.unique(test_labels)
        specificities = []

        for label in labels:
            cm = confusion_matrix(test_labels, test_predictions, labels=labels)
            tn = cm.sum() - (cm[label, :].sum() + cm[:, label].sum() - cm[label, label])
            fp = cm[:, label].sum() - cm[label, label]
            specificity = tn / (tn + fp)
            specificities.append(specificity)

        spe = np.mean(specificities)
    print(time_list)
    print('Average time: ', np.mean(time_list))
    print('std time: ', np.std(time_list))

    return acc, auc, spe, sen, f1



if __name__=='__main__':
    
    main()