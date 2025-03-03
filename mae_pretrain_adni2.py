from torchvision.models import resnet18, ResNet18_Weights, VisionTransformer
import torch
import torch.nn as nn
import utils
from my_parser import parse_args_adni2_pet_mae
from tqdm import tqdm
from data import build_data_adni2_paired_data_pretrain
from utils import pretrain_mae, focal_loss, save_model
from models import vim_model
from simple_vit import SimpleViT
import os


args = parse_args_adni2_pet_mae()

import psutil
def use_cpus(gpus: list):
    cpus = []
    for gpu in gpus:
        cpus.extend(list(range(gpu*24, (gpu+1)*24)))
    p = psutil.Process()
    p.cpu_affinity(cpus)
    print("A total {} CPUs are used, making sure that num_worker is small than the number of CPUs".format(len(cpus)))
use_cpus(gpus=[int(args.gpu)])

utils.lock_random_seed(args.seed)

model = vim_model(args)


if True :
    print('loading weights form '+ 'save_models/multi_B_mae_800.pt')
    checkpoint = torch.load('save_models/multi_B_mae_800.pt')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
loss_func = nn.MSELoss()
if args.optimizer == 'Adam' :
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
elif args.optimizer == 'AdamW' :
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'SGD' :
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=1e-7)
train_loader = build_data_adni2_paired_data_pretrain(args.batch_size, args.input_D, args.input_H, args.input_W, workers=args.workers)

pretrain_mae(args, model, train_loader, loss_func, optimizer, scheduler, args.epochs)

save_model(model, args.save_path, args.model_name)