from torchvision.models import resnet18, ResNet18_Weights, VisionTransformer
import torch
import torch.nn as nn
import utils
from my_parser import parse_args_adni2_pet_mae_fine_tune
from tqdm import tqdm
from data import build_data_adni2_pet
from utils import fine_tune, focal_loss, save_model
from models import ViT, resnet50
from simple_vit import SimpleViT
import os
from models import vit_model

args = parse_args_adni2_pet_mae_fine_tune()
utils.lock_random_seed(args.seed)

model = vit_model(args)

if args.load_model :
    model.load_state_dict(torch.load(os.path.join(args.load_path, args.load_name + '.pt')), strict=False)

loss_func = focal_loss(num_classes=3, gamma=3)
if args.optimizer == 'Adam' :
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
elif args.optimizer == 'AdamW' :
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'SGD' :
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=1e-7)
train_loader, val_loader = build_data_adni2_pet(args.batch_size, args.input_D, args.input_H, args.input_W, workers=args.workers)

fine_tune(model, train_loader, val_loader, loss_func, optimizer, scheduler, args.epochs)

save_model(model, args.save_path, args.model_name)