from torchvision.models import resnet18, ResNet18_Weights, VisionTransformer
import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
from data import build_data_adni2_paired_data_pretrain

from models import vim_model

import os
import argparse

parser = argparse.ArgumentParser()
### Dataset
parser.add_argument("--path_dataset", type=str, default="/data/krdu/dataset")

### Model S/L
parser.add_argument("--load_path", type=str, default="/data/krdu/pretrained_models")
parser.add_argument("--load_model", action='store_true', default=False)
parser.add_argument("--load_name", type=str, default="")
parser.add_argument("--save_path", type=str, default="save_models/")
parser.add_argument("--model_type", type=str, default='vit_B_mae')

### Training
parser.add_argument("--batch_size", type=int, default=48)
parser.add_argument("--input_D", type=int, default=128)
parser.add_argument("--input_H", type=int, default=128)
parser.add_argument("--input_W", type=int, default=128)
parser.add_argument("--epochs", type=int, default=800)
parser.add_argument("--optimizer", type=str, default='AdamW')
parser.add_argument("--scheduler", type=str, default='cosLR')
parser.add_argument("--lr", type=float, default=0.00015)
parser.add_argument("--weight_decay", type=float, default=0.05)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--workers", type=int, default="24")
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

utils.lock_random_seed(args.seed)


model = vim_model(args)


if False :
    model.load_state_dict(torch.load(os.path.join('save_models/multi_B_mae_600.pt')), strict=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_func = nn.MSELoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=1e-7)
train_loader = build_data_adni2_paired_data_pretrain(args.batch_size, args.input_D, args.input_H, args.input_W, workers=args.workers)

log_dir = './log_dir_128_3_'+args.model_type+'_'+str(args.epochs) +'_resume'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)
modality = ['T1_TregT1_SS','PET-FDG_TregPET-FDG_SS']
epochs = args.epochs
warmup_epochs = 30
def warmup_lambda(epoch):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0

warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

total_epochs = args.epochs
min_lr = 1e-6
import math 
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr)

for num_epochs in range(0, epochs + 1) :

    epoch_training_loss = 0
    epoch_training_tot = 0
    epoch_training_loss_mri =0
    epoch_training_loss_pet = 0
    epoch_training_reg_loss_mri = 0
    epoch_training_reg_loss_pet = 0
    
    for image_data, condition in tqdm(train_loader) :
        optimizer.zero_grad()
        image_data = {k: v.cuda() for k, v in image_data.items()}
        x1, x2 = image_data[modality[0]], image_data[modality[1]]
        condition = condition.cuda()
        condition_mri = condition[:,0]
        condition_pet = condition[:,1]

        condition_mri_mask = condition_mri > 0
        condition_pet_mask = condition_pet > 0


        o1, o2, mask1, mask2, score_mri, score_pet = model(x1, x2, condition)

        loss_mri = loss_func(o1*mask1, x1*mask1)
        loss_pet = loss_func(o2*mask2, x2*mask2)
    
        score_mri1 = score_mri.T
        labels = torch.arange(x1.shape[0],dtype=torch.long).cuda()

        # Step 2: Mask the rows and columns of score_mri
        valid_indices = torch.nonzero(condition_mri_mask).flatten()   # Indices of valid samples
        #print('valid_indices: ', valid_indices)
        #print("score_mri shape:", score_mri.shape)
        score_mri_masked = score_mri[valid_indices][:, valid_indices]  # Mask rows and columns
        score_mri1_masked = score_mri1[valid_indices][:, valid_indices]  # Mask transpose

        # Step 3: Mask the labels to include only valid samples
        labels_masked = labels[valid_indices]  # Filter labels based on valid indices
        remap = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(valid_indices)}
        labels_masked_remapped = torch.tensor([remap[label.item()] for label in labels_masked], device=labels_masked.device)
        # Step 4: Compute the loss
        if labels_masked_remapped.numel() == 0:
            loss0 = 0
            loss1 = 0
        else:
            loss0 = F.cross_entropy(score_mri_masked, labels_masked_remapped)
            loss1 = F.cross_entropy(score_mri1_masked, labels_masked_remapped)

        score_pet1 = score_pet.T

        valid_indices = torch.nonzero(condition_pet_mask).flatten()   # Indices of valid samples
        #print('valid_indices: ', valid_indices)
        #print("score_pet shape:", score_pet.shape)
        score_pet_masked = score_pet[valid_indices][:, valid_indices]  # Mask rows and columns
        score_pet1_masked = score_pet1[valid_indices][:, valid_indices]  # Mask transpose

        # Step 3: Mask the labels to include only valid samples
        labels_masked = labels[valid_indices]  # Filter labels based on valid indices
        remap = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(valid_indices)}
        labels_masked_remapped = torch.tensor([remap[label.item()] for label in labels_masked], device=labels_masked.device)

        if labels_masked_remapped.numel() == 0:
            #print("condition_pet_mask: ", condition_pet_mask)
            loss2 = 0
            loss3 = 0
        # Step 4: Compute the loss
        else:
            loss2 = F.cross_entropy(score_pet_masked, labels_masked_remapped)
            loss3 = F.cross_entropy(score_pet1_masked, labels_masked_remapped)

        loss_reg_mri = (loss0 + loss1)/2
        loss_reg_pet = (loss2 + loss3)/2
    
        loss = loss_mri + loss_pet + 0.05* (loss_reg_mri + loss_reg_pet)
        #if reg is not None :
        #    loss += reg(model)
        loss.backward()

        epoch_training_loss += loss.detach().sum().item() * x1.shape[0]
        epoch_training_loss_mri += loss_mri.detach().sum().item() * x1.shape[0]
        epoch_training_loss_pet += loss_pet.detach().sum().item() * x1.shape[0]
        epoch_training_reg_loss_mri += loss_reg_mri.detach().sum().item() * x1.shape[0]
        epoch_training_reg_loss_pet += loss_reg_pet.detach().sum().item() * x1.shape[0]
        epoch_training_tot += x1.shape[0]
        optimizer.step()

    if num_epochs < warmup_epochs:
        warmup_scheduler.step()
    else:
        cosine_scheduler.step()
    # visulize learning rate
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], num_epochs)
    writer.add_scalar('Loss/Total', round(epoch_training_loss / epoch_training_tot, 5), num_epochs)
    writer.add_scalar('MAE Loss/MRI', round(epoch_training_loss_mri / epoch_training_tot, 5), num_epochs)
    writer.add_scalar('MAE Loss/PET', round(epoch_training_loss_pet / epoch_training_tot, 5), num_epochs)
    writer.add_scalar('CL Loss/MRI', round(0.05*epoch_training_reg_loss_mri / epoch_training_tot, 5), num_epochs)
    writer.add_scalar('CL Loss/PET', round(0.05*epoch_training_reg_loss_pet / epoch_training_tot, 5), num_epochs)

    print(f"Epoch: {num_epochs} | Training Loss: {round(epoch_training_loss / epoch_training_tot, 5)}  | Training MRI Loss: {round(epoch_training_loss_mri / epoch_training_tot, 5)}  | Training PET Loss: {round(epoch_training_loss_pet / epoch_training_tot, 5)}")
    if num_epochs % 50 == 0:
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'cosine_scheduler_state_dict': cosine_scheduler.state_dict(),
        }, os.path.join(args.save_path, args.model_type+'_'+str(args.epochs)+'.pt'))


torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'cosine_scheduler_state_dict': cosine_scheduler.state_dict(),
}, os.path.join(args.save_path, args.model_type+'_'+str(args.epochs)+'.pt'))