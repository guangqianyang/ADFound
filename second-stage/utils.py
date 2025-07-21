import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import nibabel as nib

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

class warmup_scheduler(torch.optim.lr_scheduler.LRScheduler) :
    def __init__(self, scheduler, warm_up_T, lr):
        self.warm_up_T = warm_up_T
        self.scheduler = scheduler
        self.lr = lr
        super().__init__(scheduler.optimizer, scheduler.last_epoch, scheduler.verbose)

    def get_lr(self):
        if (self.last_epoch < self.warm_up_T) :
            return lr * ((self.last_epoch + 1) / (self.warm_up_T))
        else :
            return self.scheduler.get_lr()

    def _get_closed_form_lr(self):
        if (self.last_epoch < self.warm_up_T) :
            return lr * ((self.last_epoch + 1) / (self.warm_up_T))
        else :
            return self.scheduler._get_closed_form_lr()

def warmup_scheduler(optimizer, scheduler, current_epoch, max_epoch, lr, warmup_epoch = 10):
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


### subs: [condition1, condition2, ...]
### condition: [substr1, substr2, ...]: strings included in name
def param_filter(named_params, subs:list, ex_subs:list=None) :
    ret = []
    for name, i in named_params :
        contains_all = True
        ex_any = False
        for cond in subs :
            if type(cond) == str :
                cond = [cond]
            for s in cond :
                if s not in name :
                    contains_all = False
                    break
        for cond in ex_subs :
            if type(cond) == str :
                cond = [cond]
            for s in cond :
                if s in name :
                    ex_any = True
                    break

        if contains_all and not ex_any :
            ret.append(i)

    return ret

def lock_random_seed(seed) : 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def fine_tune(model, train_loader, val_loader, loss_func, optimizer, scheduler=None, epochs=10, lock_BN=False, reg=None) :
    for num_epochs in range(1, epochs + 1) :

        if lock_BN :
            model.eval()
        else :
            model.train()

        training_total = 0
        training_correct = 0

        for x, label in tqdm(train_loader) :
            optimizer.zero_grad()
            x, label = x.cuda(), label.cuda()
            o = model(x)
            loss = loss_func(o, label)
            if reg is not None :
                loss += reg(model)
            loss.backward()
            optimizer.step()
            training_total += label.numel()
            training_correct += (o.argmax(dim=1) == label).sum().item()

        if scheduler is not None :
            scheduler.step()

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad() :
            for x, label in tqdm(val_loader) :
                x, label = x.cuda(), label.cuda()
                o = model(x)
                total += label.numel()
                correct += (o.argmax(dim=1) == label).sum().item()
        
        print(f"Epoch: {num_epochs} Training Acc: {round(training_correct / training_total * 100, 2)} Validation Acc: {round(correct / total * 100, 2)}")


def pretrain_mae(args, model, train_loader, loss_func, optimizer, scheduler=None, epochs=10, reg=None) :

    log_dir = './log_dir_128_3_'+args.model_type+'_'+str(args.epochs)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    for num_epochs in range(1, epochs + 1) :

        epoch_training_loss = 0
        epoch_training_tot = 0
        epoch_training_loss_mri =0
        epoch_training_loss_pet = 0
        epoch_training_reg_loss_mri = 0
        epoch_training_reg_loss_pet = 0

  
        for x1, x2 in tqdm(train_loader) :
            optimizer.zero_grad()
            x1, x2 = x1.cuda(), x2.cuda()
            o1, o2, mask1, mask2, score_mri, score_pet = model(x1, x2)

            loss_mri = loss_func(o1*mask1, x1*mask1)
            loss_pet = loss_func(o2*mask2, x2*mask2)
     
            score_mri1 = score_mri.T
            labels = torch.arange(x1.shape[0],dtype=torch.long).cuda()
            loss0 = F.cross_entropy(score_mri, labels)
            loss1 = F.cross_entropy(score_mri1, labels)

            score_pet1 = score_pet.T
            loss2 = F.cross_entropy(score_pet, labels)
            loss3 = F.cross_entropy(score_pet1, labels)

            loss_reg_mri = (loss0 + loss1)/2
            loss_reg_pet = (loss2 + loss3)/2
      
            loss = loss_mri + loss_pet + 0.05* (loss_reg_mri + loss_reg_pet)
            if reg is not None :
                loss += reg(model)
            loss.backward()

            epoch_training_loss += loss.detach().sum().item() * x1.shape[0]
            epoch_training_loss_mri += loss_mri.detach().sum().item() * x1.shape[0]
            epoch_training_loss_pet += loss_pet.detach().sum().item() * x1.shape[0]
            epoch_training_reg_loss_mri += loss_reg_mri.detach().sum().item() * x1.shape[0]
            epoch_training_reg_loss_pet += loss_reg_pet.detach().sum().item() * x1.shape[0]
            epoch_training_tot += x1.shape[0]
            optimizer.step()

        if scheduler is not None :
            scheduler.step()
        writer.add_scalar('Loss/Total', round(epoch_training_loss / epoch_training_tot, 5), num_epochs)
        writer.add_scalar('MAE Loss/MRI', round(epoch_training_loss_mri / epoch_training_tot, 5), num_epochs)
        writer.add_scalar('MAE Loss/PET', round(epoch_training_loss_pet / epoch_training_tot, 5), num_epochs)
        writer.add_scalar('CL Loss/MRI', round(0.05*epoch_training_reg_loss_mri / epoch_training_tot, 5), num_epochs)
        writer.add_scalar('CL Loss/PET', round(0.05*epoch_training_reg_loss_pet / epoch_training_tot, 5), num_epochs)

        print(f"Epoch: {num_epochs} | Training Loss: {round(epoch_training_loss / epoch_training_tot, 5)}  | Training MRI Loss: {round(epoch_training_loss_mri / epoch_training_tot, 5)}  | Training PET Loss: {round(epoch_training_loss_pet / epoch_training_tot, 5)}")
    torch.save(model.state_dict(), os.path.join(args.save_path, args.model_type+'_'+str(args.epochs)+'.pt'))


def pretrain_mae_mri(args, model, train_loader, loss_func, optimizer, scheduler=None, epochs=10, reg=None) :
    
    for num_epochs in range(1, epochs + 1) :

        epoch_training_loss = 0
        epoch_training_tot = 0
        for x, label in tqdm(train_loader) :
            optimizer.zero_grad()
            x, label = x.cuda(), label.cuda()
            o, mask = model(x)
            loss = loss_func(o*mask, x*mask)
            if reg is not None :
                loss += reg(model)
            loss.backward()

            epoch_training_loss += loss.detach().sum().item() * x.shape[0]
            epoch_training_tot += x.shape[0]

            optimizer.step()

        if scheduler is not None :
            scheduler.step()
        
        print(f"Epoch: {num_epochs}  Training Loss: {round(epoch_training_loss / epoch_training_tot, 5)}")
        torch.save(model.state_dict(), os.path.join(args.save_path, args.model_type+'_'+str(args.epochs)+'_mri.pt'))


def pretrain_mae_pet(args, model, train_loader, loss_func, optimizer, scheduler=None, epochs=10, reg=None) :
    
    for num_epochs in range(1, epochs + 1) :

        epoch_training_loss = 0
        epoch_training_tot = 0
        for x, label in tqdm(train_loader) :
            optimizer.zero_grad()
            x, label = x.cuda(), label.cuda()
            o, mask = model(x)
            loss = loss_func(o*mask, x*mask)
            if reg is not None :
                loss += reg(model)
            loss.backward()

            epoch_training_loss += loss.detach().sum().item() * x.shape[0]
            epoch_training_tot += x.shape[0]

            optimizer.step()

        if scheduler is not None :
            scheduler.step()
        
        print(f"Epoch: {num_epochs}  Training Loss: {round(epoch_training_loss / epoch_training_tot, 5)}")
        torch.save(model.state_dict(), os.path.join(args.save_path, args.model_type+'_'+str(args.epochs)+'_pet.pt'))

def mae_vis(model, train_loader) :
    with torch.no_grad() :
        for x1, x2, label in tqdm(train_loader) :
            x1, x2, label = x1.cuda(), x2.cuda(), label.cuda()
            o1, o2, mask1, mask2 = model(x1, x2)

            plt.matshow(o1[0, 0, 31, :, :].cpu())
            plt.savefig('mri_gen.png')
            plt.matshow(x1[0, 0, 31, :, :].cpu())
            plt.savefig('mri_ori.png')
            plt.matshow(o2[0, 0, 31, :, :].cpu())
            plt.savefig('pet_gen.png')
            plt.matshow(x2[0, 0, 31, :, :].cpu())
            plt.savefig('pet_ori.png')
            break


def mae_vis_paired_data(model, train_loader) :
    with torch.no_grad() :
        index = 0
        for x1, x2, label in tqdm(train_loader) :
            x1, x2, label = x1.cuda(), x2.cuda(), label.cuda()
            o1, o2, mask1, mask2 = model(x1, x2)
            o1 = o1*mask1
            o2 = o2*mask2
            o1_save = o1[0].cpu().detach().numpy()
            o1_save = o1_save[0]
            o2_save = o2[0].cpu().detach().numpy()
            o2_save = o2_save[0]
            if not os.path.exists("results/"+str(index)):
                os.mkdir("results/"+str(index))
            nifti_o1 = nib.Nifti1Image(o1_save, np.eye(4))
            nifti_o2 = nib.Nifti1Image(o2_save, np.eye(4))
            nib.save(nifti_o1, 'results/'+str(index)+'/Generated_MRI.nii')
            nib.save(nifti_o2, 'results/'+str(index)+'/Generated_PET.nii')
            x1_save = x1[0].cpu().detach().numpy()
            x1_save = x1_save[0]
            x2_save = x2[0].cpu().detach().numpy()
            x2_save = x2_save[0]
            nifti_x1 = nib.Nifti1Image(x1_save, np.eye(4))
            nifti_x2 = nib.Nifti1Image(x2_save, np.eye(4))
            nib.save(nifti_x1, 'results/'+str(index)+'/Original_MRI.nii')
            nib.save(nifti_x2, 'results/'+str(index)+'/Original_PET.nii')


            mask = mask2[0].cpu().detach().numpy()
            mask = mask[0]
            nifti_mask = nib.Nifti1Image(mask, np.eye(4))
            nib.save(nifti_mask, 'results/'+str(index)+'/Mask.nii')


            index = index + 1


def position_embedding_sim(pe) :
    # [1, n, hidden_d]
    mat = pe[0].detach()
    vec_len = (mat ** 2).sum(dim = 1) ** 0.5
    mat_len = vec_len[:, None] @ vec_len[None, :]
    sim_mat = (mat @ mat.transpose(-1, -2)) / mat_len
    plt.matshow(sim_mat.cpu())
    plt.savefig("PE_sim.png")
    return sim_mat

def fine_tune_distiller(model, train_loader, val_loader, loss_func, optimizer, scheduler=None, epochs=10, lock_BN=False, reg=None, distill_func=None) :
    if distill_func == None :
        distill_func = loss_func
    
    for num_epochs in range(1, epochs + 1) :

        if lock_BN :
            model.student.eval()
        else :
            model.student.train()

        for x, label in tqdm(train_loader) :
            optimizer.zero_grad()
            x, label = x.cuda(), label.cuda()
            o, loss = model(x, distill_func)
            loss += loss_func(o, label)
            if reg is not None :
                loss += reg(model)
            loss.backward()
            optimizer.step()

        if scheduler is not None :
            scheduler.step()

        model.student.eval()
        total = 0
        correct = 0
        with torch.no_grad() :
            for x, label in tqdm(val_loader) :
                x, label = x.cuda(), label.cuda()
                o = model.student(x)
                total += label.numel()
                correct += (o.argmax(dim=1) == label).sum().item()
        
        print(f"Epoch: {num_epochs}  Validation Acc: {round(correct / total * 100, 2)}")


def test_acc(model, val_loader) :
    model.eval()
    total = 0
    correct = 0
    for x, label in tqdm(val_loader) :
        x, label = x.cuda(), label.cuda()
        o = model(x)
        total += label.numel()
        correct += (o.argmax(dim=1) == label).sum().item()
    return round(correct / total * 100, 2)


def save_model(model, save_path, model_name) :
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if type(model) == nn.DataParallel :
        torch.save(model.module.state_dict(), os.path.join(save_path, model_name + '.pt'))
    else :
        torch.save(model.state_dict(), os.path.join(save_path, model_name + '.pt'))