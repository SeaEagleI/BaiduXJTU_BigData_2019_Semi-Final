import torch
import re,time,math,json
import os,os.path as op
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable

from Dataloader.MultiModal_BDXJTU2019 import MM_BDXJTU2019, MM_BDXJTU2019_TTA, Augmentation
from basenet.ResNeXt101_64x4d import ResNeXt101_64x4d
from basenet.senet import se_resnet50,se_resnext101_32x4d
from basenet.oct_resnet import oct_resnet26,oct_resnet101
from basenet.nasnet import nasnetalarge
from basenet.multiscale_resnet import multiscale_resnet
from basenet.multimodal import MultiModalNet
from basenet.multiscale_se_resnext import multiscale_se_resnext
from torch.utils.data.sampler import WeightedRandomSampler
import torch.multiprocessing as mp
import argparse,re


parser = argparse.ArgumentParser(description = 'BDXJTU')
parser.add_argument('--dataset_root', default = 'data', type = str)
parser.add_argument('--class_num', default = 9, type = int)
parser.add_argument('--batch_size', default =128, type = int)
parser.add_argument('--num_workers', default = 1, type = int)
parser.add_argument('--start_iter', default = 0, type = int)
parser.add_argument('--adjust_iter', default = 40000, type = int)
parser.add_argument('--end_iter', default = 60000, type = int)
parser.add_argument('--lr', default = 0.01, type = float)
parser.add_argument('--momentum', default = 0.9, type = float)
parser.add_argument('--weight_decay', default = 1e-5, type = float)
parser.add_argument('--gamma', default = 0.1, type = float)
parser.add_argument('--resume', default = None, type = str)
parser.add_argument('--basenet', default = 'se_resnext101_32x4d', type = str)
parser.add_argument('--print-freq', '-p', default=20, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')

#parser.add_argument('--fixblocks', default = 2, type = int)

args = parser.parse_args()
class_num = [9542, 7538, 3590, 1358, 3464, 5507, 3517, 2617, 2867]
class_ration = [40000.0/i for i in class_num]
class_ration = torch.tensor(class_ration)

diag_pred = [0.76765499, 0.68981794, 0.6128591, 0.58947368, 0.90697674, 0.58221024, 0.6407767,  0.54887218, 0.61148649]
#[0.76765499, 0.68981794, 0.6128591, 0.58947368, 0.90697674, 0.58221024, 0.6407767,  0.54887218, 0.61148649]
#[0.76765499, 0.68981794, 0.6128591, 0.58947368, 0.90697674, 0.58221024, 0.6407767,  0.54887218, 0.61148649] _1
MAX = max(diag_pred)
weights = [MAX/i for i in diag_pred]
weights = torch.tensor(weights)#torch.nn.functional.normalize(torch.tensor([2.0, 3.0, 4.0, 4.0, 1.0, 4.0, 4.0, 5.0, 3.0]))
std_log = './log/Net5_HR_fold1.log'
best_log = './log/best_predicts.json'

def log(s,log_path=std_log):
    open(log_path,'a').writelines(s+'\n')
    print(s)

def log_dict(epoch,best_pred1,log_path=best_log):
    best_preds = {}
    best_preds['best_epoch'] = epoch
    best_preds['best_pred1'] = best_pred1
    json.dump(best_preds,open(log_path,'w+'))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (math.sqrt(0.9) ** (epoch)) ##origin 25 epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the predision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(Dataloader,model, criterion, optimizer, epoch):
    # Priors

    # Dataset
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
        
    model.train()
    model = model.cuda()
    DatasetLen = len(Dataloader)
    warmup_list = [0,1]
    warmup_len = DatasetLen*len(warmup_list)

    #cl = nn.CrossEntropyLoss()
    # Optimizer
    #Optimizer = optim.RMSprop(net.parameters(), lr = args.lr, momentum = args.momentum,
                          #weight_decay = args.weight_decay)

    # train
    end = time.time()
    for i, (input_img, input_vis, anos) in enumerate(Dataloader):
        data_time.update(time.time() - end)
        target = anos.cuda(async=True)
        if epoch in warmup_list:
            for param_group in optimizer.param_groups:
                cur_iter = float(i + 1 + DatasetLen*epoch)
                param_group['lr'] = args.lr*(cur_iter/warmup_len)
        with torch.no_grad():
            input_img_var = Variable(input_img.cuda())
            input_vis_var = Variable(input_vis.cuda())
            target_var = Variable(target.cuda())
        # compute output
        output = model(input_img_var.cuda(), input_vis_var.cuda())
        #print(target_var)
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        pred1, pred5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input_img.size(0))
        top1.update(pred1.item(), input_img.size(0))
        top5.update(pred5.item(), input_img.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % 250 == 0:
            log('Train: [{0}][{1}/{2}]\t'
                  'Time {batch_time.sum:.3f}\t'
                  'Data {data_time.val:.3f}\t'
                  'Loss {loss.val:.4f}\t'
                  'Acc@1 {top1.val:.3f}\t'.format(
                epoch, i+1, len(Dataloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))        
            losses.reset()
            top1.reset()
            top5.reset()
        torch.cuda.empty_cache()

def validate(val_loader,model, criterion, printable=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input_img,input_vis,target) in enumerate(val_loader):
        input_img = input_img.cuda()
        input_vis = input_vis.cuda()
        target = target.cuda()
        with torch.no_grad():
            input_img_var = Variable(input_img.cuda())
            input_vis_var = Variable(input_vis.cuda())
            target_var = Variable(target.cuda())
        # compute output
        output = model(input_img_var.cuda(), input_vis_var.cuda())
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        pred1, pred5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input_img.size(0))
        top1.update(pred1.item(), input_img.size(0))
        top5.update(pred5.item(), input_img.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # empty cache
        torch.cuda.empty_cache()

    if printable:
        log('Validate: [{0}/{1}]\t'
          'Time {batch_time.sum:.3f}\t'
          'Loss {loss.val:.4f}\t'
          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
           i+1, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))
    return top1.avg, top5.avg

def main():
    ###Enter Main Func:
    mp.set_start_method('spawn')
    #create model
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)

    MODEL_NAME = 'multiscale_se_resnext_HR'
    MODEL_DIR = op.join('weights',MODEL_NAME)
    BEST_DIR = op.join('weights','best_models')
    if not op.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    if not op.isdir(BEST_DIR):
        os.mkdir(BEST_DIR)

    #    if args.basenet == 'MultiModal':
    model = MultiModalNet(MODEL_NAME, 'DPN26', 0.5)    
    #        #net = Networktorch.nn.DataParallel(Network, device_ids=[0])
    #    elif  args.basenet == 'oct_resnet101':
    #        model = oct_resnet101()
    #        #net = Networktorch.nn.DataParallel(Network, device_ids=[0])

    model = model.cuda()
    cudnn.benchmark = True
    RESUME = False
    # MODEL_PATH = './weights/best_models/se_resnext50_32x4d_SGD_w_46.pth'
    pthlist = [i for i in os.listdir(MODEL_DIR) if i[-4:]=='.pth']
    if len(pthlist)>0:
        pthlist.sort(key=lambda x:eval(re.findall(r'\d+',x)[-1]))
        MODEL_PATH = op.join(MODEL_DIR,pthlist[-1])
        model.load_state_dict(torch.load(MODEL_PATH))
        RESUME = True

    # Dataset
    Aug = Augmentation()
    Dataset_train = MM_BDXJTU2019(root = args.dataset_root, mode = 'train', 
                                  transform = Aug, TRAIN_IMAGE_DIR = 'train_image_raw')
    #weights = [class_ration[label] for data,label in Dataset_train]
    Dataloader_train = data.DataLoader(Dataset_train, args.batch_size, 
                                 num_workers = args.num_workers,
                                 shuffle = True, pin_memory = True)

    Dataset_val = MM_BDXJTU2019(root = args.dataset_root, mode = 'val', 
                                TRAIN_IMAGE_DIR = 'train_image_raw')
    Dataloader_val = data.DataLoader(Dataset_val, batch_size = 64,
                                 num_workers = args.num_workers,
                                 shuffle = True, pin_memory = True)

    criterion = nn.CrossEntropyLoss(weight = weights).cuda()
    Optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr, momentum = args.momentum,
                          weight_decay = args.weight_decay)

    args.start_epoch = eval(re.findall(r'\d+',MODEL_PATH)[-1]) if RESUME else 0
    best_pred1,best_preds = 0,{}
    if RESUME and op.isfile(best_log):
        best_preds = json.load(open(best_log))
        best_pred1 = best_preds['best_pred1']
    elif len(os.listdir(BEST_DIR))>0:
        best_model = MultiModalNet(MODEL_NAME, 'DPN26', 0.5).cuda()
        pthlist = [i for i in os.listdir(BEST_DIR) if i[-4:]=='.pth']
        pthlist.sort(key=lambda x:eval(re.findall(r'\d+',x)[-1]))
        best_model.load_state_dict(torch.load(op.join(BEST_DIR,pthlist[-1])))
        best_pred1 = validate(Dataloader_val, best_model, criterion, printable=False)[0]
        log_dict(eval(re.findall(r'\d+',pthlist[-1])[-1]),best_pred1)

    log('#Resume: Another Start from Epoch {}'.format(args.start_epoch+1))
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(Optimizer, epoch)
        # train for one epoch
        train(Dataloader_train, model, criterion, Optimizer, epoch)    #train(Dataloader_train, Network, criterion, Optimizer, epoch)
        # evaluate on validation set
        pred1,pred5 = validate(Dataloader_val, model, criterion)  #pred1 = validate(Dataloader_val, Network, criterion)
        # remember best pred@1 and save checkpoint

        COMMON_MODEL_PATH = op.join(MODEL_DIR,'SGD_fold1_{}.pth'.format(epoch+1))
        BEST_MODEL_PATH   = op.join(BEST_DIR,'{}_SGD_fold1_{}.pth'.format(MODEL_NAME,epoch+1))
        torch.save(model.state_dict(), COMMON_MODEL_PATH)
        if pred1 > best_pred1:
            best_pred1 = max(pred1, best_pred1)
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            log_dict(epoch+1,best_pred1)
        log('Epoch:{}\tpred1:{}\tBest_pred1:{}\n'.format(epoch+1,pred1,best_pred1))

if __name__ == "__main__":
    main()
