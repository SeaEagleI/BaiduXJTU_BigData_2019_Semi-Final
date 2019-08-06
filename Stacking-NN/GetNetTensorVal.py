import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import os,os.path as op
from tqdm import tqdm
import numpy as np
import torch.multiprocessing as mp
import re

from Dataloader.MultiModal_BDXJTU2019 import MM_BDXJTU2019, BDXJTU2019_test, Augmentation
#from basenet.ResNeXt101_64x4d import ResNeXt101_64x4d
from basenet.senet import se_resnet50,se_resnext101_32x4d
from basenet.octave_resnet import octave_resnet50
from basenet.nasnet import nasnetalarge
from basenet.multimodal import MultiModalNet

PROJECT_NAME = 'Net8_MS_cat_fold1'
NET_VAL_DATA_PATH  = "./Tensors/{}_Val.npz".format(PROJECT_NAME)
CLASSES = ['00{}'.format(i) for i in range(1,10)]

def TensorGenerator(Dataloader,Data_Path,net):
#    if op.isfile(Data_Path):
#        return

    X,targets,predicts = [],[],[]
    for (image_tensor,visit_tensor,anos) in tqdm(Dataloader):
        Tensor = net.forward(image_tensor.cuda(), visit_tensor.cuda())
        Concat_Matrix = torch.cat([Tensor]).cpu().detach().numpy()
        X.append(Concat_Matrix)
        targets.append(anos[0].cpu().detach().numpy())
        preds = torch.nn.functional.normalize(Tensor)
        _, pred = preds.data.topk(1, 1, True, True)
        predicts.append(pred[0][0].cpu().detach().numpy())
    np.savez(Data_Path, X=np.array(X), targets=np.array(targets), predicts=np.array(predicts))

def GetNetTensor():
    # Priors
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    Dataset_val = MM_BDXJTU2019(root = 'data', mode = 'val', TRAIN_IMAGE_DIR = 'train_image_raw')
    Dataloader_val = data.DataLoader(Dataset_val, 1, num_workers = 0, shuffle = False, pin_memory = False)

    # Network
    cudnn.benchmark = True
    MODEL_NAME = [i for i in os.listdir('./weights') if i!='best_models'][0]
    BEST_DIR = op.join('weights','best_models')

    # Find & Load Best_Model
    if op.isdir(BEST_DIR) and len(os.listdir(BEST_DIR))>0:
        best_model = MultiModalNet(MODEL_NAME, 'DPN26', 0.5).cuda()
        pthlist = [i for i in os.listdir(BEST_DIR) if i[-4:]=='.pth']
        pthlist.sort(key=lambda x:eval(re.findall(r'\d+',x)[-1]))
        best_model.load_state_dict(torch.load(op.join(BEST_DIR,pthlist[-1])))
        best_model.eval()
        TensorGenerator( Dataloader_val, NET_VAL_DATA_PATH, best_model)
    else:
        print('No best_model pth found in ./weights/best_models.')

if __name__ == '__main__':
#    mp.set_start_method('spawn')
    GetNetTensor()
