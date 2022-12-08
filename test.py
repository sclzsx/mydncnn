import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=15, help='noise level; ignored when mode=B')
parser.add_argument("--noiseT", type=str, default='G', help='noise type, G for Gaussian, I for Impluse')
parser.add_argument("--save_dir", type=str, default='results', help='save dir of visualizing results')
opt = parser.parse_args()

def normalize(data): # 归一化
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers) # 加载网络
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth'))) # 加载模型
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png')) # 获取测试图片路径
    files_source.sort()
    # process data
    psnr_test = 0
    noiseL_B=[0,55]
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:,:,0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1) # 修改shape为（NCHW）
        ISource = torch.Tensor(Img)

        if opt.noiseT == 'I': # 噪声类型为脉冲噪声
            INoisy = torch.zeros(ISource.size())
            if opt.mode == 'S': # 非盲去噪，指定sigma
                prob = opt.noiseL / 255
            else: # 盲区噪
                prob = np.random.uniform(noiseL_B[0], noiseL_B[1]) / 255
            prob = min(max(prob, 0), 1)
            for n in range(ISource.size()[0]):
                input = ISource[n,:,:,:].clone()
                noise_tensor=torch.rand(input.size())
                salt=torch.max(input)
                pepper=torch.min(input)
                input[noise_tensor<prob/2]=salt
                input[noise_tensor>1-prob/2]=pepper
                INoisy[n,:,:,:] = input
            noise = INoisy - ISource
        else: # 噪声类型为高斯噪声
            if opt.mode == 'S': # 非盲去噪，指定sigma
                noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.noiseL/255.)
            else: # 盲区噪
                noise = torch.zeros(ISource.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0]) # 同一批次中，各样本的sigma不同，这些sigma均匀分布
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.) # 根据sigma生成均值为0，标准差为sigma/255的高斯噪声
            INoisy = ISource + noise # 加性噪声

        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        INoisy = torch.clamp(INoisy, 0, 1)
        Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        ## if you are using older version of PyTorch, torch.no_grad() may not be supported
        # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
        # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        psnr = batch_PSNR(Out, ISource, 1.)
    
        if not os.path.exists(opt.save_dir): os.makedirs(opt.save_dir) # 新建文件夹
        filename = f.split(opt.test_data)[1].split('.')[0] # 获取文件名
        ISource = (ISource*255).squeeze().cpu().numpy().astype('uint8') # 修改shape为（HW），再转为cpu模式，再转为numpy，再转为8位
        Out = (Out*255).squeeze().cpu().numpy().astype('uint8')
        INoisy = (INoisy*255).squeeze().cpu().numpy().astype('uint8')
        cv2.imwrite(opt.save_dir + filename + '_gt.png', ISource)
        cv2.imwrite(opt.save_dir + filename + '_out.png', Out)
        cv2.imwrite(opt.save_dir + filename + '_in.png', INoisy)

        psnr_test += psnr # 累加评价指标
        print("%s PSNR %f" % (f, psnr))
    psnr_test /= len(files_source) # 平均评价指标
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    with torch.no_grad(): # this can save much memory
        main()
