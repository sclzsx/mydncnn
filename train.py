import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN
from dataset import prepare_data, Dataset
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 加载参数。在default中修改。
parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="B", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--noiseT", type=str, default='G', help='noise type, G for Gaussian, I for Impluse')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
opt = parser.parse_args()

def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers) # 对于彩图，channels=3
    net.apply(weights_init_kaiming) # 初始化网络权重
    criterion = nn.MSELoss(size_average=False) # 定义MSE损失函数
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr) # 定义Adam优化器
    # training
    writer = SummaryWriter(opt.outf) # tensorboard记录训练日志
    step = 0
    noiseL_B=[0,55] # ingnored when opt.mode=='S' 盲去噪的sigma变化范围
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10. # 达到milestone次epoch后，学习率每个epoch下降
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train() # 训练模式，批次归一化权重可变
            model.zero_grad() # 梯度清零
            optimizer.zero_grad() # 优化器梯度清零
            ISource = data

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
            
            ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda()) #把tensor转为可记录梯度
            noise = Variable(noise.cuda())
            out_train = model(INoisy) # 前向传播
            loss = criterion(out_train, noise) / (INoisy.size()[0]*2) # 计算平均损失
            loss.backward() # 反向传播
            optimizer.step() # 更新学习率
            # results
            with torch.no_grad():
                model.eval() # 预测模式，固定BN权重
                out_train = torch.clamp(INoisy-model(INoisy), 0., 1.)
                psnr_train = batch_PSNR(out_train, ISource, 1.) #计算PSNR指标
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                    (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
                # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
                if step % 10 == 0:
                    # Log the scalar values
                    writer.add_scalar('loss', loss.item(), step) # 存log
                    writer.add_scalar('PSNR on training data', psnr_train, step)
                step += 1
        ## the end of each epoch
        with torch.no_grad():
            model.eval()
            # validate
            psnr_val = 0
            for k in range(len(dataset_val)):
                img_val = torch.unsqueeze(dataset_val[k], 0)
                noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
                imgn_val = img_val + noise
                img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
                out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
                psnr_val += batch_PSNR(out_val, img_val, 1.)
            psnr_val /= len(dataset_val)
            print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
            writer.add_scalar('PSNR on validation data', psnr_val, epoch)
            # log the images
            out_train = torch.clamp(INoisy-model(INoisy), 0., 1.)
            Img = utils.make_grid(ISource.data, nrow=8, normalize=True, scale_each=True)
            Imgn = utils.make_grid(INoisy.data, nrow=8, normalize=True, scale_each=True)
            Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
            writer.add_image('clean image', Img, epoch)
            writer.add_image('noisy image', Imgn, epoch)
            writer.add_image('reconstructed image', Irecon, epoch)
            # save model
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    main()
