from __future__ import print_function
import argparse
import os
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from logger import Logger

# hard-wire the gpu id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', default='', help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--cuda', default=1, action='store_true', help='enables cuda')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--localSize', type=int, default=64, help='the height / width of the region around the mask')
parser.add_argument('--hole_min', type=int, default=32, help='min height / width of the mask')
parser.add_argument('--hole_max', type=int, default=48, help='max height / width of the mask')
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs')
parser.add_argument('--preniter', type=int, default=50, help='number of epochs for generator pretraining')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--alpha', type=float, default=0.01, help='the weight of discriminator loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.99, help='beta2 for adam. default=0.99')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

device = torch.device("cuda:0" if opt.cuda else "cpu")
ndf = int(opt.ndf)
nc = 3

# save feedback
logger = Logger('./logs')

# Computes and stores the average and current value
class AverageMeter(object):
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

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(

            nn.Conv2d(nc, ndf, 5, stride=1, padding=2, dilation=1, bias=True),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.Conv2d(ndf, ndf * 2, 2, stride=2, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            nn.Conv2d(ndf * 2, ndf * 2, 3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            nn.Conv2d(ndf * 2, ndf * 4, 2, stride=2, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=2, dilation=2, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=4, dilation=4, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=8, dilation=8, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=16, dilation=16, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 4, ndf * 2, 4, stride=2, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            nn.Conv2d(ndf * 2, ndf * 2, 3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ndf * 2, ndf, 4, stride=2, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.Conv2d(ndf, int(ndf / 2), 3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(int(ndf / 2)),
            nn.ReLU(True),
            nn.Conv2d(int(ndf / 2), nc, 3, stride=1, padding=1, dilation=1, bias=True),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)

        return output

  return output.view(-1, 1).squeeze(1)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc_d = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
            nn.Conv2d(ndf * 8, ndf * 8, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
            nn.Conv2d(ndf * 8, ndf * 8, 4, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True)
        )

        self.disc_l = nn.Sequential(
            nn.Conv2d(nc, ndf * 2, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
            nn.Conv2d(ndf * 8, ndf * 8, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
            nn.Conv2d(ndf * 8, ndf * 8, 4, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(ndf * 16, 1, 1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x_global = self.disc_d(x1)
        x_local = self.disc_l(x2)
        x = torch.cat((x_global, x_local), 1)
        output = self.classifier(x)

        return output.view(-1, 1).squeeze(1)


# weight initialization
def weights_init(m):
    for m in m.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2./n))
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(0, 1)
            m.bias.data.zero_()

# generate random masks
def get_points():
    points = []
    mask = []
    for i in range(opt.batchSize):
        x1, y1 = np.random.randint(0, opt.imageSize - opt.localSize + 1, 2)
        x2, y2 = np.array([x1, y1]) + opt.localSize
        points.append([x1, y1, x2, y2])

        w, h = np.random.randint(opt.hole_min, opt.hole_max + 1, 2)
        p1 = x1 + np.random.randint(0, opt.localSize - w)
        q1 = y1 + np.random.randint(0, opt.localSize - h)
        p2 = p1 + w
        q2 = q1 + h

        m = np.zeros((1, opt.imageSize, opt.imageSize), dtype=np.uint8)
        m[:, q1:q2 + 1, p1:p2 + 1] = 1
        mask.append(m)

    return np.array(points), np.array(mask)

# crop image around the masked region
def crop_local_patches(images, points, bsize):
    cropped_images = torch.zeros([bsize, nc, opt.imageSize // 2, opt.imageSize // 2], dtype=torch.float32, device='cuda:0')
    for i in range(bsize):
        t = Variable(images[i])
        cropped_images[i] = t[:, points[i][0]:points[i][2], points[i][1]:points[i][3]]

    return cropped_images.cuda()

def save_checkpoint(state, curr_epoch):
    torch.save(state, './models/netG_e%d.pth.tar' % (curr_epoch))

# initialize Generator & Discriminator
netG = Generator().to(device)
weights_init(netG)
print(netG)

netD = Discriminator().to(device)
weights_init(netD)
print(netD)

# choose loss function
criterion = nn.BCELoss()
criterion2 = nn.MSELoss()

# fake/real labels
real_label = 1
fake_label = 0

# setup optimizers
optG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
optD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

# data loading process
traindir = os.path.join(opt.dataPath, 'train')

train_loader = torch.utils.data.DataLoader(
        dsets.ImageFolder(traindir, transforms.Compose([
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
        ])),
        batch_size=opt.batchSize, shuffle=True,
        num_workers=opt.workers, pin_memory=True)

errD_all = AverageMeter()
errG_all = AverageMeter()

# start training
for epoch in range(opt.niter):
    t0 = time.time()
    for i, data in enumerate(train_loader, 0):

        # generate masks
        points_batch, mask_batch = get_points()
        real_data = data[0].to(device)
        batch_size = real_data.size(0)

        temp = torch.from_numpy(mask_batch)
        masks = temp.type(torch.FloatTensor).cuda()
        if real_data.size(0) < opt.batchSize:
            masks_narrowed = masks.narrow(0, 0, real_data.size(0))
            masked_data = real_data * (1 - masks_narrowed)
        else:
            masked_data = real_data * (1 - masks)

        if epoch <= opt.preniter:
            optG.zero_grad()
            gen_data = netG(masked_data)
            errG = criterion2(gen_data, real_data)
            errG.backward()
            optG.step()

            print('PRETRAIN [%d/%d][%d/%d] Loss_G: %.4f'
                  % (epoch, opt.niter, i, len(train_loader), errG.item()))

            errG_all.update(errG.item())

        else:
	    optD.zero_grad()
            # train Discriminator with real samples
            local_real_data = crop_local_patches(real_data, points_batch, batch_size)
            label = torch.full((batch_size,), real_label, device=device)
            out = netD(real_data, local_real_data)
            errD_real = criterion(out, label)

            # train Discriminator with fake samples
            temp = torch.from_numpy(mask_batch)
            masks = temp.type(torch.FloatTensor).cuda()
            if real_data.size(0) < opt.batchSize:
                masks_narrowed = masks.narrow(0, 0, real_data.size(0))
                masked_data = real_data * (1 - masks_narrowed)
            else:
                masked_data = real_data * (1 - masks)

            gen_data = netG(masked_data)
	    label_f = label.clone()
            label_f.fill_(fake_label)
            local_gen_data = crop_local_patches(gen_data, points_batch, batch_size)
            out = netD(gen_data.detach(), local_gen_data.detach())
            errD_fake = criterion(out, label_f)
            errD = (errD_real + errD_fake) * opt.alpha
	    errD.backward()
            optD.step()

            # update Generator
            optG.zero_grad()
            out = netD(gen_data, local_gen_data)
            errG = criterion2(gen_data, real_data)
            errG.backward()
            optG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                % (epoch, opt.niter, i, len(train_loader), errD.item(), errG.item()))

            errD_all.update(errD.item())
            errG_all.update(errG.item())

        # TensorBoard logging
        # logging some indicative real/masked/generated images
        if (i % 500) == 0:
            info = {
                '3_ground_truth': real_data.view(-1, 3, opt.imageSize, opt.imageSize)[:3].cpu().detach().numpy(),
                '2_masked': masked_data.view(-1, 3, opt.imageSize, opt.imageSize)[:3].cpu().numpy(),
                '1_generated': gen_data.view(-1, 3, opt.imageSize, opt.imageSize)[:3].cpu().detach().numpy()
            }
            for tag, images in info.items():
                logger.image_summary(tag, images, epoch)

    print('Time elapsed Epoch %d: %d seconds'
          % (epoch, time.time() - t0))

    # TensorBoard logging
    # scalar values
    info = {
        'D loss': errD_all.avg,
        'G loss': errG_all.avg
    }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)

    # values and gradients of the parameters (histogram)
    for tag, value in netG.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.cpu().detach().numpy(), epoch)

    # save model parameters (last epoch)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': netG.state_dict(),
        'optimizer': optG.state_dict(),
    }, epoch+1)
