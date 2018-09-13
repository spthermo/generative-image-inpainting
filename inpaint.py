from __future__ import print_function
import argparse
import os
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import scipy.misc

# hard-wire the gpu id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', default='', help='path to dataset')
parser.add_argument('--netG', default='', help='path to trained generator')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--localSize', type=int, default=64, help='the height / width of the region around the mask')
parser.add_argument('--hole_min', type=int, default=32, help='min height / width of the mask')
parser.add_argument('--hole_max', type=int, default=48, help='max height / width of the mask')
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--batchSize', type=int, default='64', help='the batch size used during training')
parser.add_argument('--cuda', default=1, action='store_true', help='enables cuda')

opt = parser.parse_args()
print(opt)

cudnn.benchmark = True

device = torch.device("cuda:0" if opt.cuda else "cpu")
ndf = int(opt.ndf)
nc = 3

#create a directory for the inpainted samples
if not os.path.exists('examples'):
    os.makedirs('examples')

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

# data loading process
valdir = os.path.join(opt.dataPath, 'val')

val_loader = torch.utils.data.DataLoader(
        dsets.ImageFolder(valdir, transforms.Compose([
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
        ])),
        batch_size=opt.batchSize, shuffle=True,
        num_workers=opt.workers, pin_memory=True)

# load the pretrained weights
netG = Generator().to(device)
netG.load_state_dict(torch.load(opt.netG))
netG.eval()
print(netG)

# sample a random batch and save 6 generated images for visualization
for i, data in enumerate(val_loader, 0):
    _, mask_batch = get_points()
    x = data[0].to(device)

    temp = torch.from_numpy(mask_batch)
    masks = temp.type(torch.FloatTensor).cuda()
    if x.size(0) < opt.batchSize:
        masks_narrowed = masks.narrow(0, 0, x.size(0))
        masked_data = x * (1 - masks_narrowed)
    else:
        masked_data = x * (1 - masks)

    generated_samples = netG(masked_data)
    if (i % 10) == 0:
        out_path = './examples/sample_' + str(i) + '.png'
        
        # save one of the generated images as example
        scipy.misc.toimage(generated_samples.view(-1, 3, 128, 128)[0].cpu().detach().numpy()).save(out_path)
