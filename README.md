# Generative Image Inpainting
The presented deep NN architecture consists of a deep convolutional autoencoder (generator) and a discriminator network with two parallel branches. The implementation is a based on the model proposed in [Globally and Locally Consistent Image Completion](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf). Note that this is not an exact replication, as there are some layer-level modifications (e.g. no linear layers are utilized). The model is developed using PyTorch and is trained/tested on [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) face dataset.

## Prerequisites
The architecture has been implemented using the following:
- Python 3.5
- Scipy
- Torchvision
- Tensorflow 1.7.0
- Tensorboard

Tensorflow and Tensorboard are used for visualization and monitoring purposes, thus they are not mandatory.

## Model architecture details
Generator details:

|     Type     | Kernel | Dilation | Stride | Output |
|:------------:|:------:|:--------:|:------:|:------:|
|     conv     |  5x5   |    1     |  1x1   |   64   |
|     conv     |  2x2   |    1     |  2x2   |   128  |
|     conv     |  3x3   |    1     |  1x1   |   128  |
|     conv     |  2x2   |    1     |  2x2   |   256  |
|     conv     |  3x3   |    1     |  1x1   |   256  |
|     conv     |  3x3   |    1     |  1x1   |   256  |
| dilated conv |  3x3   |    2     |  1x1   |   256  |
| dilated conv |  3x3   |    4     |  1x1   |   256  |
| dilated conv |  3x3   |    8     |  1x1   |   256  |
| dilated conv |  3x3   |   16     |  1x1   |   256  |
|     conv     |  3x3   |    1     |  1x1   |   256  |
|     conv     |  3x3   |    1     |  1x1   |   256  |
|    deconv    |  4x4   |    1     |  2x2   |   128  |
|     conv     |  3x3   |    1     |  1x1   |   128  |
|    deconv    |  4x4   |    1     |  2x2   |   64   |
|     conv     |  3x3   |    1     |  1x1   |   32   |
|    output    |  3x3   |    1     |  1x1   |   3    |

Global Discriminator details:

|     Type     | Kernel | Dilation | Stride | Output |
|:------------:|:------:|:--------:|:------:|:------:|
|     conv     |  4x4   |    1     |  2x2   |   64   |
|     conv     |  4x4   |    1     |  2x2   |   128  |
|     conv     |  4x4   |    1     |  2x2   |   256  |
|     conv     |  4x4   |    1     |  2x2   |   512  |
|     conv     |  4x4   |    1     |  2x2   |   512  |
|     conv     |  4x4   |    1     |  1x1   |   512  |

Local Discriminator details:

|     Type     | Kernel | Dilation | Stride | Output |
|:------------:|:------:|:--------:|:------:|:------:|
|     conv     |  4x4   |    1     |  2x2   |   128  |
|     conv     |  4x4   |    1     |  2x2   |   256  |
|     conv     |  4x4   |    1     |  2x2   |   512  |
|     conv     |  4x4   |    1     |  2x2   |   512  |
|     conv     |  4x4   |    1     |  1x1   |   512  |


## Training with CelebA
To start training the generative image inpainting model use:

```
python main.py --dataPath /path/to/celebA/train
```

The ```logger.py``` file is used to create and update the model's instance for Tensorboard. To monitor the training process use:

```
tensorboard --logdir='./logs' --port 6006
```
and use your browser to access the localhost at the specified port.


## Testing on unseen data
To test the model performance on unseen data of CelebA validation set use:

```
python inpaint.py --dataPath /path/to/celebA/val --netG /path/to/pretrained/generator/model
```

for example:

```
python inpaint.py --dataPath /datasets/celebA/val --netG /models/netG_e25.pth
```

the script selects a random batch of images (batchsize=64) from the CelebA validation set and saves 6 of them under "examples" directory.

## Inpainting examples
Some indicative CelebA samples in "real-masked-generated" triplet form:

<img src="https://github.com/spthermo/generative-image-inpainting/blob/master/examples/1_1.png" width="150"> <img src="https://github.com/spthermo/generative-image-inpainting/blob/master/examples/1_2.png" width="150"> <img src="https://github.com/spthermo/generative-image-inpainting/blob/master/examples/1_3.png" width="150">

<img src="https://github.com/spthermo/generative-image-inpainting/blob/master/examples/2_1.png" width="150"> <img src="https://github.com/spthermo/generative-image-inpainting/blob/master/examples/2_2.png" width="150"> <img src="https://github.com/spthermo/generative-image-inpainting/blob/master/examples/2_3.png" width="150">

<img src="https://github.com/spthermo/generative-image-inpainting/blob/master/examples/3_1.png" width="150"> <img src="https://github.com/spthermo/generative-image-inpainting/blob/master/examples/3_2.png" width="150"> <img src="https://github.com/spthermo/generative-image-inpainting/blob/master/examples/3_3.png" width="150">


## Acknowledgement
The Tensorboard support is provided from [yunjey](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard)

### Similar projects:
[GLCIC](https://github.com/tadax/glcic) is a similar approach developed with TensorFlow.
