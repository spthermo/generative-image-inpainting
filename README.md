# Generative Image Inpainting
The presented deep NN architecture consists of a deep convolutional autoencoder (generator) and a discriminator network. The implementation is a based on the model proposed in [Globally and Locally Consistent Image Completion](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf). Note that this is not an exact replication, as there are some layer-level modifications. The model is developed using PyTorch and is trained/tested on CelebA dataset.

## Prerequisites
The architecture has been implemented using the following:
-Python 3.5
-Scipy
-Torchvision
-Tensorflow 1.7.0
-Tensorboard

Tensorflow and Tensorboard are used for visualization and monitoring purposes, thus they are not mandatory.

## Model architecture details
Generator details:

|     Type     | Kernel | Dilation | Stride | Output |
|:------------:|:------:|:--------:|:------:|:------:|
|     conv     |  5x5   |    1     |  1x1   |   64   |
|     conv     |  5x5   |    1     |  1x1   |   128  |
|     conv     |  5x5   |    1     |  1x1   |   128  |
|     conv     |  5x5   |    1     |  1x1   |   256  |
|     conv     |  5x5   |    1     |  1x1   |   256  |
|     conv     |  5x5   |    1     |  1x1   |   256  |
| dilated conv |  5x5   |    2     |  1x1   |   256  |
| dilated conv |  5x5   |    4     |  1x1   |   256  |
| dilated conv |  5x5   |    8     |  1x1   |   256  |
| dilated conv |  5x5   |   16     |  1x1   |   256  |
|     conv     |  5x5   |    1     |  1x1   |   256  |
|     conv     |  5x5   |    1     |  1x1   |   256  |
|    deconv    |  5x5   |    1     |  1x1   |   128  |
|     conv     |  5x5   |    1     |  1x1   |   128  |
|    deconv    |  5x5   |    1     |  1x1   |   64   |
|     conv     |  5x5   |    1     |  1x1   |   32   |
|    output    |  5x5   |    1     |  1x1   |   3    |



## Training with CelebA


## Testing on unseen data


## Inpainting examples


## Acknowledgement
The Tensorboard support is provided from [yunjey](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard)
