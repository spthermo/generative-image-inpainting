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


## Training with CelebA


## Testing on unseen data


## Inpainting examples


## Acknowledgement
The Tensorboard support is provided from [yunjey](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard)
