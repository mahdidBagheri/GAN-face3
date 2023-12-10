import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models

import os
import random
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import glob
import cv2
import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(42)

data_dir = '/home/mahdi/edu/GAN-face3/img_align_celeba/'
dataset_size = -1
n_epochs = 100
transform_img = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor()
    ])

class FaceDataset(Dataset):
    def __init__(self, dir, start, end):
        super(FaceDataset, self).__init__()

        # transform_list_input = [transforms.ToTensor(),
        #                         transforms.Normalize((0.5,), (0.5,))]

        transform_list_input = [transforms.ToTensor()]

        self.input_transform = transforms.Compose(transform_list_input)

        transform_list_output = [transforms.ToTensor()]
        self.output_transform = transforms.Compose(transform_list_output)

        image_path = os.path.join(dir)
        self.image_list = glob.glob(image_path + "*.jpg")[:dataset_size]

        self.image_list = self.image_list[int(start*len(self.image_list)):int(end*len(self.image_list))]
        a = 0

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        assert index <= len(self)
        image_path = self.image_list[index]
        img = cv2.imread(image_path)
        img_norm_t = (self.input_transform(img))
        return img_norm_t

    def test(self, index):
        img = self[index]

def get_dataloader(batch_size, data_dir, transforms):
    """
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of images in a batch
    :param data_dir: Directory where image data is located
    :param transforms: data augmentations
    :return: DataLoader with batched data
    """
    
    ds = FaceDataset(data_dir, 0, 1)
    data_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=True)

    return data_loader

batch_size = 32 # Instead of individual samples, the data loader produces batched samples of given number
train_loader = get_dataloader(batch_size,data_dir, transform_img)

print(len(train_loader.dataset))

def scale_images(x, max = 1.00 , min = -1.00):
    x = x * (max - min) + min
    return x

#let's check the scaling
img = train_loader.dataset[5]
print('Before scaling min: ', img.min())
print('Before scaling max: ', img.max())

scaled_img = scale_images(img)

print('After Scaling Min: ', scaled_img.min())
print('After Scaling Max: ', scaled_img.max())


def conv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, bias=False):
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding)
    # appending convolutional layer
    layers.append(conv_layer)
    # appending batch norm layer
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        """
        Initializing the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer based on which we will create the  next ones where next  layer depth = 2 * previous layer depth
        """
        super(Discriminator, self).__init__()

        # complete init function
        self.conv_dim = conv_dim

        self.conv1 = conv(3, conv_dim, batch_norm=False)
        self.conv2 = conv(conv_dim, conv_dim * 2)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8)
        self.fc = nn.Linear(conv_dim * 4 * 4 * 2, 1)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network
        :return: Discriminator logits; the output of the neural network
        """
        # define feedforward behavior
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)

        x = x.view(-1, self.conv_dim * 4 * 2 * 4)

        x = self.fc(x)

        return x


def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, bias=False):
    layers = []

    # append transpose conv layer -- we are not using bias terms in conv layers
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))

    # optional batch norm layer
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


class Generator(nn.Module):

    def __init__(self, z_size, conv_dim):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()
        self.conv_dim = conv_dim

        self.fc = nn.Linear(z_size, conv_dim * 4 * 4 * 4)
        # complete init function

        self.de_conv1 = deconv(conv_dim * 4, conv_dim * 2)
        self.de_conv2 = deconv(conv_dim * 2, conv_dim)
        self.de_conv3 = deconv(conv_dim, 3, 4, batch_norm=False)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior
        x = self.fc(x)
        x = self.dropout(x)

        x = x.view(-1, self.conv_dim * 4, 4, 4)

        x = F.relu(self.de_conv1(x))
        x = F.relu(self.de_conv2(x))
        x = self.de_conv3(x)
        x = F.tanh(x)

        return x


def weights_init_normal(m):
    """
    :param m: A module or layer in a network
    """
    # like `Conv`, `BatchNorm2d`, `Linear`, etc.
    classname = m.__class__.__name__

    #  initial weights to convolutional and linear layers
    if (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal(m.weight.data, 0.0, 0.2)

    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant(m.bias.data, 0.0)

# model hyperparameters
d_conv_dim = 64
g_conv_dim = 128
z_size = 100
# building discriminator and generator from the classes defined above
discriminator = Discriminator(d_conv_dim)
generator = Generator(z_size=z_size, conv_dim=g_conv_dim)

# initialize model weights
discriminator.apply(weights_init_normal)
generator.apply(weights_init_normal)
print("done")

# let's look at our discriminator model
print(discriminator)

# let's look at our generator model
print(generator)

use_gpu = torch.cuda.is_available()

lr = 0.0002 #learning rate
beta1=0.5
beta2=0.999

# optimizers for the discriminator D and generator G
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr, (beta1, beta2)) # for discriminator
generator_optimizer = torch.optim.Adam(generator.parameters(), lr, (beta1, beta2)) # for generator


def real_loss(D_out, smooth=False):
    '''Calculates how close discriminator outputs are to being real.
       param, D_out: discriminator logits
       return: real loss'''
    batch_size = D_out.size(0)

    if smooth:
        labels = torch.ones(batch_size) * 0.9
    else:
        labels = torch.ones(batch_size)

    if use_gpu:
        labels = labels.cuda()

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)

    return loss


def fake_loss(D_out):
    '''Calculates how close discriminator outputs are to being fake.
       param, D_out: discriminator logits
       return: fake loss'''
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)

    if use_gpu:
        labels = labels.cuda()

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)

    return loss

def plot_imgs():
    z = torch.randn(8, 1, 1, z_size)
    sample_imgs = generator(z).cpu()

    fig = plt.figure()
    for i in range(sample_imgs.size(0)):
        plt.subplot(2, 4, i + 1)

        plt.imshow(sample_imgs.detach()[i, 0, :, :], cmap='gray_r', interpolation='none')
        plt.title(f"{i}")
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    plt.show()

def train(D, G, n_epochs, train_on_gpu, print_every=50):
    '''Trains adversarial networks for some number of epochs
       param, D: the discriminator network
       param, G: the generator network
       param, n_epochs: number of epochs to train for
       param, print_every: when to print and record the models' losses
       return: D and G losses'''

    # move models to GPU
    if train_on_gpu:
        D.cuda()
        G.cuda()

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size = 16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    # move z to GPU if available
    if train_on_gpu:
        fixed_z = fixed_z.cuda()

    # epoch training loop
    for epoch in range(n_epochs):
        plot_imgs()
        # batch training loop
        for batch_i, real_images in enumerate(train_loader):

            batch_size = real_images.size(0)
            real_images = scale_images(real_images)

            # Train the discriminator on real and fake images
            discriminator_optimizer.zero_grad()

            if train_on_gpu:
                real_images = real_images.cuda()

            D_real = D(real_images)
            d_real_loss = real_loss(D_real)

            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()

            if train_on_gpu:
                z = z.cuda()

            fake_images = G(z)

            D_fake = D(fake_images)
            d_fake_loss = fake_loss(D_fake)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            discriminator_optimizer.step()

            # 2. Train the generator with an adversarial loss
            generator_optimizer.zero_grad()

            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()

            if train_on_gpu:
                z = z.cuda()

            fake_images = G(z)

            D_fake = D(fake_images)

            g_loss = real_loss(D_fake)

            g_loss.backward()
            generator_optimizer.step()

            # Print some loss stats
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print(f'Epoch [{epoch + 1}/{n_epochs}] | d_loss: {d_loss.item()} | g_loss: {g_loss.item()} | batch {batch_i}/{len(train_loader)}')
                plot_imgs()

        ## AFTER EACH EPOCH##
        # this code assumes your generator is named G, feel free to change the name
        # generate and save sample, fake images
        G.eval()  # for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train()  # back to training mode

    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    # finally return losses
    return losses


losses = train(discriminator, generator, n_epochs=n_epochs, train_on_gpu = use_gpu)

fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()

plot_imgs()