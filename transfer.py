
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import os
import torchvision.utils as utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('------------------------------------------------------------------')
print(torch.__version__,device)
print('------------------------------------------------------------------')
def load_image(img_path, img_size=None):
    '''
        Resize the input image so we can make content image and style image have same size, 
        change image into tensor and normalize it
    '''
    
    image = Image.open(img_path).convert('RGB')
    if img_size is not None:
        image = image.resize((img_size, img_size))  # change image size to (3, img_size, img_size)
    
    transform = transforms.Compose([
                        # convert the (H x W x C) PIL image in the range(0, 255) into (C x H x W) tensor in the range(0.0, 1.0) 
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),   # this is from ImageNet dataset
                        ])   

    # change image's size to (b, 3, h, w)
    image = transform(image)[:3, :, :].unsqueeze(0)

    return image


def im_convert(tensor):
    """ Display a tensor as an image. """

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0)    # change size to (channel, height, width)

    '''
        tensor (batch, channel, height, width)
        numpy.array (height, width, channel)
        to transform tensor to numpy, tensor.transpose(1,2,0) 
    '''
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))   # change into unnormalized image
    image = image.clip(0, 1)    # in the previous steps, we change PIL image(0, 255) into tensor(0.0, 1.0), so convert it

    return image

# Instant Style Transfer
class IST(nn.Module):
    def __init__(self, VGG, content, style, alpha=20, resolution=512):
        super(IST, self).__init__()

        self.VGG = VGG
        self.content = content
        self.style = transforms.functional.resize(style, self.content.shape[2:])
        self.alpha = alpha
        self.resolution = resolution

        self.content_features = self.get_features(transforms.functional.resize(content, self.resolution), self.VGG)
        self.style_features = self.get_features(transforms.functional.resize(style, self.resolution), self.VGG)
        self.style_gram_matrixs = {layer: self.get_grim_matrix(self.style_features[layer]) for layer in self.style_features}
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=1)
        )
        
    def forward(self):
        x = self.content
        x = transforms.functional.resize(x, self.resolution)

        # Encoder path
        x1 = self.encoder(x)
        
        # Decoder path
        x2 = self.decoder(x1)
        x2 = transforms.functional.resize(x2, self.content.shape[2:])
        
        return x2+self.content
    
    def get_grim_matrix(self, tensor):
        b, c, h, w = tensor.size()
        tensor = tensor.view(b * c, h * w)
        gram_matrix = torch.mm(tensor, tensor.t())
        return gram_matrix
    
    def get_features(self, image, model):
        layers = {'0': 'conv1_1',   # default style layer
                '5': 'conv2_1',   # default style layer
                '10': 'conv3_1',  # default style layer
                '19': 'conv4_1',  # default style layer
                '21': 'conv4_2',  # default content layer
                '28': 'conv5_1'}  # default style layer
        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)    
            if name in layers:
                features[layers[name]] = x
        
        return features
    
    def get_loss(self, target):
        target_features = self.get_features(transforms.functional.resize(target, self.resolution), self.VGG)
        content_loss = torch.mean((target_features['conv4_2'] - self.content_features['conv4_2']) ** 2)
        style_loss = 0
        style_weights = {'conv1_1': 0.1, 'conv2_1': 0.2, 'conv3_1': 0.4, 'conv4_1': 0.8, 'conv5_1': 1.6}
        for layer in style_weights:
            target_feature = target_features[layer]  
            target_gram_matrix = self.get_grim_matrix(target_feature)
            style_gram_matrix = self.style_gram_matrixs[layer]

            layer_style_loss = style_weights[layer] * torch.mean((target_gram_matrix - style_gram_matrix) ** 2)
            b, c, h, w = target_feature.shape
            style_loss = style_loss + layer_style_loss / (c * h * w)


        return self.alpha * content_loss + style_loss
    
    def transfer(self, visualize=False):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        if visualize:
            fig, ax = plt.subplots()
            ax.axis('off')
            plt.ion()  # Enable interactive mode
            ax.imshow(im_convert(self.content))
            plt.title('Press any key to start style tranfer...')
            plt.waitforbuttonpress()
            t = tqdm(range(1, 101))
            for epoch in t:
                target = self.forward()
                loss = self.get_loss(target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch%20==0:
                    ax.imshow(im_convert(target))
                    plt.title("Elapsed Time: {:.2f} seconds".format(t.format_dict['elapsed']))
                    plt.draw()
                    plt.pause(0.00001)    
            plt.ioff() 
            plt.show()
        else:
            for epoch in range(0, 100):
                target = self.forward()
                loss = self.get_loss(target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return im_convert(target)
         
VGG = models.vgg19(pretrained=True).features
VGG.to(device)
for parameter in VGG.parameters():
    parameter.requires_grad_(False)
for i in tqdm(range(1, 60)):
    torch.manual_seed(0)
    try:
        content_image = load_image(f'../dataset/input/{i}.png')
        style_image = load_image(f'../dataset/style/{i}.png')
    except:
        continue
    # content_image = load_image('./4k.jpg')
    # style_image = load_image('./4k2.jpg')
    content_image = content_image.to(device)
    style_image = style_image.to(device)

    style_net = IST(VGG, content_image, style_image)
    style_net.to(device)
    result = style_net.transfer(False)
    plt.imsave(f'./outputs/{i}.png', result)




