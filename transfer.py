
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


def load_image(img_path):
    '''
        Resize the input image so we can make content image and style image have same size, 
        change image into tensor and normalize it
    '''
    
    image = Image.open(img_path).convert('RGB')
    
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
    def __init__(self, VGG, content, style):
        super(IST, self).__init__()

        self.VGG = VGG
        self.content = content
        self.style = transforms.functional.resize(style, self.content.shape[2:])
        self.resolution = 480

        self.content_features = self.get_features(transforms.functional.resize(content, self.resolution), self.VGG)
        self.style_features = self.get_features(transforms.functional.resize(style, self.resolution), self.VGG)
        self.style_gram_matrixs = {layer: self.get_grim_matrix(self.style_features[layer]) for layer in self.style_features}
        
        self.style_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=1)
        )
        
    def forward(self):

        downsample_content = transforms.functional.resize(self.content, self.resolution)

        style = self.style_net(downsample_content)

        style = transforms.functional.resize(style, self.content.shape[2:])

        return style + self.content
    
    def get_grim_matrix(self, tensor):
        b, c, h, w = tensor.size()
        tensor = tensor.view(b * c, h * w)
        gram_matrix = torch.mm(tensor, tensor.t())
        return gram_matrix
    
    def get_features(self, image, model):
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
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
        return content_loss, style_loss
    
    def transfer(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        best_loss = float('inf')  
        patience = 10 
        early_stop_counter = 0  
        for epoch in range(0, 150):
            target = self.forward()
            content_loss, style_loss = self.get_loss(target)
            if epoch==0:
                inital_content_loss = content_loss.item()
                loss = content_loss + style_loss
                initial_loss = loss.item()
            loss = torch.exp(content_loss/inital_content_loss - 1) * content_loss + style_loss # maybe leave the selection of content to style ratio here·····
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            normalized_loss = loss.item()/initial_loss

            if normalized_loss < best_loss - 0.01:
                best_loss = normalized_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    break
        return im_convert(target)
         

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__,device)
VGG = models.vgg19(pretrained=True).features
VGG.to(device)
for parameter in VGG.parameters():
    parameter.requires_grad_(False)

for i in tqdm(os.listdir('./dataset/input/')):
    torch.manual_seed(0)
    content_image = load_image(f'./dataset/input/'+i)
    style_image = load_image(f'./dataset/style/'+i)
    content_image = content_image.to(device)
    style_image = style_image.to(device)

    ist = IST(VGG, content_image, style_image)
    ist.to(device)
    result = ist.transfer()
    plt.imsave(f'./outputs/{i}', result)


