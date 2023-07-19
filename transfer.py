
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import os
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import argparse
import torchvision.io as io

def load_image(img_path):
    
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),   # ImageNet 
                        ])   
    # change image's size to (b, 3, h, w)
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image


def im_convert(tensor):

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0) 
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))   # unnormalized image
    image = image.clip(0, 1)    

    return image

# Instant Style Transfer
class IST(nn.Module):
    def __init__(self, VGG, content, style):
        super(IST, self).__init__()

        self.VGG = VGG
        self.content = content
        self.style = transforms.functional.resize(style, self.content.shape[2:], antialias=True)
        self.resolution = 480

        self.content_features = self.get_features(transforms.functional.resize(content, self.resolution, antialias=True), self.VGG)
        self.style_features = self.get_features(transforms.functional.resize(style, self.resolution, antialias=True), self.VGG)
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
        
    def forward(self, x):

        downsample_content = transforms.functional.resize(x, self.resolution, antialias=True)
        # print(downsample_content.shape)

        style = self.style_net(downsample_content)

        style = transforms.functional.resize(style, self.content.shape[2:], antialias=True)

        return style + x
    
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
        target_features = self.get_features(transforms.functional.resize(target, self.resolution, antialias=True), self.VGG)
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
            target = self.forward(self.content)
            content_loss, style_loss = self.get_loss(target)
            if epoch==0:
                inital_content_loss = content_loss.item()
                loss = content_loss + style_loss
                initial_loss = loss.item()
            loss = torch.exp(content_loss/inital_content_loss - 1) * content_loss + style_loss
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
        return target
        

def main():
    parser = argparse.ArgumentParser(description='Image Style Transfer')

    # Specify content and style image paths
    parser.add_argument('--content-image', type=str, help='Path to the content image')
    parser.add_argument('--content-video', type=str, help='Path to the content video')
    parser.add_argument('--style-image', type=str, help='Path to the style image')

    # Specify content and style image folder paths
    parser.add_argument('--content-image-folder', type=str, help='Path to the folder containing content images')
    parser.add_argument('--style-image-folder', type=str, help='Path to the folder containing style images')

    args = parser.parse_args()

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.__version__, device)

    # Load VGG model
    print('Loading VGG model')
    VGG = models.vgg19(weights='DEFAULT').features
    VGG.to(device)
    for parameter in VGG.parameters():
        parameter.requires_grad_(False)
    print('Starting transfer')

    # Transfer images
    if args.content_image and args.style_image:
        # Single content and style image
        content_image = load_image(args.content_image)
        style_image = load_image(args.style_image)
        content_image = content_image.to(device)
        style_image = style_image.to(device)

        ist = IST(VGG, content_image, style_image)
        ist.to(device)
        result = im_convert(ist.transfer())
        plt.imsave('./result.png', result)

    elif args.content_image_folder and args.style_image_folder:
        # Content and style image folders
        content_folder = args.content_image_folder
        style_folder = args.style_image_folder

        # Create output folder if it doesn't exist
        output_folder = './outputs'
        os.makedirs(output_folder, exist_ok=True)

        # Process each image in the folders
        for i in tqdm(os.listdir(content_folder)):
            try:
                content_image = load_image(os.path.join(content_folder, i))
                style_image = load_image(os.path.join(style_folder, i))
            except:
                print(f'{os.path.join(style_folder, i)} not exists')
            content_image = content_image.to(device)
            style_image = style_image.to(device)

            ist = IST(VGG, content_image, style_image)
            ist.to(device)
            result = im_convert(ist.transfer())
            plt.imsave(os.path.join(output_folder, i), result)
    
    elif args.content_video and args.style_image:

        reader = io.VideoReader(args.content_video)
        fps = reader.get_metadata()['video']['fps'][0]
        print('Loading the video')
        frames = []
        for frame in tqdm(reader):
            frames.append(frame['data'])
        frames = torch.stack(frames, 0).float()/255

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        frames = (frames - mean[:, None, None]) / std[:, None, None]
        frames = frames.to(device)
        
        for i in tqdm(range(len(frames))):
            if i == 0:
                content_image = frames[0].unsqueeze(0)
                style_image = load_image(args.style_image)
                style_image = style_image.to(device)

                ist = IST(VGG, content_image, style_image)
                ist.to(device)
                frames[0] = ist.transfer()
                print(frames[0].shape)
                # plt.imsave('./result.png', )
            else:
                with torch.no_grad():
                    frames[i] = ist.forward(frames[i])

        frames = frames.cpu().detach()
        print(frames.shape)
        frames = (frames * std[:, None, None]) + mean[:, None, None]
        frames = frames * 255.0
        frames = frames.permute(0, 2, 3, 1)
        io.write_video('./result.mp4', frames, fps)

    else:
        print('Please provide either --content-image and --style-image paths or --content-image-folder and --style-image-folder paths.')

if __name__ == '__main__':
    main()