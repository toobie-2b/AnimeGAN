import matplotlib.pyplot as plt
import torch
import numpy as np
from model_utils import Discriminator, Generator
import argparse
import os

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', help='Specify the path to save the images')
    parser.add_argument('--num_images',dest='num_images', help='number of characters to be created')
    options = parser.parse_args()

    if not options.path:
        options.path = './generated_characters/'

    if not options.num_images:
        parser.error('[-] Number of characters not defined')

    return options

def create_directories(path):
    if not os.path.exists(path):
        os.mkdir(path)

def load_model():
    model = Generator()
    model.load_state_dict(torch.load('Generator.pth'))
    return model

def generate(model, num_images):
    images = model(torch.randn(num_images, 100, 1, 1))
    return images.detach().numpy()

def save_anime(path, images):
    for i in range(len(images)):
        anime = images[i].reshape(3, 64, 64)
        anime = np.moveaxis(anime, 0, 2)
        anime = anime * np.array((0.5, 0.5, 0.5)) + np.array((0.5,0.5,0.5))
        anime = np.clip(anime, 0, 1)
        plt.imsave(path + f'character{i+1}.png', anime)

if __name__ == '__main__':
    arguments = get_arguments()
    create_directories(arguments.path)
    model = load_model()
    print('Generating Characters...')
    images = generate(model, int(arguments.num_images))
    print('Saving Characters...')
    save_anime(arguments.path, images)
    print('Done')
