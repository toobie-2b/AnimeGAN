import matplotlib.pyplot as plt
import torch
import torch.nn
import numpy as np
from model_utils import Discriminator, Generator

def load_model():
    model = Generator()
    model.load_state_dict(torch.load('Generator.pth'))
    return model

def generate(model):
    images = model(torch.randn(128, 100, 1, 1))
    return images.detach().numpy()

def save_anime(images):
    for i in range(len(images)):
        anime = images[i].reshape(3, 64, 64)
        anime = np.moveaxis(anime, 0, 2)
        anime = np.clip(anime, 0, 1)
        plt.imsave(f'./generated/character{i}.png', anime)

if __name__ == '__main__':
    model = load_model()
    print('Generating Characters...')
    images = generate(model)
    print('Saving Characters...')
    save_anime(images)
    print('Done')