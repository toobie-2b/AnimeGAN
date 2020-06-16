import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import argparse
import random
from model_utils import Discriminator, Generator, apply_weights

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', dest='dataset', help='Path of the dataset')
    parser.add_argument('--epoch', dest='epoch', help='Number of training epochs')
    parser.add_argument('--device', dest='device', help='Specify the training device (default: CPU)')
    parser.add_argument('--continue', dest='cont_train', help='Continue training?')
    parser.add_argument('--seed', dest='seed', help='Specify Random Seed')

    options = parser.parse_args()

    if not options.dataset:
        parser.error('[-] Dataset path not given')

    if not options.epoch:
        parser.error('[-] Number of epochs not specified')

    if not options.device:
        options.device = 'cpu'
    else:
        print(f'Training on {options.device}')

    if not options.cont_train:
        options.cont_train=False

    if not options.seed:
        options.seed = random.randint(1, 1000)
    random.seed(int(options.seed))
    torch.manual_seed(int(options.seed))

    return options


def load_data(data_dir):
    transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            )])
    dataset =  datasets.ImageFolder(data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=128)
    return dataloader


def models_init():
    netD = Discriminator()
    netG = Generator()

    apply_weights(netD)
    apply_weights(netG)

    return netD, netG


def train(netD, netG, dataloader, num_epochs, device, check=False):
    if check:
        netD.load_state_dict(torch.load('./Discriminator.pth'))
        netG.load_state_dict(torch.load('./Generator.pth'))
        print('\nContinuing Training...\n')
    else:
        print('Starting Training...\n')

    netD.to(device)
    netG.to(device)
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        torch.save(netD.state_dict(), f'Discriminator.pth')
        torch.save(netG.state_dict(), f'Generator.pth')
        for idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            optimizerD.zero_grad()
            output = netD(images).reshape(-1)
            
            smooth_real = round(random.uniform(0.7, 1.12), 2)
            labels = (smooth_real*torch.ones(images.shape[0])).to(device)
            lossD_real = criterion(output, labels)

            fake = netG(torch.randn(images.shape[0], 100, 1, 1).to(device))
            output = netD(fake.detach()).reshape(-1)

            smooth_fake = round(random.uniform(0.0, 0.3), 2)
            labels = (smooth_fake*torch.ones(images.shape[0])).to(device)
            lossD_fake = criterion(output, labels)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            optimizerG.zero_grad()
            output = netD(fake).reshape(-1)
            labels = torch.ones(images.shape[0]).to(device)
            lossG = criterion(output, labels)
            lossG.backward()
            optimizerG.step()

            if idx % 50 == 0 and idx!=0:
                print(f'epoch[{epoch+1:3d}/{num_epochs}]=> lossD: {lossD.item():.4f}\tlossG: {lossG.item():.4f}')


if __name__ == '__main__':
    arguments = get_arguments()
    dataloader = load_data(arguments.dataset)
    netD, netG = models_init()
    train(netD, netG, dataloader, int(arguments.epoch), arguments.device, arguments.cont_train)
