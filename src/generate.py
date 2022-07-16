import torch
import matplotlib.pyplot as plt
import argparse
from generator import Generator
from config import config
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description="generate image for a number")
parser.add_argument("-n", "--number", type=int, required=True, help="the number that you want to draw")
args = parser.parse_args()

print(args.number)

x = torch.tensor([args.number])
print(x.shape)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNELS_IMG = config["CHANNELS_IMG"]
NOISE_DIM = config["NOISE_DIM"]
FEATURES_GEN = config["FEATURES_GEN"]
IMAGE_SIZE = config["IMAGE_SIZE"]
GEN_EMBEDDING = config["GEN_EMBEDDING"]
NUM_CLASSES = config["NUM_CLASSES"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMAGE_SIZE, GEN_EMBEDDING).to(device)
    gen.load_state_dict(torch.load("gen.pt"))
    noise = torch.randn(1, NOISE_DIM, 1, 1).to(device)
    image = gen(noise, x.to(device))
    print(image.shape)
    plt.imshow(image.squeeze(0).to("cpu").permute(1, 2, 0))
    plt.show()
    save_image(image, "results.jpg", normalize=True)
