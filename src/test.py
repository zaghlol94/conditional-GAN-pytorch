import torch
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from generator import Generator
from discriminator import Discriminator
from utils import initialize_weights, gradient_penalty
from config import config

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = config["LEARNING_RATE"] # could also use two lrs, one for gen and one for disc
BATCH_SIZE = config["BATCH_SIZE"]
IMAGE_SIZE = config["IMAGE_SIZE"]
CHANNELS_IMG = config["CHANNELS_IMG"]
Z_DIM = config["NOISE_DIM"]
GEN_EMBEDDING = config["GEN_EMBEDDING"]
NUM_CLASSES = config["NUM_CLASSES"]
NUM_EPOCHS = config["NUM_EPOCHS"]
FEATURES_CRITIC = config["FEATURES_DISC"]
FEATURES_GEN = config["FEATURES_GEN"]
CRITIC_ITERATIONS = config["CRITIC_ITERATIONS"]
LAMBDA_GP = config["LAMBDA_GP"]

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
#comment mnist and uncomment below if you want to train on CelebA dataset
#dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
data, label = next(iter(loader))
print(label.shape)