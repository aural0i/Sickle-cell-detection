"""PyTorch Dataset and transform definitions for the primary dataset.

Augmentation is applied ONLY when train=True. Every choice here is
biologically motivated - see docs/data_pipeline.md for the full reasoning -
and deliberately excludes anything that would alter apparent staining (no
hue/saturation jitter), since Field/Leishman stain color could itself carry
real diagnostic signal that augmentation shouldn't scramble.
"""
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224  # standard input size for ImageNet-pretrained ResNet18/MobileNetV2


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class SickleCellDataset(Dataset):
    """Expects a DataFrame with columns 'path' and 'label' (see
    src/data/manifest.py for the label definition: 1 = positive/sickle,
    0 = negative/normal).
    """

    def __init__(self, df, train: bool):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.transform = build_transforms(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return image, label
