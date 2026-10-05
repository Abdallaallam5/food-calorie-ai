"""تحميل الداتا وتجهيز الصور."""
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Food101

from . import config as C


def get_transforms(train: bool):
    """Train: فيه augmentation عشوائي. Test: resize و crop من النص بس."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(C.IMG_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(C.MEAN, C.STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(C.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(C.MEAN, C.STD),
    ])


def get_datasets(with_transforms: bool = True):
    train_ds = Food101(C.DATA_DIR, split="train", download=True,
                       transform=get_transforms(True) if with_transforms else None)
    test_ds = Food101(C.DATA_DIR, split="test", download=True,
                      transform=get_transforms(False) if with_transforms else None)
    return train_ds, test_ds


def get_loaders():
    train_ds, test_ds = get_datasets()
    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=C.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                             num_workers=C.NUM_WORKERS, pin_memory=True)
    return train_loader, test_loader, train_ds.classes
