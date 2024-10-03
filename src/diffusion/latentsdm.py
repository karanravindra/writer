import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule


class LatentsDM(LightningDataModule):
    train_dataset: Dataset
    test_dataset: Dataset

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 128,
        image_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            torch.load("checkpoints/train_latents.pt", weights_only=False),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            torch.load("checkpoints/val_latents.pt", weights_only=False),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            torch.load("checkpoints/val_latents.pt", weights_only=False),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    dm = LatentsDM(data_dir="data")
    dm.prepare_data()
    dm.setup()
    print(dm.train_dataloader())
    print(dm.val_dataloader())
    print(dm.test_dataloader())
    
    print(next(iter(dm.train_dataloader())))
