from torch import float32
import torchvision.transforms.v2 as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from lightning import LightningDataModule


class MNISTDM(LightningDataModule):
    train_dataset: MNIST
    test_dataset: MNIST

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
        transform = [
            T.Resize((self.image_size, self.image_size)),
            T.ToImage(),
            T.ToDtype(float32, scale=True),
        ]

        self.train_dataset = MNIST(
            self.data_dir + "/train",
            transform=T.Compose(transform),
            train=True,
            download=True,
        )
        self.test_dataset = MNIST(
            self.data_dir + "/test",
            transform=T.Compose(transform),
            train=False,
            download=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    dm = MNISTDM(data_dir="data")
    dm.prepare_data()
    dm.setup()
    print(dm.train_dataloader())
    print(dm.val_dataloader())
    print(dm.test_dataloader())
