import torch
import torchvision.transforms.functional as TF
import wandb
from torchvision.utils import make_grid
from lightning import LightningModule
from torchmetrics.image import StructuralSimilarityIndexMeasure


def psnr(loss):
    return 10 * torch.log10(1 / loss)


class AutoEncoderTrainer(LightningModule):
    def __init__(self, encoder, decoder, criterion, optimizer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion
        self.optimizer = optimizer

        self._ssim = StructuralSimilarityIndexMeasure()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        loss = self.criterion(x_hat, x)

        self.log("train/loss", loss)
        self.log("train/psnr", psnr(loss))
        self.log("train/ssim", self._ssim(x_hat, x))

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        loss = self.criterion(x_hat, x)

        self.log("val/loss", loss)
        self.log("val/pnsr", psnr(loss))
        self.log("val/ssim", self._ssim(x_hat, x))

        if batch_idx == 0:
            self.logger.experiment.log(  # type: ignore
                {
                    "val/recon": wandb.Image(TF.to_pil_image(make_grid(x_hat, nrow=8))),
                }
            )

        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        loss = self.criterion(x_hat, x)

        self.log("test/loss", loss)
        self.log("test/psnr", psnr(loss))
        self.log("test/ssim", self._ssim(x_hat, x))

        if batch_idx == 0:
            self.logger.experiment.log(  # type: ignore
                {
                    "test/recon": wandb.Image(
                        TF.to_pil_image(make_grid(x_hat[:64], nrow=8))
                    ),
                }
            )

        return loss

    def configure_optimizers(self):
        return self.optimizer
