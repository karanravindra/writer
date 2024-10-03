from torch.nn.functional import mse_loss
from torch.optim import AdamW
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger

from src import MNISTDM, Encoder, Decoder, AutoEncoderTrainer

HPARAMS = {
    "data/batch_size": 128,
    "data/image_size": 32,
    "data/num_workers": 4,
    "model/width": 6,
    "model/in_channels": 1,
    "train/epochs": 100,
    "train/lr": 8e-4,
    "train/weight_decay": 1e-2,
    "train/loss": mse_loss,
    "train/optimizer": AdamW,
}

if __name__ == "__main__":
    dm = MNISTDM(
        "data",
        HPARAMS["data/batch_size"],
        HPARAMS["data/image_size"],
        HPARAMS["data/num_workers"],
    )
    dm.prepare_data()
    dm.setup()

    encoder = Encoder(HPARAMS["model/in_channels"], HPARAMS["model/width"])
    decoder = Decoder(HPARAMS["model/in_channels"], HPARAMS["model/width"])
    autoencoder_trainer = AutoEncoderTrainer(
        encoder,
        decoder,
        mse_loss,
        HPARAMS["train/optimizer"](
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=HPARAMS["train/lr"],
            weight_decay=HPARAMS["train/weight_decay"],
        ),
    )

    logger = WandbLogger(project="writer", log_model=True, anonymous=True)
    logger.watch(encoder, log="all", log_graph=True)
    logger.watch(decoder, log="all", log_graph=True)
    logger.log_hyperparams(HPARAMS)
    trainer = Trainer(
        logger=logger,
        precision="bf16-mixed",
        max_epochs=HPARAMS["train/epochs"],
        check_val_every_n_epoch=1,
    )

    trainer.fit(autoencoder_trainer, dm)
    trainer.test(autoencoder_trainer, dm)
