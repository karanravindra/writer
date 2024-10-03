import torch
from torch.nn.functional import mse_loss
from torch.optim import AdamW
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger

from src import LatentsDM, DiT, RectFlowTrainer
    
def main(HPARAMS):
    torch.manual_seed(42)
    torch.mps.manual_seed(42)
    
    dm = LatentsDM(
        "data",
        HPARAMS["data/batch_size"],
        HPARAMS["data/num_workers"],
    )
    dm.prepare_data()
    dm.setup()

    dit = DiT(
        emb_dim=HPARAMS["model/emb_dim"],
        num_heads=HPARAMS["model/num_heads"],
        mlp_ratio=HPARAMS["model/mlp_ratio"],
        num_layers=HPARAMS["model/num_layers"],
    )
    autoencoder_trainer = RectFlowTrainer(
        dit,
        HPARAMS["train/optimizer"](
            dit.parameters(),
            lr=HPARAMS["train/lr"],
            weight_decay=HPARAMS["train/weight_decay"],
        ),
        rf_ln=True,
    )

    logger = WandbLogger(project="rectflow", log_model=True, anonymous=True)
    logger.watch(dit, log="all", log_graph=True)
    logger.log_hyperparams(HPARAMS)
    trainer = Trainer(
        logger=logger,
        precision="bf16-mixed",
        max_epochs=HPARAMS["train/epochs"],
        check_val_every_n_epoch=1,
        
    )

    trainer.fit(autoencoder_trainer, dm)
    trainer.test(autoencoder_trainer, dm)
    
    logger.finalize("success")


if __name__ == "__main__":
    HPARAMS = {
        "data/batch_size": 128,
        "data/num_workers": 4,
        "model/emb_dim": 32,
        "model/num_heads": 16,
        "model/mlp_ratio": 4,
        "model/num_layers": 6,
        "train/epochs": 200,
        "train/lr": 8e-4,
        "train/weight_decay": 1e-2,
        "train/optimizer": AdamW,
    }
    
    main(HPARAMS)
