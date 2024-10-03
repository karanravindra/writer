import torch
import torchvision.transforms.functional as TF
import wandb
from torchvision.utils import make_grid
from lightning import LightningModule

from src import Decoder

class RF:
    def __init__(self, model, ln=False):
        self.model = model
        self.ln = ln
        
    def __call__(self, x, cond):
        return self.forward(x, cond)

    def forward(self, x, cond):
        B, C, H, W = x.shape
        if self.ln:
            nt = torch.randn((B,), device=x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((B,), device=x.device)

        texp = t.view([B, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, cond)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b, device=z.device).view(
            [b, *([1] * len(z.shape[1:]))]
        )
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b, device=z.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images


class RectFlowTrainer(LightningModule):
    def __init__(self, model, optimizer, rf_ln=True):
        super().__init__()
        self.model = model
        self.rf = RF(model, ln=rf_ln)
        self.optimizer = optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # shift y
        y += 1
        
        # dropout y
        p = torch.rand(y.shape, device=y.device) < 0.01
        y[p] = 0

        loss, _ = self.rf(x, y)
        self.log("train/loss", loss)

        return loss
    
    def on_validation_epoch_start(self):
        decoder = Decoder(1, 6)
        decoder.load_state_dict(torch.load("checkpoints/decoder.pt", weights_only=False))
        decoder.to(self.device)
        
        B = 64
        # generate annd save samples
        cond = torch.randint(0, 11, size=(B,), device=self.device)
        uncond = torch.zeros((B,), device=self.device).long()
        
        init_noise = torch.randn((B, 4, 4, 4), device=self.device)
        
        samples = self.rf.sample(init_noise, cond, null_cond=uncond, sample_steps=50, cfg=2)
        images = torch.cat(samples, dim=0)
        images = decoder(images)
        images = images.split(B, dim=0)
        images = [TF.to_pil_image(make_grid(img, nrow=8, normalize=True)) for img in images]
        
        images[0].save(
            f"generated/{self.current_epoch}.gif",
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0,
        )
        
        self.logger.experiment.log({"sample": wandb.Image(images[-1])})

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # shift y
        y += 1
        
        # dropout y
        p = torch.rand(y.shape, device=y.device) < 0.01
        y[p] = 0

        loss, _ = self.rf(x, y)
        self.log("val/loss", loss)

        return loss

    def on_test_epoch_start(self) -> None:
        decoder = Decoder(1, 6)
        decoder.load_state_dict(torch.load("checkpoints/decoder.pt", weights_only=False))
        decoder.to(self.device)
        
        B = 64
        # generate annd save samples
        cond = torch.randint(0, 11, size=(B,), device=self.device)
        uncond = torch.zeros((B,), device=self.device).long()
        
        init_noise = torch.randn((B, 4, 4, 4), device=self.device)
        
        samples = self.rf.sample(init_noise, cond, null_cond=uncond, sample_steps=50, cfg=2)
        images = torch.cat(samples, dim=0)
        images = decoder(images)
        images = images.split(B, dim=0)
        images = [TF.to_pil_image(make_grid(img, nrow=8, normalize=True)) for img in images]
        
        images[0].save(
            f"generated/{self.current_epoch}.gif",
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0,
        )
        
        self.logger.experiment.log({"test/sample": wandb.Image(images[-1])})

    def test_step(self, batch, batch_idx):
        x, y = batch

        # shift y
        y += 1
        
        # dropout y
        p = torch.rand(y.shape, device=y.device) < 0.01
        y[p] = 0

        loss, _ = self.rf(x, y)
        self.log("test/loss", loss)

        return loss

    def configure_optimizers(self):
        return self.optimizer
