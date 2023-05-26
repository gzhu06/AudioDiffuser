from typing import Any
import os
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric

class DiffwaveModule(LightningModule):
    """
        Example DiffwaveModule.
    """
    def __init__(
        self,
        net: torch.nn.Module,
        noise_scheduler: torch.nn.Module,
        sampler: torch.nn.Module,
        diffusion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        generated_sample_length: int
    ):
        super().__init__()

        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # diffusion components
        self.net = net
        self.sampler = sampler
        self.diffusion = diffusion
        self.noise_scheduler = noise_scheduler()
        self.generated_sample_length = generated_sample_length
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()
            
    def forward(self, x: torch.Tensor, noise_level: torch.Tensor):
        # predict noise

        audio = x['audio']
        loss = self.diffusion(audio, self.net, noise_level.to(audio.device))
        return loss

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        pass

    def model_step(self, batch: Any):
        noise_level = torch.cumprod(1 - self.noise_scheduler, 0)
        loss = self.forward(batch, noise_level)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, 
                 on_epoch=True, prog_bar=True, rank_zero_only=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any):
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, 
                 on_epoch=True, prog_bar=True, rank_zero_only=True)

        return {"loss": loss}
    
    def validation_epoch_end(self):
        
        with torch.no_grad():
            device = next(self.net.parameters()).device
            initial_noise = torch.randn(1, self.generated_sample_length).to(device)
            audio_sample = self.sampler(initial_noise, self.net, self.noise_scheduler)
            audio_sample = audio_sample.cpu()
        
        audio_path = os.path.join(self.logger.save_dir, 'val_' + str(self.global_step) + '.wav')
        torchaudio.save(audio_path, audio_sample, 16000)
        
        self.val_loss_best(self.val_loss.compute())  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

    def test_step(self, batch: Any):

        loss = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def test_epoch_end(self):
        
        iteration = 10
        test_batch = 32
        test_sample_folder = os.path.join(self.logger.save_dir, 'test_samples')
        os.makedirs(test_sample_folder, exist_ok=True)
        for i in tqdm(range(iteration)):
            with torch.no_grad():
                device = next(self.net.parameters()).device
                initial_noise = torch.randn(test_batch, self.generated_sample_length).to(device)
                audio_sample = self.sampler(initial_noise, self.net, self.noise_scheduler)
                audio_sample = audio_sample.cpu()
            
            
            for j in range(audio_sample.shape[0]):
                audio_path = os.path.join(test_sample_folder, 'test_' + str(i*test_batch+j)+ '.wav')
                torchaudio.save(audio_path, audio_sample[j].unsqueeze(0), 16000)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "diffwave.yaml")
    _ = hydra.utils.instantiate(cfg)