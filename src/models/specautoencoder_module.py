from typing import Any, List, Optional, Union
import os
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from ema_pytorch import EMA
import torchaudio.transforms as TAT
from torchmetrics import MeanMetric, MinMetric
import matplotlib.pylab as plt

class SpecAutoEncoderModule(LightningModule):
    """ 
    Vanilla Autoencoder Module.
    """

    def __init__(
        self,
        autoencoder: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        bottleneck: Union[nn.Module, List[nn.Module]],
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        stft_normalized: bool,
        use_ema: bool,
        ema_beta: float,
        ema_power: float,
        loss_type: Optional[str] = None,
    ):
        super().__init__()

        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # diffusion components
        self.autoencoder = autoencoder
        self.bottleneck = nn.ModuleList(bottleneck)
        self.use_ema = use_ema
        if self.use_ema:
            self.autoencoder_ema = EMA(self.autoencoder, beta=ema_beta, power=ema_power)
            
        # loss setup
        self.loss_type = loss_type
        if self.loss_type == "mse":
            self.loss_fn = nn.MSELoss()
            
        # generation
        self.melspec_transform = TAT.MelSpectrogram(sample_rate = sample_rate,
                                                    n_fft = n_fft,
                                                    win_length = n_fft,
                                                    hop_length = hop_length,
                                                    n_mels = n_mels,
                                                    normalized = stft_normalized)
        
        # loss averaging through time
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()
        
    def spectral_normalize(self, magnitudes, C=1, clip_val=1e-7):
        return torch.log(torch.clamp(magnitudes, min=clip_val) * C)

    def spectral_de_normalize(self, magnitudes, C=1):
        return torch.exp(magnitudes) / C


    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        pass

    def model_step(self, batch: Any):
        x = batch['audio']
        # compute mel spectrogram]
        x_mel = self.melspec_transform(x).unsqueeze(1)
        x_mel = self.spectral_normalize(x_mel)
        
        y, info = self.autoencoder(x_mel, bottlenecks=self.bottleneck)

        loss = self.loss_fn(x_mel, y)
        if bool(info):
            loss = loss + info['variational_kl_loss']

        return loss, info

    def training_step(self, batch: Any, batch_idx: int):
        loss, info = self.model_step(batch)

        self.log("train_loss", loss)

        # update and log metrics
        self.train_loss(loss)
        
        if self.use_ema:
            # Update EMA model and log decay
            self.autoencoder_ema.update()
            self.log("ema_decay", self.autoencoder_ema.get_current_decay())
        return {"loss": loss}

    def on_train_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        
        for module in self.bottleneck:
            module.training = False
        
        x = batch['audio']
        x_mel = self.melspec_transform(x).unsqueeze(1)
        x_mel = self.spectral_normalize(x_mel)
        
        x_recon, var_info = self.autoencoder(x_mel, self.bottleneck) 
        val_loss = self.loss_fn(x_mel, x_recon)
        
        if bool(var_info):
            val_loss = val_loss + var_info['variational_kl_loss']

        # update and log metrics
        self.val_loss(val_loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        mel_save_dir = os.path.join(self.logger.save_dir, 'val_audio_mel')
        os.makedirs(mel_save_dir, exist_ok=True)
        mel_path = os.path.join(mel_save_dir, 'val_' + str(self.global_step) + '.png')

        if batch_idx == 0:
            with torch.no_grad():
                fig, ax = plt.subplots(2, 1, figsize=(8, 6))
                ax[0].imshow(x_mel[0][0].cpu(), aspect='auto', origin='lower', interpolation='none')
                ax[1].imshow(x_recon[0][0].cpu(), aspect='auto', origin='lower', interpolation='none')
                
                plt.savefig(mel_path)
                plt.close()

    def on_validation_epoch_end(self, outputs: List[Any]):

        self.val_loss_best(self.val_loss.compute())  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        
        pass

    def test_epoch_end(self, outputs: List[Any]):
        if self.use_ema:
            torch.save(self.autoencoder_ema, os.path.join(self.logger.save_dir, 'ae_ema_model.pt'))
        else:
            torch.save(self.autoencoder, os.path.join(self.logger.save_dir, 'ae_model.pt'))
            torch.save(self.bottleneck, os.path.join(self.logger.save_dir, 'btnk_model.pt'))

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
