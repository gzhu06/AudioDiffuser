from typing import Any, List, Optional, Union
import os
import torch
import torch.nn as nn
import torchaudio
from pytorch_lightning import LightningModule
from ema_pytorch import EMA
import auraloss
from torchmetrics import MeanMetric, MinMetric

class AutoEncoderModule(LightningModule):
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
        if self.loss_type == "mrstft":
            self.loss_fn = auraloss.freq.MultiResolutionSTFTLoss(sample_rate=sample_rate)
        elif self.loss_type == "sdstft":
            scales = [2048, 1024, 512, 256, 128]
            hop_sizes, win_lengths, overlap = [], [], 0.75
            for scale in scales:
                hop_sizes += [int(scale * (1.0 - overlap))]
                win_lengths += [scale]
            self.loss_fn = auraloss.freq.SumAndDifferenceSTFTLoss(
                fft_sizes=scales, hop_sizes=hop_sizes, win_lengths=win_lengths
            )
        elif self.loss_type == "mse":
            self.loss_fn = nn.MSELoss()
            
        # generation
        self.sample_rate = sample_rate
        
        # loss averaging through time
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()


    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        pass

    def model_step(self, batch: Any):
        x = batch['audio'].unsqueeze(1)
        y, info = self.autoencoder(x, with_info=True, bottlenecks=self.bottleneck)
        loss = self.loss_fn(x, y)

        return loss, info

    def training_step(self, batch: Any, batch_idx: int):
        loss, info = self.model_step(batch)

        if "loss" in info:
            loss_bottleneck = info["loss"]
            loss += self.loss_bottleneck_weight * loss_bottleneck
            self.log("loss_bottleneck", loss_bottleneck)

        self.log("train_loss", loss)

        # update and log metrics
        self.train_loss(loss)
        
        if self.use_ema:
            # Update EMA model and log decay
            self.autoencoder_ema.update()
            self.log("ema_decay", self.autoencoder_ema.get_current_decay())
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        
        loss, _ = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):

        self.val_loss_best(self.val_loss.compute())  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        
        loss, _ = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if batch_idx == 0:
            x = batch['audio'].unsqueeze(1)
            y, info = self.autoencoder(x, with_info=True, bottlenecks=self.bottleneck)

            test_sample_folder = os.path.join(self.logger.save_dir, 'test_samples')

            os.makedirs(test_sample_folder, exist_ok=True)
            for j in range(y.shape[0]):
                audio_sample = y[j].cpu()
                audio_filename = 'test_' + str(j) + '.wav'
                audio_path = os.path.join(test_sample_folder, audio_filename)
                torchaudio.save(audio_path, audio_sample, self.sample_rate)

                audio_sample = x[j].cpu()
                audio_filename = 'gt_' + str(j) + '.wav'
                audio_path = os.path.join(test_sample_folder, audio_filename)
                torchaudio.save(audio_path, audio_sample, self.sample_rate)

        return {"loss": loss}

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
