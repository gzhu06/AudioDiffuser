from typing import Any, List, Optional, Union
import os
import torch
from torch import Tensor
import torchaudio
import numpy as np
from tqdm import tqdm
from pytorch_lightning import LightningModule
from ema_pytorch import EMA
from torchmetrics import MeanMetric, MinMetric

class LatentDiffusionModule(LightningModule):
    """ AudioDiffModule.
    https://github.com/archinetai/audio-diffusion-pytorch

    """

    def __init__(
        self,
        net: torch.nn.Module,
        autoencoder_path: str,
        autoencoder_latent_scale: float,
        noise_scheduler: torch.nn.Module,
        noise_distribution: torch.nn.Module,
        sampler: torch.nn.Module,
        diffusion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        generated_sample_length: int,
        generated_sample_class: int,
        use_ema: bool,
        ema_beta: float,
        ema_power: float,
        audio_sample_rate: int,
        total_test_samples: Optional[int] = None
    ):
        super().__init__()

        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # diffusion components
        self.net = net
        self.autoencoder = torch.load(os.path.join(autoencoder_path, 'ae_model.pt'), 
                                      map_location=self.device)
        self.bottleneck = torch.load(os.path.join(autoencoder_path, 'btnk_model.pt'), 
                                     map_location=self.device)
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()
        self.bottleneck.requires_grad_(False)
        self.bottleneck.eval()
        
        self.autoencoder_latent_scale = autoencoder_latent_scale
        
        self.use_ema = use_ema
        if self.use_ema:
            self.net_ema = EMA(self.net, beta=ema_beta, power=ema_power)
        self.sampler = sampler
        self.diffusion = diffusion
        self.noise_distribution = noise_distribution # for training
        self.noise_scheduler = noise_scheduler()     # for sampling
        
        # generation
        self.generated_sample_length = generated_sample_length
        self.audio_sample_rate = audio_sample_rate
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()
        
    @torch.no_grad()
    def encode_latent(self, x: Tensor) -> Tensor:
        for module in self.bottleneck:
            module.training = False
        z, info = self.autoencoder.encode(x, with_info=True, bottlenecks=self.bottleneck)
        if "mean" in info:
            z = info["mean"]
        return z * self.autoencoder_latent_scale
    
    @torch.no_grad()
    def decode_latent(self, z: Tensor) -> Tensor:
        return self.autoencoder.decode(z / self.autoencoder_latent_scale)

    def forward(self, x: torch.Tensor):
        # predict noise
        
        audio = x['audio'].unsqueeze(1)
        audio_classes = x['label'] # kwargs
        
        z = self.encode_latent(audio)
        loss = self.diffusion(z, audio_classes, self.net, 
                              sigma_distribution=self.noise_distribution)
        return loss

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        pass

    def model_step(self, batch: Any):
        loss = self.forward(batch)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
#         wandb.log({"train/loss": loss.item()})
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        
        if self.use_ema:
            # Update EMA model and log decay
            self.net_ema.update()
            self.log("ema_decay", self.net_ema.get_current_decay())
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):

        self.val_loss_best(self.val_loss.compute())  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)
        
        with torch.no_grad():
            device = next(self.net.parameters()).device
            initial_noise = torch.randn(1, 1, self.generated_sample_length).to(device)
            z_initial_noise = self.encode_latent(initial_noise)
            target_class = torch.from_numpy(np.zeros(1).astype(int)).to(device)
            
            diff_net = self.net_ema if self.use_ema else self.net
            z_latent = self.sampler(z_initial_noise, target_class,
                                    fn=self.diffusion.denoise_fn, 
                                    net=diff_net, sigmas=self.noise_scheduler.to(device))
            audio_sample = self.decode_latent(z_latent)
            audio_sample = audio_sample.squeeze(1).cpu()
        audio_save_dir = os.path.join(self.logger.save_dir, 'val_audio')
        os.makedirs(audio_save_dir, exist_ok=True)
        audio_path = os.path.join(audio_save_dir, 'val_' + str(self.global_step) + '.wav')
        torchaudio.save(audio_path, audio_sample, self.audio_sample_rate)

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        print('Generating test samples....................')
        iteration = 1
        test_batch = 32
        target_classes = list(range(10))
        test_sample_folder = os.path.join(self.logger.save_dir, 'test_samples')
        os.makedirs(test_sample_folder, exist_ok=True)
        for i in tqdm(range(iteration)):
            with torch.no_grad():
                device = next(self.net.parameters()).device
                initial_noise = torch.randn(test_batch, self.generated_sample_length).to(device)
                target_class = torch.from_numpy(np.random.choice(target_classes, 32)).to(device)
                
                diff_net = self.net_ema if self.use_ema else self.net
                audio_sample = self.sampler(initial_noise.unsqueeze(1), target_class,
                                            self.diffusion.denoise_fn,
                                            diff_net, self.noise_scheduler.to(device))
                audio_sample = audio_sample.squeeze(1).cpu()
            
            for j in range(audio_sample.shape[0]):
                audio_filename = 'test_'+str(target_class[j].item())+'_'+str(i*test_batch+j)+'.wav'
                audio_path = os.path.join(test_sample_folder, audio_filename)
                torchaudio.save(audio_path, audio_sample[j].unsqueeze(0), self.audio_sample_rate)

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
