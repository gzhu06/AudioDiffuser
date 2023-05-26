from typing import Any, List, Union
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchaudio
from einops import repeat
from pytorch_lightning import LightningModule
from ema_pytorch import EMA
from torchmetrics import MeanMetric, MinMetric

class DiffAutoEncoderModule(LightningModule):
    """ 
    Diffusion Autoencoder Module, based on https://diff-ae.github.io/
    """

    def __init__(
        self,
        net: torch.nn.Module,
        encoder: Union[nn.Module, List[nn.Module]],
        bottleneck: nn.Module,
        noise_scheduler: torch.nn.Module,
        noise_distribution: torch.nn.Module,
        sampler: torch.nn.Module,
        diffusion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        reconstructed_sample_length: int,
        sample_rate: int,
        use_ema: bool,
        ema_beta: float,
        ema_power: float,
    ):
        super().__init__()

        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # diffusion components
        self.net = net
        self.encoder = nn.ModuleList(encoder)
        self.bottleneck = bottleneck
        self.use_ema = use_ema
        if self.use_ema:
            self.net_ema = EMA(self.net, beta=ema_beta, power=ema_power)
        self.sampler = sampler
        self.diffusion = diffusion
        self.noise_distribution = noise_distribution # for training
        self.sigmas = noise_scheduler()
            
        # generation
        self.sample_rate = sample_rate
        self.reconstructed_sample_length = reconstructed_sample_length
        
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

    def forward(self, x: torch.Tensor):
        # predict noise
        
        audio = x['audio'].unsqueeze(1)
        audio_classes = x['label'] # kwargs

        loss = self.diffusion(audio, audio_classes, self.net, 
                              sigma_distribution=self.noise_distribution, 
                              x_start=audio, 
                              x_encoders=self.encoder,
                              bottleneck=self.bottleneck)

        return loss

    def model_step(self, batch: Any):
        loss = self.forward(batch)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

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
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            
            with torch.no_grad():
                device = next(self.net.parameters()).device
                initial_noise = torch.randn(1, 1, self.reconstructed_sample_length).to(device)
                diff_net = self.net_ema if self.use_ema else self.net
                target_class = torch.from_numpy(np.zeros(1).astype(int)).to(device)
                condition_sample = batch['audio'][-1].unsqueeze(0)
                audio_sample = self.sampler(initial_noise, target_class,
                                            fn=self.diffusion.denoise_fn, 
                                            net=diff_net, sigmas=self.sigmas.to(device), 
                                            x_start=condition_sample.unsqueeze(0), 
                                            x_encoders=self.encoder,
                                            bottleneck=self.bottleneck)

                audio_sample = audio_sample.squeeze(1).cpu()
            audio_save_dir = os.path.join(self.logger.save_dir, 'val_audio')
            os.makedirs(audio_save_dir, exist_ok=True)
            audio_path_recon = os.path.join(audio_save_dir, 'val_recon'+str(self.global_step)+'.wav')
            torchaudio.save(audio_path_recon, audio_sample, self.sample_rate)
            audio_path_gt = os.path.join(audio_save_dir, 'val_gt.wav')
            torchaudio.save(audio_path_gt, condition_sample.cpu(),
                            self.sample_rate)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):

        self.val_loss_best(self.val_loss.compute())  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)
        
    def interpolate(self, x, target_class, net):
        
        device = next(self.net.parameters()).device
        
        # stochastic encoder: spherical interpolation for stochastic latents
        def cos(a, b):
            a = a.view(-1)
            b = b.view(-1)
            a = F.normalize(a, dim=0)
            b = F.normalize(b, dim=0)
            return (a * b).sum()
        
        xT = self.sampler.encode(x, target_class, 
                                 fn=self.diffusion.denoise_fn,
                                 net=net, decode=False, 
                                 sigmas=self.sigmas.to(device),
                                 x_start=x, 
                                 x_encoders=self.encoder,
                                 bottleneck=self.bottleneck)

        theta = torch.arccos(cos(xT[0], xT[1]))
        alpha = torch.tensor(np.linspace(0, 1, 10, dtype=np.float32)).to(x.device)
        xT_intp = (torch.sin((1 - alpha[:, None]) * theta) * xT[0].flatten(1)[None] + torch.sin(alpha[:, None] * theta) * xT[1].flatten(1)[None]) / torch.sin(theta)
        xT_intp = xT_intp.view(-1, *xT[0].shape)
        
        # semantic encoder: linear interpolation for semantic latents
        xse_intp = net.interpolate_semantic(x, x_encoders=self.encoder,
                                            bottleneck=self.bottleneck)
        
        # combined interpolations
        audio_samples = self.sampler(xT_intp/self.sigmas[0], target_class,
                                     fn=self.diffusion.denoise_fn, 
                                     net=net, sigmas=self.sigmas.to(device), 
                                     encoded_x=xse_intp)
        
        return audio_samples

    def test_step(self, batch: Any, batch_idx: int):
        
        # update and log metrics
        loss = self.model_step(batch)
        self.test_loss(loss)
        
        audio_save_dir = os.path.join(self.logger.save_dir, 'test_audio')
        os.makedirs(audio_save_dir, exist_ok=True)
        gen_bs = 16
        if batch_idx == 0:
            # semantic inputs
            condition_sample = batch['audio'][-1].unsqueeze(0)
            audio_path_gt = os.path.join(audio_save_dir, 'test_gt.wav')
            torchaudio.save(audio_path_gt, condition_sample.cpu(), self.sample_rate)
            
            with torch.no_grad():
                device = next(self.net.parameters()).device
                diff_net = self.net_ema if self.use_ema else self.net
                
                # exp 1: reconstructions
                # subcode xT
                initial_noise = torch.randn(gen_bs, 1, self.reconstructed_sample_length).to(device)
                target_class = torch.from_numpy(np.zeros(1).astype(int)).to(device)
                
#                 #### debug
#                 audio_samples = self.sampler.encode(condition_sample.unsqueeze(1), target_class,
#                                             fn=self.diffusion.denoise_fn, 
#                                             net=diff_net, sigmas=self.sigmas.to(device), 
#                                             decode=True, x_encoder=self.encoder,
#                                             x_start=condition_sample.unsqueeze(1), 
#                                             bottleneck=self.bottleneck)
#                 ####

                # reconstructed samples with given noise and condition audio
                condition_sample = repeat(condition_sample, 'h w -> (repeat h) w', repeat=gen_bs)
                audio_samples = self.sampler(initial_noise, target_class,
                                            fn=self.diffusion.denoise_fn, 
                                            net=diff_net, sigmas=self.sigmas.to(device), 
                                            x_start=condition_sample.unsqueeze(1), 
                                            x_encoders=self.encoder,
                                            bottleneck=self.bottleneck)
                
#                 # exp 2: interpolations
#                 intp_gts = batch['audio'][2:4].unsqueeze(1)
#                 intp_samples = self.interpolate(intp_gts, target_class, diff_net)
                
#                 for j, intp_gt in enumerate(intp_gts):
#                     audio_path_gt = os.path.join(audio_save_dir, 'intp_gt'+str(j)+'.wav')
#                     torchaudio.save(audio_path_gt, intp_gt.squeeze(1).cpu(), self.sample_rate)
                    
#             for i, audio_sample in enumerate(intp_samples):
#                 audio_path_intp = os.path.join(audio_save_dir, 'intp'+str(i)+'.wav')
#                 torchaudio.save(audio_path_intp, audio_sample.cpu(), self.sample_rate)
                
            for i, audio_sample in enumerate(audio_samples):
                audio_path_recon = os.path.join(audio_save_dir, 'recon'+str(i)+'.wav')
                torchaudio.save(audio_path_recon, audio_sample.cpu(), self.sample_rate)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        pass

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
