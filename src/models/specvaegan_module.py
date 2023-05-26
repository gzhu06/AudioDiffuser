from typing import Any, List, Union
import os
from functools import partial
import torch
import torchaudio
import torch.nn as nn
from pytorch_lightning import LightningModule
import torchaudio.transforms as TAT
from torchaudio.functional import amplitude_to_DB
from ema_pytorch import EMA
import torch.nn.functional as F
from einops import rearrange
from torchmetrics import MeanMetric, MinMetric
from .vqgan_module import hinge_gen_loss, hinge_discr_loss, gradient_penalty
from .module_utils import exists
import matplotlib.pylab as plt

class SpecVAEGANModule(LightningModule):
    """ 
    Spec-VAE with GAN, using manual optimization
    """

    def __init__(
        self,
        autoencoder: torch.nn.Module,
        discriminator: torch.nn.Module,
        optimizer_gener: torch.optim.Optimizer,
        optimizer_discr: torch.optim.Optimizer,
        accum_grads: int,
        apply_grad_penalty: bool,
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
        warmup_steps: int = 10_000,
        max_grad_norm = 0.5,
        discr_max_grad_norm = None,
        recon_loss_weight: float = 1.,
        vae_loss_weight: float = 1.,
        adversarial_loss_weight: float = 1.,
        feature_loss_weight: float = 100
    ):
        super().__init__()

        # for VQGAN training, there will be two optimizers
        self.optimizer_gener = optimizer_gener
        self.optimizer_discr = optimizer_discr
        self.scheduler = scheduler
        self.apply_grad_penalty = apply_grad_penalty
        self.warmup_steps = warmup_steps
        
        # GAN components
        ## patchgan discriminator
        self.discriminator = discriminator

        ## autoencoder and vae bottleneck
        self.autoencoder = autoencoder
        self.bottleneck = nn.ModuleList(bottleneck)

        self.use_ema = use_ema
        if self.use_ema:
            self.autoencoder_ema = EMA(self.autoencoder, beta=ema_beta, power=ema_power)

        self.recon_loss_weight = recon_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.vae_loss_weight = vae_loss_weight
        self.recon_loss_fn = nn.MSELoss()
        
        # asp parameters
        self.melspec_transform = TAT.MelSpectrogram(sample_rate = sample_rate,
                                                    n_fft = n_fft,
                                                    win_length = n_fft,
                                                    hop_length = hop_length,
                                                    n_mels = n_mels,
                                                    normalized = stft_normalized)
            
        # loss averaging through time
        self.train_recon_loss = MeanMetric()
        self.val_recon_loss = MeanMetric()
        self.test_recon_loss = MeanMetric()
        
        # for tracking best so far validation accuracy
        self.val_recon_loss_best = MinMetric()
        
    def spectral_normalize(self, magnitudes, C=1, clip_val=1e-7):
        return torch.log(torch.clamp(magnitudes, min=clip_val) * C)

    def spectral_de_normalize(self, magnitudes, C=1):
        return torch.exp(magnitudes) / C

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        pass

    def compute_dis_loss(self, orig_x, recon_x):

        grad_penalty = None

        # discriminator
        real, fake = orig_x.clone(), recon_x.detach()
        real_logits, fake_logits = map(self.discriminator, (real.requires_grad_(), fake))
        discr_loss = hinge_discr_loss(fake_logits, real_logits)

        if self.apply_grad_penalty:
            grad_penalty = gradient_penalty(real, discr_loss)
        
        return discr_loss

#         return [('discr', discr_loss), ('grad penalty', grad_penalty)]

    def compute_gen_loss(self, x_orig, x_recon):

        ## 1. recon loss: reconstruct from auto encoder
        recon_loss = self.recon_loss_fn(x_orig, x_recon)

        ## 2. adversarial loss
        adversarial_losses = []
        discr_intermediates = []

        real, fake = x_orig, x_recon

        ### features from stft
        (real_logits, real_intermediates), (fake_logits, fake_intermediates) =\
            map(partial(self.discriminator, return_intermediates=True), (real, fake))
        discr_intermediates.append((real_intermediates, fake_intermediates))

        adversarial_losses.append(hinge_gen_loss(fake_logits))
        adversarial_loss = torch.stack(adversarial_losses).mean()

        ## 3. feature_losses
        feature_losses = []
        for real_intermediates, fake_intermediates in discr_intermediates:
            losses = [F.l1_loss(real_intermediate, fake_intermediate) 
                        for real_intermediate, fake_intermediate 
                        in zip(real_intermediates, fake_intermediates)]
            
            feature_losses.extend(losses)
        feature_loss = torch.stack(feature_losses).mean()
        
        return recon_loss, adversarial_loss, feature_loss

    def training_step(self, batch: Any, batch_idx: int, optimizer_idx):
        
        # access data
        x = batch['audio']

        # compute mel spectrogram]
        x_mel = self.melspec_transform(x).unsqueeze(1)
        x_mel = self.spectral_normalize(x_mel)
        
        x_recon, var_info = self.autoencoder(x_mel, self.bottleneck)

        # hack: half for generator, half for discriminator
        if optimizer_idx == 0:
            ## vae loss
            vae_loss = var_info['variational_kl_loss']

            ## the rest loss
            recon_loss, adversarial_loss, feature_loss =\
                  self.compute_gen_loss(x_mel, x_recon)

            # total loss
            adv_loss_weight = self.adversarial_loss_weight if self.global_step >= self.warmup_steps else 0
            feature_loss_weight = self.feature_loss_weight if self.global_step >= self.warmup_steps else 0
            total_gen_loss = recon_loss * self.recon_loss_weight +\
                adversarial_loss * adv_loss_weight + self.vae_loss_weight * vae_loss +\
            feature_loss * feature_loss_weight
            
            self.log("gen loss", total_gen_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("recon loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True)
            
            return total_gen_loss
            
        if optimizer_idx == 1:
            
            real, fake = x_mel, x_recon.detach()
            discr_loss = self.compute_dis_loss(real, fake)
            
            return discr_loss


    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass
    
    def validation_step(self, batch: Any, batch_idx: int):

        x = batch['audio']
        x_mel = self.melspec_transform(x).unsqueeze(1)
        x_mel = self.spectral_normalize(x_mel)
        x_recon, var_info = self.autoencoder(x_mel, self.bottleneck)
        val_recon_loss = self.recon_loss_fn(x_mel, x_recon)

        # update and log metrics
        self.val_recon_loss(val_recon_loss)
        self.log("val/loss", self.val_recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        mel_save_dir = os.path.join(self.logger.save_dir, 'val_audio_mel')
        os.makedirs(mel_save_dir, exist_ok=True)
        mel_path = os.path.join(mel_save_dir, 'val_' + str(self.global_step) + '.png')

        if batch_idx == 0:
            with torch.no_grad():
                fig, ax = plt.subplots(2, 1, figsize=(8, 6))
                ax[0].imshow(x_mel[0][0].cpu(), aspect='auto', origin='lower', interpolation='none')
                ax[1].imshow(x_recon[0][0].cpu(), aspect='auto', origin='lower', 
                             interpolation='none')
                
                plt.savefig(mel_path)
                plt.close()
                
            
    def validation_epoch_end(self, outputs: List[Any]):

        pass

        # self.val_loss_best(self.val_loss.compute())  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        
        loss, _ = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if batch_idx == 0:
            x = batch['audio']
            x_recon = self.autoencoder(x, vq_bottleneck=self.bottleneck, return_recons_only=True)
            test_sample_folder = os.path.join(self.logger.save_dir, 'test_samples')

            os.makedirs(test_sample_folder, exist_ok=True)
            for j in range(x_recon.shape[0]):
                audio_sample = x_recon[j].cpu()
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

    def multiscale_discriminator_iter(self):
        for ind, discr in enumerate(self.discriminators):
            yield f'multiscale_discr_optimizer_{ind}', discr

    def configure_optimizers(self): # TODO
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizers_list = []
        optimizer_gener = self.optimizer_gener(params=self.autoencoder.parameters())
        optimizers_list.append(optimizer_gener)
        optimizer_discr = self.optimizer_discr(params=self.discriminator.parameters())
        optimizers_list.append(optimizer_discr)


        # if self.scheduler is not None:
        #     scheduler = self.scheduler(optimizer=optimizer_gener)
        #     return {
        #         "optimizer": optimizer_gener,
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "monitor": "val/loss",
        #             "interval": "epoch",
        #             "frequency": 1,
        #         },
        #     }
        return optimizers_list

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "diffwave.yaml")
    _ = hydra.utils.instantiate(cfg)
