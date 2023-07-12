from typing import Any, List
import os
from functools import partial
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as TAT
from torch.autograd import grad as torch_grad
from pytorch_lightning import LightningModule
from ema_pytorch import EMA
from torch.linalg import vector_norm
import torch.nn.functional as F
from einops import rearrange
from torchmetrics import MeanMetric, MinMetric
from itertools import zip_longest
from .module_utils import default, cast_tuple, exists

# helper functions
def hinge_gen_loss(fake):
    return -fake.mean()

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def log(t, eps=1e-8):
    return torch.log(t.clamp(min=eps))

def gradient_penalty(wave, output, weight = 10):

    gradients = torch_grad(
        outputs = output,
        inputs = wave,
        grad_outputs = torch.ones_like(output),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((vector_norm(gradients, dim = 1) - 1) ** 2).mean()

class VQGANModule(LightningModule):
    """ 
    VQ-VAE with GAN. Don't use precision 16 for training.
    """

    def __init__(
        self,
        autoencoder: torch.nn.Module,
        vqdiscriminator: torch.nn.Module,
        vqstftdiscriminator: torch.nn.Module,
        optimizer_gener: torch.optim.Optimizer,
        optimizer_discr: torch.optim.Optimizer,
        discr_multi_scales: list,
        accum_grads: int,
        apply_grad_penalty: bool,
        scheduler: torch.optim.lr_scheduler,
        bottleneck: torch.nn.Module,
        sample_rate: int,
        use_ema: bool,
        ema_beta: float,
        ema_power: float,
        warmup_steps: int = 10_000,
        max_grad_norm = 0.5,
        discr_max_grad_norm = None,
        recon_loss_weight: float = 1.,
        multi_spectral_recon_loss_weight: float = 1.,
        adversarial_loss_weight: float = 1.,
        feature_loss_weight: float = 100,
        multi_spectral_window_powers_of_two = tuple(range(6, 12)),
        multi_spectral_n_ffts = 512,
        multi_spectral_n_mels = 64,
        stft_normalized=True,
    ):
        super().__init__()

        # for VQGAN training, there will be multiple optimizers
        self.automatic_optimization = False
        self.accum_grads = accum_grads
        self.optimizer_gener = optimizer_gener
        self.optimizer_discr = optimizer_discr
        self.scheduler = scheduler
        self.apply_grad_penalty = apply_grad_penalty
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.discr_max_grad_norm = discr_max_grad_norm
        
        # GAN components
        ## stft discriminator
        self.vqstftdiscriminator = vqstftdiscriminator
        ## multi-scale discriminators
        self.discr_multi_scales = discr_multi_scales
        self.discriminators = nn.ModuleList([vqdiscriminator for _ in range(len(discr_multi_scales))])
        discr_rel_factors = [int(s1 / s2) for s1, s2 in zip(discr_multi_scales[:-1], discr_multi_scales[1:])]
        self.downsamples = nn.ModuleList([nn.Identity()] + [nn.AvgPool1d(2 * factor, stride=factor, padding=factor) for factor in discr_rel_factors])
        
        ## autoencoder and vqbottleneck
        self.autoencoder = autoencoder
        self.bottleneck = bottleneck

        self.use_ema = use_ema
        if self.use_ema:
            self.autoencoder_ema = EMA(self.autoencoder, beta=ema_beta, power=ema_power)

        self.recon_loss_weight = recon_loss_weight
        self.multi_spectral_recon_loss_weight = multi_spectral_recon_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.register_buffer('zero', torch.tensor([0.]), persistent=False)
        
        # multi spectral reconstruction
        self.mel_spec_transforms = nn.ModuleList([])
        self.mel_spec_recon_alphas = []

        num_transforms = len(multi_spectral_window_powers_of_two)
        multi_spectral_n_ffts = cast_tuple(multi_spectral_n_ffts, num_transforms)
        multi_spectral_n_mels = cast_tuple(multi_spectral_n_mels, num_transforms)

        for powers, n_fft, n_mels in zip_longest(multi_spectral_window_powers_of_two, multi_spectral_n_ffts, multi_spectral_n_mels):
            win_length = 2 ** powers
            alpha = (win_length / 2) ** 0.5

            calculated_n_fft = default(max(n_fft, win_length), win_length)

            melspec_transform = TAT.MelSpectrogram(
                sample_rate = sample_rate,
                n_fft = calculated_n_fft,
                win_length = win_length,
                hop_length = win_length // 4,
                n_mels = n_mels,
                normalized = stft_normalized
            )

            self.mel_spec_transforms.append(melspec_transform)
            self.mel_spec_recon_alphas.append(alpha)
            
        # generation
        self.sample_rate = sample_rate
            
        # loss averaging through time
        self.train_recon_loss = MeanMetric()
        self.val_recon_loss = MeanMetric()
        self.test_recon_loss = MeanMetric()
        
        # for tracking best so far validation accuracy
        self.val_recon_loss_best = MinMetric()

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        pass

    def compute_dis_loss(self, orig_x, recon_x):

        stft_discr_loss = None
        stft_grad_penalty = None
        discr_losses = []
        discr_grad_penalties = []

        # STFT discriminator
        real, fake = orig_x.clone(), recon_x.detach()
        stft_real_logits, stft_fake_logits = map(self.vqstftdiscriminator, 
                                                 (real.requires_grad_(), fake))
        stft_discr_loss = hinge_discr_loss(stft_fake_logits, stft_real_logits)

        if self.apply_grad_penalty:
            stft_grad_penalty = gradient_penalty(real, stft_discr_loss)

        # multi-scale disctiminator
        scaled_real, scaled_fake = real, fake
        for discr, downsample in zip(self.discriminators, self.downsamples):
            scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))

            real_logits, fake_logits = map(discr, (scaled_real.requires_grad_(), scaled_fake))
            one_discr_loss = hinge_discr_loss(fake_logits, real_logits)

            discr_losses.append(one_discr_loss)
            if self.apply_grad_penalty:
                discr_grad_penalties.append(gradient_penalty(scaled_real, one_discr_loss))

        # return a list of discriminator losses with List[Tuple[str, Tensor]]
        discr_losses_pkg = []
        discr_losses_pkg.extend([(f'scale:{scale}', multi_scale_loss) for scale, multi_scale_loss in zip(self.discr_multi_scales, discr_losses)])
        discr_losses_pkg.extend([(f'scale_grad_penalty:{scale}', discr_grad_penalty) for scale, discr_grad_penalty in zip(self.discr_multi_scales, discr_grad_penalties)])

        if exists(stft_discr_loss):
            discr_losses_pkg.append(('stft', stft_discr_loss))

        if exists(stft_grad_penalty):
            discr_losses_pkg.append(('stft_grad_penalty', stft_grad_penalty))

        all_discr_losses = torch.stack(discr_losses).mean()
        if exists(stft_discr_loss):
            all_discr_losses = all_discr_losses + stft_discr_loss

        return discr_losses_pkg, all_discr_losses

    def compute_gen_loss(self, x_orig, x_recon):
        ## 1. recon loss: reconstruct from auto encoder
        recon_loss = F.mse_loss(x_orig, x_recon)

        ## 2. multispectral recon loss - eq (4) and (5) in https://arxiv.org/abs/2107.03312
        multi_spectral_recon_loss = self.zero
        if self.multi_spectral_recon_loss_weight > 0:
            for mel_transform, alpha in zip(self.mel_spec_transforms, self.mel_spec_recon_alphas):
                orig_mel, recon_mel = map(mel_transform, (x_orig, x_recon))

                l1_mel_loss = (orig_mel - recon_mel).abs().sum(dim=-2).mean()
                
#                 multi_spectral_recon_loss = multi_spectral_recon_loss + l1_mel_loss

                # temp ignore l2 term
                log_orig_mel, log_recon_mel = map(log, (orig_mel, recon_mel))
                l2_log_mel_loss = alpha * vector_norm(log_orig_mel - log_recon_mel, dim=-2).mean()
                multi_spectral_recon_loss = multi_spectral_recon_loss + l1_mel_loss + l2_log_mel_loss

        ## 3. adversarial loss for multi-scale discriminators
        adversarial_losses = []
        discr_intermediates = []

        real, fake = x_orig, x_recon

        ### features from stft
        (stft_real_logits, stft_real_intermediates), (stft_fake_logits, stft_fake_intermediates) =\
            map(partial(self.vqstftdiscriminator, return_intermediates=True), (real, fake))
        discr_intermediates.append((stft_real_intermediates, stft_fake_intermediates))

        scaled_real, scaled_fake = real, fake
        for discr, downsample in zip(self.discriminators, self.downsamples):
            scaled_real, scaled_fake = map(downsample, (scaled_real, scaled_fake))

            (real_logits, real_intermediates), (fake_logits, fake_intermediates) =\
                    map(partial(discr, return_intermediates = True), (scaled_real, scaled_fake))

            discr_intermediates.append((real_intermediates, fake_intermediates))

            one_adversarial_loss = hinge_gen_loss(fake_logits)
            adversarial_losses.append(one_adversarial_loss)

        adversarial_losses.append(hinge_gen_loss(stft_fake_logits))
        adversarial_loss = torch.stack(adversarial_losses).mean()

        ## 3. feature_losses
        feature_losses = []

        for real_intermediates, fake_intermediates in discr_intermediates:
            losses = [F.l1_loss(real_intermediate, fake_intermediate) 
                        for real_intermediate, fake_intermediate 
                        in zip(real_intermediates, fake_intermediates)]
            
            feature_losses.extend(losses)
        feature_loss = torch.stack(feature_losses).mean()
        
        return recon_loss, multi_spectral_recon_loss, adversarial_loss, feature_loss

    def model_step(self, batch: Any, batch_idx: int, model_type: str):

        x_orig, x_recon, commit_loss = self.autoencoder(batch, self.bottleneck)

        if model_type == 'generator':
            # update vae (generator)
            opt_gener = self.optimizers()[0]
            
            ## 1. sum commitment loss
            all_commitment_loss = commit_loss.sum()

            ## 2-4 the rest loss
            recon_loss, multi_spectral_recon_loss, adversarial_loss, feature_loss =\
                  self.compute_gen_loss(x_orig, x_recon)

            # total loss
            adv_loss_weight = self.adversarial_loss_weight if self.global_step >= self.warmup_steps else 0
            total_gen_loss = recon_loss * self.recon_loss_weight +\
                  multi_spectral_recon_loss * self.multi_spectral_recon_loss_weight +\
                      adversarial_loss * adv_loss_weight +\
                        feature_loss * self.feature_loss_weight + all_commitment_loss

            self.manual_backward(total_gen_loss / self.accum_grads)
            
            self.log("gen loss", total_gen_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("recon loss", self.train_recon_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("multispec recon loss", multi_spectral_recon_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("adv loss", adversarial_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("feature loss", feature_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("commitment_loss", all_commitment_loss, on_step=True, on_epoch=True, prog_bar=True)

            if exists(self.max_grad_norm):
                self.clip_gradients(opt_gener, 
                                    gradient_clip_val=self.max_grad_norm, 
                                    gradient_clip_algorithm="norm")
            
            if (batch_idx + 1) % self.accum_grads == 0:
                opt_gener.step()
                opt_gener.zero_grad()

            return recon_loss

        elif model_type == 'discriminator':

            real, fake = x_orig, x_recon.detach()

            # update discriminator
            opt_discr = self.optimizers()[1]
            opt_multi_discr = self.optimizers()[2:]
            
            discr_losses, all_discr_losses = self.compute_dis_loss(real, fake)

            for name, discr_loss in discr_losses:
                self.manual_backward(discr_loss/self.accum_grads, retain_graph=True)
                self.log(name + "_loss", discr_loss, on_step=True, on_epoch=True, prog_bar=True)

            if exists(self.discr_max_grad_norm):
                self.clip_gradients(opt_discr, gradient_clip_val=self.discr_max_grad_norm)

            # gradient step for all discriminators
            if (batch_idx + 1) % self.accum_grads == 0:
                opt_discr.step()
                opt_discr.zero_grad()
                for multiscale_discr_optim in opt_multi_discr:
                    multiscale_discr_optim.step()
                    multiscale_discr_optim.zero_grad()

    def training_step(self, batch: Any, batch_idx: int):
        
        # access data
        x = batch['audio']
        
        train_recon_loss = self.model_step(x, batch_idx, model_type='generator')
        self.train_recon_loss(train_recon_loss)

        self.model_step(x, batch_idx, model_type='discriminator')

#         # hack: half for generator, half for discriminator
#         idx_shuffle = torch.randperm(x.shape[0])
#         train_recon_loss = self.model_step(x[idx_shuffle[:x.shape[0]//2]], 
#                                            batch_idx, model_type='generator')
#         self.train_recon_loss(train_recon_loss)

#         self.model_step(x[idx_shuffle[x.shape[0]//2]:], batch_idx, model_type='discriminator')

    def on_train_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):

        x = batch['audio']
        x_orig, x_recon, _ = self.autoencoder(x, self.bottleneck)
        val_recon_loss = F.mse_loss(x_orig, x_recon)

        # update and log metrics
        self.val_recon_loss(val_recon_loss)
        self.log("val/loss", self.val_recon_loss, on_step=False, on_epoch=True, prog_bar=True)

        audio_save_dir = os.path.join(self.logger.save_dir, 'val_audio')
        os.makedirs(audio_save_dir, exist_ok=True)
        audio_path = os.path.join(audio_save_dir, 'val_' + str(self.global_step) + '.wav')
        audio_gt_path = os.path.join(audio_save_dir, 'val_gt_' + str(self.global_step) + '.wav')
        if batch_idx == 0:
            with torch.no_grad():
                if x_orig.ndim == 2:
                    x_orig = rearrange(x_orig, 'b n -> b 1 n')
                    
                torchaudio.save(audio_gt_path, x_orig[0].cpu(), self.sample_rate)
                torchaudio.save(audio_path, x_recon[0].cpu(), self.sample_rate)
            
    def on_validation_epoch_end(self):

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
            for j in range(x.shape[0]):
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
        optimizer_discr = self.optimizer_discr(params=self.vqstftdiscriminator.parameters())
        optimizers_list.append(optimizer_discr)

        # multi-scale optimizer
        for discr_optimizer_key, discr in self.multiscale_discriminator_iter():
            one_multiscale_discr_optimizer = self.optimizer_discr(discr.parameters())
            optimizers_list.append(one_multiscale_discr_optimizer)

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
