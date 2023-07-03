from typing import Any, Dict, Optional, Tuple

import torch
import os, glob
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from torch.nn import functional as F

class DcaseDevDataset(Dataset):
    
    
    def __init__(self, 
                 paths, 
                 n_fft, 
                 hop_length, 
                 num_frames, 
                 shuffle_spec=False):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.stft_args = dict(n_fft=n_fft, hop_length=hop_length, 
                              center=True, return_complex=True)
            
        self.shuffle_spec = shuffle_spec
        filenames = []
        for path in paths:
            filenames += glob.glob(f'{path}/**/*.wav', recursive=True)
        
        # magnitude sanity check
        self.filenames = []
        for filename in filenames:
            x, _ = torchaudio.load(filename)
            if x.abs().max().numpy() < 0.001:
                continue
            else:
                self.filenames.append(filename)
        
        self.label_idx_dict = {'DogBark': 0, 'Footstep': 1, 'GunShot': 2, 
                               'Keyboard': 3, 'MovingMotorVehicle': 4, 
                               'Rain': 5, 'Sneeze_Cough': 6}
            
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        x, _ = torchaudio.load(audio_filename)
        
        # padding
        target_len = (self.num_frames - 1) * self.hop_length # + n_fft?
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len-target_len)) # random starting point
            else:
                # start = int((current_len-target_len)/2) # start at half the difference
                start = 0
            x = x[..., start:start+target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='reflect')
#             x = F.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')


        # STFT
        window = torch.hann_window(self.n_fft, periodic=True).to(x.device)
        X = torch.stft(x, window=window, normalized=True, **self.stft_args)

        # label
        class_name = audio_filename.split('/')[-2]
        return {'audio': X, 'label': self.label_idx_dict[class_name]}
    
class Collator:
    """ 
    Zero-pads model inputs and targets based on number of samples per step
    """
    def __init__(self):
        pass
        
    def collate(self, minibatch):
    
        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
#         wav_mag = np.stack([record['wav_mag'] for record in minibatch if 'wav_mag' in record])
        class_labels = [record['label'] for record in minibatch if 'label' in record]
        class_labels = np.array(class_labels)
        return {'audio': torch.from_numpy(audio), 
                'label': torch.from_numpy(class_labels)}
    

class DcaseDevDataModule(LightningDataModule):
    """A DataModule implements 5 key methods:
    
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "./",
        n_fft: int = 510, 
        hop_length: int = 128, 
        num_frames: int = 256, 
        shuffle_spec: bool = False,
        spec_factor: float = 0.15,
        spec_abs_exponent: float = 0.5,
        num_class: int = 8,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.stft_args = dict(n_fft=n_fft, 
                              hop_length=hop_length,
                              center=True)

    @property
    def num_classes(self):
        return self.hparams.num_class

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if self.data_train is None:
            dataset = DcaseDevDataset(paths=[self.hparams.data_dir], 
                                          n_fft=self.hparams.n_fft, 
                                          hop_length=self.hparams.hop_length, 
                                          num_frames=self.hparams.num_frames, 
                                          shuffle_spec=False)
            train_size = int(0.95 * len(dataset))
            val_size = int(0.05 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            self.data_train, self.data_val, self.data_test = random_split(dataset, [train_size, val_size, test_size])
        
    def spec_fwd(self, spec):
        '''
            # only do this calculation if spec_exponent != 1, 
            # otherwise it's quite a bit of wasted computation
            # and introduced numerical error
        '''
        if self.spec_abs_exponent != 1:
            e = self.spec_abs_exponent
            spec = spec.abs()**e * torch.exp(1j * spec.angle())
        spec = spec * self.spec_factor

        return spec
    
    def spec_back(self, spec):
        spec = spec / self.spec_factor
        if self.spec_abs_exponent != 1:
            e = self.spec_abs_exponent
            spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())

        return spec

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            collate_fn=Collator().collate,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            collate_fn=Collator().collate,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            collate_fn=Collator().collate,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils
    
    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "dcaseDev_complex.yaml")
    cfg.data_dir = '/storageNVME/ge/DCASEFoleySoundSynthesisDevSet'
    cfg.num_workers = 0
    cfg.batch_size = 1
    cfg.hop_length = 128
    cfg.num_frames = 768
    cfg.n_fft = 510
    dataset = hydra.utils.instantiate(cfg)
    dataset.setup()
    wav_mags = []
    for data_item in (dataset.train_dataloader()):
        
        wav_mag = data_item['wav_mag'].numpy()
        if wav_mag < 0.01:
            wav_mags.append(wav_mag)
#             print(wav_mag)
    print(len(wav_mags))

