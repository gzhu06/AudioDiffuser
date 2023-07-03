from typing import Any, Dict, Optional, Tuple

import torch
import os, glob, random
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from torch.nn import functional as F

class PercussionDataset(Dataset):
    """
    {'Cymbals': 0, 'Floor Tom': 1, 'Floor Tom SNoff': 2, 
    'Kick': 3, 'Kick SNoff': 4, 'Rack Tom': 5, 
    'Rack Tom SNoff': 6, 'Snare': 7, 'Snare SNoff': 8}
     """
    # speech with diverse length, could use mask to adjust
    
    def __init__(self, path, 
                 n_fft, hop_length, 
                 num_frames):
        super().__init__()
        self.filenames = glob.glob(f'{path}/**/*.wav', recursive=True)
        self.label_to_idx = self.get_label()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.stft_args = dict(n_fft=n_fft, hop_length=hop_length, 
                              center=True, return_complex=True)
            
    def __len__(self):
        return len(self.filenames)
    
    def get_label(self):
        
         # create filename: label dict
        all_labels = []
        for filename in self.filenames:
            all_labels.append(filename.split('/')[5]) # coarse label
        
        # collect all labels and label_to_idx dict
        label_to_idx = {}
        label_set = list(set(all_labels))
        label_set.sort()
        for i, class_label in enumerate(label_set):
            label_to_idx[class_label] = i

        return label_to_idx
    
    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        x, _ = torchaudio.load(audio_filename)
        
        # padding
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            x = x[..., :target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (0, pad), mode='constant')

        # normalization
        normfac = x.abs().max()
        x = x / normfac
            
        # STFT
        window = torch.hann_window(self.n_fft, periodic=True).to(x.device)
        X = torch.stft(x, window=window, normalized=True, **self.stft_args)

        # get label
        class_label = audio_filename.split('/')[5]
        return {'audio': X, 'label': self.label_to_idx[class_label]}
    
class Collator:
    """ 
    Zero-pads model inputs and targets based on number of samples per step
    """
    def __init__(self):
        pass
        
    def collate(self, minibatch):
    
        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
        class_labels = [record['label'] for record in minibatch if 'label' in record]
        class_labels = np.array(class_labels)
        return {'audio': torch.from_numpy(audio), 'label': torch.from_numpy(class_labels)}

class PercussionDataModule(LightningDataModule):
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
        train_val_split: Tuple[float, float] = (0.92, 0.04),
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
        percussion_dataset = PercussionDataset(self.hparams.data_dir,
                                               n_fft=self.hparams.n_fft, 
                                               hop_length=self.hparams.hop_length, 
                                               num_frames=self.hparams.num_frames)
        
        train_len = int(self.hparams.train_val_split[0]*len(percussion_dataset))
        val_len = int(self.hparams.train_val_split[1]*len(percussion_dataset))
        test_len = len(percussion_dataset) - train_len - val_len
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train, self.data_val, self.data_test = random_split(
                    dataset=percussion_dataset,
                    lengths=[train_len, val_len, test_len],
                    generator=torch.Generator().manual_seed(42))

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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "medleydb.yaml")
    cfg.data_dir = str(root / "sc09")
    _ = hydra.utils.instantiate(cfg)
