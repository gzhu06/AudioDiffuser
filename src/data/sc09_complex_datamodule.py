from typing import Any, Dict, Optional, Tuple

import torch
import os, glob
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from torch.nn import functional as F

class SC09Dataset(Dataset):
    # speech with diverse length, could use mask to adjust
    
    def __init__(self, path, data_splits, 
                 n_fft, hop_length, 
                 num_frames, 
                 shuffle_spec=False):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.stft_args = dict(n_fft=n_fft, hop_length=hop_length, 
                              center=True, return_complex=True)
            
        self.shuffle_spec = shuffle_spec

        self.filenames = []
        for data_split in data_splits:
            self.filenames += glob.glob(f'{path}/{data_split}/*.wav', recursive=True)
        
        self.label_idx_dict = {'Zero': 0, 'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 
                               'Five': 5, 'Six': 6, 'Seven': 7, 'Eight': 8, 'Nine': 9}
            
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        x, _ = torchaudio.load(audio_filename)
        
        # padding
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len-target_len))
            else:
                start = int((current_len-target_len)/2)
            x = x[..., start:start+target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')
            
        # normalization
        normfac = x.abs().max()
        x = x / normfac
        
        # STFT
        window = torch.hann_window(self.n_fft, periodic=True).to(x.device)
        X = torch.stft(x, window=window, normalized=True, **self.stft_args)

        # label
        class_name = audio_filename.split('/')[-1].split('_')[0]
        return {'audio': X, 'label': self.label_idx_dict[class_name]}
    
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
    

class SC09DataModule(LightningDataModule):
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
        self.data_train = SC09Dataset(path=self.hparams.data_dir, 
                                          data_splits=['train', 'valid'], 
                                          n_fft=self.hparams.n_fft, 
                                          hop_length=self.hparams.hop_length, 
                                          num_frames=self.hparams.num_frames, 
                                          shuffle_spec=True)
        self.data_val = SC09Dataset(path=self.hparams.data_dir, 
                                        data_splits=['valid'],
                                        n_fft=self.hparams.n_fft, 
                                        hop_length=self.hparams.hop_length, 
                                        num_frames=self.hparams.num_frames, 
                                        shuffle_spec=False)
        self.data_test = SC09Dataset(path=self.hparams.data_dir, 
                                         data_splits=['test'],
                                         n_fft=self.hparams.n_fft,
                                         hop_length=self.hparams.hop_length, 
                                         num_frames=self.hparams.num_frames, 
                                         shuffle_spec=False)
        
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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "sc09_complex.yaml")
    cfg.data_dir = '/storageNVME/ge/sc09'
    dataset = hydra.utils.instantiate(cfg)
    dataset.setup()
    print(len(dataset.data_train))
    for data_item in (dataset.train_dataloader()):
        audio_compess = dataset.spec_fwd(data_item['audio'])
        isnan = torch.sum(torch.isnan(audio_compess.abs()))
        print(audio_compess.real.max(), audio_compess.imag.max())
