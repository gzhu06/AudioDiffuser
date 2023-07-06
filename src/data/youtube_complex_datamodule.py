from typing import Any, Dict, Optional, Tuple

import torch
import os, glob
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from scipy.io import wavfile
from torch.nn import functional as F

class YoutubeDataset(Dataset):
    # speech with diverse length, could use mask to adjust
    
    def __init__(self, paths, 
                 n_fft, hop_length, 
                 num_frames, sample_rate,
                 epoch_iters,
                 shuffle_spec=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.stft_args = dict(n_fft=n_fft, hop_length=hop_length, 
                              center=True, return_complex=True)
        self.shuffle_spec = shuffle_spec

        self.filenames = []
        for path in paths:
            self.filenames += glob.glob(f'{path}/**/*.wav', recursive=True)
            
        self.filenames *= epoch_iters
        
        self.label_idx_dict = {'Piano': 0, 'Andrew Langdon': 1, 'Bach': 2, 'Baroque': 3, 
                               'Chillstep': 4, 'City of Gamers': 5, 'Cyberpunk 2077': 6, 
                               'Dark Techno': 7, 'Drum and Bass': 8, 'Dubstep Mix': 9, 
                               'EDM': 10, 'French 79': 11, 'GHOSTS': 12, 'Hans Zimmer': 13, 
                               'Jazz': 14, 'Kiasmos': 15, 'Lofi beats': 16, 'Ludovico Einaudi': 17, 
                               'Ludwig Goransson': 18, 'Joachim Pastor Mix': 19, 'Metal': 20,
                               'Techno Mix': 21, 'Synthwave Mix': 22}
            
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        _, x = wavfile.read(audio_filename, mmap=True)
        
        target_len = (self.num_frames - 1) * self.hop_length # + n_fft?
        current_len = len(x)
        start = int(np.random.uniform(0, current_len-target_len)) # random starting point
        x = x[start:start+target_len]
        x = x / 32767.0
        x = torch.from_numpy(x).unsqueeze(0)
            
        # STFT
        window = torch.hann_window(self.n_fft, periodic=True).to(x.device)
        X = torch.stft(x.to(window.dtype), window=window, normalized=True, **self.stft_args)

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
        class_labels = [record['label'] for record in minibatch if 'label' in record]
        class_labels = np.array(class_labels)
        return {'audio': torch.from_numpy(audio), 'label': torch.from_numpy(class_labels)}
    

class YoutubeDataModule(LightningDataModule):
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
        sample_rate: int = 22050,
        n_fft: int = 510, 
        hop_length: int = 128, 
        num_frames: int = 256, 
        shuffle_spec: bool = False,
        spec_factor: float = 0.15,
        spec_abs_exponent: float = 0.5,
        num_class: int = 36,
        epoch_iters: int = 100,
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
        self.epoch_iters = epoch_iters
        self.sample_rate = sample_rate
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
            dataset = YoutubeDataset(paths=[self.hparams.data_dir], 
                                          n_fft=self.hparams.n_fft, 
                                          hop_length=self.hparams.hop_length, 
                                          num_frames=self.hparams.num_frames, 
                                          sample_rate=self.sample_rate,
                                          epoch_iters=self.epoch_iters,
                                          shuffle_spec=True)
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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "youtube_complex.yaml")
    cfg.data_dir = str("/storageNVME/yutong/youtube_dataset/youtube_dataset_mono_remov_silence/")
    dataset = hydra.utils.instantiate(cfg)
    dataset.setup()
    print(len(dataset.data_train))
    print(len(dataset.data_train[0]['audio']))
    train_loader = dataset.train_dataloader()
    for batch in train_loader:
        print(batch['audio'].shape)
        print(batch['label'].shape)
        break