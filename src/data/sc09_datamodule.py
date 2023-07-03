from typing import Any, Dict, Optional, Tuple

import torch
import os, glob, random
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from scipy.io import wavfile
import torchaudio

class SC09Dataset(Dataset):
    
    def __init__(self, paths):
        super().__init__()
        self.filenames = []
        for path in paths:
            self.filenames += glob.glob(f'{path}/**/*.wav', recursive=True)
        self.label_idx_dict = {'Zero': 0, 'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 
                               'Five': 5, 'Six': 6, 'Seven': 7, 'Eight': 8, 'Nine': 9}
            
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]

        signal, _ = torchaudio.load(audio_filename)
        class_name = audio_filename.split('/')[-1].split('_')[0]
        class_label = self.label_idx_dict[class_name]
        return {'audio': signal[0], 'label': class_label}
    
class Collator:
    def __init__(self, audio_len):
        self.audio_len = audio_len
        
    def collate(self, minibatch):
        class_labels = []
        for record in minibatch:
            if len(record['audio']) > self.audio_len:
                start = random.randint(0, record['audio'].shape[-1] - self.audio_len)
                end = start + self.audio_len
                record['audio'] = record['audio'][start:end]
            elif len(record['audio']) < self.audio_len:
                record['audio'] = np.pad(record['audio'], (0, self.audio_len-len(record['audio'])), mode='constant')
                
            class_labels.append(record['label'])
                
        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
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
        audio_len: int = 16000,
        num_class: int = 10,
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
        self.data_train = SC09Dataset([os.path.join(self.hparams.data_dir, 'train'), 
                                       os.path.join(self.hparams.data_dir, 'valid')])
        self.data_val = SC09Dataset([os.path.join(self.hparams.data_dir, 'valid')])
        self.data_test = SC09Dataset([os.path.join(self.hparams.data_dir, 'test')])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            collate_fn=Collator(self.hparams.audio_len).collate,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            collate_fn=Collator(self.hparams.audio_len).collate,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            collate_fn=Collator(self.hparams.audio_len).collate,
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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "sc09.yaml")
    cfg.data_dir = str(root / "sc09")
    _ = hydra.utils.instantiate(cfg)
