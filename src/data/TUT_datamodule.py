from typing import Any, Dict, Optional, Tuple

import torch
import os, glob, json
import numpy as np
import random
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio

class TUTDataset(Dataset):
    '''
    # ambient sound label to index: 
    {'beach': 0, 'bus': 1, 'cafe/restaurant': 2, 'car': 3, 'city_center': 4, 
     'forest_path': 5, 'grocery_store': 6, 'home': 7, 'library': 8, 
     'metro_station': 9, 'office': 10, 'park': 11, 'residential_area': 12, 
     'train': 13, 'tram': 14}
    '''
    
    def __init__(self, path, data_split):
        super().__init__()
        self.filenames = []
        self.filenames = glob.glob(f'{path}/{data_split}/*.wav')
        self.meta_json = os.path.join(path, f'meta_{data_split}.json')
        
        self.label_dict = self.get_label()
            
    def __len__(self):
        return len(self.filenames)
    
    def get_label(self):
        
        # load file label dict
        with open(self.meta_json, 'r') as f:
            file_meta_dict = json.load(f)
        
        # collect all labels and label_to_idx dict
        all_labels = list(file_meta_dict.values())
        label_to_idx = {}
        label_set = list(set(all_labels))
        label_set.sort()
        for i, class_label in enumerate(label_set):
            label_to_idx[class_label] = i
        
         # create filename: label dict
        label_dict = {}
        for filename in self.filenames:
            filename_in_dict = filename.split('/')[-1]
            file_label = file_meta_dict[filename_in_dict]
            label_dict[filename_in_dict] = label_to_idx[file_label]
            
        return label_dict
    
    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        signal, _ = torchaudio.load(audio_filename)
        class_label = self.label_dict[audio_filename.split('/')[-1]]
        
        return {'audio': signal[0], 'label': class_label}
    
class Collator:
    def __init__(self, audio_len):
        self.audio_len = audio_len
        
    def collate(self, minibatch):
    
        class_labels = []
        for record in minibatch:
            class_labels.append(record['label'])
            if len(record['audio']) > self.audio_len:
                start = random.randint(0, record['audio'].shape[-1] - self.audio_len)
                end = start + self.audio_len
                record['audio'] = record['audio'][start:end]
            else:
                print('error audio files!')
                exit()

        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
        class_labels = np.array(class_labels)
        
        return {'audio': torch.from_numpy(audio), 'label': torch.from_numpy(class_labels)}

class TUTDataModule(LightningDataModule):
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
        num_class: int = 8,
        audio_len: int = 16000,
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
        self.data_train = TUTDataset(self.hparams.data_dir, 'train')
        self.data_val = TUTDataset(self.hparams.data_dir, 'eval')
        self.data_test = TUTDataset(self.hparams.data_dir, 'eval')

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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "medleydb.yaml")
    cfg.data_dir = str(root / "sc09")
    _ = hydra.utils.instantiate(cfg)
