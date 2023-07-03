from typing import Any, Dict, Optional, Tuple

import torch
import os, glob, random
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio

class PercussionDataset(Dataset):
    """
    {'Cymbals': 0, 'Floor Tom': 1, 'Floor Tom SNoff': 2, 
    'Kick': 3, 'Kick SNoff': 4, 'Rack Tom': 5, 
    'Rack Tom SNoff': 6, 'Snare': 7, 'Snare SNoff': 8}
     """
    # speech with diverse length, could use mask to adjust
    
    def __init__(self, path, max_audio_length):
        super().__init__()
        self.filenames = glob.glob(f'{path}/**/*.wav', recursive=True)
        self.label_to_idx = self.get_label()
        self.max_audio_length = max_audio_length
            
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
        signal, _ = torchaudio.load(audio_filename)

        class_label = audio_filename.split('/')[5]
        return {'audio': signal[0][:self.max_audio_length], 'label': self.label_to_idx[class_label]}
    
class Collator:
    """ 
    Zero-pads model inputs and targets based on number of samples per step
    """
    def __init__(self, audio_len):
        self.audio_len = audio_len
        
    def collate(self, minibatch):

        """Collate's training batch from linear-spectrogram
        PARAMS
        ------
        batch: [text_normalized, spec_normalized]
        """
        # Right zero-pad linear-spec
        wf_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor([x['audio'].shape[-1] for x in minibatch]), dim=0, descending=True)
        max_target_len = self.audio_len
        
        # include linear padded & sid
        wf_padded = torch.FloatTensor(len(minibatch), max_target_len)
        wf_padded.zero_()
        output_lengths = torch.LongTensor(len(minibatch))
        
        # collect labels
        class_labels = []
        for i in range(len(ids_sorted_decreasing)):
            record = minibatch[ids_sorted_decreasing[i]]
            wf = record['audio']
            wf_padded[i, :wf.shape[0]] = wf
            output_lengths[i] = wf.shape[0]
            class_labels.append(record['label'])
            
        class_labels = torch.from_numpy(np.array(class_labels))

        return {'audio': wf_padded, 'label': class_labels, 'lengths': output_lengths}

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
        audio_len: int = 16000,
        train_val_split: Tuple[float, float] = (0.92, 0.04),
        num_class: int = 9,
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
        percussion_dataset = PercussionDataset(self.hparams.data_dir, self.hparams.audio_len)
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
