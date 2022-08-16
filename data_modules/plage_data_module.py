import os
import torch
import pytorch_lightning as pl
from plage_dataset import PlageDataset
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast

class PlageDatasetModule(pl.LightningDataModule):
    def __init__(self,
                 train_data: DatasetDict, 
                 test_data: DatasetDict, 
                 tokenizer: T5TokenizerFast,
                 batch_size: int = 8):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
    
    def setup(self, stage = None):
        self.train_dataset = PlageDataset(
            self.train_data,
            self.tokenizer
        )
        self.test_dataset = PlageDataset(
            self.test_data,
            self.tokenizer
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = os.cpu_count()
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = os.cpu_count()
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = os.cpu_count()
        )
