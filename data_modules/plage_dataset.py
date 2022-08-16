import torch
from torch.utils.data import Dataset, DataLoader
from datasets.dataset_dict import DatasetDict
from transformers import T5TokenizerFast

class PlageDataset(Dataset):
    def __init__(self, 
                 df: DatasetDict, 
                 tokenizer: T5TokenizerFast):
        self.inputs = df['func_code_string']
        self.target = df['func_documentation_string']
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        code = self.inputs[idx]
        summary = self.target[idx]

        code_tokens = self.tokenizer.encode_plus(
            code,
            add_special_tokens = True,
            padding = 'max_length',
            return_attention_mask = True,
            max_length = 512,
            truncation = True,
            return_tensors = 'pt'
        )

        summary_encoding = self.tokenizer.encode_plus(
            summary,
            add_special_tokens = True,
            return_attention_mask = True,
            padding = 'max_length',
            max_length = 128,
            truncation = True,
            return_tensors = 'pt'
        )
        labels = summary_encoding['input_ids']
        labels[labels == 0] == -100
        
        return dict(
            text = code,
            summary = summary,
            text_input_ids =  code_tokens['input_ids'].flatten(),
            text_attention_mask = code_tokens['attention_mask'].flatten(),
            labels = labels.flatten(),
            labels_attention_mask = summary_encoding['attention_mask'].flatten()
        )
