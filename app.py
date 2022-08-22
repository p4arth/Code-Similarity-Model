import torch
import os
import numpy as np
import gradio as gr
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datasets.dataset_dict import DatasetDict
from transformers import AdamW, T5ForConditionalGeneration, T5TokenizerFast
from tqdm.auto import tqdm
import warnings
warnings.simplefilter('ignore')

from models.summarizer import SummarizerModel
from transformers import AutoTokenizer
MODEL_NAME = 'Salesforce/codet5-base-multi-sum'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SummarizerModel(MODEL_NAME)

def summarize(text: str, 
              tokenizer = tokenizer,
              trained_model = model):
    """
    Summarizes a given code in text format.
    Args:
        text: The code in string format that needs to be summarized.
        tokenizer: The tokeniszer used in the trained T5 model.
        trained_model: A SummarizerModel fine-tuned instance of 
        T5 model family.
    """
    text_encoding = tokenizer.encode_plus(
            text,
            padding = 'max_length',
            max_length = 512,
            add_special_tokens = True,
            return_attention_mask = True,
            truncation = True,
            return_tensors = 'pt'
        )
    generated_ids = trained_model.model.generate(
        input_ids = text_encoding['input_ids'],
        attention_mask = text_encoding['attention_mask'],
        max_length = 150,
        num_beams = 2,
        repetition_penalty = 2.5,
        length_penalty = 1.0,
        early_stopping = True
    )
    preds = [tokenizer.decode(gen_id, skip_special_tokens = True,
                              clean_up_tokenization_spaces=True)
                                for gen_id in generated_ids]
    return "".join(preds)

outputs = gr.outputs.Textbox()
iface = gr.Interface(fn=summarize, 
                   inputs=['text'], 
                   outputs=outputs,
                   description="This is the summarization")
iface.launch()
