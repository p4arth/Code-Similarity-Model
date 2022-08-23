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
from summarizer import SummarizerModel
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import warnings
warnings.simplefilter('ignore')

MODEL_NAME = 'Salesforce/codet5-base-multi-sum'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SummarizerModel(MODEL_NAME)
model.load_state_dict(torch.load('codet5-base-1_epoch-val_loss-0.80.pth'))
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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

def find_similarity_score(code_1, code_2, model = embedding_model):
    summary_code_1 = summarize(text = code_1)
    summary_code_2 = summarize(text = code_2)
    embedding_1 = model.encode(summary_code_1)
    embedding_2 = model.encode(summary_code_2)
    score = np.dot(embedding_1, embedding_2)/(np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))
    return summary_code_1, summary_code_2, round(score, 2)

outputs = gr.outputs.Textbox()
iface = gr.Interface(fn=find_similarity_score, 
                     inputs=[gr.Textbox(label = 'First Code snippet'), 
                             gr.Textbox(label = 'Second Code snippet')], 
                     outputs=[gr.Textbox(label = 'Summary of first Code snippet'), 
                              gr.Textbox(label = 'Summary of second Code snippet'),
                              gr.Textbox(label = 'The similarity score')],
                     description='The similarity score')
iface.launch()