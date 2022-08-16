import numpy as np

def remove_docstrings(batch):
    """
    Removes Docstrings from a given code file.
    Args:
        batch: A target data belonging to HuggingFace
        DatasetDict class.
    """
    for i in range(len(batch['func_code_string'])):
        batch['func_code_string'][i] = batch['func_code_string'][i].replace(batch['func_documentation_string'][i], '')
    return batch

def summarize(text: str, 
              tokenizer,
              trained_model):
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
