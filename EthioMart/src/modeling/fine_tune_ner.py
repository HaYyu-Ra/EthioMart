# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset  # Import Hugging Face's datasets library

# Step 2: Install necessary libraries (Run in a separate cell in Google Colab or local environment)
# !pip install transformers datasets

# Step 3: Load the pre-trained model and tokenizer
model_name = 'xlm-roberta-base'  # You can also use 'bert-tiny-amharic' or 'afroxmlr'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 4: Load the labeled dataset in CoNLL format
def load_conll_data(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        label = []
        for line in f:
            if line.strip():  # If line is not empty
                word, tag = line.strip().split()
                sentence.append(word)
                label.append(tag)
            else:
                if sentence:  # End of a sentence
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []
        # Add the last sentence if exists
        if sentence:
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels

train_file_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\EthioMart\EthioMart\data\raw\labeled_messages.conll'
val_file_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\EthioMart\EthioMart\data\raw\labeled_messages_val.conll'

train_sentences, train_labels = load_conll_data(train_file_path)
val_sentences, val_labels = load_conll_data(val_file_path) if os.path.exists(val_file_path) else (None, None)

# Step 5: Create label mapping
unique_labels = set(tag for label in train_labels for tag in label)
label_to_id = {label: i for i, label in enumerate(unique_labels)}
num_labels = len(unique_labels)

# Step 6: Tokenize the data and align labels
def tokenize_and_align_labels(sentences, labels):
    tokenized_inputs = tokenizer(sentences, padding=True, truncation=True, is_split_into_words=True)  # Removed clean_up_tokenization_spaces
    aligned_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to original words
        label_ids = [-100] * len(tokenized_inputs['input_ids'][i])  # Default to -100 for ignored tokens
        for j in range(len(label)):
            if word_ids[j] is not None:  # Ensure word_ids is not None
                label_ids[word_ids[j]] = label_to_id[label[j]]  # Assign label id to the corresponding token
        aligned_labels.append(label_ids)
    return tokenized_inputs, aligned_labels

# Tokenize training and validation datasets
tokenized_train_inputs, aligned_train_labels = tokenize_and_align_labels(train_sentences, train_labels)
tokenized_val_inputs, aligned_val_labels = tokenize_and_align_labels(val_sentences, val_labels) if val_sentences else (None, None)

# Step 7: Create custom Dataset
class NERDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = NERDataset(tokenized_train_inputs, aligned_train_labels)
val_dataset = NERDataset(tokenized_val_inputs, aligned_val_labels) if val_sentences else None

# Step 8: Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch' if val_dataset is not None else 'no',  # Use 'no' if eval dataset is not available
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=10,       # Log every 10 steps
)

# Step 9: Initialize Trainer
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Pass eval_dataset only if it exists
)

# Step 10: Fine-tune the model
trainer.train()

# Step 11: Evaluate the fine-tuned model
if val_dataset:
    metrics = trainer.evaluate()
    print(f"Validation metrics: {metrics}")

# Step 12: Save the fine-tuned model for future use
model.save_pretrained('./fine_tuned_ner_model')
tokenizer.save_pretrained('./fine_tuned_ner_model')

print("Model fine-tuning complete. The model and tokenizer have been saved.")
