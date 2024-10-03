import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    XLMRobertaForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)

# Define your dataset class
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

# Function to load and preprocess data from the CoNLL format
def load_data(file_path):
    texts = []
    labels = []
    
    # Define a mapping for NER labels to numeric ids
    label_map = {'O': 0, 'B-PRODUCT': 1, 'I-PRODUCT': 2, 'B-PRICE': 3, 'I-PRICE': 4, 'B-LOCATION': 5, 'I-LOCATION': 6}
    
    # Load data from .conll file
    with open(file_path, 'r', encoding='utf-8') as f:
        tokens = []
        ner_tags = []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    texts.append(tokens)
                    labels.append([label_map[tag] for tag in ner_tags])
                    tokens = []
                    ner_tags = []
            else:
                token, tag = line.split()  # Assuming each line has a token and its NER tag
                tokens.append(token)
                ner_tags.append(tag)
    
    return texts, labels

def align_labels_with_tokens(texts, labels, encodings, tokenizer):
    aligned_labels = []
    
    for i, label in enumerate(labels):
        word_ids = encodings.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore this token
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # Use the label for the first token of the word
            else:
                label_ids.append(-100)  # Ignore subsequent tokens of the same word
            previous_word_idx = word_idx
        
        aligned_labels.append(label_ids)
    
    return aligned_labels

def main():
    # Use Google Colab or any other environment with GPU support for faster training
    # Install necessary libraries
    # !pip install transformers datasets

    # Load the labeled dataset in CoNLL format
    data_file_path = 'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/EthioMart/EthioMart/data/raw/labeled_messages.conll'
    texts, labels = load_data(data_file_path)

    # Debugging: Print the number of unique labels
    unique_labels = set(np.concatenate(labels))
    print(f"Unique labels in the dataset: {unique_labels}")
    print(f"Number of unique labels: {len(unique_labels)}")

    # Load pre-trained XLM-Roberta model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    model = XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-base', num_labels=len(unique_labels))

    # Tokenize data and align labels with tokens
    encodings = tokenizer(texts, truncation=True, padding=True, is_split_into_words=True)
    aligned_labels = align_labels_with_tokens(texts, labels, encodings, tokenizer)

    # Create dataset with tokenized inputs and aligned labels
    dataset = NERDataset(encodings, aligned_labels)

    # Split the dataset into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy='epoch',  # Updated deprecated `evaluation_strategy` to `eval_strategy`
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Save the model
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')

if __name__ == "__main__":
    main()
