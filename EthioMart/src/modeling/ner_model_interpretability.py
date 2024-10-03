# Step 1: Import necessary libraries 
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
import shap
import lime
from lime.lime_text import LimeTextExplainer
import warnings
from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizer

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Step 2: Load the pre-trained models and tokenizers
models_to_evaluate = [
    'xlm-roberta-base',
    'distilbert-base-multilingual-cased',
    'bert-base-multilingual-cased'
]

# Step 3: Load the labeled dataset in CoNLL format
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

# Define the path for the training data
train_file_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\EthioMart\EthioMart\data\raw\labeled_messages.conll'

# Load training dataset
train_sentences, train_labels = load_conll_data(train_file_path)

# Step 4: Create label mapping
unique_labels = set(tag for label in train_labels for tag in label)
label_to_id = {label: i for i, label in enumerate(unique_labels)}
num_labels = len(unique_labels)

# Step 5: Tokenize the data and align labels
def tokenize_and_align_labels(sentences, labels, tokenizer):
    tokenized_inputs = tokenizer(sentences, padding=True, truncation=True, is_split_into_words=True)
    aligned_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100] * len(tokenized_inputs['input_ids'][i])  # Default to -100 for ignored tokens
        for j in range(len(label)):
            if word_ids[j] is not None:  # Ensure word_ids is not None
                label_ids[word_ids[j]] = label_to_id[label[j]]  # Assign label id to the corresponding token
        aligned_labels.append(label_ids)
    return tokenized_inputs, aligned_labels

# Step 6: Create custom Dataset class
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

# Step 7: Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=1,
)

# Directory to save fine-tuned models
model_save_dir = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\EthioMart\fine_tuned_ner_model'

# Initialize a dictionary to store metrics for each model
metrics_results = {}

for model_name in models_to_evaluate:
    print(f"Fine-tuning model: {model_name}")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

    # Tokenize the training dataset
    tokenized_train_inputs, aligned_train_labels = tokenize_and_align_labels(train_sentences, train_labels, tokenizer)

    # Split the training data into training and validation sets
    train_size = int(0.8 * len(tokenized_train_inputs['input_ids']))  # 80% for training
    val_size = len(tokenized_train_inputs['input_ids']) - train_size  # Remaining for validation
    train_encodings = {key: val[:train_size] for key, val in tokenized_train_inputs.items()}
    val_encodings = {key: val[train_size:] for key, val in tokenized_train_inputs.items()}
    train_labels_subset = aligned_train_labels[:train_size]
    val_labels_subset = aligned_train_labels[train_size:]

    # Create datasets
    train_dataset = NERDataset(train_encodings, train_labels_subset)
    val_dataset = NERDataset(val_encodings, val_labels_subset)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Step 11: Evaluate the fine-tuned model
    metrics = trainer.evaluate()
    print(f"Metrics for {model_name}: {metrics}")
    metrics_results[model_name] = metrics  # Store metrics for the model

    # Save the fine-tuned model
    model_save_path = os.path.join(model_save_dir, model_name.replace("/", "_"))
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print(f"Fine-tuned model saved to: {model_save_path}")

    # SHAP analysis
    def explain_with_shap(model, tokenizer, examples):
        # Tokenize examples for SHAP
        tokenized_examples = tokenizer(examples, padding=True, truncation=True, return_tensors="pt")
        
        # SHAP for transformer-based models requires a masker, use a simple text masker
        masker = shap.maskers.Text(tokenizer, padding="max_length")
        
        # Initialize the SHAP explainer with a masker
        explainer = shap.Explainer(model, masker=masker)
        
        # Calculate SHAP values
        shap_values = explainer(tokenized_examples['input_ids'])  # Use input_ids for SHAP analysis
        return shap_values

    # Example difficult cases for analysis
    difficult_cases_examples = ["The bank can refuse to lend money."]
    
    try:
        shap_values = explain_with_shap(model, tokenizer, difficult_cases_examples)
        print(f"SHAP values for difficult cases: {shap_values}")
    except Exception as e:
        print(f"Error during SHAP analysis: {e}")

    # LIME analysis
    def explain_with_lime(model, tokenizer, examples):
        # Initialize LIME text explainer
        explainer = LimeTextExplainer(class_names=list(label_to_id.keys()))
        
        # LIME expects text input directly, not tokenized inputs
        explanations = []
        for example in examples:
            explanation = explainer.explain_instance(
                example, 
                classifier_fn=lambda x: model(tokenizer(x, return_tensors="pt")['input_ids']).logits.detach().numpy(),
                num_features=10
            )
            explanations.append(explanation)
        return explanations

    # Perform LIME interpretation
    try:
        lime_explanations = explain_with_lime(model, tokenizer, difficult_cases_examples)
        print(f"LIME explanations for difficult cases: {lime_explanations}")
    except Exception as e:
        print(f"Error during LIME analysis: {e}")

# Step 12: Report generation and analysis
report_path = './interpretability_report.txt'
with open(report_path, 'w') as f:
    f.write("Model Interpretability Report\n")
    f.write("=============================\n")
    for model_name, metrics in metrics_results.items():
        f.write(f"Model: {model_name}\n")
        f.write(f"Metrics: {metrics}\n")
        f.write("\n")
    f.write("SHAP and LIME Analysis Completed\n")
    
print(f"Interpretability report generated at: {report_path}")
