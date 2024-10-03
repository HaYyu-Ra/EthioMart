import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import spacy
from spacy.training import Example

# Load the labeled data
Labeled_CSV_PATH = r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\EthioMart\data\raw\labeled_data.csv"

def load_data(file_path):
    """Load labeled data from CSV."""
    return pd.read_csv(file_path)

def train_spacy_ner(train_data):
    """Train a Spacy NER model."""
    nlp = spacy.blank("am")  # Replace "am" with the language code for Amharic
    ner = nlp.add_pipe("ner", last=True)

    for _, annotations in train_data:
        for ent in annotations['entities']:
            ner.add_label(ent[2])  # Add the entity label to the model

    # Train the model
    nlp.begin_training()
    for itn in range(30):
        for text, annotations in train_data:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example])

    return nlp

def evaluate_model(nlp, test_data):
    """Evaluate the model."""
    y_true, y_pred = [], []
    for text, annotations in test_data:
        doc = nlp(text)
        predictions = [(ent.text, ent.label_) for ent in doc.ents]
        y_true.append(annotations['entities'])
        y_pred.append(predictions)

    # Compute classification report
    print(classification_report(y_true, y_pred))

def main():
    """Main function to load data, train and evaluate models."""
    df = load_data(Labeled_CSV_PATH)
    train_data = [(row['Content'], {'entities': []}) for index, row in df.iterrows()]  # Prepare training data
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Train the NER model
    nlp_model = train_spacy_ner(train_data)

    # Evaluate the model
    evaluate_model(nlp_model, test_data)

if __name__ == '__main__':
    main()
