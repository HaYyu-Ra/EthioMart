import pandas as pd
import re

# Load the CSV file (Extract)
data_path = r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\EthioMart\EthioMart\data\raw\messages.csv"
df = pd.read_csv(data_path)

# Drop empty columns
df.dropna(axis=1, how='all', inplace=True)

# Drop empty rows
df.dropna(axis=0, how='all', inplace=True)

# Define a function to clean the text
def clean_text(text):
    if isinstance(text, str):  # Ensure the text is a string
        # Remove English words, emojis, and any non-Amharic characters
        text = re.sub(r'[^\u1200-\u137F\s]', '', text)  # Keep only Amharic characters and spaces
        return text.strip()  # Remove leading and trailing whitespace
    return ''  # Return an empty string if the text is not a string (like NaN)

# Transform the 'Content' column by cleaning unwanted text
df['Content'] = df['Content'].apply(clean_text)

# Drop any rows that are now empty after cleaning
df.dropna(axis=0, how='all', inplace=True)

# Load the cleaned data to a new CSV file (Load)
cleaned_data_path = r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\EthioMart\EthioMart\data\raw\cleaned_messages.csv"
df.to_csv(cleaned_data_path, index=False)

print(f"Cleaned data saved to: {cleaned_data_path}")
