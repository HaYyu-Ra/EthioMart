import os
import csv
import re

# File path for the existing messages CSV
RAW_CSV_FILE_PATH = r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\EthioMart\EthioMart\data\raw\messages.csv"
# File path for saving cleaned messages
CLEANED_CSV_FILE_PATH = r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\EthioMart\EthioMart\data\raw\messages\cleaned_messages.csv"

# Create directories if they do not exist
os.makedirs(os.path.dirname(CLEANED_CSV_FILE_PATH), exist_ok=True)

# Function to clean the message text
def clean_text(text):
    # Remove English letters and words, emojis, and non-Amharic characters
    text = re.sub(r'[A-Za-z0-9]+', '', text)  # Remove English letters and digits
    text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)  # Remove emojis (smileys)
    text = re.sub(r'[\U0001F300-\U0001F5FF]', '', text)  # Remove other emojis
    text = re.sub(r'[^፡-፞ዐ-ዚ]+', '', text)  # Keep only Amharic characters
    return text.strip()

# Function to process the CSV file
def process_messages():
    try:
        with open(RAW_CSV_FILE_PATH, mode='r', newline='', encoding='utf-8') as raw_file:
            reader = csv.DictReader(raw_file)
            
            # Open the cleaned CSV file to save cleaned messages
            with open(CLEANED_CSV_FILE_PATH, mode='w', newline='', encoding='utf-8') as cleaned_file:
                fieldnames = ["Channel Title", "Channel Username", "Channel ID", "Message ID", "Date", "Sender", "Cleaned Content"]
                writer = csv.DictWriter(cleaned_file, fieldnames=fieldnames)
                
                # Write CSV header
                writer.writeheader()
                
                for row in reader:
                    if row['Content']:
                        # Clean the message content
                        cleaned_content = clean_text(row['Content'])
                        # Write cleaned message details to the new CSV file
                        writer.writerow({
                            "Channel Title": row['Channel Title'],
                            "Channel Username": row['Channel Username'],
                            "Channel ID": row['Channel ID'],
                            "Message ID": row['Message ID'],
                            "Date": row['Date'],
                            "Sender": row['Sender'],
                            "Cleaned Content": cleaned_content
                        })
                        print(f"Message ID: {row['Message ID']} - Cleaned Content: {cleaned_content}")
                    else:
                        print("Received an empty message.")

            print(f"Cleaned messages saved to {CLEANED_CSV_FILE_PATH}")

    except FileNotFoundError:
        print(f"File not found: {RAW_CSV_FILE_PATH}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    process_messages()
