import pandas as pd
import re

def load_data(file_path):
    # Load the CSV data
    df = pd.read_csv(file_path)

    # Display DataFrame columns and a sample of data
    print(f"DataFrame columns: {df.columns}")
    print("Sample data:")
    print(df.head())

    # Clean DataFrame: Drop empty columns and rows
    df.dropna(axis=0, how='all', inplace=True)  # Drop rows that are completely NaN
    df.dropna(axis=1, how='all', inplace=True)  # Drop columns that are completely NaN
    df.reset_index(drop=True, inplace=True)     # Reset index after dropping

    return df

def label_data(df):
    labeled_data = []

    # Counter to keep track of how many messages are labeled
    labeled_message_count = 0
    max_messages_to_label = 50  # Set a limit for the number of messages to label

    for index, row in df.iterrows():
        # Access the 'Cleaned Content' column
        message = row.get('Cleaned Content')  # Changed from 'Message' to 'Cleaned Content'

        # Check for NaN values in message and handle them
        if pd.isna(message):
            print(f"Skipping index {index} due to NaN message.")
            continue  # Skip this iteration if message is NaN

        # Tokenization and labeling
        tokens = message.split()  # Split message into tokens
        labeled_tokens = []

        for token in tokens:
            # Default label is O (Outside)
            label = "O"
            
            # Labeling logic for entities
            # Example products
            if re.search(r'(?=.*\b(የተለያዩ|ማህደር|ከርስ|ክበብ|ቀይበ|ማዕደ|ማዕበር)\b).+', token):
                if labeled_tokens and labeled_tokens[-1][1] == "B-Product":
                    label = "I-Product"  # Continue product entity
                else:
                    label = "B-Product"  # Start product entity

            # Example locations
            elif token in ["አዲስ አበባ", "ቦሌ", "ጎደኛ", "ማህሌ", "ሳቢ"]:  # Example locations
                if labeled_tokens and labeled_tokens[-1][1] == "B-LOC":
                    label = "I-LOC"  # Continue location entity
                else:
                    label = "B-LOC"  # Start location entity
            
            # Example prices
            elif re.match(r'^(ዋጋ\s*\d+|\d+\s*ብር)', token):  # Matches patterns like "ዋጋ 1000" or "100 ብር"
                if labeled_tokens and labeled_tokens[-1][1] == "B-PRICE":
                    label = "I-PRICE"  # Continue price entity
                else:
                    label = "B-PRICE"  # Start price entity
            
            # Append the token and its label
            labeled_tokens.append((token, label))

        # Add labeled tokens to the final labeled data
        labeled_data.append(labeled_tokens)

        # Increment the counter and break if the limit is reached
        labeled_message_count += 1
        if labeled_message_count >= max_messages_to_label:
            break

    return labeled_data

def save_labeled_data(labeled_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for tokens in labeled_data:
            for token, label in tokens:
                f.write(f"{token} {label}\n")
            f.write("\n")  # Newline to separate sentences

def main():
    # Specify the path to the input and output files
    input_file_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\EthioMart\EthioMart\data\raw\labeled_messages.csv'
    output_file_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\EthioMart\EthioMart\data\raw\labeled_messages.conll'

    # Load data
    df = load_data(input_file_path)

    # Check if the 'Cleaned Content' column exists
    if 'Cleaned Content' not in df.columns:
        raise ValueError("The 'Cleaned Content' column does not exist in the dataset. Please check the column name.")

    # Label the data
    labeled_data = label_data(df)

    # Save labeled data
    save_labeled_data(labeled_data, output_file_path)

    print(f"Labeled messages saved to {output_file_path}")

if __name__ == '__main__':
    main()
