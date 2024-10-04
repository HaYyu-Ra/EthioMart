import asyncio
import csv
import re
from telethon import TelegramClient
from telethon.errors import UsernameInvalidError, ChannelPrivateError, PeerFloodError

# Your API credentials
API_ID = '22719059'  # Your API ID
API_HASH = '2a3f5d1d5e677274fc404071bb6bf1bd'  # Your API Hash

# Define the channel username
CHANNEL_USERNAME = 'Fashiontera'  # Public username for the Fashiontera channel

# File paths
RAW_CSV_FILE_PATH = r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\EthioMart\EthioMart\data\raw\messages.csv"
CLEANED_CSV_FILE_PATH = r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\EthioMart\EthioMart\data\raw\cleaned_messages.csv"
LABELED_CONLL_FILE_PATH = r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\EthioMart\EthioMart\data\raw\labeled_messages.conll"

# Function to clean the message content
def clean_message(content):
    # Remove unnecessary spaces, symbols, etc. (adjust the cleaning steps as needed)
    content = re.sub(r'\s+', ' ', content).strip()  # Removing extra spaces
    content = re.sub(r'[^\w\s]', '', content)  # Removing punctuation
    return content

# Function to label entities (dummy labelling, modify based on your actual entity extraction logic)
def label_message(content):
    # In a real scenario, use an NER model here to label the entities
    words = content.split()
    labeled_data = []
    for word in words:
        if re.match(r'\b\d+\b', word):  # Example: Price (numbers)
            labeled_data.append(f"{word} B-PRICE")
        elif word.isupper():  # Example: Product name (all caps)
            labeled_data.append(f"{word} B-PRODUCT")
        else:
            labeled_data.append(f"{word} O")  # O for non-entity words
    return labeled_data

# Function to scrape messages from a channel
async def main():
    async with TelegramClient('session_name', API_ID, API_HASH) as client:
        try:
            print(f"Attempting to get the entity for username: {CHANNEL_USERNAME}")
            # Get the channel entity using the public username
            channel = await client.get_entity(CHANNEL_USERNAME)
            print(f"Scraping messages from: {CHANNEL_USERNAME}")
            print(f"Channel ID: {channel.id}")

            # Open the raw CSV file to save original messages
            with open(RAW_CSV_FILE_PATH, mode='w', newline='', encoding='utf-8') as raw_file:
                raw_writer = csv.writer(raw_file)
                raw_writer.writerow(["Channel Title", "Channel Username", "Channel ID", "Message ID", "Date", "Sender", "Content"])

                # Open the cleaned CSV file to save cleaned messages
                with open(CLEANED_CSV_FILE_PATH, mode='w', newline='', encoding='utf-8') as cleaned_file:
                    cleaned_writer = csv.writer(cleaned_file)
                    cleaned_writer.writerow(["Message ID", "Cleaned Content"])

                    # Open the CONLL file to save labeled messages
                    with open(LABELED_CONLL_FILE_PATH, mode='w', encoding='utf-8') as conll_file:

                        # Asynchronously iterate through all messages in the channel
                        async for message in client.iter_messages(channel):
                            if message and message.message:
                                # Write raw message to CSV
                                raw_writer.writerow([
                                    channel.title, channel.username, channel.id, message.id,
                                    message.date, message.sender_id, message.message
                                ])

                                # Clean the message content
                                cleaned_content = clean_message(message.message)
                                cleaned_writer.writerow([message.id, cleaned_content])

                                # Label the cleaned message for NER
                                labeled_message = label_message(cleaned_content)
                                
                                # Write labeled message in CoNLL format (word by word)
                                for labeled_word in labeled_message:
                                    word, label = labeled_word.split()
                                    conll_file.write(f"{word} {label}\n")
                                conll_file.write("\n")  # Blank line after each message
                                
                                print(f"Message ID: {message.id} - Cleaned: {cleaned_content}")
                            else:
                                print("Received an empty message.")

            print(f"Messages saved to {RAW_CSV_FILE_PATH}, cleaned messages saved to {CLEANED_CSV_FILE_PATH}, and labeled messages saved to {LABELED_CONLL_FILE_PATH}")

        except UsernameInvalidError:
            print("The specified username is invalid or does not exist.")
        except ChannelPrivateError:
            print("The channel is private, and you do not have access.")
        except PeerFloodError:
            print("You're being rate-limited. Try again later.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == '__main__':
    asyncio.run(main())
