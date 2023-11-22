import telebot
from telebot import types
import model1
import model2
import json
import string
import nltk
import re
import secretdata

best_weights = [0.55, 0.6]
threshold = 0.5
bot_token = secretdata.token
bot = telebot.TeleBot(bot_token)
import re

def simple_sent_tokenize(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences

def text_to_json(text):
    sentences = simple_sent_tokenize(text)

    # Create a dictionary with IDs and sentences
    result = [dict.fromkeys(['text'], sentence) for i, sentence in enumerate(sentences)]

    return result

def save_to_json(data, filename='output.json'):
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, 'Приветствую! Отправьте мне файл в виде json или просто текста (тогда сравнение будет происходить среди предложений).')

@bot.message_handler(func=lambda message: True)
def process_input(message):
    # Check if the message has a document (JSON file)
    if message.document:
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        with open('input.json', 'wb') as new_file:
            new_file.write(downloaded_file)
        result = process_json('input.json')
        bot.reply_to(message, json.dumps(result, indent=4))
    else:
        # Process text separated by points
        text = message.text
        result = process_text(text)
        with open('result.json', encoding='utf-8') as fl:
            kk = fl.read()
            bot.reply_to(message, kk)
        with open('result.json', 'rb') as file:
            bot.send_document(message.chat.id, file)

result_dict = {}
def process_json(file_path: str) -> dict:
    path = file_path
    result1 = model1.get_result(path)
    result2 = model2.get_result(path)
    with open('output.json', 'r') as file:
        data = json.load(file)
    clean = data
    clean = [item['text'] for item in clean]
    data = [sentence['text'].lower() for sentence in data]
    data = [re.sub(r'[,\.\?\!\-]', '', sentence) for sentence in data]
    result_dict = {key : [] for key in clean}
    for i in range(len(result1)):
        final_score = best_weights[0] * result1[i][2] + best_weights[1] * result2[i][2]
        if final_score > threshold:
            result_dict[clean[result1[i][0]]].append(clean[result1[i][1]])
    return result_dict

def process_text(text: str) -> dict:
    json_data = text_to_json(text)

    # Save JSON to a file
    save_to_json(json_data)

    # Split text by points
    sentences = text.split('.')
    result1 = model1.get_result('output.json')
    result2 = model2.get_result('output.json')
    with open('output.json', 'r') as file:
        data = json.load(file)
    clean = data
    clean = [item['text'] for item in clean]
    clean = [re.sub(r'[,\.\?\!\-]', '', sentence) for sentence in clean]
    data = [sentence['text'].lower() for sentence in data]
    data = [re.sub(r'[,\.\?\!\-]', '', sentence) for sentence in data]
    result_dict = {key : [] for key in clean}
    for i in range(len(result1)):
        final_score = best_weights[0] * result1[i][2] + best_weights[1] * result2[i][2]
        if final_score > threshold:
            print(final_score)
            result_dict[clean[result1[i][0]]].append(clean[result1[i][1]])

    with open('result.json', 'w') as file:
        json.dump(result_dict, file, ensure_ascii=False, indent=4)
    return result_dict


# Start the bot
bot.polling()