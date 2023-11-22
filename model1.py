import json
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import string
import pymorphy2
import re

custombannedwords = ['!', ',', '.', '№', '-']
similoef = 1
path = "output.json"


def del_signs(sentence):
    return re.sub(r'[^\w\s]', '', sentence)



morph = pymorphy2.MorphAnalyzer()

def preprocess_sentence(sentence, word_set):
    sentence = remove_punctuation(sentence)
    sentence = replace_unknown_words(sentence, word_set)
    sentence = replace_pronouns(sentence)
    sentence = replace_prepositions(sentence)
    sentence = lemmatize_sentence(sentence)
    return sentence

def lemmatize_sentence(sentence):
    return ' '.join([morph.parse(word)[0].normal_form for word in sentence.split()])

def compare_sentences(words1, words2):
    lemmas1 = set(words1.split())
    lemmas2 = set(words2.split())

    common_words = lemmas1 & lemmas2
    similarity = len(common_words) / len(lemmas1.union(lemmas2))

    if ('не' in lemmas1 and 'не' in lemmas2) or ('не' not in lemmas1 and 'не' not in lemmas2):
        similarity *= 1  # Увеличиваем схожесть, если обе частицы "не" присутствуют
    else:
        similarity = similarity - 0.3

    parsed_words1 = [morph.parse(word)[0].normal_form for word in words1.split() if morph.parse(word)[0].tag.POS in ('VERB', 'INFN')]
    parsed_words2 = [morph.parse(word)[0].normal_form for word in words2.split() if morph.parse(word)[0].tag.POS in ('VERB', 'INFN')]

    if set(parsed_words2) != set(parsed_words1):
        similarity = similarity - 0.3
    else:
        similarity = similarity

    return similarity

def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))

def replace_pronouns(sentence):
    return ' '.join(['<PRONOUN>' if 'NPRO' in morph.parse(word)[0].tag else word for word in sentence.split()])

def replace_prepositions(sentence, replacement_token='PREPOSITION'):
    return ' '.join([replacement_token if 'PREP' in morph.parse(word)[0].tag else word for word in sentence.split()])

def levenshtein_distance(first_word, second_word):
    if len(first_word) < len(second_word):
        return levenshtein_distance(second_word, first_word)

    if len(second_word) == 0:
        return len(first_word)

    previous_row = list(range(len(second_word) + 1))

    for i, c1 in enumerate(first_word):
        current_row = [i + 1]

        for j, c2 in enumerate(second_word):
            # Calculate insertions, deletions and substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)

            # Get the minimum to append to the current row
            current_row.append(min(insertions, deletions, substitutions))

        # Store the previous row
        previous_row = current_row

    # Returns the last element (distance)
    return previous_row[-1]

def find_similar(s1, arr):
    for j in range(8):
        for word in arr:
            if levenshtein_distance(s1, word) <= j:
                return word

def replace_unknown_words(sentence, word_set):
    words = sentence.split()
    corrected_sentence = []

    for i, word in enumerate(words):
        parsed_word = morph.parse(word)[0]

        if "UnknownPrefixAnalyzer" in str(parsed_word.methods_stack):
            # Убираем слово с ошибкой из множества перед поиском наиболее похожего
            word_set_without_error = word_set - {word}

            # Найдем ближайшее слово из множества и заменим
            corrected_word = find_similar(word, word_set_without_error)

            words[i] = corrected_word
            corrected_sentence.append(f"{word} -> {corrected_word}")
        else:
            corrected_sentence.append(word)

    return ' '.join(corrected_sentence)


def get_result(path = "sample.json"):
    with open(path, 'r', encoding='utf-8') as file:
        sentences = json.load(file)
    sentences = [elem['text'] for elem in sentences]
    sentences = [elem.lower() for elem in sentences]
    sentences = [del_signs(elem) for elem in sentences]
    # Создаем множество всех слов в предложениях
    all_words_set = set(word for sentence in sentences for word in sentence.split())

    # Создайте пустой словарь для хранения сходства между предложениями
    similarity_matrix = {}

    # Соберем все предложения после предобработки в список
    preprocessed_sentences = [preprocess_sentence(sentence, all_words_set) for sentence in sentences]

    # Используем TF-IDF векторизацию
    vectorizer = TfidfVectorizer(analyzer=lambda x: x.split())
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)

    # Сравните все пары предложений в датасете
    for i in tqdm(range(len(preprocessed_sentences))):
        for j in range(len(preprocessed_sentences)):
            if i == j:
                continue
            similarity = compare_sentences(preprocessed_sentences[i], preprocessed_sentences[j])
            similarity_matrix[(i, j)] = similarity

    result = [(i, j, similarity)
              for (i, j), similarity in similarity_matrix.items()]

    return result
