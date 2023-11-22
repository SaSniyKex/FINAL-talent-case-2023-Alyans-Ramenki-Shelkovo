#!/usr/bin/env python
# coding: utf-8

# In[135]:


'''

Идея: 
1) использовать фонетический алгоритм aka soundEx, но более эффективный вариант: адаптированный Metaphone к русскому языку
Хороший способ привести слова с орфографическими ошибками к нормальным формам
2) далее использовать алгоритм шинглов: нормализуем, удаляем предлоги и неинформативные прилагательные

+хороший скор
-ресурсозатратно    
'''


# In[14]:


import numpy as np
import json
import matplotlib.pyplot as plt
from functools import reduce
from fuzzywuzzy import fuzz, process
import pymorphy2
import rapidfuzz.distance.JaroWinkler as jw
import re
from sklearn.feature_extraction.text import TfidfVectorizer

morph = pymorphy2.MorphAnalyzer()

path_to_js = 'sample.json'
custombanned = r'[!,.№-]'
similoef = 1
levin_thres = 89
jack_thres = 0.6
cos_thres = 0.559


# In[15]:


#нормализация 
def norm(text : list):
    mor = pymorphy2.MorphAnalyzer()
    if type(text[0]) != list:
        text = [text]
        
    for indsec in range(len(text)):
        for indword in range(len(text[indsec])):
            text[indsec][indword] = mor.parse(text[indsec][indword])[0].normal_form.lower()
    return text

def tokenize(sentence : list):
    for ind in range(len(sentence)):
        sentence[ind] = re.sub(custombanned, '', sentence[ind])
        sentence[ind] = re.sub(r'\?', ' ?', sentence[ind])
        sentence[ind] = sentence[ind].split()
    return sentence

def usePipeline(text : list, funcs : list):
    for funcPipe in funcs:
        text = funcPipe(text)
    return text

#функция получения схожести по косинусовой мере
def getCos(Fsen, Ssen):
    return np.dot(Fsen, Ssen.T) / (np.linalg.norm(Fsen) * np.linalg.norm(Ssen))


# In[16]:


#кастом датасет для удобной работы
class Customdataset:
    def __init__(self, path : str(), lang='ru', pipeline=[tokenize]):
        self.mor = pymorphy2.MorphAnalyzer()
        with open(path) as fl:
            self.dataset = json.loads(fl.read())
        assert('ru' == lang)
        self.names = ['кен', 'том', 'мэри']
        self.data = [text['text'] for text in self.dataset]
        self.unitedData = None
        self.vocab = None
        self.vecwords = None
        self.model = TfidfVectorizer()
        
        if pipeline == []:
            return
        self.data = usePipeline(self.data, pipeline)
        self.unitedData = [' '.join(sen) for sen in self.data]
        #build vocab
        self.vocab = set(reduce(lambda x, y: x + y, self.data))
        #build vectors
        self.vecwords = self.model.fit_transform(self.unitedData).toarray()
        #build PRONS and INFN,VERBS
        self.prons = []
        self.verbs = []
        for sent in self.data:
            pronsnow = set([morph.parse(word)[0].normal_form for word in sent if word in self.names or 'NPRO' in morph.parse(word)[0].tag])
            verbsnow = set([morph.parse(word)[0].normal_form for word in sent if morph.parse(word)[0].tag.POS in ['VERB', 'INFN']])
            self.prons.append(pronsnow)
            self.verbs.append(verbsnow)

        

        
            
            
    def getvec(self, ind):
        return self.vecwords[ind]
    

    #получить i предложение с отделенными токенами
    def __getitem__(self, ind):
        if isinstance(ind, slice):
            return self.data[ind.start:ind.stop]
        return self.data[ind]


    #получить словарь слов
    def getVocab(self):
        return self.vocab
    
    #длина датасета
    def __len__(self):
        return len(self.data)

    #input: токенизированное предл.
    #return: (ind, is_rewrite, sum_scores, [first_score, second_score, ...])
    def score(self):
        pass



# In[17]:


cst = Customdataset(path_to_js)


# In[18]:


def getReplaced(string :str, rules : dict) -> str:
    for key, value in rules.items():
        string = string.replace(key, value)
    return string


def convertIntoSounds(sentences : list) -> str:
    rulesVowels = {
        'йо' : 'и',
        'ио' : 'и',
        'йе' : 'и',
        'ие' : 'и',
        'о' : 'а',
        'ы' : 'а',
        'я' : 'а',
        'е' : 'и',
        'ё' : 'и',
        'э' : 'и',
        'ю' : 'у'
    }

    rulesConsonants = {
        'б' : 'п',
        'з' : 'с',
        'д' : 'т',
        'в' : 'ф',
        'г' : 'к'
    }

    rulesSpecial = {
        'ТС' : 'Ц'
    }

    '''
    rulesSpecial ->
    rulesConsonants -> 
    rulesVowels
    '''
    for indSent in range(len(sentences)):
        for indWord in range(len(sentences[indSent])):
            sentences[indSent][indWord] = getReplaced(sentences[indSent][indWord], rulesSpecial)
            sentences[indSent][indWord] = getReplaced(sentences[indSent][indWord], rulesConsonants)
            sentences[indSent][indWord] = getReplaced(sentences[indSent][indWord], rulesVowels)
    return sentences


# In[31]:


#stop прилаг. и предлоги

def filtrSen(sentence):
    result = []

    for ind in range(len(sentence)):
        wordtemp = []
        for wordind in range(len(sentence[ind])):
            pos = morph.parse(sentence[ind][wordind])[0].tag.POS
            if pos not in ['ADJF', 'PREP']:
                wordtemp.append(sentence[ind][wordind])
        result.append(wordtemp)
    return result



# In[99]:


def shingling(text, k):
    """
    Функция для создания шинглов из текста.

    :param text: Входной текст
    :param k: Размер шингла (длина подстроки)
    :return: Список шинглов
    """
    shingles = set()
    words = text

    for i in range(len(words) - k + 1):
        shingle = tuple(words[i:i + k])
        shingles.add(shingle)
    return shingles

#сравнение множеств шинглов разной длины от min(len(первое предл), len(второе предл))
#считаем количество совпадений всех шинглов разной длины
#чем меньше размер шингла, тем меньше вес
def comparePairs(firstSet, secondSet):
    sumscore = 0
    for indKi in range(len(firstSet)):
        set1 = set(firstSet[indKi])
        set2 = set(secondSet[indKi])

        common_elements = set1.intersection(set2)
        all_elements = set1.union(set2)

        if len(all_elements) == 0:
            return 0.0

        sumscore += (len(common_elements) / len(all_elements)) * (len(firstSet) - indKi)*4.5
    return sumscore 



# In[106]:


def jaro_similarity(s1, s2):
    if s1 == s2:
        return 1.0

    len_s1 = len(s1)
    len_s2 = len(s2)

    max_dist = max(len_s1, len_s2) // 2 - 1

    matches = 0
    transpositions = 0

    match_flags_s1 = [False] * len_s1
    match_flags_s2 = [False] * len_s2

    for i in range(len_s1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len_s2)

        for j in range(start, end):
            if not match_flags_s2[j] and s1[i] == s2[j]:
                match_flags_s1[i] = True
                match_flags_s2[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len_s1):
        if match_flags_s1[i]:
            while not match_flags_s2[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

    jaro = (matches / len_s1 + matches / len_s2 + (matches - transpositions) / matches) / 3.0

    prefix_length = 0
    for i in range(min(4, min(len_s1, len_s2))):
        if s1[i] == s2[i]:
            prefix_length += 1
        else:
            break

    jaro_winkler = jaro + 0.1 * prefix_length * (1 - jaro)

    return jaro_winkler




# In[125]:


#tokenized inp
def getResult(path = 'sample.json', percentage = 100):
    with open(path, 'r') as file:
        data = json.load(file)
    threshold = 10
    cst = Customdataset(path, pipeline=[tokenize, norm])
    cstnew = Customdataset(path, pipeline=[tokenize, norm, filtrSen, convertIntoSounds])
    result = []
    for i in range(int(len(cst) * percentage / 100)):
        for j in range(0, int(len(cst) * percentage / 100)):
            if i == j:
                continue
                
            minDiv = min(len(cst[i]), len(cst[j]))
            a = [shingling(cstnew[i], k) for k in reversed(range(minDiv))]
            b = [shingling(cstnew[j], k) for k in reversed(range(minDiv))]
            resScore = comparePairs(a, b)
            if resScore > threshold and jaro_similarity(cstnew[i], cstnew[j]) > 0.87:
                result.append((i, j, resScore))
    result = [(i, j, k / max(result,key=lambda x:x[2])[2]) for i, j, k in result]
    return result
    

