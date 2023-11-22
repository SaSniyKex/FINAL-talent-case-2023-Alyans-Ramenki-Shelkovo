#!/usr/bin/env python
# coding: utf-8

# In[13]:


import re
import json
from pymorphy2 import MorphAnalyzer


# In[14]:


with open('sample.json', 'r') as file:
    data = json.load(file)
data = [elem['text'].lower() for elem in data]
data = [re.sub(r'[,\.\?\!\-]', '', elem) for elem in data]
data


# In[15]:


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


# In[16]:


morph = MorphAnalyzer()
normal_data = []
for sentence in data:
    normal_sentence = []
    for word in sentence.split():
        if word not in ("не", "нет", "ни"):
            normal_word = morph.parse(word)[0]
            if normal_word.tag.POS == 'NPRO':
                normal_sentence.append('<PRONOUN>')
            else:
                normal_sentence.append(normal_word.normal_form)
        else:
            normal_sentence.append(word)
    normal_data.append(' '.join(normal_sentence))
normal_data


# In[17]:


def compare(first_sentence, second_sentence):
    if len(first_sentence.split()) > len(second_sentence.split()):
        return compare(second_sentence, first_sentence)
    summary_distance = 0
    first_sentence = sorted(first_sentence.split())
    second_sentence = sorted(second_sentence.split())
    for i in range(len(first_sentence)):
        summary_distance += levenshtein_distance(first_sentence[i], second_sentence[i])
    if len(first_sentence) < len(second_sentence):
        summary_distance += sum(map(len, second_sentence[len(first_sentence):len(second_sentence) - 1]))
    return summary_distance


# In[18]:


def dbscan(sentences: list[str], epsilon=2):
    visited = []
    clusters = []
    while len(sentences) != len(visited):
        root_point = ""
        cluster = []
        current_index = 0
        for i, elem in enumerate(sentences):
            if i not in visited:
                root_point = elem
                cluster.append((root_point, i))
                visited.append(i)
                break
        while current_index < len(cluster):
            for i, elem in enumerate(sentences):
                if (elem, i) not in cluster:
                    if compare(elem, root_point) <= epsilon:
                        cluster.append((elem, i))
                        visited.append(i)
            current_index += 1
            if current_index < len(cluster):
                root_point = cluster[current_index][0]
        clusters.append([data[elem[1]] for elem in cluster])
    return clusters


# In[19]:


result = dbscan(normal_data)
len(result)


# In[20]:


result_dict_normal = {}
for cluster in result:
    for i, elem in enumerate(cluster):
        result_dict_normal[elem] = cluster[:i] + cluster[i + 1:]


# In[21]:


def get_score(array1, array2):
    max_score = max(len(array1), len(array2))
    if max_score == 0:
        return 100

    score = max_score
    for elem in array1:
        if elem not in array2:
            score -= 1
    for elem in array2:
        if elem not in array1:
            score -= 1

    return score / max_score * 100


# In[22]:


with open('target.json', 'r') as file:
    target = json.load(file)


# In[23]:


score = 0
for key in result_dict_normal.keys():
    score += get_score(result_dict_normal[key], target[key])
score /= float(len(target))
score


# In[12]:


with open('result_normal.json', 'w') as file:
    json.dump(result_dict_normal, file, ensure_ascii=False, indent=4)



# In[ ]:




