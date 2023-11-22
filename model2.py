import numpy as np
import json
from functools import reduce
from fuzzywuzzy import fuzz
import pymorphy2
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

morph = pymorphy2.MorphAnalyzer()
path_to_js = 'output.json'
path_to_names = 'bdnames.txt'
custombanned = r'[!,.№-]'
similoef = 1
levin_thres = 89
jack_thres = 0.6
cos_thres = 0.559

def negEq(senF, senS):
    return abs(len([word for word in senF if word == 'не']) - len([word for word in senS if word == 'не']))


def eqls(pronsF : set, verbsF : set, pronsS : set, verbsS : set):
    return (pronsF == pronsS) and (verbsF == verbsS)


def getProns(text, names):
    pronsNnames = []
    for word in text:
        if word in names:
            pronsNnames.append(word)
        if 'NPRO' in morph.parse(word)[0].tag:
            pronsNnames.append(morph.parse(word)[0].normal_form)
    return set(pronsNnames)

def getVerbs(text):
    return set([morph.parse(word)[0].normal_form for word in text if morph.parse(word)[0].tag.POS in ['VERB', 'INFN']])


#равенство исходя из разности множеств уник. слов
def eq(senF, senS):
    if(len(set(senF) - set(senS)) <= 1):
        return True
    return False

#схожесть по метрике Джекарда
def eqJack(senF, senS):
    intersec = set(senF) & set(senS)
    fset = set(senF)
    sset = set(senS)
    return len(intersec) / len((fset | sset - intersec))

#метод получения местоимений, имен собственных и глаголов
def getRelation(tokens : list, names : list):
    mor = pymorphy2.MorphAnalyzer()
    ims, verbs = [], []
    for indword in range(len(tokens)):
        word = tokens[indword]
        if mor.parse(word)[0].tag.POS in ['VERB', 'INFN']:
            if tokens[indword - 1] == 'не':
                verbs.append('не '+mor.parse(word)[0].normal_form)
            else:
                verbs.append(mor.parse(word)[0].normal_form)
        if word in names or mor.parse(word)[0].tag.POS in ['NPRO']:
            ims.append(word)
    return [ims, verbs]

#метод схожести двух предложений, исходя из полученных местоимений, глаголов, имен собственных
def eqs(senF : list, senS : list, names=[]):
    tksF = getRelation(senF, names)
    tksS = getRelation(senS, names)
    if len(tksF[0]) != len(tksS[0]) or len(tksS[1]) != len(tksF[1]):
        return False
    if len(tksF[0]) == len(tksF[1]) == len(tksS[0]) == len(tksS[1]) == 0:
        return True

    for i in range(len(tksF[0])):
        if tksF[0][i] != tksS[0][i]:
            return False

    for i in range(len(tksF[1])):
        if tksF[1][i] != tksS[1][i]:
            return False

    return True


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


#кастом датасет для удобной работы
class Customdataset:
    def __init__(self, path : str(), lang='ru', pipeline=[tokenize]):
        self.mor = pymorphy2.MorphAnalyzer()
        with open(path) as fl:
            self.dataset = json.loads(fl.read())
        assert('ru' == lang)
        self.names = ['кен', 'том', 'мэри']
        with open(path_to_names) as fl:
            self.names = fl.read().lower().split('\n') + self.names
        self.data = [text['text'] for text in self.dataset]
        self.unitedData = None
        self.vocab = None
        self.vecwords = None
        self.model = TfidfVectorizer()

        assert(pipeline != [])
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
        assert(ind < len(self.data))
        return self.data[ind]


    #получить словарь слов
    def getVocab(self):
        return self.vocab

    #длина датасета
    def __len__(self):
        return len(self.data)

    #input: токенизированное предл.
    #return: (ind, is_rewrite, sum_scores, [first_score, second_score, ...])
    def score(self, dataTest, levinWeight=1, jackWeight=1, cosSimWeight=1, strParamsWeight=1,
              NegativeWeight=-100, strParams=True):
        # assert(type(dataTest) == list, 'GIMME TKNZD sentence (чекай нормализован ли датасет)')
        # assert(self.vocab != None, 'need tokenize pipeline')
        # print(' '.join(dataTest))
        vecdataTest = self.model.transform([' '.join(dataTest)]).toarray()
        PronsdataTest = getProns(dataTest, self.names)
        VerbsdataTest = getVerbs(dataTest)

        results = []

        for i in range(len(self.data)):
            levincoef = fuzz.token_set_ratio(dataTest, self.data[i])
            jackcoef = eqJack(dataTest, self.data[i])
            cosSim = getCos(self.vecwords[i], vecdataTest)
            negAbs = negEq(self.data[i], dataTest)

            isRewrite = bool(levincoef >= levin_thres and \
                             jackcoef >= jack_thres and cosSim > cos_thres and negAbs == 0)
            coefs = [levincoef, jackcoef, cosSim, negAbs]
            sumScores = levinWeight * levincoef + jackWeight * jackcoef + cosSimWeight * cosSim + int(negAbs) * NegativeWeight
            if strParams:
                isTrueOrder = eqls(PronsdataTest, VerbsdataTest, self.prons[i], self.verbs[i])
                isRewrite &= isTrueOrder
                coefs.append(isTrueOrder)
                sumScores += strParamsWeight * int(isTrueOrder)



            results.append((i, isRewrite, sumScores, coefs))
        return results

def get_result(path = "sample.json"):
    #датасет без нормализации
    cst = Customdataset(path, pipeline=[tokenize, norm])
    cst2 = Customdataset(path)
    result = []
    for i in tqdm(range(len(cst))):
        res = cst.score(cst[i], strParams=True)
        for j in range(len(res)):
            if res[j][0] != i:
                result.append([i, j, res[j][2][0]])

    for elem in result:
        elem[2] /= 100.0
    return result