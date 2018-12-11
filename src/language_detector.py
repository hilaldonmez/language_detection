import re
from collections import defaultdict, Counter
import numpy as np
from sklearn.model_selection import KFold, train_test_split

# languages = ['bg', 'bs', 'cz', 'es-AR', 'es-ES', 'hr', 'id', 'mk', 'my', 'pt-BR', 'pt-PT', 'sk', 'sr']
input_file = 'Project (Application 2) (Corpus).txt'
remove_list = [" ", ".", ",", "[", "]", "(", ')', '"', "”", "„", "“", '%', '!', ';', '?', '–', '-', '&', "'", '/', '…', '→', '’', '‘', "`", '@', '$', '»', '«', '€', '•', '½', '+', '#', '*', '‰', '®', '<', '>', '=', '©', '°', '²']

def read_file(input_file, remove_list):
    with open(input_file, encoding='utf-16') as f:
        lines = f.read().split("\n")
        sentences_by_language = {}              
        for index, line in enumerate(lines):
            sentence = line.split("\t")
            if len(sentence) < 2:
                continue
            lang = sentence[1]
            sent = sentence[0]
            for i in remove_list: 
                sent = sent.replace(i, "")              
            if lang in sentences_by_language:
                sentences_by_language[lang].append(sent)              
            else:
                sentences_by_language[lang] = [sent]                
        return sentences_by_language    
    
def train_preprocessing(sentences):
    total_sentences = []
    for lang in sentences:
        temp_sent = ""
        for sent in lang:
            temp_sent = temp_sent + sent 
        total_sentences.append(temp_sent)        
    return total_sentences
            
def get_train_test(sentences):
    sentences_train = []
    sentences_test = []
    for lang in sentences:
        train, test = train_test_split(sentences[lang], test_size=0.1, random_state=42)   
        sentences_train.append(train)
        sentences_test.append(test)
    return sentences_train, sentences_test

def get_letter_dict(train_sentences):
    temp = ''.join(train_sentences)
    return list(set(temp))

def get_conditional_prob(train_sentences, letter_dict, apply_smoothing=False):
    len_letter = len(letter_dict)
    conditional_prob = np.zeros((len_letter, len_languages))
    for lang in range(len(train_sentences)):
        counter = Counter(train_sentences[lang])
        for char in range(len(letter_dict)):
            real_char = letter_dict[char] 
            conditional_prob[char][lang] = counter[real_char]
        if apply_smoothing:
            conditional_prob[:,lang] = np.array([(element + 1) for element in conditional_prob[:,lang]]) / (lang_total_char[lang] + len_letter)
        else:
            conditional_prob[:,lang] = conditional_prob[:,lang] / lang_total_char[lang] 
    return conditional_prob

sentences = read_file(input_file, remove_list)
sentences_train, sentences_test = get_train_test(sentences)
train_sentences = train_preprocessing(sentences_train)
len_languages = len(sentences)
language_prob = 1/len_languages
lang_total_char = [len(i) for i in train_sentences]       
letter_dict = get_letter_dict(train_sentences)
conditional_prob =  get_conditional_prob(train_sentences, letter_dict)

#%%
        
