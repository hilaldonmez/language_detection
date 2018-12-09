# -*- coding: utf-8 -*-
import re
from collections import defaultdict
import numpy as np
from sklearn.model_selection import KFold

# languages = ['bg', 'bs', 'cz', 'es-AR', 'es-ES', 'hr', 'id', 'mk', 'my', 'pt-BR', 'pt-PT', 'sk', 'sr']
input_file = 'Project (Application 2) (Corpus).txt'

def read_file(input_file):
	with open(input_file, encoding='utf-16') as f:
		lines = f.read().split("\n")
		sentences_by_language = {}
		for index, line in enumerate(lines):
			sentence = line.split("\t")
			if len(sentence) < 2:
				continue
			lang = sentence[1]
			if lang in sentences_by_language:
				sentences_by_language[lang].append(sentence[0])
			else:
				sentences_by_language[lang] = [sentence[0]]
		return sentences_by_language

def get_sentence_id(sentences):
    lang_id = dict()
    count = 0
    for i in sentences.keys():
        lang_id[i] = count 
        count = count + 1
    return lang_id

# remove later
def preprocessing(sentences,languages):
    remove_list = [" ","." ,"," ,"[" ,"]","(" ,')','"',"”","„","“",'%','!',';','?','–','-','&',"'",'/','…','→','’','‘',"`",'@','$','»','«','€','•','½','+','#','*','‰','®','<','>','=','©','°','²' ]	
    
    pre_sentences = dict()
    for l in sentences:
        l_id = languages[l]
        pre_sentences[l_id] = []
        for sent in sentences[l]:
            temp_sent = sent
            for i in remove_list:               
                temp_sent = temp_sent.replace(i, "")
            pre_sentences[l_id].append(temp_sent)

    return pre_sentences


#%%
def get_frequencies(sentences,languages):
    remove_list = [" ","." ,"," ,"[" ,"]","(" ,')','"',"”","„","“",'%','!',';','?','–','-','&',"'",'/','…','→','’','‘',"`",'@','$','»','«','€','•','½','+','#','*','‰','®','<','>','=','©','°','²' ]	
            
    pre_sentences = dict()
    letter_dict = defaultdict()
    total_char = []  # total num of letter for each language  
    lang_letter = [] # each letter frequency for each language  
    count  = 0
    for l in sentences:
        lang_letter_freq = dict() # the num letter for a language
        letter_count = 0
        l_id = languages[l]
        pre_sentences[l_id] = []
        for sent in sentences[l]:
            sentence_letter_freq = dict() #letter freq in the sentence
            temp_sent = sent
            for i in remove_list:               
                temp_sent = temp_sent.replace(i, "")
            for letter in temp_sent:
                letter_count = letter_count + 1 
                if letter not in letter_dict:
                    letter_dict[letter] = count
                    count = count + 1
                letter_id = letter_dict[letter]
               
                if letter_id in lang_letter_freq:
                    lang_letter_freq[letter_id] = lang_letter_freq[letter_id] + 1
                else:
                    lang_letter_freq[letter_id] = 1
                
                if letter_id in sentence_letter_freq:
                    sentence_letter_freq[letter_id] = sentence_letter_freq[letter_id] + 1
                else:
                    sentence_letter_freq[letter_id] = 1
                         
            pre_sentences[l_id].append(sentence_letter_freq)
            
        total_char.append(letter_count)
        lang_letter.append(lang_letter_freq)
    
    return pre_sentences, total_char, lang_letter, letter_dict


# P(c_i | l)
def get_conditional_prob(letter_dict, languages , lang_letter_count, total_char) :
    
    letter_len = len(letter_dict)
    lang_len = len(languages)
    letter_lang_prob = np.zeros((letter_len, lang_len ))
    for letter in letter_dict:
        letter_id = letter_dict[letter]
        for lang in languages:
            lang_id = languages[lang]
            freq = 0
            
            if letter_id in lang_letter_count[lang_id]:
                freq = lang_letter_count[lang_id][letter_id]
            
            lang_count = total_char[lang_id]
            letter_lang_prob[letter_id][lang_id] = freq / lang_count
    
    return letter_lang_prob

#  sentence containing letter id
def naive_bayes(letter_lang_prob, language_prob,sentence):
    sentence_prob = 1 
    lang_id = 0
    for letter_id in sentence:
        sentence_prob = sentence_prob * letter_lang_prob[letter_id][lang_id] * language_prob  
    
    return sentence_prob


        
#%%

sentences = read_file(input_file)
languages = get_sentence_id(sentences)
len_languages = len(languages)
clean_sentences = preprocessing(sentences, languages)
language_prob = 1/len_languages
pre_sentences, total_char, lang_letter_count , letter_dict = get_frequencies(sentences,languages)    
letter_lang_prob = get_conditional_prob(letter_dict, languages, lang_letter_count, total_char)    

#%%
# letter id 
temp_sentences = pre_sentences[0][0].keys()
sentence_prob = naive_bayes(letter_lang_prob, language_prob, temp_sentences)








