import re
from collections import defaultdict, Counter
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import math
import operator

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

def get_train_test(sentences):
    sentences_train = []
    sentences_test = []
    interval =  0
    for lang in sentences:
        train, test = train_test_split(sentences[lang], test_size=0.1, random_state=42)   
        interval = len(test)
        sentences_train.append(train)
        sentences_test.append(test)
    return sentences_train, sentences_test, interval
    
def train_preprocessing(sentences):
    total_sentences = []
    for lang in sentences:
        temp_sent = ""
        for sent in lang:
            temp_sent = temp_sent + sent 
        total_sentences.append(temp_sent)        
    return total_sentences
            

def get_letter_dict(train_sentences):
    temp = ''.join(train_sentences)
    return list(set(temp))

def get_conditional_prob(train_sentences, letter_dict, apply_smoothing = True):
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

def naivebayes(sentences_test,conditional_prob, len_languages):        
    sent_label = []
    for lang in sentences_test:
        sent_prob = np.ones(len_languages)*language_prob
        for sent in lang:
            c = Counter(sent)
            for poss_lang in range(len_languages):
                for i in c:
                    letter_prob = 0
                    if i in letter_dict:
                        letter_id = letter_dict.index(i)
                        letter_prob = math.log10(conditional_prob[letter_id][poss_lang])
                        
                    sent_prob[poss_lang] = sent_prob[poss_lang] + letter_prob*c[i]
            sent_label.append(np.argmax(sent_prob))
    return sent_label        

def evaluation(label, len_languages, interval , NB):
    real_label = np.ones(len_languages) * interval
    if not NB:
        labelled_count = [label.count(i+1) for i in range(len_languages)]
        TP = np.asarray([label[(interval*i):(interval*(i+1))].count(i+1)  for i in range(len_languages)])
    else:
        labelled_count = [label.count(i) for i in range(len_languages)]
        TP = np.asarray([label[(interval*i):(interval*(i+1))].count(i)  for i in range(len_languages)])
    
    FP = np.asarray(list(map(operator.sub, labelled_count, TP)))
    FN = np.asarray(list(map(operator.sub, real_label , TP)))
    
    lang_accuracy = []
    for i in range(len_languages):
        acc  = (interval * len_languages - FN[i]- FP[i]) / (interval * len_languages) 
        lang_accuracy.append(acc)

    total_accuracy = sum(TP) / (len_languages * interval)
    
    TP_sum = np.sum(TP)  
    TP_FP_sum = np.sum(TP + FP)
    TP_FN_sum = np.sum(TP + FN)
    micro_average_recall = TP_sum / TP_FP_sum
    micro_average_precision = TP_sum / TP_FN_sum
    micro_average_f = 2* micro_average_recall * micro_average_precision / (micro_average_recall + micro_average_precision) 
     
    recall = TP / (TP + FP)
    precision = TP / (TP + FN)
    F = 2 * recall * precision / (recall + precision )
    macro_average_recall = np.sum(recall) / len_languages
    macro_average_precision = np.sum(precision) / len_languages
    macro_average_f = np.sum(F) / len_languages
    
    return micro_average_recall , micro_average_precision , micro_average_f , macro_average_recall , macro_average_precision , macro_average_f , lang_accuracy , total_accuracy


sentences = read_file(input_file, remove_list)
sentences_train, sentences_test, interval = get_train_test(sentences)
train_sentences = train_preprocessing(sentences_train)
len_languages = len(sentences)
language_prob = math.log10(1/len_languages)
lang_total_char = [len(i) for i in train_sentences]       
letter_dict = get_letter_dict(train_sentences)
conditional_prob =  get_conditional_prob(train_sentences, letter_dict)
label = naivebayes(sentences_test, conditional_prob , len_languages)
micro_average_recall , micro_average_precision , micro_average_f , macro_average_recall , macro_average_precision , macro_average_f , lang_accuracy , total_accuracy = evaluation(label, len_languages, interval , True)


        