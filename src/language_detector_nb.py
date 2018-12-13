import Util as pr
from collections import  Counter
import numpy as np
import math

def get_conditional_prob(train_sentences, letter_dict, len_languages, lang_total_char, apply_smoothing = True):
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

def naivebayes(sentences_test, conditional_prob, len_languages, letter_dict, language_prob):        
    sent_label = []
    for lang in sentences_test:
        for sent in lang:
            label = find_lang_tag(sent, len_languages, language_prob, letter_dict, conditional_prob)
            sent_label.append(label)
    return sent_label

def find_lang_tag(sentence, len_languages, language_prob, letter_dict, conditional_prob):
    sent_prob = np.ones(len_languages)*math.log10(language_prob)
    c = Counter(sentence)
    for poss_lang in range(len_languages):
        for i in c:
            letter_prob = 0
            if i in letter_dict:
                letter_id = letter_dict.index(i)
                letter_prob = math.log10(conditional_prob[letter_id][poss_lang])
                
            sent_prob[poss_lang] = sent_prob[poss_lang] + letter_prob*c[i]
    return np.argmax(sent_prob)

def get_language(sentence):
    for i in pr.remove_list: 
        sentence = sentence.replace(i, "")
    lang_total_char = [len(i) for i in pr.train_sentences]       
    conditional_prob =  get_conditional_prob(pr.train_sentences, pr.letter_dict, pr.len_languages, lang_total_char)
    language = find_lang_tag(sentence, pr.len_languages, pr.language_prob, pr.letter_dict, conditional_prob)
    return pr.lang_dict[language]

def apply_naive_bayes():     
    lang_total_char = [len(i) for i in pr.train_sentences]       
    conditional_prob =  get_conditional_prob(pr.train_sentences, pr.letter_dict, pr.len_languages, lang_total_char)
    label = naivebayes(pr.sentences_test, conditional_prob, pr.len_languages, pr.letter_dict, pr.language_prob)
    micro_average_recall, micro_average_precision, micro_average_f, macro_average_recall, macro_average_precision, macro_average_f, lang_accuracy, total_accuracy = pr.evaluation(label, pr.len_languages, pr.interval, pr.lang_dict, True)
