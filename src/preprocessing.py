import numpy as np
from sklearn.model_selection import  train_test_split
import operator

# languages = ['bg', 'bs', 'cz', 'es-AR', 'es-ES', 'hr', 'id', 'mk', 'my', 'pt-BR', 'pt-PT', 'sk', 'sr']
input_file = 'Project (Application 2) (Corpus).txt'
remove_list = [" ", ".", ",", "[", "]", "(", ')', '"', "”", "„", "“", '%', '!', ';', '?', '–', '-', '&', "'", '/', '…', '→', '’', '‘', "`", '@', '$', '»', '«', '€', '•', '½', '+', '#', '*', '‰', '®', '<', '>', '=', '©', '°', '²', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', "\xa0", "\xad", "\u2002"]

               
def read_file(input_file, remove_list, NB = False):
    with open(input_file, encoding='utf-16') as f:
        lines = f.read().split("\n")
        sentences_by_language = {} 
        average_word_list = {}
        lang_dict = {}
        count = 0             
        for index, line in enumerate(lines):
            sentence = line.split("\t")
            if len(sentence) < 2:
                continue
            lang = sentence[1]
            sent = sentence[0]
            if not NB:
                word_list = sent.split(" ")
                total_avg = sum( map(len, word_list) ) / len(word_list)
                if lang in average_word_list:
                    average_word_list[lang].append(total_avg)
                else:
                    average_word_list[lang] = [total_avg]
            
            for i in remove_list: 
                sent = sent.replace(i, "")              
            if lang in sentences_by_language:
                sentences_by_language[lang].append(sent)              
            else:
                lang_dict[count] = lang
                count = count + 1
                sentences_by_language[lang] = [sent]                
        return sentences_by_language , lang_dict , average_word_list   

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



def evaluation(label, len_languages, interval ,lang_dict , NB):
    real_label = np.ones(len_languages) * interval
    if not NB:
        labelled_count = [label.count(i+1) for i in range(len_languages)]
        TP = np.asarray([label[(interval*i):(interval*(i+1))].count(i+1)  for i in range(len_languages)])
    else:
        labelled_count = [label.count(i) for i in range(len_languages)]
        TP = np.asarray([label[(interval*i):(interval*(i+1))].count(i)  for i in range(len_languages)])
    
    FP = np.asarray(list(map(operator.sub, labelled_count, TP)))
    FN = np.asarray(list(map(operator.sub, real_label , TP)))
    TN = np.asarray(np.ones(len_languages)* interval * len_languages - FN - FP - TP)
    
    if not NB:
        TP = TP + 10**-3
        FP = FP + 10**-3
        FN = FN + 10**-3
        TN = TN + 10**-3

    for i in range(len(TP)):
        print("TP for ",lang_dict[i],": ",TP[i])
        print("FP for ",lang_dict[i],": ",FP[i])
        print("FN for ",lang_dict[i],": ",FN[i])
        print("TN for ",lang_dict[i],": ",TN[i])

    
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
    
    print("Accuracy for each language : ")
    for i in range(len(lang_accuracy)):
        print(lang_dict[i],": ",lang_accuracy[i])
    
    
    print("\nAccuracy for entire set: ",total_accuracy)
    print("Micro average recall: ",micro_average_recall)
    print("Micro average precision: ",micro_average_precision)
    print("Micro average f: ",micro_average_f)
    print("Macro average recall: ",macro_average_recall)
    print("Macro average precision: ",macro_average_precision)
    print("Macro average f: ",macro_average_f)
    
    return micro_average_recall , micro_average_precision , micro_average_f , macro_average_recall , macro_average_precision , macro_average_f , lang_accuracy , total_accuracy


sentences, lang_dict, average_word_list = read_file(input_file, remove_list)
sentences_train, sentences_test, interval = get_train_test(sentences)
train_sentences = train_preprocessing(sentences_train)
letter_dict = get_letter_dict(train_sentences)
len_languages = len(sentences)
language_prob = (1/len_languages)




