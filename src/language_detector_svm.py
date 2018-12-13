import Util as pr

def get_full_dict(original_sentences):
    full_sent = ""    
    count = 0
    for lang in original_sentences:
        count = count + 1
        for sent in original_sentences[lang]:
            full_sent = full_sent + sent    
    
    letters = list(set(full_sent))  
    letter_dict = {}    
    # id starts with 1 in SVM library
    for i in range(len(letters)):
        letter_dict[letters[i]] = i+1           
    return letter_dict    
    

# lang id and letter id start with 1 in SWM library    
# sentence length is added as a feature
# 10000 means the length of sentence 
# 10001 means the number of capital -> number of occurences
# 10002 means the average length of word in sentences    
def write_data(file_name, sentences, average_word_list, lang_dict, letter_dict, sentence_length = False, capital = False, word_list = False):
    file = open(file_name, "w")         
    for lang_id in range(len(sentences)):
        sent_count = 0
        for sent in sentences[lang_id]:   
            capital_count = 0
            len_sent = len(sent)            
            temp_string = str(lang_id + 1) + " "
            letters = list(set(sent))
            increasing_order = []
            for l in letter_dict:
                letter_id = letter_dict[l]
                if l in letters:
                    if l.isupper():
                        capital_count = capital_count + 1 
                    increasing_order.append(letter_id)                              
            for i in sorted(increasing_order):    
                temp_string = temp_string + str(i) + ":1 "
        
            if sentence_length:
                temp_string = temp_string + str(10000) + ":" + str(len_sent) + " "
            if capital:
                temp_string = temp_string + str(10001) + ":" + str(capital_count) + " " 
            if word_list:
                lang = lang_dict[lang_id]
                temp_string = temp_string + str(10002) + ":" + str(average_word_list[lang][sent_count]) + " "
                
            temp_string = temp_string + " \n"
            sent_count = sent_count + 1
            file.write(temp_string)

def get_result_label(file_name):
    result_label = []
    with open(file_name) as f:
        for line in f:
            result_label.append(line.split()[0])
    return result_label

def apply_svm(sentence_length = False, capital = False, word_list = False):
    letter_dict = get_full_dict(pr.sentences)    
    write_data("train.txt", pr.sentences_train, pr.average_word_list, pr.lang_dict, letter_dict, sentence_length,capital, word_list)
    write_data("test.txt", pr.sentences_test, pr.average_word_list, pr.lang_dict, letter_dict, sentence_length,capital, word_list)

    #%%
    result_label = get_result_label("result.txt")            
    result_label = list(map(int, result_label))
    micro_average_recall, micro_average_precision, micro_average_f, macro_average_recall, macro_average_precision, macro_average_f, lang_accuracy, total_accuracy = pr.evaluation(result_label, pr.len_languages, pr.interval, pr.lang_dict, False)  


