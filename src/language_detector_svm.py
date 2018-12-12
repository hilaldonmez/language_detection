import language_detector as ld

original_sentences = ld.sentences
sentences_train, sentences_test, interval = ld.get_train_test(original_sentences)
len_languages = ld.len_languages
#%%
def get_full_dict(original_sentences):
    full_sent = ""
    
    for lang in original_sentences:
        for sent in original_sentences[lang]:
            full_sent = full_sent + sent    
    
    letters = list(set(full_sent))  
    letter_dict = {}
    
    for i in range(len(letters)):
        letter_dict[letters[i]] = i+1           
    return letter_dict    
    
#%%
# lang id ve letter id 1 den baslıyor    
def write_data(file_name, sentences , train_file = True):
    file = open(file_name ,"w")         
        
    for lang_id in range(len(sentences)):
        for sent in sentences[lang_id]:            
            temp_string = str(lang_id + 1) + " "
            letters = list(set(sent))
            increasing_order = []
            for l in letter_dict:
                letter_id = letter_dict[l]
                if l in letters:
                    increasing_order.append(letter_id)               
                
            for i in sorted(increasing_order):    
                temp_string = temp_string + str(i) + ":1 "
                
            temp_string = temp_string + " \n"
            file.write(temp_string)

def get_result_label(file_name):
    result_label = []
    with open(file_name) as f:
        for line in f:
            result_label.append(line.split()[0])
    return result_label

            
#%%
letter_dict = get_full_dict(original_sentences)    
write_data("train.txt", sentences_train)
write_data("test.txt", sentences_test , train_file = False)

#%% 
result_label = get_result_label("result.txt")            
result_label = list(map(int, result_label))

micro_average_recall , micro_average_precision , micro_average_f , macro_average_recall , macro_average_precision , macro_average_f , lang_accuracy , total_accuracy = ld.evaluation(result_label,  len_languages, interval , False)  









     