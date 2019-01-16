# Language Detection
Language Identification System is implemented via both Naive Bayes and Support Vector Machine. Dataset is Discriminating between Similar Languages (DSL) Shared Task 2015 corpus which consists of 13 languages and 2000 sentences for each language. In the dataset, each line has a sentence and related language label. </br>
Code is implemented using Python3. Below are python3 packages needed to run code:
* numpy
* scikit-learn
* [SVM multiclass library](https://www.cs.cornell.edu/people/tj/svm_light/svm_multiclass.html)
</br>

After the environment is set, there are two ways to execute the language detector.
1. Evaluation of the system by splitting input data to training and test sets, then calculating the accuracy, precision, recall and F-measure values for different methods. Default classication system
is Naive Bayes Classification if no extra parameter is given.</br>
    python3 evaluate.py </br>
    **Parameters:** </br>
    -  -svm: To use Support Vector Machine as classification method. </br>
          python3 evaluate.py -svm </br>
    -   -word-len: To use average length of the words as a feature in SVM implementation </br>
          python3 evaluate.py -svm -word-len
    -   -sen-len: To use sentence length values as a feature in SVM implementation. </br>
          python3 evaluate.py -svm -sen-len
    -   -capital: To use capitalization as a feature in SVM implementation.</br>
          python3 evaluate.py -svm -capital
2. To take a sentence as an input and determining the language of the text. </br>
    python3 detect_language.py
    
    



