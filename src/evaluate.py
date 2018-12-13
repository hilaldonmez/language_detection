import language_detector_nb as nb
import language_detector_svm as svm
import sys

APPLY_SVM = False
APPLY_WORD = False
APPLY_SENTENCE = False
APPLY_CAPITAL = False

arguments = sys.argv
for index, arg in enumerate(arguments):
    if arg == "-svm":
        APPLY_SVM = True
    elif arg == "-word-len" and APPLY_SVM:
        APPLY_WORD = True
    elif arg == "-sen-len" and APPLY_SVM:
        APPLY_SENTENCE = True
    elif arg == "-capital" and APPLY_SVM:
        APPLY_CAPITAL = True

if APPLY_SVM:
	print("Applying support vector machine")
	if APPLY_SENTENCE:
		print("\tApplied feature: Length of sentence")
	if APPLY_WORD:
		print("\tApplied feature: Avg length of words")
	if APPLY_CAPITAL:
		print("\tApplied feature: Capitalization")
else:
	print("Applying naive bayes")

if APPLY_SVM:
	svm.apply_svm(APPLY_SENTENCE, APPLY_CAPITAL, APPLY_WORD)
else:
	nb.apply_naive_bayes()
