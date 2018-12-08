# -*- coding: utf-8 -*-
import re

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

sentences = read_file(input_file)
for lang in sentences.keys():
	print(lang)

for czech_sentence in sentences['cz']:
	print(czech_sentence)