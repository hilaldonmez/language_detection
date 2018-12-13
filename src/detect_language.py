import language_detector_nb as nb

language_names = {'bg': 'Bulgarian', 'bs': 'Bosnian', 'cz': 'Czech', 'es-AR': 'Argentinian Spanish', 'es-ES': 'Peninsular Spanish', 'hr': 'Croatian', 'id': 'Indonesian', 'mk': 'Macedonian', 'my': 'Malay', 'pt-BR': 'Brazilian Portuguese', 'pt-PT': 'European Portuguese', 'sk': 'Slovak', 'sr': 'Serbian'}

while True:
	sentence = input("Enter a sentence (or \'-exit\' to exit program): ")
	if sentence == '-exit':
		break
	language = nb.get_language(sentence)
	if language in language_names:
		language = language_names[language]
	print("Detected Language: ", language)
