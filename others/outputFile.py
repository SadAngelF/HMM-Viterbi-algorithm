
# this function takes in data in lists format [[], []]
def exportDictionary(data, new_words, var):
	newData = []
	for i in range(len(data)):
		newData.append(data[i+1])
	f = open('dev.p'+var+'.out', 'w')

	for i in range(len(newData)):
		for j in range(len(newData[i])):
			f.write(new_words[i][j] + ' ' + newData[i][j] + '\n')
		f.write('\n')
	print("Done writing file!") 
	f.close()

def export(data, new_words, var):
	

	f = open('dev.p'+var+'.out', 'w')

	for i in range(len(data)):
		for j in range(len(data[i])):
			f.write(new_words[i][j] + ' ' + data[i][j] + '\n')
		f.write('\n')
	print("Done writing file!") 
	f.close()

def exportDictionaryW(data, new_words, var):
	newData = []
	for i in range(len(data)):
		newData.append(data[i+1])
	f = open('dev.p'+var+'.out', 'w', encoding='utf8')

	for i in range(len(newData)):
		for j in range(len(newData[i])):
			f.write(new_words[i][j] + ' ' + newData[i][j] + '\n')
		f.write('\n')
	print("Done writing file!") 
	f.close()

def exportW(data, new_words, var):
	

	f = open('dev.p'+var+'.out', 'w', encoding='utf8')

	for i in range(len(data)):
		for j in range(len(data[i])):
			f.write(new_words[i][j] + ' ' + data[i][j] + '\n')
		f.write('\n')
	print("Done writing file!") 
	f.close()