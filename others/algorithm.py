from helper import *
from math import log


"""This function generates the potential tag sequence based on
the emission emissionParameters """
def part2analysis(X, Y, newX):
	a = getUniqueY(Y)
	y2x = YtoX(X, Y)
	com = combineX(X)


	output = [] # create a var for output list
	unique_tag_Y = a # create list of unique tag, Y
	count = 0
	for s in  newX:
		temp = []
		for w in s:
			potential_Y = [] # create a potential y list
			for i in range(len(unique_tag_Y)):
				potential_Y.append(emissionParameters(y2x, com, Y, w, unique_tag_Y[i]))
			argmax_Y = unique_tag_Y[potential_Y.index(max(potential_Y))] # the tag with highest possiblity\
			print("argmax: ",argmax_Y)
			# output[newX.index(s)][s.index(w)] = argmax_Y
			temp.append((argmax_Y))
			# print(output)
		count += 1
		output.append((temp))
		print("%d sentence is done. Still have %d"%(count, len(output) - count))
	print("Completed")

	return output




"""Viterbi Algorithm"""
def viterbi(X, Y, newX):
	
	a = getUniqueY(Y) # unique Y tag list
	modi = modifiedY(Y)	# Y list with 'START' & 'STOP'
	join = joinY(modi) # join the modified Y tgt to form a list
	y2x = YtoX(X, Y)  # emission list
	com = combineX(X) # combine X to one list
	trans = transTable(Y, a, modi, join)
	lengthOfNewX = len(newX)
	lengthOfUniY = len(a)

	# Initialised the viterbi
	output = {}
	wholeText = []
	subSentences = []
	newSubSentences = []
	maxSeg = []
	text = []

	for i in range(lengthOfNewX): # for each sentence
		print(i, " sentence")
		tupperware = ()
		subSentences = []
		newSubSentences = []
		maxSeg = []
		layer = []
		word = []

		if(len(newX[i]) > 1):
			"""This part is from START to layer 1"""
			for j in range(lengthOfUniY):
				
				pattern = ("START", a[j])
				piAxB = 0

				if trans[pattern] == 0 or emissionParameters(y2x, com, Y, newX[i][0], a[j]) == 0:
					piAxB = -10000
				else:
					piAxB = log(1) + log(trans[pattern]) + log(emissionParameters(y2x, com, Y, newX[i][0], a[j]))

				word.append(("START", a[j], piAxB)) #[()]
				
			sort = sorted(word, key=lambda x: x[2], reverse=True)
			layer.append(sort) # Here we have the first transition score in order


			print("start layer is done, move on to layer one")
			"""this part is from second layer up to last"""
			for k in range(len((newX)[i])-1):
				word = []

				temp = []

				# from first layer
				if k == 0:

					for n in range(lengthOfUniY):
						
						for m in range(len(layer[k])):
							# print("layer :",layer[k][m])
							pattern = (layer[k][m][1], a[n]) # index[i][k-1][m][1] = previous node
							piAxB = 0

							if trans[pattern] == 0 or emissionParameters(y2x, com, Y, newX[i][k+1], a[n]) == 0:
								piAxB = -10000
							else:
								piAxB = layer[k][m][2] + log(trans[pattern]) + log(emissionParameters(y2x, com, Y, newX[i][k+1], a[n]))

							word.append((layer[k][m][1], a[n], piAxB))

						sort = sorted(word, key=lambda x: x[2], reverse=True) # [(), (), ()] in sorted form

						temp.append(sort[0])
					sort = sorted(temp, key=lambda x: x[2], reverse=True) # one last sort for that layer
					layer.append(sort)




				else:
					for n in range(lengthOfUniY):
						
						for m in range(len(layer[k])):

							pattern = (layer[k][m][1], a[n]) # index[i][k-1][m][1] = previous node
							piAxB = 0

							if trans[pattern] == 0 or emissionParameters(y2x, com, Y, newX[i][k+1], a[n]) == 0:
								piAxB = -10000
							else:
								piAxB = layer[k][m][2] + log(trans[pattern]) + log(emissionParameters(y2x, com, Y, newX[i][k+1], a[n]))
							
							word.append((layer[k][m][1], a[n], piAxB))

						sort = sorted(word, key=lambda x: x[2], reverse=True) # [(), (), ()] in sorted form
						temp.append(sort[0])
					sort = sorted(temp, key=lambda x: x[2], reverse=True)
					layer.append(sort)
					# text.append(layer)
					# print("Length of layer: ",len(layer))					
				
			print("Done with hidden layer")
			"""This is from last layer to 'STOP'"""
			word = []
			for j in range(len(layer[-1])):
				
				pattern = (layer[-1][j][1], "STOP")
				piAxB = 0

				if trans[pattern] == 0:
					piAxB = -10000
				else:
					piAxB = layer[-1][j][2] * log(trans[pattern])

				word.append((layer[-1][j][1], "STOP", piAxB))
			sort = sorted(word, key=lambda x: x[2], reverse=True) # [(), (), ()] in sorted form
			layer.append(list(sort[0]))
			

			path = []
			path.append(layer[-1][0]) # first append the node before 'STOP'

			temp_path = deepcopy(layer)
			temp_path.pop()


			o = len(temp_path)

			while(o > 1):
				# at this stage the output format should be like [[(),(),(),()],[(),(),()],[(),(),()]]
				for j in range(len(temp_path[-1])):

					re = path[0]

					if re == temp_path[-1][j][1]:
						path.insert(0, temp_path[-1][j][0])
						temp_path.pop()
						break
					else:
						pass

				o -= 1
				output[i+1] = path
				# print(output)
		else:
			"""This part is from START to layer 1"""
			for j in range(lengthOfUniY):
				
				pattern = ("START", a[j])
				piAxB = 0

				if trans[pattern] == 0 or emissionParameters(y2x, com, Y, newX[i][0], a[j]) == 0:
					piAxB = -10000
				else:
					piAxB = log(1) + log(trans[pattern]) + log(emissionParameters(y2x, com, Y, newX[i][0], a[j]))

				word.append(("START", a[j], piAxB)) #[()]
				
			sort = sorted(word, key=lambda x: x[2], reverse=True)
			layer.append(sort[0][1]) # Here we have the first transition score in order
			output[i+1] = layer
	return output



def topK(X, Y, newX, K):
	K = int(K)
	a = getUniqueY(Y) # unique Y tag list
	modi = modifiedY(Y)	# Y list with 'START' & 'STOP'
	join = joinY(modi) # join the modified Y tgt to form a list
	y2x = YtoX(X, Y)  # emission list
	com = combineX(X) # combine X to one list
	trans = transTable(Y, a, modi, join)
	lengthOfNewX = len(newX)
	lengthOfUniY = len(a)

	# Initialised the viterbi
	output = {}
	wholeText = []
	subSentences = []
	newSubSentences = []
	maxSeg = []
	text = []

	for i in range(lengthOfNewX): # for each sentence
		print(i, " sentence")
		tupperware = ()
		subSentences = []
		newSubSentences = []
		maxSeg = []
		layer = []
		word = []

		if(len(newX[i]) > 1):
			"""This part is from START to layer 1"""
			for j in range(lengthOfUniY):
				
				pattern = ("START", a[j])
				piAxB = 0

				if trans[pattern] == 0 or emissionParameters(y2x, com, Y, newX[i][0], a[j]) == 0:
					piAxB = -10000
				else:
					piAxB = log(1) + log(trans[pattern]) + log(emissionParameters(y2x, com, Y, newX[i][0], a[j]))

				word.append(("START", a[j], piAxB)) #[()]
				
			sort = sorted(word, key=lambda x: x[2], reverse=True)
			layer.append(sort) # Here we have the first transition score in order


			print("start layer is done, move on to layer one")
			"""this part is from second layer up to last"""
			for k in range(len((newX)[i])-1):
				word = []

				temp = []

				# from first layer
				if k == 0:

					for n in range(lengthOfUniY):
						
						for m in range(len(layer[k])):
							# print("layer :",layer[k][m])
							pattern = (layer[k][m][1], a[n]) # index[i][k-1][m][1] = previous node
							piAxB = 0

							if trans[pattern] == 0 or emissionParameters(y2x, com, Y, newX[i][k+1], a[n]) == 0:
								piAxB = -10000
							else:
								piAxB = layer[k][m][2] + log(trans[pattern]) + log(emissionParameters(y2x, com, Y, newX[i][k+1], a[n]))

							word.append((layer[k][m][1], a[n], piAxB))

						sort = sorted(word, key=lambda x: x[2], reverse=True) # [(), (), ()] in sorted form

						temp.append(sort[K-1])
					sort = sorted(temp, key=lambda x: x[2], reverse=True) # one last sort for that layer
					layer.append(sort)




				else:
					for n in range(lengthOfUniY):
						
						for m in range(len(layer[k])):

							pattern = (layer[k][m][1], a[n]) # index[i][k-1][m][1] = previous node
							piAxB = 0

							if trans[pattern] == 0 or emissionParameters(y2x, com, Y, newX[i][k+1], a[n]) == 0:
								piAxB = -10000
							else:
								piAxB = layer[k][m][2] + log(trans[pattern]) + log(emissionParameters(y2x, com, Y, newX[i][k+1], a[n]))
							
							word.append((layer[k][m][1], a[n], piAxB))

						sort = sorted(word, key=lambda x: x[2], reverse=True) # [(), (), ()] in sorted form
						temp.append(sort[K-1])
					sort = sorted(temp, key=lambda x: x[2], reverse=True)
					layer.append(sort)
					# text.append(layer)
					# print("Length of layer: ",len(layer))					
				
			print("Done with hidden layer")
			"""This is from last layer to 'STOP'"""
			word = []
			for j in range(len(layer[-1])):
				
				pattern = (layer[-1][j][1], "STOP")
				piAxB = 0

				if trans[pattern] == 0:
					piAxB = -10000
				else:
					piAxB = layer[-1][j][2] * log(trans[pattern])

				word.append((layer[-1][j][1], "STOP", piAxB))
			sort = sorted(word, key=lambda x: x[2], reverse=True) # [(), (), ()] in sorted form
			layer.append(list(sort[K-1]))
			

			path = []
			path.append(layer[-1][0]) # first append the node before 'STOP'

			temp_path = deepcopy(layer)
			temp_path.pop()


			o = len(temp_path)

			while(o > 1):
				# at this stage the output format should be like [[(),(),(),()],[(),(),()],[(),(),()]]
				for j in range(len(temp_path[-1])):

					re = path[0]

					if re == temp_path[-1][j][1]:
						path.insert(0, temp_path[-1][j][0])
						temp_path.pop()
						break
					else:
						pass

				o -= 1
				output[i+1] = path
				# print(output)
		else:
			"""This part is from START to layer 1"""
			for j in range(lengthOfUniY):
				
				pattern = ("START", a[j])
				piAxB = 0

				if trans[pattern] == 0 or emissionParameters(y2x, com, Y, newX[i][0], a[j]) == 0:
					piAxB = -10000
				else:
					piAxB = log(1) + log(trans[pattern]) + log(emissionParameters(y2x, com, Y, newX[i][0], a[j]))

				word.append(("START", a[j], piAxB)) #[()]
				
			sort = sorted(word, key=lambda x: x[2], reverse=True)
			layer.append(sort[K-1][1]) # Here we have the first transition score in order
			output[i+1] = layer
	return output

def viterbiImproved(X, Y, newX):
	
	a = getUniqueY(Y) # unique Y tag list
	modi = modifiedY(Y)	# Y list with 'START' & 'STOP'
	join = joinY(modi) # join the modified Y tgt to form a list
	y2x = YtoX(X, Y)  # emission list
	com = combineX(X) # combine X to one list
	trans = transTable(Y, a, modi, join)
	lengthOfNewX = len(newX)
	lengthOfUniY = len(a)

	# Initialised the viterbi
	output = {}
	wholeText = []
	subSentences = []
	newSubSentences = []
	maxSeg = []
	text = []

	for i in range(lengthOfNewX): # for each sentence
		print(i, " sentence")
		tupperware = ()
		subSentences = []
		newSubSentences = []
		maxSeg = []
		layer = []
		word = []

		if(len(newX[i]) > 1):
			"""This part is from START to layer 1"""
			for j in range(lengthOfUniY):
				
				pattern = ("START", a[j])
				piAxB = 0

				if trans[pattern] == 0 or emissionParameters(y2x, com, Y, newX[i][0], a[j]) == 0:
					piAxB = -10000
				else:
					piAxB = log(1) + log(trans[pattern]) + log(emissionParameters(y2x, com, Y, newX[i][0], a[j]))

				word.append(("START", a[j], piAxB)) #[()]
				
			sort = sorted(word, key=lambda x: x[2], reverse=True)
			layer.append(sort) # Here we have the first transition score in order


			print("start layer is done, move on to layer one")
			"""this part is from second layer up to last"""
			for k in range(len((newX)[i])-1):
				word = []

				temp = []

				# from first layer
				if k == 0:

					for n in range(lengthOfUniY):
						
						for m in range(len(layer[k])):
							# print("layer :",layer[k][m])
							pattern = (layer[k][m][1], a[n]) # index[i][k-1][m][1] = previous node
							piAxB = 0

							if trans[pattern] == 0 or emissionParameters(y2x, com, Y, newX[i][k+1], a[n]) == 0:
								piAxB = -10000
							else:
								piAxB = layer[k][m][2] + log(trans[pattern]) + log(emissionParameters(y2x, com, Y, newX[i][k+1], a[n]))

							word.append((layer[k][m][1], a[n], piAxB))

						sort = sorted(word, key=lambda x: x[2], reverse=True) # [(), (), ()] in sorted form

						temp.append(sort[0])
					sort = sorted(temp, key=lambda x: x[2], reverse=True) # one last sort for that layer
					layer.append(sort)




				else:
					for n in range(lengthOfUniY):
						
						for m in range(len(layer[k])):

							pattern = (layer[k][m][1], a[n]) # index[i][k-1][m][1] = previous node
							piAxB = 0

							if trans[pattern] == 0 or emissionParameters(y2x, com, Y, newX[i][k+1], a[n]) == 0:
								piAxB = -10000
							else:
								piAxB = layer[k][m][2] + log(trans[pattern]) + log(emissionParameters(y2x, com, Y, newX[i][k+1], a[n]))
							
							word.append((layer[k][m][1], a[n], piAxB))

						sort = sorted(word, key=lambda x: x[2], reverse=True) # [(), (), ()] in sorted form
						temp.append(sort[0])
					sort = sorted(temp, key=lambda x: x[2], reverse=True)
					layer.append(sort)
					# text.append(layer)
					# print("Length of layer: ",len(layer))					
				
			print("Done with hidden layer")
			"""This is from last layer to 'STOP'"""
			word = []
			for j in range(len(layer[-1])):
				
				pattern = (layer[-1][j][1], "STOP")
				piAxB = 0

				if trans[pattern] == 0:
					piAxB = -10000
				else:
					piAxB = layer[-1][j][2] * log(trans[pattern])

				word.append((layer[-1][j][1], "STOP", piAxB))
			sort = sorted(word, key=lambda x: x[2], reverse=True) # [(), (), ()] in sorted form
			layer.append(list(sort[0]))
			

			path = []
			path.append(layer[-1][0]) # first append the node before 'STOP'

			temp_path = deepcopy(layer)
			temp_path.pop()


			o = len(temp_path)

			while(o > 1):
				# at this stage the output format should be like [[(),(),(),()],[(),(),()],[(),(),()]]
				for j in range(len(temp_path[-1])):

					re = path[0]

					if re == temp_path[-1][j][1]:
						path.insert(0, temp_path[-1][j][0])
						temp_path.pop()
						break
					else:
						pass

				o -= 1
				output[i+1] = path
				# print(output)
		else:
			"""This part is from START to layer 1"""
			for j in range(lengthOfUniY):
				
				pattern = ("START", a[j])
				piAxB = 0

				if trans[pattern] == 0 or emissionParameters(y2x, com, Y, newX[i][0], a[j]) == 0:
					piAxB = -10000
				else:
					piAxB = log(1) + log(trans[pattern]) + log(emissionParameters(y2x, com, Y, newX[i][0], a[j]))

				word.append(("START", a[j], piAxB)) #[()]
				
			sort = sorted(word, key=lambda x: x[2], reverse=True)
			layer.append(sort[0][1]) # Here we have the first transition score in order
			output[i+1] = layer
	return output