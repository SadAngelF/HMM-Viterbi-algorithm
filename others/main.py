"""
Singapore University of Technology and Design
Term 6 Machine Learning Project



This is the main file to be excuted for this project
"""

import sys
import os
import platform
from algorithm import *

from outputFile import export, exportDictionary, exportW, exportDictionaryW

# get file input
# argv[1]: train
#argv[2]: dev.in

if __name__ == '__main__':

	print("This program will generate a sequence labelling model using Hidden Markov Model(HMM)\n")



	# determine which platform the machine is running on
	if platform.system() == 'Windows':
		train = os.getcwd() + '\\' + sys.argv[1]
		data = os.getcwd() + '\\' + sys.argv[2]

		print("Getting input data...\n\n")

		# read file using the path we got from above
		X = []
		Y = []
		newX = []
		readW(train, X, Y)
		readNewXW(data, newX)

		print("Done importing data.")

		while(True):
		
			print("Part2 sentiment analysis: 2\nPart3 viterbi: 3 \nPart4 top-k: 4\nPart5 design challenge\nYou can quit the program by pressing 'q'")
			part = input("Enter your choice(2/3/4/5): ")
			if(part == '2'):
				out = part2analysis(X, Y, newX)
				exportW(out, newX, part)
			elif(part == '3'):
				out = viterbi(X, Y, newX)
				exportDictionaryW(out, newX, part)
			elif(part == '4'):
				K = input("Which top-K best path you want to get?(1-5)\n")
				out = topK(X, Y, newX, K)
				exportDictionaryW(out, newX, part)
			elif(part == '5'):
				out = viterbiImproved(X, Y, newX)
				exportDictionaryW(out, newX, part)
			elif(part == 'q'):
				exit()

	else:
		train = os.getcwd() + '/' + sys.argv[1]
		data = os.getcwd() + '/' + sys.argv[2]

		print("Getting input data...\n\n")

		# read file using the path we got from above
		X = []
		Y = []
		newX = []
		read(train, X, Y)
		readNewX(data, newX)

		print("Done importing data.")
	
		while(True):
		
			print("Part2 sentiment analysis: 2\nPart3 viterbi: 3 \nPart4 top-k: 4\nPart5 design challenge\nYou can quit the program by pressing 'q'")
			part = input("Enter your choice(2/3/4/5): ")
			if(part == '2'):
				out = part2analysis(X, Y, newX)
				export(out, newX, part)
			elif(part == '3'):
				out = viterbi(X, Y, newX)
				exportDictionary(out, newX, part)
			elif(part == '4'):
				K = input("Which top-K best path you want to get?(1-5)\n")
				out = topK(X, Y, newX, K)
				exportDictionary(out, newX, part)
			elif(part == '5'):
				out = viterbiImproved(X, Y, newX)
				exportDictionary(out, newX, part)
			elif(part == 'q'):
				exit()

