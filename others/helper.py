from copy import deepcopy
import re

"""This function filter noises in X dataset and update Y tag list accordingly"""
def clean(X, Y):
    
    for i in range(len(X)):
        for j in range(len(X[i])):
            
            symbol = re.match(r'\w+', X[i][j])
            url = re.match(r'\/([^\/]*\.+[^\/]*)$', X[i][j])

            if((symbol != None and Y[i][j]!='O') or (url != None and Y[i][j]!='O')):
                Y[i][j] = 'O'

            if('#' in X[i][j] or '@' in X[i][j] and Y[i][j] != 'O'):
                Y[i][j] = 'O'

"""This function read train data"""
def read(file, X, Y):
    words = list()
    labels = list()
    with open(file) as infile:
        for line in infile:
            if line == '\n':
                # print(line)
                X.append(words)
                Y.append(labels)
                words = list()
                labels = list()
            else:
                word, label = line.strip().split(' ')
                # print(word)
                words.append(word)
                labels.append(label)


"""This function read testing data"""
def readNewX(testdata, newX):
    words = list()
    # labels = list()
    with open(testdata) as infile:
        for line in infile:
            if line == '\n':

                newX.append(words)
                
                words = list()

            else:
                word = line.strip()
                # print(word)
                words.append(word)


"""This function read train data"""
def readW(file, X, Y):
    words = list()
    labels = list()
    with open(file, encoding='utf8') as infile:
        for line in infile:
            if line == '\n':
                # print(line)
                X.append(words)
                Y.append(labels)
                words = list()
                labels = list()
            else:
                word, label = line.strip().split(' ')
                # print(word)
                words.append(word)
                labels.append(label)


"""This function read testing data"""
def readNewXW(testdata, newX):
    words = list()
    # labels = list()
    with open(testdata, encoding='utf8') as infile:
        for line in infile:
            if line == '\n':

                newX.append(words)
                
                words = list()

            else:
                word = line.strip()
                # print(word)
                words.append(word)

"""This function combine x into a list"""
def combineX(X):
    sq = list()
    for word_seq in X:
        for word in word_seq:
            sq.append(word)
    
    return sq

"""This function output the count of emission from y to x"""
def countY2X(YX, x, y):
    pattern = y + "-->" + x
    return YX.count(pattern)

"""This function output unique Y list"""
def getUniqueY(Y):
    temp = deepcopy(Y)
    a = set(y for i in temp for y in i)
    return list(a)

"""This function put  Y transition into a list"""
def joinY(newY):
    sq = []

    for y in newY:
        # print(y)
        for t in range(len(y)-1):
        	sq.append(y[t]+"-->"+y[t+1])
    return sq

"""This function join each y and x respectively and form a string"""
def YtoX(X, Y):
    Y2X = []
    count = 0
    # pattern = x + "-->" + y
    for i in range(len(X)):
        for j in range(len(X[i])):
            Y2X.append(Y[i][j] + "-->" + X[i][j])
    
    # count  =Y2X.count(pattern)
    return Y2X

"""This function adds 'START' and 'STOP to the Y sequence"""
def modifiedY(Y):
    temp = deepcopy(Y)
    
    for i in range(len(Y)):
        temp[i].insert(0, "START")
        temp[i].insert(len(temp[i]) + 1, "STOP")

    return temp

"""This function count the total # of y in a tag sequence"""
def countY(Y, y):
    num_Y = 0
    for sub_y in Y:
        num_Y += sub_y.count(y)
    return num_Y

"""This function check if the new x appear in the training X set"""
def checkNewXinX(X, x):
    if x in combineX(X):
        return True
    else:
        return False

def transitionParameter(Y, prev_y, current_y, modi, join):
    
    # probability of transition that we want to get
    transition = prev_y + "-->" +current_y

    # A modified Y sequence with 'START' & 'STOP' state added
    newY = modi
  
    # get the total number of prev Y
    count_prevY = countY(newY,prev_y)
    # print(count_prevY)

    # get the number of transition count Yi-1 -> Yi
    transition_count = join.count(transition)
    # print(transition_count)
    

    return transition_count/count_prevY

"""This function will form a transition params table"""
def transTable(Y, uniY, modi, join):
    """TODO:
        Add 'START' & 'STOP' for the 2 unique Y seq
        and form a probability table"""
    

    # unique Y sequence at y-axis
    uniY_vertical = deepcopy(uniY)
    uniY_vertical.insert(0, "START") 

    # unique X sequence at x-axis
    uniY_horizontal = deepcopy(uniY)
    uniY_horizontal.insert(len(uniY_vertical) + 1, "STOP")

    # A dict to store the state(i-1, i) mapped with prob
    prob_dict = {}

    for prev_state in uniY_vertical:

        for current_state in uniY_horizontal:
            state = (prev_state, current_state)
            # print(state)
            prob_dict[state] = transitionParameter(Y, prev_state, current_state, modi, join) 
            

    
    return prob_dict


"""Function that estimates the emission parameters 
from the training set using MLE"""
def part2EmissionParameter(YX, Y, x, y):
  return countY2X(YX, x, y)/((countY(Y, y)))

def emissionXExist(YX, Y, x, y):
    # if(countY2X(YX, x, y) == 0):
    #     return 0
    # else:
    return countY2X(YX, x, y)/((countY(Y, y)+1))

def emissionXnotExist(Y, y):
    return 1/(countY(Y, y) + 1)


"""This function calculate the probability for
the emission from each state to the observation"""
def emissionParameters(YX, comX, Y, x, y):

    # Check if x apper in the training set
    if x in comX:
        output = emissionXExist(YX, Y, x, y)
    else:
        output = emissionXnotExist(Y, y)
    return output

def transList(Y, uniY, modi, join):
    """TODO:
        Add 'START' & 'STOP' for the 2 unique Y seq
        and form a probability table"""
    

    # unique Y sequence at y-axis
    uniY_vertical = deepcopy(uniY)
    # uniY_vertical.insert(0, "START")

    # unique X sequence at x-axis
    uniY_horizontal = deepcopy(uniY)
    # uniY_horizontal.insert(len(uniY_vertical) + 1, "STOP")

    # A dict to store the state(i-1, i) mapped with prob
    prob_dict = []

    for prev_state in uniY_vertical:

        for current_state in uniY_horizontal:
            state = (prev_state, current_state, transitionParameter(Y, prev_state, current_state, modi, join))
            prob_dict.append(state)

    return prob_dict