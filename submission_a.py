import numpy as np
from collections import defaultdict
import re

def file_reader(State_File, Symbol_File,q_3):
    num_state = 0
    state_dict = dict()
    trans_dict = dict()
    symbol_dict = dict()
    emission_dict = dict()
    trans_matix = np.matrix
    emission_matrix = np.matrix
    init_matrix = np.matrix
    if q_3:
        smooth = 0.0001
    else:
        smooth = 1
    with open(State_File, 'r') as state_f:
        num_state = int(state_f.readline().rstrip())
        for i in range(num_state):
            line = state_f.readline().rstrip()
            state_dict[line] = i
        while True:
            line = state_f.readline()
            if not line:
                break
            line_ = line.rstrip().split(' ')
            line_ = list(map(int, line_))
            trans_dict.setdefault(line_[0], {})
            trans_dict[line_[0]][line_[1]] = line_[2]

        trans_matix = np.zeros((num_state,num_state),float)
        for i in range(num_state):
            if state_dict['END'] != i:
                if i in trans_dict.keys():
                    sum_v = sum(trans_dict[i].values())
                else:
                    sum_v = 0
                for j in range(num_state):
                    if j == state_dict['BEGIN']:
                        continue
                    if j in trans_dict[i].keys() and i in trans_dict.keys():
                        trans_matix[i,j] = (smooth + trans_dict[i][j])/(sum_v+(smooth*num_state)-1)
                    else:
                        trans_matix[i,j] = smooth/(sum_v+(smooth*num_state)-1)
                if i == state_dict['BEGIN']:
                    init_matrix=trans_matix[i,:]
    # print(trans_matix)
    # print(init_matrix)

    with open(Symbol_File, 'r') as symbol_f:
        num_symbol = int(symbol_f.readline().rstrip())
        for i in range(num_symbol):
            line = symbol_f.readline().rstrip()
            symbol_dict[line] = i
        while True:
            line = symbol_f.readline()
            if not line:
                break
            line_ = line.rstrip().split(' ')
            line_ = list(map(int, line_))
            emission_dict.setdefault(line_[0], {})
            emission_dict[line_[0]][line_[1]] = line_[2]

        emission_matrix = np.zeros((num_state,num_symbol+1))
        for i in range(num_state):
            if state_dict['BEGIN'] != i and state_dict['END'] != i:
                if i in emission_dict.keys():
                    sum_v = sum(emission_dict[i].values())
                else:
                    sum_v = 0
                for j in range(num_symbol+1):
                    if j in emission_dict[i].keys() and i in emission_dict.keys():
                        emission_matrix[i,j] = (smooth+emission_dict[i][j])/(sum_v+smooth*num_symbol+1)
                    else:
                        emission_matrix[i,j] = smooth/(sum_v+smooth*num_symbol+1)
    # print(emission_matrix)
    return num_state,num_symbol,state_dict,symbol_dict,trans_matix,emission_matrix,init_matrix


def viterbi_alg(num_state,num_symbol,state_dict,symbol_dict,trans_matix,emission_matrix,init_matrix,query,k):
    # DP
    dpmatrix = []
    for i in range(num_state):
        dpmatrix.append([])
        for j in range(len(query)+2):
            # dpmatrix[i].append([(0.0,[])]*k)
            dpmatrix[i].append([])
            for m in range(k):
                dpmatrix[i][j].append([])
                dpmatrix[i][j][m].append(0.0)
                dpmatrix[i][j][m].append([])
    # for row in dpmatrix:
    #     print(row)
    # init
    dpmatrix[state_dict['BEGIN']][0][0] = [1,[]]
    for i in range(num_state):
        # print(dpmatrix[i][1][0][0])
        if query[0] in symbol_dict.keys():
            dpmatrix[i][1][0][0] = init_matrix[i] * emission_matrix[i, symbol_dict[query[0]]]
        else:
            dpmatrix[i][1][0][0] = init_matrix[i] * emission_matrix[i, symbol_dict['UNK']]
        dpmatrix[i][1][0][1].append(state_dict['BEGIN'])

    for j in range(2,len(query)+1):
        for i in range(num_state):
            tmp = []
            for m in range(num_state):
                for y in range(k):
                    if query[j-1] in symbol_dict.keys():
                        tmp.append( (dpmatrix[m][j - 1][y][0] * trans_matix[m, i] * emission_matrix[i, symbol_dict[query[j-1]]],dpmatrix[m][j - 1][y][1]+[m]) )
                    else:
                        tmp.append((dpmatrix[m][j - 1][y][0] * trans_matix[m, i] * emission_matrix[i, symbol_dict['UNK']], dpmatrix[m][j - 1][y][1]+[m]))
            tmp = sorted(tmp, key=lambda x: x[0], reverse=True)
            #print('tmp',tmp)
            for n in range(k):
                #print(tmp[n])
                dpmatrix[i][j][n][0] = tmp[n][0]
                dpmatrix[i][j][n][1].extend(tmp[n][1])

    for i in range(num_state):
        for j in range(k):
            end_matrix = trans_matix[:,state_dict['END']]
            dpmatrix[i][len(query)+1][j][0] = end_matrix[i] * dpmatrix[i][len(query)][j][0]
            dpmatrix[i][len(query) + 1][j][1].extend(dpmatrix[i][len(query)][j][1]+[i])

    # for row in dpmatrix:
    #     print(row)

    r = []


    for i in range(num_state):
        r.extend(dpmatrix[i][len(query)+1])
    r = sorted(r,key=lambda x: x[0], reverse=True)
    # print(r)
    result = []
    result_l = []
    end = state_dict['END']
    for i in range(k):
        result = r[i][1] +[end]+[np.log(r[i][0])]
        result_l.append(result)
    # print(result_l)
    return result_l


# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    num_state,num_symbol,state_dict,symbol_dict,trans_matix,emission_matrix,init_matrix = file_reader(State_File, Symbol_File,False)
    #with open("./abc1.txt",'w') as f:
    #    f.write(str(emission_matrix))
    pattern = r"[0-9A-Za-z.]+|[,&-/()]"
    result = []
    with open(Query_File, 'r') as query_f:
        while True:
            line = query_f.readline().rstrip()
            if not line:
                break
            query = re.compile(pattern).findall(line)
            # print('query', query)
            symbol_dict["UNK"] = num_symbol

            result.extend(viterbi_alg(num_state,num_symbol,state_dict,symbol_dict,trans_matix,emission_matrix,init_matrix,query,1))
    return result

# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    num_state, num_symbol, state_dict, symbol_dict, trans_matix, emission_matrix, init_matrix = file_reader(State_File, Symbol_File,False)
    pattern = r"[0-9A-Za-z.]+|[,&-/()]"
    result = []
    with open(Query_File, 'r') as query_f:
        while True:
            line = query_f.readline().rstrip()
            if not line:
                break
            query = re.compile(pattern).findall(line)
            # print('query', query)
            symbol_dict["UNK"] = num_symbol
            #print('### ')
            #print(init_matrix)
            result.extend(viterbi_alg(num_state, num_symbol, state_dict, symbol_dict, trans_matix, emission_matrix,init_matrix, query, k))
    return result

# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    num_state, num_symbol, state_dict, symbol_dict, trans_matix, emission_matrix, init_matrix = file_reader(State_File,Symbol_File,True)
    pattern = r"[0-9A-Za-z.]+|[,/-()&]"
    result = []
    with open(Query_File, 'r') as query_f:
        while True:
            line = query_f.readline().rstrip()
            if not line:
                break
            query = re.compile(pattern).findall(line)
            # print('query', query)
            symbol_dict["UNK"] = num_symbol
            result.extend(viterbi_alg(num_state, num_symbol, state_dict, symbol_dict, trans_matix, emission_matrix, init_matrix, query, 1))
    return result
