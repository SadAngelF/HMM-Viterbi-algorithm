# Import your files here...
import re
import numpy as np
import math
import sys
import copy

def log(x):
    if x == 0:
        return math.log(sys.float_info.min)
    else:
        return math.log(x)

def read_S_files(file):
    '''
    read the Symbol or State file, output the number, the states(or symbols) and transitions(or emissions)
    '''
    dict_t = dict()
    with open(file,'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        line = line.strip('\n')
    N = int(lines[0])         # the number of states
    States = []               # the list of the name of states
    for i in range(N):
        a = lines[i+1]
        States.append(a)
        dict_t[a]=i
    lines = lines[N+1:]
    trans = []                # the transitions between two states
    for line in lines:
        l = list(map(int, line.split()))
        trans.append(l)

    return N, States, trans, dict_t

def read_Q_file(file):
    '''
    read the Q file, output the address to be parsed
    '''
    with open(file,'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        line = line.strip('\n')
    adds = []                # the addresses to be parsed
    for line in lines:
        l = re.split(r"([- , ( ) / & ])", line)
        while ' ' in l:
            l.remove(' ')
        while '' in l:
            l.remove('')
        adds.append(l)

    return adds

# viterbi:
def Viterbit(obs, states, symbol, t_pro, e_pro, start, end):
    path = { s:[] for s in range(len(states))} # init path: path[s] represents the path ends with s
    curr_pro = {}
    for s in range(len(states)):
        if obs[0] in symbol:
            obs_i = symbol.index(obs[0])
        else:
            obs_i = len(symbol)
        #print(len(symbol))
        curr_pro[s] = t_pro[start,s]*e_pro[s,obs_i]
    for i in range(1, len(obs)):
        last_pro = copy.copy(curr_pro)
        #print(last_pro)
        curr_pro = {}
        if obs[i] in symbol:
            obs_i = symbol.index(obs[i])
        else:
            obs_i = len(symbol)
        for curr_state in range(len(states)):
            #print(last_pro)
            #print(t_pro)
            #print(e_pro)
            #print(curr_state)
            #print(obs_i)
            line = [(last_pro[last_state]*t_pro[last_state, curr_state]*e_pro[curr_state, obs_i], last_state) for last_state in range(len(states))] 
            #print(line)
            #probabilities_and_previous_symbols.sort(key=lambda probability_and_previous_symbol: probability_and_previous_symbol[0], reverse=True)
            line.sort(key=lambda a:a[0], reverse=True)
            #print(line)
            #max_pro, last_sta = max(line)
            max_pro, last_sta = line[0]
            #print(line[0])
            #print(last_sta)
            #print(max_pro,last_sta)
            curr_pro[curr_state] = max_pro
            path[curr_state].append(last_sta)
        #print(curr_pro)
        #print(last_pro)
    #print('###')
    #print(path)
    #print('###')
	# find the final largest probability
    max_pro = -1
    max_path = []
    len_obs = len(obs)
    #print(path)
    #print(len_obs)
    for s in range(len(states)):
        if curr_pro[s]*t_pro[s,end] > max_pro:
            temp = s
            max_path = []
            max_path.insert(0,s)
            for i in range(len(obs)-1):
                max_path.insert(0,path[temp][len_obs - 2 - i])
                temp = max_path[0]
            max_pro = curr_pro[s]*t_pro[s,end]
		# print '%s: %s'%(curr_pro[s], path[s]) # different path and their probability
    return max_path, max_pro

#viterbit for problem 3
def Viterbit3(obs, states, symbol, t_pro, e_pro, start, end):
    path = { s:[] for s in range(len(states))} # init path: path[s] represents the path ends with s
    curr_pro = {}
    for s in range(len(states)):
        if obs[0] in symbol:
            obs_i = symbol.index(obs[0])
        else:
            obs_i = len(symbol)
        #print(len(symbol))
        curr_pro[s] = t_pro[start,s]*e_pro[s,obs_i]
        if obs[0][0] != 'U':
            curr_pro[0] = 0
    for i in range(1, len(obs)):
        last_pro = copy.copy(curr_pro)
        #print(last_pro)
        curr_pro = {}
        if obs[i] in symbol:
            obs_i = symbol.index(obs[i])
        else:
            obs_i = len(symbol)
        for curr_state in range(len(states)):
            #print(last_pro)
            #print(t_pro)
            #print(e_pro)
            #print(curr_state)
            #print(obs_i)
            line = [(last_pro[last_state]*t_pro[last_state, curr_state]*e_pro[curr_state, obs_i], last_state) for last_state in range(len(states))] 
            #print(line)
            #probabilities_and_previous_symbols.sort(key=lambda probability_and_previous_symbol: probability_and_previous_symbol[0], reverse=True)
            line.sort(key=lambda a:a[0], reverse=True)
            #print(line)
            #max_pro, last_sta = max(line)
            max_pro, last_sta = line[0]
            #print(line[0])
            #print(last_sta)
            #print(max_pro,last_sta)
            curr_pro[curr_state] = max_pro
            path[curr_state].append(last_sta)
        #print(curr_pro)
        #print(last_pro)
    #print('###')
    #print(path)
    #print('###')
	# find the final largest probability
    max_pro = -1
    max_path = []
    len_obs = len(obs)
    #print(path)
    #print(len_obs)
    for s in range(len(states)):
        if curr_pro[s]*t_pro[s,end] > max_pro:
            temp = s
            max_path = []
            max_path.insert(0,s)
            for i in range(len(obs)-1):
                max_path.insert(0,path[temp][len_obs - 2 - i])
                temp = max_path[0]
            max_pro = curr_pro[s]*t_pro[s,end]
		# print '%s: %s'%(curr_pro[s], path[s]) # different path and their probability
    return max_path, max_pro

def sum_l(list):
    a = str(list[0])
    for i in range(1,len(list)):
        a += ' '
        a += str(list[i])
    return a


def top_k(k, query, states, symbol, trans_matix, emission_matrix, start, end, init_matrix, state_dict, symbol_dict, num_state, num_symbol):
    #print(k)
    # num_state,num_symbol,state_dict,symbol_dict,trans_matix,emission_matrix,init_matrix,query,k
    
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
            dpmatrix[i][1][0][0] = init_matrix[i] * emission_matrix[i, num_symbol]
        dpmatrix[i][1][0][1].append(state_dict['BEGIN'])

    for j in range(2,len(query)+1):
        for i in range(num_state):
            tmp = []
            for m in range(num_state):
                for y in range(k):
                    if query[j-1] in symbol_dict.keys():
                        tmp.append( (dpmatrix[m][j - 1][y][0] * trans_matix[m, i] * emission_matrix[i, symbol_dict[query[j-1]]],dpmatrix[m][j - 1][y][1]+[m]) )
                    else:
                        tmp.append((dpmatrix[m][j - 1][y][0] * trans_matix[m, i] * emission_matrix[i, num_symbol], dpmatrix[m][j - 1][y][1]+[m]))
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
    
    # read State_File
    N, States, trans, state_dict = read_S_files(State_File)

    # read Symbol_File
    M, Symbols, emissions, symbol_dict = read_S_files(Symbol_File)
    #print(state_dict)

    # read Query_File
    adds = read_Q_file(Query_File)


    for i in range(len(States)):
        if States[i] == 'BEGIN':
            begin = i
        if States[i] == 'END':
            end = i 

    # the content above is to read the files, the following will work out it!
    # compute the probablity for trans and emissions:
    A = np.zeros((N, N))
    sum_trans = np.zeros((N))
    for temp in trans:
        sum_trans[temp[0]] += temp[2]
    for temp in trans:
        A[temp[0],temp[1]] = (temp[2]+1)/(sum_trans[temp[0]]+N-1) 
    for i in range(N):
        for j in range(N):
            if A[i,j] == 0 and States[j] != 'BEGIN' and States[i] != 'END':
                A[i,j] = (1)/(sum_trans[i]+N-1) 
    
    
    B = np.zeros((N,M+1))
    sum_emiss = np.zeros((N))
    for temp in emissions:
        sum_emiss[temp[0]] += temp[2]
    for temp in emissions:
        B[temp[0],temp[1]] = ((temp[2]+1))/(sum_emiss[temp[0]] + M + 1)
    for i in range(N):
        if i != end and i != begin:
            B[i,M] = 1/(sum_emiss[i] + M + 1)
    for i in range(N):
        for j in range(M):
            if B[i,j] == 0 and States[i] != 'BEGIN' and States[i] != 'END':
                B[i,j] = 1/(sum_emiss[i] + M + 1)
    #with open("./abc.txt",'w') as f:
    #    f.write(str(B))
    #print(N,M)
    #print(B)

    # viterbi algorithm:
    #print(adds)

    final = []
    for obs in adds:
        #pass
        #print(begin)
        max_path, max_pro = Viterbit(obs, States, Symbols, A, B, begin, end)
        #print(max_path, np.log(max_pro))
        #max_path.append(np.log(max_pro))
        max_path.insert(0,begin)
        max_path.append(end)
        max_path.append(np.log(max_pro))
        final.append(max_path)
    return final


# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    
    # read State_File
    N, States, trans, state_dict = read_S_files(State_File)

    # read Symbol_File
    M, Symbols, emissions, symbol_dict = read_S_files(Symbol_File)
    #print(state_dict)

    # read Query_File
    adds = read_Q_file(Query_File)


    for i in range(len(States)):
        if States[i] == 'BEGIN':
            begin = i
        if States[i] == 'END':
            end = i 

    # the content above is to read the files, the following will work out it!
    # compute the probablity for trans and emissions:
    A = np.zeros((N, N))
    sum_trans = np.zeros((N))
    for temp in trans:
        sum_trans[temp[0]] += temp[2]
    for temp in trans:
        A[temp[0],temp[1]] = (temp[2]+1)/(sum_trans[temp[0]]+N-1) 
    for i in range(N):
        for j in range(N):
            if A[i,j] == 0 and States[j] != 'BEGIN' and States[i] != 'END':
                A[i,j] = (1)/(sum_trans[i]+N-1) 
    
    
    B = np.zeros((N,M+1))
    sum_emiss = np.zeros((N))
    for temp in emissions:
        sum_emiss[temp[0]] += temp[2]
    for temp in emissions:
        B[temp[0],temp[1]] = ((temp[2]+1))/(sum_emiss[temp[0]] + M + 1)
    for i in range(N):
        if i != end and i != begin:
            B[i,M] = 1/(sum_emiss[i] + M + 1)
    for i in range(N):
        for j in range(M):
            if B[i,j] == 0 and States[i] != 'BEGIN' and States[i] != 'END':
                B[i,j] = 1/(sum_emiss[i] + M + 1)
    #with open("./abc.txt",'w') as f:
    #    f.write(str(B))
    #print(N,M)
    #print(B)
    
    # viterbi algorithm:
    #print(adds)
    for i in range(len(States)):
        if States[i] == 'BEGIN':
            begin = i
        if States[i] == 'END':
            end = i 

    init_matrix = np.matrix
    for i in range(N):
        if i == begin:
            init_matrix = A[i,:]


    final = []
    for obs in adds:
        #pass
        #print(begin)
        top_k_path= top_k(k, obs, States, Symbols, A, B, begin, end, init_matrix,state_dict, symbol_dict, N, M)
        final.extend(top_k_path)
        #print(top_k_path)
        #max_path.append(np.log(max_pro))
        #final.append(max_path)
    return final

# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    
    # read State_File
    N, States, trans, state_dict = read_S_files(State_File)

    # read Symbol_File
    M, Symbols, emissions, symbol_dict = read_S_files(Symbol_File)
    #print(state_dict)

    # read Query_File
    adds = read_Q_file(Query_File)


    for i in range(len(States)):
        if States[i] == 'BEGIN':
            begin = i
        if States[i] == 'END':
            end = i 

    smooth = 1
    # the content above is to read the files, the following will work out it!
    # compute the probablity for trans and emissions:
    A = np.zeros((N, N))
    sum_trans = np.zeros((N))
    for temp in trans:
        sum_trans[temp[0]] += temp[2]
    for temp in trans:
        A[temp[0],temp[1]] = (temp[2]+smooth)/(sum_trans[temp[0]]+smooth*N-1) 
    for i in range(N):
        for j in range(N):
            if A[i,j] == 0 and States[j] != 'BEGIN' and States[i] != 'END':
                A[i,j] = smooth/(sum_trans[i]+smooth*N-1) 
    
    
    B = np.zeros((N,M+1))
    sum_emiss = np.zeros((N))
    for temp in emissions:
        sum_emiss[temp[0]] += temp[2]
    for temp in emissions:
        B[temp[0],temp[1]] = ((temp[2]+smooth))/(sum_emiss[temp[0]] + smooth*M + 1)
    for i in range(N):
        if i != end and i != begin:
            B[i,M] = smooth/(sum_emiss[i] + smooth*M + 1)
    for i in range(N):
        for j in range(M):
            if B[i,j] == 0 and States[i] != 'BEGIN' and States[i] != 'END':
                B[i,j] = smooth/(sum_emiss[i] +smooth*M + 1)
    #with open("./abc.txt",'w') as f:
    #    f.write(str(B))
    #print(N,M)
    #print(B)

    # viterbi algorithm:
    #print(adds)
    for i in range(len(States)):
        if States[i] == 'BEGIN':
            begin = i
        if States[i] == 'END':
            end = i 

    final = []
    for obs in adds:
        #pass
        #print(begin)
        max_path, max_pro = Viterbit3(obs, States, Symbols, A, B, begin, end)
        #print(max_path, np.log(max_pro))
        #max_path.append(np.log(max_pro))
        max_path.insert(0,begin)
        max_path.append(end)
        max_path.append(np.log(max_pro))
        final.append(max_path)
    return final