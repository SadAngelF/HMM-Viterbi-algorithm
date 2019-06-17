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
    with open(file,'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        line = line.strip('\n')
    N = int(lines[0])         # the number of states
    States = []               # the list of the name of states
    for i in range(N):
        States.append(lines[i+1])
    lines = lines[N+1:]
    trans = []                # the transitions between two states
    for line in lines:
        l = list(map(int, line.split()))
        trans.append(l)

    return N, States, trans

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
        l = re.split(r"([, ( ) / - & ])", line)
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


def sum_l(list):
    a = str(list[0])
    for i in range(1,len(list)):
        a += ' '
        a += str(list[i])
    return a


def top_k(m, obs, states, symbol, t_pro, e_pro, start, end):
    scores = {}
    n = len(obs)
    for k in range(n + 2):
        scores[k] = {}
        for state in range(len(states)):
            scores[k][state] = []
    if obs[0] in symbol:
        obs_i = symbol.index(obs[0])
    else:
        obs_i = len(symbol)
    for s in range(len(states)): 
        scores[0][s]= [(0, "NA")]
        scores[1][s] = [(log(t_pro[start,s]) + log(e_pro[s, obs_i]), "START")]
    scores[0]["STOP"]= [(0, "NA")]
    scores[0]["START"]= [(1, "NA")]

    for k in range(2, n + 1):
        if obs[k-1] in symbol:
            obs_i = symbol.index(obs[k-1])
        else:
            obs_i = len(symbol)
        for v in range(len(states)):
            # Get the k highest probability scores and associated previous symbols
            probabilities_and_previous_symbols = [(max([score_and_symbol[0] + log(t_pro[u][v]) + log(e_pro[v, obs_i]) for score_and_symbol in scores[k-1][u]]), u) for u in range(len(states))]
            # Eliminate duplicate scores
            probabilities_and_previous_symbols.sort(key=lambda probability_and_previous_symbol: probability_and_previous_symbol[0], reverse=True)
            scores[k][v] = probabilities_and_previous_symbols[0:m]

    # Final entry
    probabilities_and_previous_symbols = [(score_and_symbol[0] + log(t_pro[u,end]), u) for u in range(len(states)) for score_and_symbol in scores[k-1][u]]
    probabilities_and_previous_symbols.sort(key=lambda probability_and_previous_symbol: probability_and_previous_symbol[0], reverse=True)
    scores[n+1][end] = probabilities_and_previous_symbols[0:m]
    #print(scores[n+1][end])
    # Get matrix of top m paths at each step in the sequence
    # The [k][symbol] element gives a list of tuples of the top-k paths to
    # the kth observation if it is tagged with symbol, and their score
    top_m_scores_and_paths= {}
    top_m_scores_and_paths[n+1] = {}
    for k in range(1, n + 1):
        top_m_scores_and_paths[k] = {}
        for state in range(len(states)):
            top_m_scores_and_paths[k][state] = []

    # Set base case
    for state in range(len(states)):
        top_m_scores_and_paths[1][state] = [(log(t_pro[start,state]), [start])]

    for k in range(2, n + 1):
        for v in range(len(states)):
            probabilities_and_paths = [(score_and_symbol[0] + score_and_path[0], score_and_path[1] + [score_and_symbol[1]]) for score_and_symbol in scores[k][v] for score_and_path in top_m_scores_and_paths[k-1][score_and_symbol[1]]]
            #print(probabilities_and_paths)
            #a= input()

            # Eliminate duplicate paths
            #print(probabilities_and_paths)
            probabilities_and_paths = set([(probability_and_path[0], sum_l(probability_and_path[1])) for probability_and_path in probabilities_and_paths])
            probabilities_and_paths = [(probability_and_path[0], probability_and_path[1].split()) for probability_and_path in probabilities_and_paths]
            probabilities_and_paths.sort(key=lambda probability_and_path: probability_and_path[0], reverse=True)
            top_m_scores_and_paths[k][v] = probabilities_and_paths[0:m]

    # Final entry
    probabilities_and_paths = [(score_and_symbol[0] + score_and_path[0], score_and_path[1] + [score_and_symbol[1]]) for score_and_symbol in scores[n+1][end] for score_and_path in top_m_scores_and_paths[n][score_and_symbol[1]]]
    probabilities_and_paths.sort(key=lambda probability_and_path: probability_and_path[0], reverse=True)
    top_m_scores_and_paths[n+1][end] = probabilities_and_paths[0:m]

    #top_m_scores_and_paths[n+1]["STOP"]

    return top_m_scores_and_paths[n+1][end]




# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    
    # read State_File
    N, States, trans = read_S_files(State_File)

    # read Symbol_File
    M, Symbols, emissions = read_S_files(Symbol_File)

    # read Query_File
    adds = read_Q_file(Query_File)

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
                A[i,j] = (temp[2]+1)/(sum_trans[temp[0]]+N-1) 
    #print(A)
    
    B = np.zeros((N,M+1))
    sum_emiss = np.zeros((N))
    for temp in emissions:
        sum_emiss[temp[0]] += temp[2]
    for temp in emissions:
        B[temp[0],temp[1]] = ((temp[2]+1))/(sum_emiss[temp[0]] + M + 1)
    for i in range(N):
        B[i,M] = 1/(sum_emiss[i] + M + 1)
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
    N, States, trans = read_S_files(State_File)

    # read Symbol_File
    M, Symbols, emissions = read_S_files(Symbol_File)

    # read Query_File
    adds = read_Q_file(Query_File)

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
                A[i,j] = (temp[2]+1)/(sum_trans[temp[0]]+N-1) 
    
    B = np.zeros((N,M+1))
    sum_emiss = np.zeros((N))
    for temp in emissions:
        sum_emiss[temp[0]] += temp[2]
    for temp in emissions:
        B[temp[0],temp[1]] = ((temp[2]+1))/(sum_emiss[temp[0]] + M + 1)
    for i in range(N):
        B[i,M] = 1/(sum_emiss[i] + M + 1)
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
        top_k_path= top_k(k, obs, States, Symbols, A, B, begin, end)
        for temp in top_k_path:
            path = []
            for i in temp[1]:
                path.append(int(i))
            path.append(end)
            path.append(temp[0])
            final.append(path)
            
        #max_path.append(np.log(max_pro))
        #final.append(max_path)
    return final

# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    pass # Replace this line with your implementation...
