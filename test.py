import submission1 as submission


# How we test your implementation for Q1
# Model Parameters...
'''
State_File ='./dev_set/State_File'
Symbol_File='./dev_set/Symbol_File'
Query_File ='./dev_set/Query_File'

State_File ='./toy_example/State_File'
Symbol_File='./toy_example/Symbol_File'
Query_File ='./toy_example/Query_File'

viterbi_result = submission.viterbi_algorithm(State_File, Symbol_File, Query_File)


# How we test your implementation for Q2
# Model Parameters...
State_File ='./dev_set/State_File'
Symbol_File='./dev_set/Symbol_File'
Query_File ='./dev_set/Query_File'
k = 5 #(It can be any valid integer...)
top_k_result = submission.top_k_viterbi(State_File, Symbol_File, Query_File, k)
'''


# How we test your implementation for Q3.
# Model Parameters...
State_File ='./dev_set/State_File'
Symbol_File='./dev_set/Symbol_File'
Query_File ='./dev_set/Query_File'
advanced_result = submission.advanced_decoding(State_File, Symbol_File, Query_File)


# Example output for Q1.
for row in advanced_result:
    with open("./abc.txt",'a') as f:
        for i in range(len(row)-2):
            f.write(str(row[i])+' ')
        f.write(str(row[i+1]))
        f.write('\n')
    print(row)
    pass