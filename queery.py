import re


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



Query_File ='./queer'
adds = read_Q_file(Query_File)
print(adds)