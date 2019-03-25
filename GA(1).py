import numpy as np
import scipy as sp
import scipy.io as io
import random as rd
import copy
import math as mt

r = 6
t = 181
m = 30
n = 40
K = 3000
Lambda = 3
num = 50

def get_right_index(a,b,c,d):
    if(a < 0):
        a = 0

    if(b > 30):
        b = 30
    if(c < 0):
        c = 0
    if(d > 40):
        d = 40
    return [a,b,c,d]

def get_Mat_P1(index):
    i1 = 2
    i2 = 1
    i3 = 0

    i = index[0] - 1
    j = index[1] - 1
    matrix = np.zeros((m,n))
    a = i - i1
    b = i + 1 + i1
    c = j - i2
    d = j + 1 + i2
    a, b, c, d = get_right_index(a,b,c,d)

    matrix[a:b, c:d] = 1
    a, b, c, d = get_right_index(i - i2, i + 1 + i2, j - i1, j + 1 + i1)
    matrix[a:b, c:d] = 1

    a, b, c, d = get_right_index(i - i2, i + 1 + i2, j - i2, j + i2 + 1)
    matrix[a:b, c:d] = 2
    matrix[i, j] = 3
    return matrix


def get_Mat_P2(index):
    i = index[0] - 1
    j = index[1] - 1
    matrix = np.zeros((m,n))

    a, b, c, d = get_right_index(i - 3, i + 3, j - 2, j + 3)
    matrix[a:b, c:d] = 1

    a, b, c, d = get_right_index(i - 2, i + 2, j - 3, j + 4)
    matrix[a:b, c:d] = 1
    a, b, c, d = get_right_index(i - 1, i + 1, j - 2, j + 3)
    matrix[a:b, c:d] = 2
    a, b, c, d = get_right_index(i - 2, i + 2, j - 1, j + 2)
    matrix[a:b, c:d] = 2
    matrix[i, j] = 3
    return matrix

def get_Mat_P3(index):
    i = index[0] - 1
    j = index[1] - 1
    matrix = np.zeros((m,n))

    a, b, c, d = get_right_index(i - 2, i + 3, j - 6, j - 1)
    matrix[a:b, c:d] = 1

    a, b, c, d = get_right_index(i - 1, i + 2, j - 2, j)
    matrix[a:b, c:d] = 2
    matrix[i, j] = 3
    return matrix

def get_Mat_P4(index):
    i = index[0] - 1
    j = index[1] - 1
    matrix = np.zeros((m,n))

    a, b, c, d = get_right_index(i - 6, i - 1, j - 2, j + 3)
    matrix[a:b, c:d] = 1

    a, b, c, d = get_right_index(i - 2, i , j - 1, j + 2)
    matrix[a:b, c:d] = 2
    matrix[i, j] = 3
    return matrix

def get_Mat_P5(index):
    i = index[0] - 1
    j = index[1] - 1
    matrix = np.zeros((m, n))

    a, b, c, d = get_right_index(i - 2, i + 3, j + 2, j + 7)
    matrix[a:b, c:d] = 1

    a, b, c, d = get_right_index(i - 1, i + 2, j + 1, j + 3)
    matrix[a:b, c:d] = 2
    matrix[i, j] = 3
    return matrix

def get_Mat_P6(index):
    i = index[0] - 1
    j = index[1] - 1
    matrix = np.zeros((m,n))

    a, b, c, d = get_right_index(i + 2, i + 7, j - 2, j + 3)
    matrix[a:b, c:d] = 1

    a, b, c, d = get_right_index(i + 1, i + 3, j - 1, j + 2)
    matrix[a:b, c:d] = 2
    matrix[i, j] = 3
    return matrix


def getNext_C():
    next1 = np.random.randint(0,7)
    return next1

def init():
    C = []
    for i in range(num):
        temp = []
        for j in range(t):
            temp.append(getNext_C())
        temp = np.array(temp)
        C.append(temp)
    return C


def get_P(c,index):
    if (c == 1):
        return get_Mat_P1(index)
    if (c == 2):
        return get_Mat_P2(index)
    if (c == 3):
        return get_Mat_P3(index)
    if (c == 4):
        return get_Mat_P4(index)
    if (c == 5):
        return get_Mat_P5(index)
    if (c == 6):
        return get_Mat_P6(index)
    else:
        return np.zeros((m,n))

M = np.ones((m,n))
M[:,:] = 2
M[:6,25:] = 0
M[8:,25:] = 3
M[:16,18:23] = 1
M[18:,18:23] = 1
M[18:,2:16] = 3
M[:16,2:16] = 3


T_index = [[2,2],[2,5],[2,8],[2,11],[2,14],[2,17],[2,20],[2,23],[4,1],[4,4],
    [4,7],[4,10],[4,13],[4,16],[4,19],[4,22],[4,25],[6,2],[6,5],[6,8],
    [6,11],[6,14],[6,17],[6,20],[6,23],[8,1],[8,4],[8,7],[8,10],[8,13],
    [8,16],[8,19],[8,22],[8,25],[8,28],[8,31],[8,34],[8,37],[8,40],[10,2],
    [10,5],[10,8],[10,11],[10,14],[10,17],[10,20],[10,23],[10,26],[10,29],[10,32],
    [10,35],[10,38],[12,1],[12,4],[12,7],[12,10],[12,13],[12,16],[12,19],[12,22],
    [12,25],[12,28],[12,31],[12,34],[12,37],[12,40],[14,2],[14,5],[14,8],[14,11],
    [14,14],[14,17],[14,20],[14,23],[14,26],[14,29],[14,32],[14,35],[14,38],[16,1],
    [16,4],[16,7],[16,10],[16,13],[16,16],[16,19],[16,22],[16,25],[16,28],[16,31],
    [16,34],[16,37],[16,40],[18,2],[18,5],[18,8],[18,11],[18,14],[18,17],[18,20],
    [18,23],[18,26],[18,29],[18,32],[18,35],[18,38],[20,1],[20,4],[20,7],[20,10],
    [20,13],[20,16],[20,19],[20,22],[20,25],[20,28],[20,31],[20,34],[20,37],[20,40],
    [22,2],[22,5],[22,8],[22,11],[22,14],[22,17],[22,20],[22,23],[22,26],[22,29],
    [22,32],[22,35],[22,38],[24,1],[24,4],[24,7],[24,10],[24,13],[24,16],[24,19],
    [24,22],[24,25],[24,28],[24,31],[24,34],[24,37],[24,40],[26,2],[26,5],[26,8],
    [26,11],[26,14],[26,17],[26,20],[26,23],[26,26],[26,29],[26,32],[26,35],[26,38],
    [28,1],[28,4],[28,7],[28,10],[28,13],[28,16],[28,19],[28,22],[28,25],[28,28],
    [28,31],[28,34],[28,37],[28,40],[30,2],[30,5],[30,8],[30,11],[30,14],[30,17],
    [30,20],[30,23],[30,26],[30,29],[30,32],[30,35],[30,38]]

print(len(T_index))

print(get_Mat_P1([20,34]).shape)

Pro_c = 0.5
Pro_r = 1

def get_cost(c):
    if(c==1):
        return 0.34
    if(c==2):
        return 0.3
    if(c==0):
        return 0
    else:
        return 0.07

def return_num(c):
    if(c==0):
        return 0
    return 1

def func(lost):
    return 1/(0.0001+lost)

def sigmo(num):
    return 1.0/(1.0+np.exp(-num))

def T_M_distance(C):
    T1 = np.zeros((m, n))
    for i in range(t):
        T1 = T1 + np.multiply(return_num(C[i]), get_P(C[i], T_index[i]))
    mat = T1 - M

    lost = np.sum(np.sum(mat ** 2))
    cost = 0.5 * lost / (m * n)
    return cost

def d_func(C):
    T1 = np.zeros((m,n))
    for i in range(t):
        T1 = T1 + np.multiply(return_num(C[i]), get_P(C[i],T_index[i]))
    mat = T1 - M

    lost = np.sum(np.sum(mat**2))
    reg = 0
    for i in range(t):
        reg = reg + get_cost(C[i])
    cost = 0.5*lost/(m*n) + Lambda*sigmo(0.1*(reg-K))
    return func(cost)

def find_best(C):
    index = 0
    for i in range(len(C)):
        if(d_func(C[index])<d_func(C[i])):
            index = i
    return C[index]


def rws(C):
    k = np.zeros(num)
    new_C = []
    for i in range(num):
        k[i] = d_func(C[i])
    Sum = np.sum(k)
    for i in range(num):
        diss = np.random.uniform(0,1)
        for j in range(num):
            if(diss <= (np.sum(k[0:j+1])/Sum)):
                new_C.append(C[j])
                break

    return new_C

def change_help(C1,C2):
    number1 = rd.randint(0,10*t)%t
    number2 = rd.randint(0,10*t)%t
    if(number1>number2):
        temp1 = number1
        number1 = number2
        number2 = temp1

    temp1 = copy.deepcopy(C1)
    temp2 = copy.deepcopy(C2)
    temp = C1[number1:number2]
    C1[number1:number2] = C2[number1:number2]
    C2[number1:number2] = temp

    if(T_M_distance(C1)>T_M_distance(temp1)):
        C1 = temp1
    if(T_M_distance(C2)>T_M_distance(temp2)):
        C2 = temp2
    return [C1,C2]

def change(C,indexing):
    number = 0
    while(number < (len(indexing)-1)):
        C[indexing[number]], C[indexing[number + 1]] = change_help(C[indexing[number]],C[indexing[number + 1]])
        number = number + 2
    return C


def exc(C):
    indexing = []
    for i in range(num):
        if(Pro_c >= np.random.uniform(0,1)):
            indexing.append(i)
    return change(C,indexing)



def diff_help(C,indexing):
    for i in range(len(indexing)):
        index = rd.randint(0,10*t)%t
        temp = copy.deepcopy(C[indexing[i]])
        C[indexing[i]][index] = getNext_C()
        if(T_M_distance(temp)<T_M_distance(C[indexing[i]])):
            C[indexing[i]] = temp

    return C


def diffs(C):
    indexing = []
    for g in range(num):
        diss = np.random.uniform(0,1)
        if(diss < Pro_r):
            indexing.append(g)
    C = diff_help(C,indexing)

    return C

print(sum(sum(M)))

#print(d(np.array([1,1,1,1,1,0,0,0,0,0])))

interate = 3000
best_s = []
dis = np.zeros(interate)
best_num = np.zeros(interate)

C =init()
for i in range(interate):
    #print(i)
    #print(len(C))
    #print(C)
    rd.shuffle(C)
    rws(C)

    exc(C)

    diffs(C)
    zin = copy.deepcopy(find_best(C))
    #print(zin)

    T = np.zeros((m, n))
    for inter in range(t):
        T = T + np.multiply(return_num(zin[inter]),get_P(zin[inter],T_index[inter]))
        #print(zin[inter])
        #print(get_P(zin[inter], T_index[inter]))
        #print(return_num(zin[inter]))
    dis[i] = (0.5*sum(sum((T-M)**2)))/(m*n)
    print(i)
    print(0.5*sum(sum((T-M)**2))/(m*n))
    print()
    best_num[i] = d_func(zin)
    print(best_num[i])
    #print(best_num[i])

    #best_num.append(d_func(zin))
    best_s.append(copy.deepcopy(zin))
    #print(np.sum(zin))

print("全局最小")
minimum = np.max(best_num)
print(np.max(best_num))

for j in range(len(best_num)):
    if(best_num[j] == minimum):
        min_index = j

print(min_index)
best = best_s[min_index]

print(np.sum(best))

print(best)
T = np.zeros((m, n))
for inter in range(t):
    T = T + np.multiply(return_num(best[inter]),get_P(best[inter],T_index[inter]))
print(0.5/(m*n)*sum(sum((T-M)**2)))
print(T-M)

reg = 0
for i in range(t):
    reg = reg + get_cost(best[i])
print(reg)
file = "_data_new12"+str(K)+".txt"
np.savetxt(file,T-M,fmt = '%.0d')

file = "_data_new"+str(K)+".txt"
np.savetxt(file,T,fmt = '%.0d')
zone = best

"""fin = []
for n in range(len(zone)):
    if(zone[n]==1):
        fin.append(copy.deepcopy(T_index[n]))
print(fin)"""



T = np.zeros((m, 40))
print(best)
print(get_P(best[inter],T_index[inter]).shape)

for inter in range(t):
    T = T + np.multiply(return_num(best[inter]),get_P(best[inter],T_index[inter]))

ins = []
for i in range(len(best)):
    if(best[i]==0):
        continue
    else:
       ins.append(T_index[i])
print(len(ins))
print(T.shape)
print(T-M)