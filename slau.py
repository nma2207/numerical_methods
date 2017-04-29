#coding:utf-8
from __future__ import  print_function
import numpy as np
import math
import numpy.linalg as alg
import matplotlib.pyplot as plt

eps=1e-2
def p(x):
    return 1+x
def u(x):
    return x**3-x**4
def g(x):
    return x+1
def F(x):
    return -6*x+3*x*x+16*x*x*x+u(x)*g(x)

#
#Генерируем заданную матрицу
#
def generate_slau(n):
    a=np.zeros((n-1))
    h=1./n
    for i in range(2, n):
        a[i-1]=-p(i*h)
    b=np.zeros((n-1))
    for i in range(1, n):
        b[i-1]=(p(i*h)+p((i+1)*h)+h*h*g(i*h))

    c=np.zeros((n-1))
    for i in range(1,n-1):
        c[i-1]=-p((i+1)*h)

    f=np.zeros((n-1))
    for i in range(1,n):
        f[i-1]=h*h*F(i*h)
    return a,b,c,f


#
#проверяем, является ли наша матрица матрицей с диагональным преобладанием
#
def check_diag_domination(a, b, c):
    if math.fabs(c[0])>=math.fabs(b[0]):
        return False
    n=b.size
    if math.fabs(a[n-1])>=math.fabs(b[n-1]):
        return False
    for i in range(1, n-1):
        if(math.fabs(a[i])+math.fabs(c[i])>=math.fabs(b[i])):
            return False
    return True
#
#Сравниваем результаты премножения матрица на вектор х и f
#
def print_result(a,b,c,f,x):
    n=a.size

    print (b[0]*x[0]+c[0]*x[1], f[0])
    for i in range(1, n-1):
        print (a[i]*x[i-1]+b[i]*x[i]+c[i]*x[i+1], f[i])
    print (a[n-1]*x[n-2]+b[n-1]*x[n-1], f[n-1])


#
#Метод прогонки, который является  точным
#
def sweep_method(a, b, c, f):
    n=b.size
    p=np.zeros((n))
    q=np.zeros((n))
    p[1]=c[0]/b[0]
    q[1]=-f[0]/b[0]
    for i in range(2,n):
        p[i]=c[i-1]/(b[i-1]-a[i-1]*p[i-1])
        q[i]=(a[i-1]*q[i-1]-f[i-1])/(b[i-1]-a[i-1]*p[i-1])
    x=np.zeros((n))
    #print 'p=',p
    #print 'q=', q
    x[n-1]=(a[n-1]*q[n-1]-f[n-1])/(b[n-1]-a[n-1]*p[n-1])
    for i in range(n-2, -1, -1):
        #print i, i+1
        x[i]=p[i+1]*x[i+1]+q[i+1]
    return x
#
#Для сравнения правильности вычислений надо убедиться, что u_i отличается от y_i не более чем на O(h^2)
#
def compare_u_and_y(y):
    n=y.size+1
    h=1./n
    print ('h =', h)
    for i in range(1, n):
        print (u(i*h), y[i-1], math.fabs(u(i*h)-y[i-1]))
def getA(i, j, a, b, c):
    if  i==j:
        return b[i]
    elif i==j+1:
        return a[i]
    elif i==j-1:
        return c[i]
    else:
        return 0

def mult_A_x(a,b,c,x):
    #x - вектор размерности n
    n=a.size
    res=np.zeros((n))
    for i in range(n):
        for j in range(n):
            res[i]+=getA(i,j,a,b, c)*x[j]
    return res

def yacobi_method(a,b,c,f):
    n=a.size
    x=np.zeros((n))
    k=0
    while alg.norm(mult_A_x(a,b,c,x)-f)>eps:
        #print('k=',k)
        k+=1
        x_prev=np.copy(x)
        for i in range(n):
            x[i]=0
            for j in range(n):
                if(j==i):
                    continue
                x[i]+=(getA(i,j,a,b,c)/getA(i,i,a,b,c))*x_prev[j]
            x[i]=-x[i]
            x[i]+=f[i]/getA(i,i,a,b,c)
    return x,k

def relax_method(a,b,c, f, w):
    n=a.size
    x=np.zeros((n))
    k=0
    while alg.norm(mult_A_x(a,b,c,x)-f)>eps:
        k+=1
        x_prev=np.copy(x)
        for i in range(n):
            x[i]=0
            for j in range(i):
                x[i]+=getA(i,j,a,b,c)/getA(i,i,a,b,c)*x[j]
            for j in range(i+1, n):
                x[i]+=getA(i,j,a,b,c)/getA(i,i,a,b,c)*x_prev[j]
            x[i]=-x[i]
            x[i]+=f[i]/getA(i,i,a,b,c)
            x[i]*=w
            x[i]+=(1-w)*x_prev[i]

    return x, k


def main():
    n=10
    a,b,c,f=generate_slau(n)
    for i in range(n-1):
        print (a[i], b[i], c[i], f[i])
    print(check_diag_domination(a,b,c))
    x=sweep_method(a,-b,c,f)
    print (x)
    print ('result:')
    print_result(a,b,c,f,x)
    print ('compare')
    compare_u_and_y(x)

    jacoby_x, j_k=yacobi_method(a,b,c,f)
    relax_x, r_k=relax_method(a,b,c,f, 2)

    print('real x:\n', x)
    print('yacobi x:\n', jacoby_x)
    print('relax x:\n', relax_x)
    print(jacoby_x-x)
    print('\nadfadf\n')
    print (mult_A_x(a,b,c,x)-f)
    print(mult_A_x(a, b, c, jacoby_x)-f)
    print('k compare', j_k, r_k)

def test_relax():
    n=40
    a,b,c,f=generate_slau(n)
    k_s=[]
    w_arr=np.arange(0.1, 2, 0.01)
    for w in w_arr:
        print('w =', w)
        x,k=relax_method(a,b,c,f,w)
        k_s.append(k)
    k_s=np.array(k_s)
    plt.figure()
    plt.plot(w_arr, k_s)
    plt.show()


if __name__=="__main__":
    test_relax()