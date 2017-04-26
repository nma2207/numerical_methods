#coding:utf-8
import numpy as np


def p(x):
    return 1+x
def u(x):
    return x**3-x**4
def g(x):
    return x+1
def F(x):
    return -6*x-3*x*x-16*x*x*x

#
#Генерируем заданную матрицу
#
def generate_slau(n):
    a=np.zeros((n))
    h=1./n
    for i in range(1, n):
        a[i]=-p(i*h)
    b=np.zeros((n))
    for i in range(n):
        b[i]=p(i*h)+p((i+1)*h)+h*h*g(i*h)

    c=np.zeros((n))
    for i in range(n-1):
        c[i]=-p((i+1)*h)

    f=np.zeros((n))
    for i in range(n):
        f[i]=h*h*F(i*h)
    return a,b,c,f


#
#проверяем, является ли наша матрица матрицей с диагональным преобладанием
#
def check_diag_domination(a, b, c):
    if c[0]>=b[0]:
        return False
    n=b.size
    if a[n-1]>=b[n-1]:
        return False
    for i in range(1, n-1):
        if(a[i]+c[i]>=b[i]):
            return False
    return True
#
#Сравниваем результаты премножения матрица на вектор х и f
#
def print_result(a,b,c,f,x):
    n=a.size

    print -b[0]*x[0]+c[0]*x[1], f[0]
    for i in range(1, n-1):
        print a[i]*x[i-1]-b[i]*x[i]+c[i]*x[i+1], f[i]
    print a[n-1]*x[n-2]-b[n-1]*x[n-1], f[n-1]


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

def main():
    n=3
    a,b,c,f=generate_slau(n)
    for i in range(n):
        print a[i], b[i], c[i], f[i]
    print check_diag_domination(a,b,c)
    x=sweep_method(a,b,c,f)
    print x
    print 'result:'
    print_result(a,b,c,f,x)
   # print check_diag_domination(a, b, c)


if __name__=="__main__":
    main()