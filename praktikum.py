import math
import numpy as np
from matplotlib import pyplot as plt
def f1(x, y):
    return y[1]

def f2(x, y):
    return -2.*y[0]*y[0]*(1-4*x*x*y[0])

def y1(x):
    return 1./(1+x*x)

def y2(x):
    return -2.*x/((1+x*x)**2)

def f(x, y):
    return np.array([f1(x, y), f2(x, y)])

def computing(a, b, n):
    x_arr=[]
    result=[]
    h=float(b-a)/n
    y=np.array([y1(a), y2(a)])
    #result.append(y)
    #x_arr.append(a)
    for i in range( n+1):
        x=a+i*h
        k1=f(x, y)
        k2=f(x+h/4, y+h*k1/2)
        k3=f(x+h/2, y+h*k2/2)
        k4=f(x+h, y+h*k1-2*h*k2+2*h*k3)
        y=y+h*(k1+4*k3+k4)/6.
        result.append(y)
        x_arr.append(x)
    return np.array(x_arr), np.array(result)
        #print y, np.array([y1(x), y2(x)])

def computing_real_values(a, b, n):
    x_arr=[]
    result=[]
    h=float(b-a)/n
    for i in range(n+1):
        x=a+i*h
        result.append(np.array([y1(x), y2(x)]))
        x_arr.append(x)
    return np.array(x_arr), np.array(result)



def main():
    a=0
    b=5
    n=100
    x1,results1 = computing(a, b, n)
    x2, results2= computing_real_values(a, b, n)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('function #1')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.plot(x1, results1.transpose()[0])
    plt.plot(x2, results2.transpose()[0])
    plt.subplot(1, 2, 2)
    plt.plot(x1, results1.transpose()[1])
    plt.plot(x2, results2.transpose()[1])
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title('function #2')
    plt.show()


if __name__=="__main__":
    main()




