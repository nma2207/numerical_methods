import math
import numpy as np

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
    h=float(b-a)/n
    y=np.array([y1(a), y2(a)])
    for i in range(n+1):
        x=a+i*h
        k1=f(x, y)
        k2=f(x+h/4, y+h*k1/2)
        k3=f(x+h/2, y+h*k2/2)
        k4=f(x+h, y+h*k1-2*h*k2+2*h*k3)
        y=y+h*(k1+4*k3+k4)/6.
        print y, np.array([y1(x), y2(x)])


def main():
    computing(0, 5, 10000)

if __name__=="__main__":
    main()




