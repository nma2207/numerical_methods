import math
import numpy as np

eps = 1e-06


def f(x, eps):
    sum = x
    a = x
    n = 0
    while (math.fabs(a) > eps):
        q = -(((math.pi / 2) ** 2) * (x ** 4) * (4 * n + 1)) / ((2 * n + 1) * (2 * n + 2) * (4 * n + 5))
        a *= q
        sum += a
        n += 1
    return sum


def l(x_arr, x, k, n):
    result = 0.
    for i in range(n+1):
        if i == k:
            continue
        p1 = 1.0
        for j in range(n+1):
            p1 *= float(x - x_arr[j])
        p2 = 1.
        for j in range(n+1):
            if (j!= k):
                p2 *= float(x_arr[k] - x_arr[j])
        result += p1 / float((x - x_arr[k]) * (x - x_arr[i]) * p2)
    return result

def l_1(x_arr, x, i, n):
    result=1.0
    for j in range(n+1):
        if(j==i):
            continue
        result*=float(x-x_arr[j])/float(x_arr[i]-x_arr[j])
    #print 'result=',result
    return result




def c(x):
    return math.cos(math.pi * x * x / 2)


def L(n, a, b, x):
    h = float(b - a) / n
    x_arr = []
    for i in range(n + 1):
        x_arr.append(a + h * i)
    result = 0.0
    for i in range(n+1):
        result += f(x_arr[i], eps) * l(x_arr, x, i, n)
    return result


def chebishev_polynom(a,b,n):
    t_arr=[]
    for k in range(n+1):
        tk=math.cos(math.pi*(2*k+1)/float(2*(n+1)))
        t_arr.append(tk)
    x_arr=[]
    alfa=float(b-a)/2
    betta=float(a+b)/2
    for t in t_arr:
        x=alfa*t+betta
        x_arr.append(x)
    return x_arr

def main():
    x = []
    a = 0
    b = 1.5
    n = 10
    h = float(b - a) / n
    for i in range(n):
        x.append(a +h/2+ i * h)
    for i in x:
        print 'x={0:.2f}       f={1:.5f}        L={2:.10f}      c={3:.10f}       dif={4:.10f} '.format(i,f(i,eps), L(n, a, b, i), c(i), L(n, a, b, i)- c(i))


def test():
    print chebishev_polynom(0, 1.5, 10)

if __name__ == "__main__":
    test()
