import numpy as np


def p(x):
    return 1+x
def u(x):
    return x**3-x**4
def g(x):
    return x+1
def F(x):
    return -6*x-3*x*x-16*x*x*x
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
def main():
    n=3
    a,b,c,f=generate_slau(n)
    for i in range(n):
        print a[i], b[i], c[i], f[i]
def sweep_method(a, b, c, f):
    #FIXME:
    #TODO:
    print 'TODO'

if __name__=="__main__":
    main()