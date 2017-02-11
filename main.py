import math
import numpy as np
eps=1e-06

def f(x,eps):
    sum=x
    a=x
    n=0
    while(math.fabs(a)>eps):
        q=-(((math.pi/2)**2)*(x**4)*(4*n+1))/((2*n+1)*(2*n+2)*(4*n+5))
        a*=q
        sum+=a
        n+=1
    return sum

def main():
    x=[]
    results=[]
    for i in np.arange(0, 1.5+0.15, 0.15):
        x.append(i)
    for i in x:
        results.append(f(i,eps))
    for i in range(len(x)):
        print 'x={0:.2f}       f={1:.5f}'.format(x[i],results[i])


if __name__ == "__main__":
    main()
