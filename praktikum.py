import math
import numpy as np
from matplotlib import pyplot as plt
import decimal as dec

toch=100

def f1(x, y):
    #return y[0]/(2+2*x)-2*x*y[1]
    #return (y[1])
    return y[1]

def f2(x, y):
    #return y[1]/(2+2*x)+2*x*y[0]
    return -2*y[0]*y[0]*(1-4*x*x*y[0])
    #return -y[0]

def y1(x):
    return 1/((1+x*x))
    #return dec.Decimal.from_float(math.cos(x * x) * (math.sqrt(1 + x)))

def y2(x):
    return -2*x/((1+x*x)*(1+x*x))
    #return dec.Decimal.from_float(math.sin(x*x)*(math.sqrt(1+x)))
    #return dec.Decimal.from_float(math.sin(x * x)) * ((1 + x) ** dec.Decimal(1)/2)

def f(x, y):
    return np.array([f1(x, y), f2(x, y)])

def computing(a, b, n):
    dec.getcontext().prec = toch
    x_arr=[]
    result=[]
    h=dec.Decimal(b-a)/n
    print type(h)
    y=(np.array([y1(dec.Decimal(a)), y2(dec.Decimal(a))]))
    result.append(y)
    x_arr.append(a)
    for i in range(n):
        dec.getcontext().prec = toch
        x=dec.Decimal(a)+i*h
        k1=f(x, y)
        k2=f(x+h/4, y+h*k1/4)
        k3=f(x+h/2, y+h*k2/2)
        k4=f(x+h, y+h*k1-2*h*k2+2*h*k3)
        print y
        y=y+h*(k1+4*k3+k4)/6
        #y=y+h/6*(k1+2*k2+2*k3+k4)

        # k1=f(x,y)
        # k2=f(x+h/2, y+h*k1/2)
        # k3=f(x+h/2, y+h*k2/2)
        # k4=f(x+h, y+h*k3)
        # y=y+h*(k1+2*k2+2*k3+k4)/6
        result.append(y)
        x_arr.append(x+h)
    return np.array(x_arr), np.array(result)
        #print y, np.array([y1(x), y2(x)])

def computing_real_values(a, b, n):
    x_arr=[]
    result=[]
    dec.getcontext().prec=toch
    h=dec.Decimal(b-a)/n
    print h
    # x_arr.append(a)
    # result.append(np.array([y1(a), y2(a)]))
    for i in range(n+1):
        x=dec.Decimal(a)+i*h
        y=np.array([y1(x), y2(x)])
        print y
        result.append(y)
        x_arr.append(x)
    return np.array(x_arr), np.array(result)



def main():
    dec.getcontext().prec=toch
    y=dec.Decimal(5)
    y/=3
    print y
    print dec.Decimal(1)/dec.Decimal(7)
    a=0
    b=5
    n=50

    x1,results1 = computing(a, b, n)
    x2, results2= computing_real_values(a, b, n)
    for i in range(len(x1)):
        print 'x={0:.3f}       y1={1:.15f}     y1_r={2:.15f}   dif={3:.15f}             y2={4:.15f}     y2_r={5:.15f}   dif={6:.15f}'.format(x1[i], results1[i,0], results2[i,0], results1[i,0]-results2[i,0], results1[i,1], results2[i,1], results1[i,1]-results2[i,1])
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('function #1')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.plot(x1, results1.transpose()[0], 'g')
    plt.plot(x2, results2.transpose()[0], 'b')
    plt.subplot(1, 2, 2)
    plt.plot(x1, results1.transpose()[1],'g')
    plt.plot(x2, results2.transpose()[1],'b')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title('function #2')
    plt.show()


if __name__=="__main__":
    main()




