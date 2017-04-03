import math
import numpy as np
from matplotlib import pyplot as plt

eps = 1e-06
int_eps = 1e-04


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


def l(x_arr, x, k):
    result = 0.
    for i in range(len(x_arr)):
        if i == k:
            continue
        p1 = 1.0
        for j in range(len(x_arr)):
            p1 *= float(x - x_arr[j])
        p2 = 1.
        for j in range(len(x_arr)):
            if (j != k):
                p2 *= float(x_arr[k] - x_arr[j])
        result += p1 / float((x - x_arr[k]) * (x - x_arr[i]) * p2)
    return result


def c(x):
    return math.cos(math.pi * x * x / 2)


def L(x_arr, x):
    result = 0.0
    for i in range(len(x_arr)):
        result += f(x_arr[i], eps) * l(x_arr, x, i)
    return result


def chebishev_polynom(a, b, n):
    t_arr = []
    for k in range(n + 1):
        tk = math.cos(math.pi * (2 * k + 1) / float(2 * (n + 1)))
        t_arr.append(tk)
    x_arr = []
    alfa = float(b - a) / 2
    betta = float(a + b) / 2
    for t in t_arr:
        x = alfa * t + betta
        x_arr.append(x)
    return x_arr


def integral_left_rect(a, b):
    N = 1
    s = 0
    s_prev = 0
    while True:
        s_prev = s
        s=0
        h = float(b - a) / N
        for i in range(N):
            s += c(a + i * h)*h
        #s *= h
        if (math.fabs(s - s_prev) < int_eps):
            break
        else:
            N +=1
    return s, N


def integral_trapezoid(a, b):
    N = 1
    s = 0
    s_prev = 0
    while True:
        s_prev = s
        s=0
        h = float(b - a) / N
        for i in range(N):
            s += (c(a + i * h) + c(a + (i + 1) * h))
        s *= (h / 2.)
        if (math.fabs(s - s_prev) < int_eps):
            break
        else:
            N +=1
    return s, N


def integral_simpson(a, b):
    N = 1
    s = 0
    s_prev = 0
    while True:
        s_prev = s
        s=0
        h = float(b - a) / N
        for i in range(N):
            s += (c(a + i * h) + 4*c(a + (i + 0.5) * h) + c(a + (i + 1) * h))
        s *= (h / 6.)
        if (math.fabs(s - s_prev) < int_eps):
            break
        else:
            N +=1
    return s, N


def main():
    a = 0
    b = 1.5
    top = 100
    bot = 50
    step = 5
    max_errors1 = []
    h_arr1 = []
    for n1 in range(bot, top, step):
        print 'n1 =', n1
        x1 = []
        h1 = float(b - a) / n1
        for i in range(n1):
            x1.append(a + h1 / 2 + i * h1)

        x_arr = []
        for i in range(n1 + 1):
            x_arr.append(a + i * h1)
        errors = []
        for i in x1:
            L_val = L(x_arr, i)
            c_val = c(i)
            errors.append(math.fabs(L_val - c_val))
            print'{0:.3f}\t{1:.15f}\t{2:.15f}\t{3:.15f}'.format(i, L_val, c_val, math.fabs(L_val-c_val))
        max_errors1.append(np.max(np.array(errors)))
        h_arr1.append(h1)

    #
    # chebishev
    #
    max_errors2 = []
    h_arr2 = []
    for n2 in range(bot, top, step):
        print'n2 = ', n2
        h2 = float(b - a) / n2
        x2 = []
        for i in range(n2):
            x2.append(a + h2 / 2 + i * h2)
        x_cheb = chebishev_polynom(a, b, n2)
        errors = []
        for i in x2:
            l_val = L(x_cheb, i)
            c_val = c(i)
            errors.append(math.fabs(l_val - c_val))
            print'{0:.3f}\t{1:.15f}\t{2:.15f}\t{3:.15f}'.format(i, L_val, c_val, math.fabs(L_val - c_val))
        max_errors2.append(np.max(np.array(errors)))
        h_arr2.append(h2)

    h_arr1 = np.array(h_arr1)
    max_errors1 = np.array(max_errors1)
    h_arr2 = np.array(h_arr2)
    max_errors2 = np.array(max_errors2)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(h_arr1, max_errors1, 'r')
    plt.title('Ravnomerno')
    plt.ylabel('max error')
    plt.xlabel('h')

    plt.subplot(1, 2, 2)
    plt.plot(h_arr2, max_errors2, 'g')
    plt.title('Chebishev')
    plt.ylabel('max error')
    plt.xlabel('h')
    plt.show()
    # x_cheb=chebishev_polynom(a,b,n2)
    # x2=[]
    # h2 = float(b - a) / n2
    # for i in range(n2):
    #     x2.append(a +h2/2+ i * h2)
    # print ('\nchebishev:\n')
    # L_arr=[]
    # for i in x2:
    #     L_arr.append(L(x_cheb, i))
    #     print ('x={0:.5f}       f={1:.5f}        L={2:.10f}      c={3:.10f}       dif={4:.10f} '.format(i,f(i,eps), L(x_cheb, i), c(i), L(x_cheb, i)- c(i)))
    # x2=np.array(x2)
    # L_arr=np.array(L_arr)
    # plt.figure()
    # plt.plot(x2, L_arr)
    # plt.show()
def test():
    a=0
    b=1.5
    n=10
    h=float(b-a)/n
    x_arr=[]
    s_arr=[]
    for i in range(n+1):
        x=a+i*h
        sum,k=f(x,eps)
        print '{0:.2f}\t{1:.10f}\t{2:.0f}'.format(x,sum,k)
        x_arr.append(x)
        s_arr.append(sum)
    x_arr=np.array(x_arr)
    s_arr=np.array(s_arr)
    plt.figure()
    plt.plot(x_arr, s_arr)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def test_int():
    a = 0
    b = 1.5
    n=10
    h=float(b-a)/n
    for i in range(n+1):
        x=a+i*h
        left=integral_left_rect(0,x)
        trap=integral_trapezoid(0,x)
        simp=integral_simpson(0,x)
        f_r=f(x, eps)
        print 'x=',x
        print 'Left rect: ', left
        print 'Trapezoid: ', trap
        print 'Simpson:   ',simp
        print 'real val:  ',f_r

if __name__ == "__main__":
    main()
