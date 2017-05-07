#coding:utf-8
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
    #print type(h)
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
        #print y
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
    #print h
    # x_arr.append(a)
    # result.append(np.array([y1(a), y2(a)]))
    for i in range(n+1):
        x=dec.Decimal(a)+i*h
        y=np.array([y1(x), y2(x)])
        #print y
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
    #n=50
    bot=200
    top=500
    step=10
    errors1 = []
    errors2=[]
    h_arr = []
    for n in range(bot, top+step, step):
        print n
        x1,results1 = computing(a, b, n)
        x2, results2= computing_real_values(a, b, n)
        print results1.shape
        error1=np.max(np.abs(results1[:,0]-results2[:,0]))
        error2=np.max(np.abs(results1[:,1]-results2[:,1]))
        h=dec.Decimal(b-a)/n
        errors1.append(error1)
        errors2.append(error2)
        h_arr.append(h)


    errors1=np.array(errors1)
    errors2 = np.array(errors2)
    h_arr=np.array(h_arr)
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(h_arr, errors1, 'r', label='y1')
    plt.plot(h_arr, errors2, 'b', label='y2')
    plt.xlabel("h")
    plt.ylabel("error")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(h_arr, errors1/(h_arr**4), 'r', label='y1')
    plt.plot(h_arr, errors2/(h_arr**4), 'b', label='y2')
    plt.xlabel("h")
    plt.ylabel("error / h^4")
    plt.legend()
    plt.show()



def calc_dx(v,theta):
    return v*math.cos(math.radians(theta))
def calc_dy(v, theta):
    return v*math.sin(math.radians(theta))
def calc_dx_dy(v,theta):
    return np.array([calc_dx(v,theta),
                    calc_dy(v,theta)])

def calc_dv(m,dm, T, C, p, S,g, v, theta):

    return 1./m*(T-0.5*C*p*S*v*v)-float(dm)/m*v-g*math.sin(math.radians(theta))
def calc_dthata(g,v,theta):
    return -g/v*math.cos(math.radians(theta))

def calc_dv_dtheta(m,dm, T, C, p, S,g, v, theta):
    return np.array([calc_dv(m,dm, T, C,p,S,g,v, theta),
                     calc_dthata(g,v,theta)])
ENGINE_WORK_TIME=3
# т.к. в начальный момент времни масса топлива =15, а через 3 секунды оно выгорает полностью,
# можно описать уравнением m=-5*t+50, dm=-5 - первые 3 секунды, потом m=35, а dm=0
def calc_m(t):
    if(t<ENGINE_WORK_TIME):
        return -5*t+50
    else:
        return 35
def calc_dm(t):
    if(t<ENGINE_WORK_TIME):
        return -5
    else:
        return 0
#тяга есть только в момент работы двигателя
def calc_T(t):
    if(t<ENGINE_WORK_TIME):
        return 8
    else:
        return 0

def calc_v_and_theta_array(t0,t_end, N, C,p, S, g, v0, theta0):
    v_theta_arr=[]
    v_theta_arr.append(np.array([v0, theta0]))
    v_theta=np.array([v0, theta0])
    t_arr=[]
    t_arr.append(0)
    h=float(t_end-t0)/N
    for i in range(N):
        t=t0+i*h
        m=calc_m(t)
        dm=calc_dm(t)
        T=calc_T(t)
        k1=calc_dv_dtheta(m,dm, T, C, p, S,g, v_theta[0], v_theta[1])
        k2=calc_dv_dtheta(m,dm, T, C, p, S,g,
                          (v_theta+h*k1/4)[0], (v_theta+h*k1/4)[1])
        k3 = calc_dv_dtheta(m, dm, T, C, p, S, g,
                            (v_theta + h * k2 / 2)[0], (v_theta + h * k2 / 2)[1])
        k4 = calc_dv_dtheta(m, dm, T, C, p, S, g,
                            (v_theta + h*k1-2*h*k2+2*h*k3)[0], (v_theta + h*k1-2*h*k2+2*h*k3)[1])
        v_theta=v_theta+h*(k1+4*k3+k4)/6
        v_theta_arr.append(v_theta)
        t_arr.append(t+h)
    return v_theta_arr, t_arr

def calc_x_y_arr(v_theta_arr, t_arr):
    x0=0.0
    y0=0.0
    x_y_arr=[]
    x_y_arr.append(np.array([x0, y0]))
    x_y=np.array([x0, y0])
    h=t_arr[1]-t_arr[0]
    print h
    for i in range(len(t_arr)):
        v,theta=v_theta_arr[i][0], v_theta_arr[i][1]
        k1 = calc_dx_dy(v, theta)
        k2 = calc_dx_dy(v+k1[0]*h/4., theta+k1[1]*h/4.)
        k3 = calc_dx_dy(v + k2[0] * h/2., theta + k2[1] * h/2.)
        k4 = calc_dx_dy(v + h*k1[0]-2*h*k2[0]+2*h*k3[0],
                        theta + h*k1[1]-2*h*k2[1]+2*h*k3[1])
        x_y=x_y+h*(k1+4*k3+k4)/6.
        x_y_arr.append(x_y)
    return x_y_arr



def task():
    C=0.25
    p=1.29
    S=0.35
    g=9.81
    v0=60
    theta0_1=0.6
    theta0_2=1.2
    v_and_theta1, t_arr=calc_v_and_theta_array(0,5,200, C, p,S,g,v0,theta0_1)
    x_y1= calc_x_y_arr(v_and_theta1, t_arr)
    x_y1=np.array(x_y1)
    x1=x_y1[:,0]
    y1=x_y1[:,1]
    plt.figure()
    plt.plot(x1,y1, 'b')

    v_and_theta2, t_arr=calc_v_and_theta_array(0,5,200, C, p,S,g,v0,theta0_2)
    x_y2= calc_x_y_arr(v_and_theta2, t_arr)
    x_y2=np.array(x_y2)
    x2=x_y2[:,0]
    y2=x_y2[:,1]
    plt.plot(x2, y2, 'g')
    plt.show()



    #print calc_v_and_theta_array(0,5,50, C, p,S,g,v0,theta0_1)



if __name__=="__main__":
    task()




