Python 3.8.2 (tags/v3.8.2:7b3ab59, Feb 25 2020, 23:03:10) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> from numpy.linalg import inv
>>> from numpy import dot
>>> import numpy as np
>>> from numpy import pi,exp,sqrt
>>> import matplotlib.pyplot as plt
>>> from mpl_toolkits.mplot3d import axes3d
>>> from mpmath import *
>>> from matplotlib.ticker import MultipleLocator
>>> import matplotlib as mpl
>>> from math import gamma
>>> import math as math
>>> from scipy import integrate
Nv=45
>>> Nv=60
>>> Mv=7
>>> pal_v = np.linspace(-Mv, Mv, 2*Nv)
>>> per_v = np.linspace(-Mv, Mv, 2*Nv)
>>> Nt=3000
>>> Mt=3000
>>> t=np.linspace(0, Mt, Nt-1)
>>> F=(t[1]-t[0])/(2*(pal_v[1]-pal_v[0]))**2
>>> F
18.074549699799995
>>> n=1
>>> n2=0
>>> n3=-1
>>> omega=-1
>>> fre=0.07
>>> k_pal0=0.245
>>> k_per0=k_pal0*tan((55*np.pi)/180)
>>> k_per0
mpf('0.34989626165181803')
>>> k_pal_max=0.28
>>> k_pal_min=0.21
>>> k_per_max=k_pal_max*tan((55*np.pi)/180)
>>> k_per_min=k_pal_min*tan((55*np.pi)/180)
>>> a_pal=0.035
>>> a_per=a_pal*tan((55*np.pi)/180)
>>> a_per
mpf('0.049985180235974008')
>>> GV=0.86
>>> B_B0=0.001
>>> def k(b):
    f = lambda x: ((besselj(0, (b*x)/(omega), 0))**2)*np.exp(-(((x-0.35)**2)/(0.05**2)))*x
    I=integrate.quad(f, k_per_min, k_per_max)
    return I[0]

>>> def coefficient_a(a,b):
    return ((0.52*np.pi**2)/(0.035*0.05**2))*(B_B0**2)*(((b)**2)/abs(a-GV))*(fre/k_pal0)**2*k(b)*(np.exp(-(((fre-a*k_pal0-n*omega)/(a-GV))**2)/(0.035**2)))**2

>>> def coefficient_a2(a,b):
    return ((0.04*np.pi**2)/(0.035*0.05**2))*(B_B0**2)*(((a)**2)/abs(a-GV))*(fre/k_pal0)**2*k(b)*(np.exp(-(((fre-a*k_pal0-n2*omega)/(a-GV))**2)/(0.035**2)))**2

>>> def coefficient_a3(a,b):
    return ((0.52*np.pi**2)/(0.035*0.05**2))*(B_B0**2)*(((b)**2)/abs(a-GV))*(fre/k_pal0)**2*k(b)*(np.exp(-(((fre-a*k_pal0-n3*omega)/(a-GV))**2)/(0.035**2)))**2

>>> AA=np.zeros(((2*Nv)*(2*Nv),(2*Nv)*(2*Nv)))
>>> delv=2*abs(pal_v[1]-pal_v[0])
>>> def Matrix_A(b):
    A=np.zeros(((2*Nv),(2*Nv)))
    for i in range(2*Nv):
        for j in range(2*Nv):
            if i==0:
                A[i,j] =1+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==0 else 0 if j==1 else -(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==2 else 0
            elif i==1:
                A[i,j] =0 if j==0 else 1+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==1 else 0 if j==2 else -(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==3 else 0
            elif i==2*Nv-1:
                A[i,j] =-(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv-3 else 0 if j==2*Nv-2 else 1+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv-1 else 0 if j==2*Nv else 0
            elif i==2*Nv:
                A[i,j] =-(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv-2 else 0 if j==2*Nv-1 else 1+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv else 0
            else:
                A[i,j] =-(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==i-2 else 0 if j==i-1 else 1+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==i else 0 if j==i+1 else -(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==i+2 else 0
    return A

>>> def Matrix_B1(b):
    B=np.zeros(((2*Nv),(2*Nv)))
    for i in range(2*Nv):
        for j in range(2*Nv):
            if i==0:
                B[i,j] =0 if j==0 else (F/2)*((pal_v[i]*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i],per_v[b]-delv/2)+(F/2)*(((pal_v[i]+delv/2)*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i]+delv/2,per_v[b])+ (F/2)*((pal_v[i]*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(F/2)*(((pal_v[i]+delv/2)*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i]+delv/2,per_v[b])+ (F/2)*((pal_v[i]*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(F/2)*(((pal_v[i]+delv/2)*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==1 else 0
            elif i==2*Nv:
                B[i,j] =0 if j==2*Nx else -(F/2)*((pal_v[i]*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i],per_v[b]-delv/2)-(F/2)*(((pal_v[i]-delv/2)*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i]-delv/2,per_v[b])- (F/2)*((pal_v[i]*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i],per_v[b]-delv/2)-(F/2)*(((pal_v[i]-delv/2)*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i]-delv/2,per_v[b])- (F/2)*((pal_v[i]*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i],per_v[b]-delv/2)-(F/2)*(((pal_v[i]-delv/2)*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv-1 else 0
            else:
                B[i,j] =0 if j==i else -(F/2)*((pal_v[i]*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i],per_v[b]-delv/2)-(F/2)*(((pal_v[i]-delv/2)*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i]-delv/2,per_v[b])- (F/2)*((pal_v[i]*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i],per_v[b]-delv/2)-(F/2)*(((pal_v[i]-delv/2)*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i]-delv/2,per_v[b])- (F/2)*((pal_v[i]*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i],per_v[b]-delv/2)-(F/2)*(((pal_v[i]-delv/2)*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==i-1 else (F/2)*((pal_v[i]*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i],per_v[b]-delv/2)+(F/2)*(((pal_v[i]+delv/2)*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i]+delv/2,per_v[b])+ (F/2)*((pal_v[i]*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(F/2)*(((pal_v[i]+delv/2)*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i]+delv/2,per_v[b])+ (F/2)*((pal_v[i]*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(F/2)*(((pal_v[i]+delv/2)*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==i+1 else 0
    return B

>>> def Matrix_B2(b):
    B=np.zeros(((2*Nv),(2*Nv)))
    for i in range(2*Nv):
        for j in range(2*Nv):
            if i==0:
                B[i,j] =0 if j==0 else -(F/2)*((pal_v[i]*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i],per_v[b]+delv/2)-(F/2)*(((pal_v[i]+delv/2)*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i]+delv/2,per_v[b])- (F/2)*((pal_v[i]*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i],per_v[b]+delv/2)-(F/2)*(((pal_v[i]+delv/2)*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i]+delv/2,per_v[b])- (F/2)*((pal_v[i]*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i],per_v[b]+delv/2)-(F/2)*(((pal_v[i]+delv/2)*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==1 else 0
            elif i==2*Nv:
                B[i,j] =0 if j==2*Nx else (F/2)*((pal_v[i]*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i],per_v[b]+delv/2)+(F/2)*(((pal_v[i]-delv/2)*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i]-delv/2,per_v[b])+ (F/2)*((pal_v[i]*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(F/2)*(((pal_v[i]-delv/2)*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i]-delv/2,per_v[b])+ (F/2)*((pal_v[i]*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(F/2)*(((pal_v[i]-delv/2)*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv-1 else 0
            else:
                B[i,j] =0 if j==i else (F/2)*((pal_v[i]*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i],per_v[b]+delv/2)+(F/2)*(((pal_v[i]-delv/2)*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i]-delv/2,per_v[b])+ (F/2)*((pal_v[i]*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(F/2)*(((pal_v[i]-delv/2)*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i]-delv/2,per_v[b])+ (F/2)*((pal_v[i]*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(F/2)*(((pal_v[i]-delv/2)*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==i-1 else -(F/2)*((pal_v[i]*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i],per_v[b]+delv/2)-(F/2)*(((pal_v[i]+delv/2)*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i]+delv/2,per_v[b])- (F/2)*((pal_v[i]*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i],per_v[b]+delv/2)-(F/2)*(((pal_v[i]+delv/2)*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i]+delv/2,per_v[b])- (F/2)*((pal_v[i]*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i],per_v[b]+delv/2)-(F/2)*(((pal_v[i]+delv/2)*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==i+1 else 0
    return B

>>> def Matrix_C1(b):
    C=np.zeros(((2*Nv),(2*Nv)))
    for i in range(2*Nv):
        for j in range(2*Nv):
            C[i,j] =-(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)-(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)-(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2) if j==i else 0
    return C

>>> def Matrix_C2(b):
    C=np.zeros(((2*Nv),(2*Nv)))
    for i in range(2*Nv):
        for j in range(2*Nv):
            C[i,j] =-(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)-(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)-(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2) if j==i else 0
    return C

>>> def Matrix_A_1(b):
    A_1=np.zeros(((2*Nv),(2*Nv)))
    for i in range(2*Nv):
        for j in range(2*Nv):
            if i==0:
                A_1[i,j] =1+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==0 else 0 if j==1 else -(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==2 else 0
            elif i==1:
                A_1[i,j] =0 if j==0 else 1+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==1 else 0 if j==2 else -(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==3 else 0
            elif i==2*Nv-1:
                A_1[i,j] =-(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv-3 else 0 if j==2*Nv-2 else 1+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv-1 else 0 if j==2*Nv else 0
            elif i==2*Nv:
                A_1[i,j] =-(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv-2 else 0 if j==2*Nv-1 else 1+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv else 0
            else:
                A_1[i,j] =-(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==i-2 else 0 if j==i-1 else 1+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==i else 0 if j==i+1 else -(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==i+2 else 0
    return A_1

>>> for a in range(2*Nv):
    for b in range(2*Nv):
        if a==b:
            AA[a*2*Nv:(a+1)*2*Nv,b*2*Nv:(b+1)*2*Nv]=Matrix_A(a)

            
>>> for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            AA[(a+1)*2*Nv:(a+2)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=Matrix_B1(a+1)

>>> for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            AA[a*2*Nv:(a+1)*2*Nv,(b+1)*2*Nv:(b+2)*2*Nv]=Matrix_B2(a)

>>> for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            AA[(a+2)*2*Nv:(a+3)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=Matrix_C1(a+2)

>>> for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            AA[a*2*Nv:(a+1)*2*Nv,(b+2)*2*Nv:(b+3)*2*Nv]=Matrix_C2(a)

>>> AA_1 = inv(AA)
>>> QQ=np.zeros(((2*Nv)*(2*Nv),(2*Nv)*(2*Nv)))
>>> for a in range(2*Nv):
    for b in range(2*Nv):
        if a==b:
            QQ[a*2*Nv:(a+1)*2*Nv,b*2*Nv:(b+1)*2*Nv]=Matrix_A_1(a)

            
>>> for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            QQ[(a+1)*2*Nv:(a+2)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=-Matrix_B1(a+1)

            
>>> for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            QQ[a*2*Nv:(a+1)*2*Nv,(b+1)*2*Nv:(b+2)*2*Nv]=-Matrix_B2(a)

            
>>> for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            QQ[(a+2)*2*Nv:(a+3)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=-Matrix_C1(a+2)

            
>>> for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            QQ[a*2*Nv:(a+1)*2*Nv,(b+2)*2*Nv:(b+3)*2*Nv]=-Matrix_C2(a)

            
>>> AQ=dot(AA_1,QQ)
>>> def Kappa_Initial_Strahl(a,b):
    kappa=150
    return (2.175)**(-1.5)*0.08*np.exp(-((b)**2)/2.175)*np.exp(-((a-Us)**2)/2.175)

>>> def Kappa_Initial_Core(a,b):
    kappa=150
    return (1.087)**(-1.5)*0.92*np.exp(-((b)**2)/1.087)*np.exp(-((a-Uc)**2)/1.087)

>>> Me=9.1094*(10**(-28))
>>> Mp=1.6726*(10**(-24))
>>> ratio=Me/Mp
>>> Us=108*ratio**(0.5)
>>> Uc=-9.3913*ratio**(0.5)
>>> cont_lev = np.linspace(-8,0,25)
>>> f_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> solu2=np.zeros(shape = (Nv, 2*Nv))
>>> fc_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> ff_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        f_1[j*2*Nv+i]=Kappa_Initial_Strahl(pal_v[i],per_v[j])

        
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        fc_1[j*2*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j])

        
>>> ff_1=f_1+fc_1
>>> Mf_1=np.max(ff_1)
>>> per_v2 = np.linspace(0, Mv, Nv)
>>> X2,Y2 = np.meshgrid(pal_v,per_v2)
>>> for k in range(5): #Numer in range indicates the minute.
    print(k)
    #ff_1=f_1+fc_1
    for j in range(Nv):
        for i in range(2*Nv):
        #solu[j,i]=(abs(f_1[j*2*Nv+i])/Mf_1)
            if abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>1:
                solu2[j,i]=0
            elif abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>10**(-5):
                solu2[j,i]=np.log10(abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1)#np.log10
            else:
                solu2[j,i]=-10
    #Mf_1=np.max(f_1)
    fig = plt.figure()
    fig.set_dpi(350)
    plt.contourf(X2, Y2,solu2, cont_lev,cmap='Blues');
    ax = plt.gca()
    ax.spines['left'].set_position('center')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_smart_bounds(True)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    #ax.yaxis.set_ticks_position('left')
    #ax.set_xlim(-Mv, Mv); ax.set_ylim(0, 0.6);
    #plt.xlim(-Mv, Mv)
    #plt.ylim(0, 0.6)
    plt.axis('equal')
    plt.yticks([2,4,6,8])
    plt.xticks([-6,-4,-2,0,2,4,6])
    plt.rc('font', size=9)
    plt.tick_params(labelsize=9)
    plt.text(-0.2,-1.6,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae}$', fontsize=9)
    plt.text(-0.2,8.3,r'$\mathcal{v}_\perp/\mathcal{v}_{Ae}$', fontsize=9)
    plt.colorbar(label=r'$Log(F/F_{MAX})$')
    #plt.savefig(QLD/Collision/qld/{k}.png)
    #plt.clf()
    plt.show()
    for t in range(100):
        ff_1=dot(AQ, ff_1)

        
0
<matplotlib.contour.QuadContourSet object at 0x0000018E2B3B6BE0>

Warning (from warnings module):
  File "<pyshell#99>", line 19
MatplotlibDeprecationWarning: 
The set_smart_bounds function was deprecated in Matplotlib 3.2 and will be removed two minor releases later.

Warning (from warnings module):
  File "<pyshell#99>", line 21
MatplotlibDeprecationWarning: 
The set_smart_bounds function was deprecated in Matplotlib 3.2 and will be removed two minor releases later.
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B3AAB80>, <matplotlib.axis.YTick object at 0x0000018E2B381B50>, <matplotlib.axis.YTick object at 0x0000018E2B3B73A0>, <matplotlib.axis.YTick object at 0x0000018E2B3B78B0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B39EB50>, <matplotlib.axis.XTick object at 0x0000018E2B3813D0>, <matplotlib.axis.XTick object at 0x0000018E2B42A0D0>, <matplotlib.axis.XTick object at 0x0000018E2B42A5E0>, <matplotlib.axis.XTick object at 0x0000018E2B42AAF0>, <matplotlib.axis.XTick object at 0x0000018E2B4310A0>, <matplotlib.axis.XTick object at 0x0000018E2B431550>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B39E040>
1
<matplotlib.contour.QuadContourSet object at 0x0000018E2BC9F1F0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B3411C0>, <matplotlib.axis.YTick object at 0x0000018E2B2A1A00>, <matplotlib.axis.YTick object at 0x0000018E2AA938B0>, <matplotlib.axis.YTick object at 0x0000018E2AA93DC0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B297FD0>, <matplotlib.axis.XTick object at 0x0000018E2B30FEB0>, <matplotlib.axis.XTick object at 0x0000018E2AA96940>, <matplotlib.axis.XTick object at 0x0000018E2AA96E50>, <matplotlib.axis.XTick object at 0x0000018E2AA96730>, <matplotlib.axis.XTick object at 0x0000018E2AA93B80>, <matplotlib.axis.XTick object at 0x0000018E2AA98550>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AAC1100>
2
<matplotlib.contour.QuadContourSet object at 0x0000018E2B47FC10>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B3B6EE0>, <matplotlib.axis.YTick object at 0x0000018E2B39E370>, <matplotlib.axis.YTick object at 0x0000018E2AADE040>, <matplotlib.axis.YTick object at 0x0000018E2AADE910>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B26C100>, <matplotlib.axis.XTick object at 0x0000018E2B259C10>, <matplotlib.axis.XTick object at 0x0000018E2AADE580>, <matplotlib.axis.XTick object at 0x0000018E2AAEE160>, <matplotlib.axis.XTick object at 0x0000018E2B31B2B0>, <matplotlib.axis.XTick object at 0x0000018E2B297730>, <matplotlib.axis.XTick object at 0x0000018E2BCA7910>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B456160>
3
<matplotlib.contour.QuadContourSet object at 0x0000018E2A9833A0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A97A550>, <matplotlib.axis.YTick object at 0x0000018E2A96F280>, <matplotlib.axis.YTick object at 0x0000018E2A9EAA60>, <matplotlib.axis.YTick object at 0x0000018E2A9ED040>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A97A0D0>, <matplotlib.axis.XTick object at 0x0000018E2B3F2CD0>, <matplotlib.axis.XTick object at 0x0000018E2A9EA940>, <matplotlib.axis.XTick object at 0x0000018E2A9ED3D0>, <matplotlib.axis.XTick object at 0x0000018E2A9F10A0>, <matplotlib.axis.XTick object at 0x0000018E2A9F1550>, <matplotlib.axis.XTick object at 0x0000018E2A9F1A60>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AA1C2B0>
4
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB435E0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB68D60>, <matplotlib.axis.YTick object at 0x0000018E2AB30310>, <matplotlib.axis.YTick object at 0x0000018E2ABA2CD0>, <matplotlib.axis.YTick object at 0x0000018E2AB36220>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB68070>, <matplotlib.axis.XTick object at 0x0000018E2AB22250>, <matplotlib.axis.XTick object at 0x0000018E2AB36AC0>, <matplotlib.axis.XTick object at 0x0000018E2AB361C0>, <matplotlib.axis.XTick object at 0x0000018E2ABA62B0>, <matplotlib.axis.XTick object at 0x0000018E2ABA67C0>, <matplotlib.axis.XTick object at 0x0000018E2ABA6CD0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ABCF4C0>
>>> for k in range(1): #Numer in range indicates the minute.
    print(k)
    #ff_1=f_1+fc_1
    for j in range(Nv):
        for i in range(2*Nv):
        #solu[j,i]=(abs(f_1[j*2*Nv+i])/Mf_1)
            if abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>1:
                solu2[j,i]=0
            elif abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>10**(-5):
                solu2[j,i]=np.log10(abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1)#np.log10
            else:
                solu2[j,i]=-10
    #Mf_1=np.max(f_1)
    fig = plt.figure()
    fig.set_dpi(350)
    plt.contourf(X2, Y2,solu2, cont_lev,cmap='Blues');
    ax = plt.gca()
    ax.spines['left'].set_position('center')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_smart_bounds(True)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    #ax.yaxis.set_ticks_position('left')
    #ax.set_xlim(-Mv, Mv); ax.set_ylim(0, 0.6);
    #plt.xlim(-Mv, Mv)
    #plt.ylim(0, 0.6)
    plt.axis('equal')
    plt.yticks([2,4,6,8])
    plt.xticks([-6,-4,-2,0,2,4,6])
    plt.rc('font', size=9)
    plt.tick_params(labelsize=9)
    plt.text(-0.2,-1.6,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae}$', fontsize=9)
    plt.text(-0.2,8.3,r'$\mathcal{v}_\perp/\mathcal{v}_{Ae}$', fontsize=9)
    plt.colorbar(label=r'$Log(F/F_{MAX})$')
    #plt.savefig(QLD/Collision/qld/{k}.png)
    #plt.clf()
    plt.show()

    
0
<matplotlib.contour.QuadContourSet object at 0x0000018E2AD76430>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC612E0>, <matplotlib.axis.YTick object at 0x0000018E2AC30E80>, <matplotlib.axis.YTick object at 0x0000018E2A9F1B50>, <matplotlib.axis.YTick object at 0x0000018E2AA335B0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB43100>, <matplotlib.axis.XTick object at 0x0000018E2AB36C10>, <matplotlib.axis.XTick object at 0x0000018E2AB5F370>, <matplotlib.axis.XTick object at 0x0000018E2AA33FA0>, <matplotlib.axis.XTick object at 0x0000018E2AB5FD60>, <matplotlib.axis.XTick object at 0x0000018E2AB5FA60>, <matplotlib.axis.XTick object at 0x0000018E2A964CD0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9F38B0>
>>> f_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> solu2=np.zeros(shape = (Nv, 2*Nv))
>>> fc_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> ff_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        f_1[j*2*Nv+i]=Kappa_Initial_Strahl(pal_v[i],per_v[j])

        
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        fc_1[j*2*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j])

        
>>> ff_1=f_1+fc_1
>>> for k in range(21): #Numer in range indicates the minute.
    print(k)
    #ff_1=f_1+fc_1
    for j in range(Nv):
        for i in range(2*Nv):
        #solu[j,i]=(abs(f_1[j*2*Nv+i])/Mf_1)
            if abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>1:
                solu2[j,i]=0
            elif abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>10**(-5):
                solu2[j,i]=np.log10(abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1)#np.log10
            else:
                solu2[j,i]=-10
    #Mf_1=np.max(f_1)
    fig = plt.figure()
    fig.set_dpi(350)
    plt.contourf(X2, Y2,solu2, cont_lev,cmap='Blues');
    ax = plt.gca()
    ax.spines['left'].set_position('center')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_smart_bounds(True)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    #ax.yaxis.set_ticks_position('left')
    #ax.set_xlim(-Mv, Mv); ax.set_ylim(0, 0.6);
    #plt.xlim(-Mv, Mv)
    #plt.ylim(0, 0.6)
    plt.axis('equal')
    plt.yticks([2,4,6,8])
    plt.xticks([-6,-4,-2,0,2,4,6])
    plt.rc('font', size=9)
    plt.tick_params(labelsize=9)
    plt.text(-0.2,-1.6,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae}$', fontsize=9)
    plt.text(-0.2,8.3,r'$\mathcal{v}_\perp/\mathcal{v}_{Ae}$', fontsize=9)
    plt.colorbar(label=r'$Log(F/F_{MAX})$')
    #plt.savefig(QLD/Collision/qld/{k}.png)
    #plt.clf()
    plt.show()
    for t in range(20):
        ff_1=dot(AQ, ff_1)

        
0
<matplotlib.contour.QuadContourSet object at 0x0000018E2BCC5760>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B47F0D0>, <matplotlib.axis.YTick object at 0x0000018E2B456430>, <matplotlib.axis.YTick object at 0x0000018E2B2AF370>, <matplotlib.axis.YTick object at 0x0000018E2B2AFCA0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A96F910>, <matplotlib.axis.XTick object at 0x0000018E2B456040>, <matplotlib.axis.XTick object at 0x0000018E2B2AFEE0>, <matplotlib.axis.XTick object at 0x0000018E2B2C08B0>, <matplotlib.axis.XTick object at 0x0000018E2B318160>, <matplotlib.axis.XTick object at 0x0000018E2B318700>, <matplotlib.axis.XTick object at 0x0000018E2B318C10>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018CFFD74C70>
1
<matplotlib.contour.QuadContourSet object at 0x0000018E2B3296A0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC60F10>, <matplotlib.axis.YTick object at 0x0000018E2AD700A0>, <matplotlib.axis.YTick object at 0x0000018E2AAF3D60>, <matplotlib.axis.YTick object at 0x0000018E2B2932B0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC60EB0>, <matplotlib.axis.XTick object at 0x0000018E2B314C70>, <matplotlib.axis.XTick object at 0x0000018E2B293AC0>, <matplotlib.axis.XTick object at 0x0000018E2B2930D0>, <matplotlib.axis.XTick object at 0x0000018E2AAF5340>, <matplotlib.axis.XTick object at 0x0000018E2AAF5850>, <matplotlib.axis.XTick object at 0x0000018E2AAF5D60>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AAE0580>
2
<matplotlib.contour.QuadContourSet object at 0x0000018E2AA96730>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A9F9C40>, <matplotlib.axis.YTick object at 0x0000018E2AAA0D60>, <matplotlib.axis.YTick object at 0x0000018E2ABD3DF0>, <matplotlib.axis.YTick object at 0x0000018E2AA09340>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2BCA7EB0>, <matplotlib.axis.XTick object at 0x0000018E2BCCBB20>, <matplotlib.axis.XTick object at 0x0000018E2AA09EE0>, <matplotlib.axis.XTick object at 0x0000018E2AA09250>, <matplotlib.axis.XTick object at 0x0000018E2ABD73D0>, <matplotlib.axis.XTick object at 0x0000018E2ABD78E0>, <matplotlib.axis.XTick object at 0x0000018E2ABD7DF0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ABFF5E0>
3
<matplotlib.contour.QuadContourSet object at 0x0000018E2B2B82E0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAE8340>, <matplotlib.axis.YTick object at 0x0000018E2A9D5C10>, <matplotlib.axis.YTick object at 0x0000018E2AAA0040>, <matplotlib.axis.YTick object at 0x0000018E2AAA0CA0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2BCCBF10>, <matplotlib.axis.XTick object at 0x0000018E2A9CCB20>, <matplotlib.axis.XTick object at 0x0000018E2AAA0F10>, <matplotlib.axis.XTick object at 0x0000018E2B47F100>, <matplotlib.axis.XTick object at 0x0000018E2B47FBB0>, <matplotlib.axis.XTick object at 0x0000018E2B47F850>, <matplotlib.axis.XTick object at 0x0000018E2B2AF670>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2BCA7880>
4
<matplotlib.contour.QuadContourSet object at 0x0000018E2AD70610>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A9A6B80>, <matplotlib.axis.YTick object at 0x0000018E2B44D400>, <matplotlib.axis.YTick object at 0x0000018E2AAD7070>, <matplotlib.axis.YTick object at 0x0000018E2B3ACFD0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A9A6A00>, <matplotlib.axis.XTick object at 0x0000018E2A9B13A0>, <matplotlib.axis.XTick object at 0x0000018E2AAD7340>, <matplotlib.axis.XTick object at 0x0000018E2B3AC910>, <matplotlib.axis.XTick object at 0x0000018E2AD89760>, <matplotlib.axis.XTick object at 0x0000018E2AD89D30>, <matplotlib.axis.XTick object at 0x0000018E2AD89610>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ABCF400>
5
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC30B20>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC243D0>, <matplotlib.axis.YTick object at 0x0000018E2AB43790>, <matplotlib.axis.YTick object at 0x0000018E2B42B640>, <matplotlib.axis.YTick object at 0x0000018E2B42BB50>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAEE4C0>, <matplotlib.axis.XTick object at 0x0000018E2B34B0A0>, <matplotlib.axis.XTick object at 0x0000018E2B42B970>, <matplotlib.axis.XTick object at 0x0000018E2ACC3700>, <matplotlib.axis.XTick object at 0x0000018E2ACC3BE0>, <matplotlib.axis.XTick object at 0x0000018E2ACC6130>, <matplotlib.axis.XTick object at 0x0000018E2ACC6640>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AC39DF0>
6
<matplotlib.contour.QuadContourSet object at 0x0000018E2AD5FBB0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB26640>, <matplotlib.axis.YTick object at 0x0000018E2AAE03A0>, <matplotlib.axis.YTick object at 0x0000018E2AB5BB50>, <matplotlib.axis.YTick object at 0x0000018E2AB5B5E0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB265E0>, <matplotlib.axis.XTick object at 0x0000018E2AB36DF0>, <matplotlib.axis.XTick object at 0x0000018E2AB65730>, <matplotlib.axis.XTick object at 0x0000018E2AB5B730>, <matplotlib.axis.XTick object at 0x0000018E2AB65340>, <matplotlib.axis.XTick object at 0x0000018E2A9649A0>, <matplotlib.axis.XTick object at 0x0000018E2A964DF0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AD70B80>
7
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAE81F0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B47F130>, <matplotlib.axis.YTick object at 0x0000018E2AA98D30>, <matplotlib.axis.YTick object at 0x0000018E2AA09CA0>, <matplotlib.axis.YTick object at 0x0000018E2AA09C40>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A97AEB0>, <matplotlib.axis.XTick object at 0x0000018E2B30F790>, <matplotlib.axis.XTick object at 0x0000018E2AA09310>, <matplotlib.axis.XTick object at 0x0000018E2AC96D00>, <matplotlib.axis.XTick object at 0x0000018E2AC96880>, <matplotlib.axis.XTick object at 0x0000018E2AC96CD0>, <matplotlib.axis.XTick object at 0x0000018E2B2C2730>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9CC340>
8
<matplotlib.contour.QuadContourSet object at 0x0000018E2BCB56A0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AA42E50>, <matplotlib.axis.YTick object at 0x0000018E2AC61640>, <matplotlib.axis.YTick object at 0x0000018E2B3A3490>, <matplotlib.axis.YTick object at 0x0000018E2B3A3160>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AA42040>, <matplotlib.axis.XTick object at 0x0000018E2B2AFFD0>, <matplotlib.axis.XTick object at 0x0000018E2B3A3CA0>, <matplotlib.axis.XTick object at 0x0000018E2AA35790>, <matplotlib.axis.XTick object at 0x0000018E2AA358B0>, <matplotlib.axis.XTick object at 0x0000018E2AA35040>, <matplotlib.axis.XTick object at 0x0000018E2AC9F340>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AADAAF0>
9
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAACC40>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AA9A3D0>, <matplotlib.axis.YTick object at 0x0000018E2AA262B0>, <matplotlib.axis.YTick object at 0x0000018E2AA9D340>, <matplotlib.axis.YTick object at 0x0000018E2AA9D850>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC7CE80>, <matplotlib.axis.XTick object at 0x0000018E2A974F10>, <matplotlib.axis.XTick object at 0x0000018E2AA9D670>, <matplotlib.axis.XTick object at 0x0000018E2AB8D400>, <matplotlib.axis.XTick object at 0x0000018E2AB8D8E0>, <matplotlib.axis.XTick object at 0x0000018E2AB8DDF0>, <matplotlib.axis.XTick object at 0x0000018E2AB7D340>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ABE9E80>
10
<matplotlib.contour.QuadContourSet object at 0x0000018E2A9E4880>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC7CEB0>, <matplotlib.axis.YTick object at 0x0000018E2ABC9BE0>, <matplotlib.axis.YTick object at 0x0000018E2B2AF220>, <matplotlib.axis.YTick object at 0x0000018E2B2AFDF0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ABDE5B0>, <matplotlib.axis.XTick object at 0x0000018E2A9D3E50>, <matplotlib.axis.XTick object at 0x0000018E2B2AFDC0>, <matplotlib.axis.XTick object at 0x0000018E2A96FEE0>, <matplotlib.axis.XTick object at 0x0000018E2A96F730>, <matplotlib.axis.XTick object at 0x0000018E2A9F97F0>, <matplotlib.axis.XTick object at 0x0000018E2A9F9670>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9CCB50>
11
<matplotlib.contour.QuadContourSet object at 0x0000018E2B2C2910>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B31E1F0>, <matplotlib.axis.YTick object at 0x0000018E2AAD09A0>, <matplotlib.axis.YTick object at 0x0000018E2AC41BE0>, <matplotlib.axis.YTick object at 0x0000018E2AC39100>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B31EA60>, <matplotlib.axis.XTick object at 0x0000018E2AAD0910>, <matplotlib.axis.XTick object at 0x0000018E2AC41BB0>, <matplotlib.axis.XTick object at 0x0000018E2AC39A60>, <matplotlib.axis.XTick object at 0x0000018E2AA3CD90>, <matplotlib.axis.XTick object at 0x0000018E2AA3CAF0>, <matplotlib.axis.XTick object at 0x0000018E2AA3C460>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB22430>
12
<matplotlib.contour.QuadContourSet object at 0x0000018E2B456DF0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ABD6C40>, <matplotlib.axis.YTick object at 0x0000018E2B31ED00>, <matplotlib.axis.YTick object at 0x0000018E2AB06340>, <matplotlib.axis.YTick object at 0x0000018E2AB06610>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAA01F0>, <matplotlib.axis.XTick object at 0x0000018E2AD63610>, <matplotlib.axis.XTick object at 0x0000018E2AB06670>, <matplotlib.axis.XTick object at 0x0000018E2AA98BE0>, <matplotlib.axis.XTick object at 0x0000018E2AA98C10>, <matplotlib.axis.XTick object at 0x0000018E2AA98DF0>, <matplotlib.axis.XTick object at 0x0000018E2ACCEF10>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB924F0>
13
<matplotlib.contour.QuadContourSet object at 0x0000018E2B42BB50>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB23A60>, <matplotlib.axis.YTick object at 0x0000018E2A9F1C70>, <matplotlib.axis.YTick object at 0x0000018E2A9F1A60>, <matplotlib.axis.YTick object at 0x0000018E2A9F1760>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ACCE430>, <matplotlib.axis.XTick object at 0x0000018E2B416760>, <matplotlib.axis.XTick object at 0x0000018E2AD88220>, <matplotlib.axis.XTick object at 0x0000018E2A9F1400>, <matplotlib.axis.XTick object at 0x0000018E2AD88B50>, <matplotlib.axis.XTick object at 0x0000018E2AAEE370>, <matplotlib.axis.XTick object at 0x0000018E2AAEEA60>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B2C2F40>
14
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB655E0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AA461F0>, <matplotlib.axis.YTick object at 0x0000018E2AAD0190>, <matplotlib.axis.YTick object at 0x0000018E2AB5BA60>, <matplotlib.axis.YTick object at 0x0000018E2AA16EB0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AD70E80>, <matplotlib.axis.XTick object at 0x0000018E2A9B1550>, <matplotlib.axis.XTick object at 0x0000018E2AA16DF0>, <matplotlib.axis.XTick object at 0x0000018E2AB5B460>, <matplotlib.axis.XTick object at 0x0000018E2AB92250>, <matplotlib.axis.XTick object at 0x0000018E2ABEE850>, <matplotlib.axis.XTick object at 0x0000018E2ABEE130>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B2AF280>
15
<matplotlib.contour.QuadContourSet object at 0x0000018E2B30F730>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC872B0>, <matplotlib.axis.YTick object at 0x0000018E2B3B6760>, <matplotlib.axis.YTick object at 0x0000018E2AA26700>, <matplotlib.axis.YTick object at 0x0000018E2AA26F40>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B47F970>, <matplotlib.axis.XTick object at 0x0000018E2BCBEAC0>, <matplotlib.axis.XTick object at 0x0000018E2AA26D60>, <matplotlib.axis.XTick object at 0x0000018E2A98C220>, <matplotlib.axis.XTick object at 0x0000018E2A98C250>, <matplotlib.axis.XTick object at 0x0000018E2AD5E6A0>, <matplotlib.axis.XTick object at 0x0000018E2AD5ED90>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AA9BD90>
16
<matplotlib.contour.QuadContourSet object at 0x0000018E2ACB7EB0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AA51E50>, <matplotlib.axis.YTick object at 0x0000018E2ACB37F0>, <matplotlib.axis.YTick object at 0x0000018E2ACC75B0>, <matplotlib.axis.YTick object at 0x0000018E2ACC7AC0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AA54CD0>, <matplotlib.axis.XTick object at 0x0000018E2AC41880>, <matplotlib.axis.XTick object at 0x0000018E2ACC78E0>, <matplotlib.axis.XTick object at 0x0000018E2AC4F670>, <matplotlib.axis.XTick object at 0x0000018E2AC4FB50>, <matplotlib.axis.XTick object at 0x0000018E2AC3B070>, <matplotlib.axis.XTick object at 0x0000018E2AC3B5B0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AC0AD60>
17
<matplotlib.contour.QuadContourSet object at 0x0000018E2AA044C0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AA540A0>, <matplotlib.axis.YTick object at 0x0000018E2A9E73A0>, <matplotlib.axis.YTick object at 0x0000018E2B34C5E0>, <matplotlib.axis.YTick object at 0x0000018E2AAB5790>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2BCB9250>, <matplotlib.axis.XTick object at 0x0000018E2AC79C70>, <matplotlib.axis.XTick object at 0x0000018E2B34CA00>, <matplotlib.axis.XTick object at 0x0000018E2AAB5D90>, <matplotlib.axis.XTick object at 0x0000018E2AAC5520>, <matplotlib.axis.XTick object at 0x0000018E2AAC5DC0>, <matplotlib.axis.XTick object at 0x0000018E2AAC51F0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9904C0>
18
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB22E80>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC59DF0>, <matplotlib.axis.YTick object at 0x0000018E2ABCFEE0>, <matplotlib.axis.YTick object at 0x0000018E2A97A310>, <matplotlib.axis.YTick object at 0x0000018E2A97A0A0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC59C10>, <matplotlib.axis.XTick object at 0x0000018E2AB7DD00>, <matplotlib.axis.XTick object at 0x0000018E2A97ADF0>, <matplotlib.axis.XTick object at 0x0000018E2AB30A30>, <matplotlib.axis.XTick object at 0x0000018E2AB30CD0>, <matplotlib.axis.XTick object at 0x0000018E2B2C2B20>, <matplotlib.axis.XTick object at 0x0000018E2B2C20A0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2BCBE070>
19
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAA08B0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AD5BA90>, <matplotlib.axis.YTick object at 0x0000018E2AB063D0>, <matplotlib.axis.YTick object at 0x0000018E2B4568E0>, <matplotlib.axis.YTick object at 0x0000018E2B456520>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AA981C0>, <matplotlib.axis.XTick object at 0x0000018E2B31E100>, <matplotlib.axis.XTick object at 0x0000018E2B456A60>, <matplotlib.axis.XTick object at 0x0000018E2AB43880>, <matplotlib.axis.XTick object at 0x0000018E2AB43820>, <matplotlib.axis.XTick object at 0x0000018E2ACCE2E0>, <matplotlib.axis.XTick object at 0x0000018E2ACCEF40>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AA3C7F0>
20
<matplotlib.contour.QuadContourSet object at 0x0000018E2B45C550>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A991F70>, <matplotlib.axis.YTick object at 0x0000018E2A9F99A0>, <matplotlib.axis.YTick object at 0x0000018E2A9F9490>, <matplotlib.axis.YTick object at 0x0000018E2AC241C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAB2C10>, <matplotlib.axis.XTick object at 0x0000018E2AAB20A0>, <matplotlib.axis.XTick object at 0x0000018E2AC244C0>, <matplotlib.axis.XTick object at 0x0000018E2AC24370>, <matplotlib.axis.XTick object at 0x0000018E2A9F96A0>, <matplotlib.axis.XTick object at 0x0000018E2B2A13A0>, <matplotlib.axis.XTick object at 0x0000018E2B2A1E80>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB222B0>
>>> density=(10**(2))
>>> B=5*10**(-4)
>>> c=2.9979*10**(10)
>>> q=4.8032*(10**(-10))
>>> time=np.linspace(0, Mt, Nt)
>>> deltt=abs(time[1]-time[0])
>>> deltt
1.0003334444814937
>>> Col_parameter_e=q**(3)*(4*np.pi)**(2.5)*density**(2.5)*B**(-4)*25*c*10**4
>>> Col_parameter_p=q**(3)*(4*np.pi)**(2.5)*density**(2.5)*B**(-4)*25*c*10**4
>>> def Collision_Core(a,b):
    return Me**(0.5)*(np.pi)**(-1.5)*(1.087)**(-1.5)*0.92*np.exp(-((b)**2)/1.087)*np.exp(-((a-Uc)**2)/1.087)

>>> def Collision_Proton(a,b):
    return Me**(0.5)*(np.pi)**(-1.5)*(1/ratio)**(3/2)*np.exp(-(1/ratio)*((b)**2))*np.exp(-(1/ratio)*((a)**2))

>>> def G_per_2(a,b):
    try:
        f = lambda y, x: 2*np.pi*((((a-x)**2+((b)-y)**2)**(-1/2))-(((b)-y)**2)*(((a-x)**2+((b)-y)**2)**(-3/2)))*Collision_Core(x,y)*y
        I=integrate.dblquad(f, -Mv, Mv, lambda y: 0, lambda y: Mv)
        return I[0]
    except ZeroDivisionError:
        return 0

>>> def G_pal_2(a,b):
    try:
        f = lambda y, x: 2*np.pi*((((a-x)**2+((b)-y)**2)**(-1/2))-((a-x)**2)*(((a-x)**2+((b)-y)**2)**(-3/2)))*Collision_Core(x,y)*y
        I=integrate.dblquad(f, -Mv, Mv, lambda y: 0, lambda y: Mv)
        return I[0]
    except ZeroDivisionError:
        return 0

>>> def G_pal_per(a,b):
    try:
        f = lambda y, x: 2*np.pi*(-(a-x)*((b)-y)*(((a-x)**2+((b)-y)**2)**(-3/2)))*Collision_Core(x,y)*y
        I=integrate.dblquad(f, -Mv, Mv, lambda y: 0, lambda y: Mv)
        return I[0]
    except ZeroDivisionError:
        return 0

>>> def H_per(a,b):
        return 0

>>> def H_pal(a,b):
        return 0

>>> def G_per_2p(a,b):
    try:
        f = lambda y, x: 2*np.pi*((((a-x)**2+((b)-y)**2)**(-1/2))-(((b)-y)**2)*(((a-x)**2+((b)-y)**2)**(-3/2)))*Collision_Proton(x,y)*y
        I=integrate.dblquad(f, -Mv, Mv, lambda y: 0, lambda y: Mv)
        return I[0]
    except ZeroDivisionError:
        return 0

>>> def G_pal_2p(a,b):
    try:
        f = lambda y, x: 2*np.pi*((((a-x)**2+((b)-y)**2)**(-1/2))-((a-x)**2)*(((a-x)**2+((b)-y)**2)**(-3/2)))*Collision_Proton(x,y)*y
        I=integrate.dblquad(f, -Mv, Mv, lambda y: 0, lambda y: Mv)
        return I[0]
    except ZeroDivisionError:
        return 0

>>> def G_pal_perp(a,b):
    try:
        f = lambda y, x: 2*np.pi*(-(a-x)*((b)-y)*(((a-x)**2+((b)-y)**2)**(-3/2)))*Collision_Proton(x,y)*y
        I=integrate.dblquad(f, -Mv, Mv, lambda y: 0, lambda y: Mv)
        return I[0]
    except ZeroDivisionError:
        return 0

>>> def H_perp(a,b):
    try:
        f = lambda y, x: 2*np.pi*(-((b)-y)*(((a-x)**2+((b)-y)**2)**(-3/2)))*Collision_Proton(x,y)*y
        I=integrate.dblquad(f, -Mv, Mv, lambda y: 0, lambda y: Mv)
        return I[0]
    except ZeroDivisionError:
        return 0

>>> def H_palp(a,b):
    try:
        f = lambda y, x: 2*np.pi*(-(a-x)*(((a-x)**2+((b)-y)**2)**(-3/2)))*Collision_Proton(x,y)*y
        I=integrate.dblquad(f, -Mv, Mv, lambda y: 0, lambda y: Mv)
        return I[0]
    except ZeroDivisionError:
        return 0

>>> def Matrix_A(b):
    A=np.zeros(((2*Nv),(2*Nv)))
    for i in range(2*Nv):
        for j in range(2*Nv):
            if i==0:
                A[i,j] =1+Col_parameter_e*(-2*np.pi*deltt*Collision_Core(pal_v[i],per_v[b])-2*np.pi*deltt*ratio*Collision_Proton(pal_v[i],per_v[b])+(F/2)*G_per_2(pal_v[i],per_v[b])+(F/2)*G_per_2p(pal_v[i],per_v[b])+(F/2)*G_pal_2(pal_v[i],per_v[b])+(F/2)*G_pal_2p(pal_v[i],per_v[b])) if j==0 else -Col_parameter_e*(deltt/delv)*0.5*H_pal(pal_v[i],per_v[b])-Col_parameter_p*(deltt/delv)*0.5*H_palp(pal_v[i],per_v[b]) if j==1 else -Col_parameter_e*(F/4)*G_pal_2(pal_v[i],per_v[b])-Col_parameter_p*(F/4)*G_pal_2p(pal_v[i],per_v[b]) if j==2 else 0
            elif i==1:
                A[i,j] =Col_parameter_e*(deltt/delv)*0.5*H_pal(pal_v[i],per_v[b])+Col_parameter_p*(deltt/delv)*0.5*H_palp(pal_v[i],per_v[b]) if j==0 else 1+Col_parameter_e*(-2*np.pi*deltt*Collision_Core(pal_v[i],per_v[b])-2*np.pi*deltt*ratio*Collision_Proton(pal_v[i],per_v[b])+(F/2)*G_per_2(pal_v[i],per_v[b])+(F/2)*G_per_2p(pal_v[i],per_v[b])+(F/2)*G_pal_2(pal_v[i],per_v[b])+(F/2)*G_pal_2p(pal_v[i],per_v[b])) if j==1 else -Col_parameter_e*(deltt/delv)*0.5*H_pal(pal_v[i],per_v[b])-Col_parameter_p*(deltt/delv)*0.5*H_palp(pal_v[i],per_v[b]) if j==2 else -Col_parameter_e*(F/4)*G_pal_2(pal_v[i],per_v[b])-Col_parameter_p*(F/4)*G_pal_2p(pal_v[i],per_v[b]) if j==3 else 0
            elif i==2*Nv-1:
                A[i,j] =-Col_parameter_e*(F/4)*G_pal_2(pal_v[i],per_v[b])-Col_parameter_p*(F/4)*G_pal_2p(pal_v[i],per_v[b]) if j==2*Nv-3 else Col_parameter_e*(deltt/delv)*0.5*H_pal(pal_v[i],per_v[b])+Col_parameter_p*(deltt/delv)*0.5*H_palp(pal_v[i],per_v[b]) if j==2*Nv-2 else 1+Col_parameter_e*(-2*np.pi*deltt*Collision_Core(pal_v[i],per_v[b])-2*np.pi*deltt*ratio*Collision_Proton(pal_v[i],per_v[b])+(F/2)*G_per_2(pal_v[i],per_v[b])+(F/2)*G_per_2p(pal_v[i],per_v[b])+(F/2)*G_pal_2(pal_v[i],per_v[b])+(F/2)*G_pal_2p(pal_v[i],per_v[b])) if j==2*Nv-1 else -Col_parameter_e*(deltt/delv)*0.5*H_pal(pal_v[i],per_v[b])-Col_parameter_p*(deltt/delv)*0.5*H_palp(pal_v[i],per_v[b]) if j==2*Nv else 0
            elif i==2*Nv:
                A[i,j] =-Col_parameter_e*(F/4)*G_pal_2(pal_v[i],per_v[b])-Col_parameter_p*(F/4)*G_pal_2p(pal_v[i],per_v[b]) if j==2*Nv-2 else Col_parameter_e*(deltt/delv)*0.5*H_pal(pal_v[i],per_v[b])+Col_parameter_p*(deltt/delv)*0.5*H_palp(pal_v[i],per_v[b]) if j==2*Nv-1 else 1+Col_parameter_e*(-2*np.pi*deltt*Collision_Core(pal_v[i],per_v[b])-2*np.pi*deltt*ratio*Collision_Proton(pal_v[i],per_v[b])+(F/2)*G_per_2(pal_v[i],per_v[b])+(F/2)*G_per_2p(pal_v[i],per_v[b])+(F/2)*G_pal_2(pal_v[i],per_v[b])+(F/2)*G_pal_2p(pal_v[i],per_v[b])) if j==2*Nv else 0
            else:
                A[i,j] =-Col_parameter_e*(F/4)*G_pal_2(pal_v[i],per_v[b])-Col_parameter_p*(F/4)*G_pal_2p(pal_v[i],per_v[b]) if j==i-2 else Col_parameter_e*(deltt/delv)*0.5*H_pal(pal_v[i],per_v[b])+Col_parameter_p*(deltt/delv)*0.5*H_palp(pal_v[i],per_v[b]) if j==i-1 else 1+Col_parameter_e*(-2*np.pi*deltt*Collision_Core(pal_v[i],per_v[b])-2*np.pi*deltt*ratio*Collision_Proton(pal_v[i],per_v[b])+(F/2)*G_per_2(pal_v[i],per_v[b])+(F/2)*G_per_2p(pal_v[i],per_v[b])+(F/2)*G_pal_2(pal_v[i],per_v[b])+(F/2)*G_pal_2p(pal_v[i],per_v[b])) if j==i else -Col_parameter_e*(deltt/delv)*0.5*H_pal(pal_v[i],per_v[b])-Col_parameter_p*(deltt/delv)*0.5*H_palp(pal_v[i],per_v[b]) if j==i+1 else -Col_parameter_e*(F/4)*G_pal_2(pal_v[i],per_v[b])-Col_parameter_p*(F/4)*G_pal_2p(pal_v[i],per_v[b]) if j==i+2 else 0
    return A

>>> def Matrix_B1(b):
    B=np.zeros(((2*Nv),(2*Nv)))
    for i in range(2*Nv):
        for j in range(2*Nv):
            if i==0:
                B[i,j] =Col_parameter_e*(deltt/delv)*0.5*H_per(pal_v[i],per_v[b])+Col_parameter_p*(deltt/delv)*0.5*H_perp(pal_v[i],per_v[b]) if j==0 else +Col_parameter_e*(F/2)*G_pal_per(pal_v[i],per_v[b])+Col_parameter_p*(F/2)*G_pal_perp(pal_v[i],per_v[b]) if j==1 else 0
            elif i==2*Nv:
                B[i,j] =Col_parameter_e*(deltt/delv)*0.5*H_per(pal_v[i],per_v[b])+Col_parameter_p*(deltt/delv)*0.5*H_perp(pal_v[i],per_v[b]) if j==2*Nx else -Col_parameter_e*(F/2)*G_pal_per(pal_v[i],per_v[b])-Col_parameter_p*(F/2)*G_pal_perp(pal_v[i],per_v[b]) if j==2*Nv-1 else 0
            else:
                B[i,j] =Col_parameter_e*(deltt/delv)*0.5*H_per(pal_v[i],per_v[b])+Col_parameter_p*(deltt/delv)*0.5*H_perp(pal_v[i],per_v[b]) if j==i else -Col_parameter_e*(F/2)*G_pal_per(pal_v[i],per_v[b])-Col_parameter_e*(F/2)*G_pal_perp(pal_v[i],per_v[b]) if j==i-1 else +Col_parameter_e*(F/2)*G_pal_per(pal_v[i],per_v[b])+Col_parameter_e*(F/2)*G_pal_perp(pal_v[i],per_v[b]) if j==i+1 else 0
    return B

>>> def Matrix_C1(b):
    C=np.zeros(((2*Nv),(2*Nv)))
    for i in range(2*Nv):
        for j in range(2*Nv):
            C[i,j] =-Col_parameter_e*(F/4)*G_per_2(pal_v[i],per_v[b])-Col_parameter_p*(F/4)*G_per_2p(pal_v[i],per_v[b]) if j==i else 0
    return C

>>> def Matrix_A_1(b):
    A_1=np.zeros(((2*Nv),(2*Nv)))
    for i in range(2*Nv):
        for j in range(2*Nv):
            if i==0:
                A_1[i,j] =1-Col_parameter_e*(-2*np.pi*deltt*Collision_Core(pal_v[i],per_v[b])-2*np.pi*deltt*ratio*Collision_Proton(pal_v[i],per_v[b])+(F/2)*G_per_2(pal_v[i],per_v[b])+(F/2)*G_per_2p(pal_v[i],per_v[b])+(F/2)*G_pal_2(pal_v[i],per_v[b])+(F/2)*G_pal_2p(pal_v[i],per_v[b])) if j==0 else Col_parameter_e*(deltt/delv)*0.5*H_pal(pal_v[i],per_v[b])+Col_parameter_p*(deltt/delv)*0.5*H_palp(pal_v[i],per_v[b]) if j==1 else Col_parameter_e*(F/4)*G_pal_2(pal_v[i],per_v[b])+Col_parameter_p*(F/4)*G_pal_2p(pal_v[i],per_v[b]) if j==2 else 0
            elif i==1:
                A_1[i,j] =-Col_parameter_e*(deltt/delv)*0.5*H_pal(pal_v[i],per_v[b])-Col_parameter_p*(deltt/delv)*0.5*H_palp(pal_v[i],per_v[b]) if j==0 else 1-Col_parameter_e*(-2*np.pi*deltt*Collision_Core(pal_v[i],per_v[b])-2*np.pi*deltt*ratio*Collision_Proton(pal_v[i],per_v[b])+(F/2)*G_per_2(pal_v[i],per_v[b])+(F/2)*G_per_2p(pal_v[i],per_v[b])+(F/2)*G_pal_2(pal_v[i],per_v[b])+(F/2)*G_pal_2p(pal_v[i],per_v[b])) if j==1 else Col_parameter_e*(deltt/delv)*0.5*H_pal(pal_v[i],per_v[b])+Col_parameter_p*(deltt/delv)*0.5*H_palp(pal_v[i],per_v[b]) if j==2 else Col_parameter_e*(F/4)*G_pal_2(pal_v[i],per_v[b])+Col_parameter_p*(F/4)*G_pal_2p(pal_v[i],per_v[b]) if j==3 else 0
            elif i==2*Nv-1:
                A_1[i,j] =Col_parameter_e*(F/4)*G_pal_2(pal_v[i],per_v[b])+Col_parameter_p*(F/4)*G_pal_2p(pal_v[i],per_v[b]) if j==2*Nv-3 else -Col_parameter_e*(deltt/delv)*0.5*H_pal(pal_v[i],per_v[b])-Col_parameter_p*(deltt/delv)*0.5*H_palp(pal_v[i],per_v[b]) if j==2*Nv-2 else 1-Col_parameter_e*(-2*np.pi*deltt*Collision_Core(pal_v[i],per_v[b])-2*np.pi*deltt*ratio*Collision_Proton(pal_v[i],per_v[b])+(F/2)*G_per_2(pal_v[i],per_v[b])+(F/2)*G_per_2p(pal_v[i],per_v[b])+(F/2)*G_pal_2(pal_v[i],per_v[b])+(F/2)*G_pal_2p(pal_v[i],per_v[b])) if j==2*Nv-1 else Col_parameter_e*(deltt/delv)*0.5*H_pal(pal_v[i],per_v[b])+Col_parameter_p*(deltt/delv)*0.5*H_palp(pal_v[i],per_v[b]) if j==2*Nv else 0
            elif i==2*Nv:
                A_1[i,j] =Col_parameter_e*(F/4)*G_pal_2(pal_v[i],per_v[b])+Col_parameter_p*(F/4)*G_pal_2p(pal_v[i],per_v[b]) if j==2*Nv-2 else -Col_parameter_e*(deltt/delv)*0.5*H_pal(pal_v[i],per_v[b])-Col_parameter_p*(deltt/delv)*0.5*H_palp(pal_v[i],per_v[b]) if j==2*Nv-1 else 1-Col_parameter_e*(-2*np.pi*deltt*Collision_Core(pal_v[i],per_v[b])-2*np.pi*deltt*ratio*Collision_Proton(pal_v[i],per_v[b])+(F/2)*G_per_2(pal_v[i],per_v[b])+(F/2)*G_per_2p(pal_v[i],per_v[b])+(F/2)*G_pal_2(pal_v[i],per_v[b])+(F/2)*G_pal_2p(pal_v[i],per_v[b])) if j==2*Nv else 0
            else:
                A_1[i,j] =Col_parameter_e*(F/4)*G_pal_2(pal_v[i],per_v[b])+Col_parameter_p*(F/4)*G_pal_2p(pal_v[i],per_v[b]) if j==i-2 else -Col_parameter_e*(deltt/delv)*0.5*H_pal(pal_v[i],per_v[b])-Col_parameter_p*(deltt/delv)*0.5*H_palp(pal_v[i],per_v[b]) if j==i-1 else 1-Col_parameter_e*(-2*np.pi*deltt*Collision_Core(pal_v[i],per_v[b])-2*np.pi*deltt*ratio*Collision_Proton(pal_v[i],per_v[b])+(F/2)*G_per_2(pal_v[i],per_v[b])+(F/2)*G_per_2p(pal_v[i],per_v[b])+(F/2)*G_pal_2(pal_v[i],per_v[b])+(F/2)*G_pal_2p(pal_v[i],per_v[b])) if j==i else Col_parameter_e*(deltt/delv)*0.5*H_pal(pal_v[i],per_v[b])+Col_parameter_p*(deltt/delv)*0.5*H_palp(pal_v[i],per_v[b]) if j==i+1 else Col_parameter_e*(F/4)*G_pal_2(pal_v[i],per_v[b])+Col_parameter_p*(F/4)*G_pal_2p(pal_v[i],per_v[b]) if j==i+2 else 0
    return A_1

>>> AA2=np.zeros(((2*Nv)*(2*Nv),(2*Nv)*(2*Nv)))
>>> for a in range(2*Nv):
    for b in range(2*Nv):
        if a==b:
            AA2[a*2*Nv:(a+1)*2*Nv,b*2*Nv:(b+1)*2*Nv]=Matrix_A(a)

            
>>> for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            AA2[(a+1)*2*Nv:(a+2)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=Matrix_B1(a+1)

            
>>> for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            AA2[a*2*Nv:(a+1)*2*Nv,(b+1)*2*Nv:(b+2)*2*Nv]=-Matrix_B1(a)

            
>>> for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            AA2[(a+2)*2*Nv:(a+3)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=Matrix_C1(a+2)

            
>>> for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            AA2[a*2*Nv:(a+1)*2*Nv,(b+2)*2*Nv:(b+3)*2*Nv]=Matrix_C1(a)

            
>>> AA_12 = inv(AA2)
>>> QQ2=np.zeros(((2*Nv)*(2*Nv),(2*Nv)*(2*Nv)))
>>> for a in range(2*Nv):
    for b in range(2*Nv):
        if a==b:
            QQ2[a*2*Nv:(a+1)*2*Nv,b*2*Nv:(b+1)*2*Nv]=Matrix_A_1(a)

            
>>> for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            QQ2[(a+1)*2*Nv:(a+2)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=-Matrix_B1(a+1)

>>> for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            QQ2[a*2*Nv:(a+1)*2*Nv,(b+1)*2*Nv:(b+2)*2*Nv]=Matrix_B1(a)

            
>>> for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            QQ2[(a+2)*2*Nv:(a+3)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=-Matrix_C1(a+2)

            
>>> for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            QQ2[a*2*Nv:(a+1)*2*Nv,(b+2)*2*Nv:(b+3)*2*Nv]=-Matrix_C1(a)

            
>>> AQ2=dot(AA_12,QQ2)
>>> f_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> solu2=np.zeros(shape = (Nv, 2*Nv))
>>> fc_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> ff_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        f_1[j*2*Nv+i]=Kappa_Initial_Strahl(pal_v[i],per_v[j])

        
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        fc_1[j*2*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j])

        
>>> ff_1=f_1+fc_1
>>> Mf_1=np.max(ff_1)
>>> for t in range(500):
        f_1=dot(AQ, f_1)

        
>>> f_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> solu2=np.zeros(shape = (Nv, 2*Nv))
>>> fc_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> ff_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        f_1[j*2*Nv+i]=Kappa_Initial_Strahl(pal_v[i],per_v[j])

        
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        fc_1[j*2*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j])

        
>>> ff_1=f_1+fc_1
>>> Mf_1=np.max(ff_1)
>>> for t in range(400):
        f_1=dot(AQ, f_1)

        
>>> for k in range(8):
	ff_1=f_1+fc_1
	for j in range(Nv):
		for i in range(2*Nv):
			if abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>1:
				solu2[j,i]=0
			elif abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>10**(-4.8):
				solu2[j,i]=np.log10(abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1)
			else:
				solu2[j,i]=-10
	fig = plt.figure()
	fig.set_dpi(350)
	plt.contourf(X2, Y2,solu2, cont_lev,cmap='Blues');
	ax = plt.gca()
	ax.spines['left'].set_position('center')
	ax.spines['left'].set_smart_bounds(True)
	ax.spines['bottom'].set_position('zero')
	ax.spines['bottom'].set_smart_bounds(True)
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	ax.xaxis.set_ticks_position('bottom')
	plt.axis('equal')
	plt.yticks([2,4,6,8])
	plt.xticks([-6,-4,-2,0,2,4,6])
	plt.rc('font', size=9)
	plt.tick_params(labelsize=9)
	plt.text(-0.2,-1.6,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae}$', fontsize=9)
	plt.text(-0.2,8.3,r'$\mathcal{v}_\perp/\mathcal{v}_{Ae}$', fontsize=9)
	plt.colorbar(label=r'$Log(F/F_{MAX})$')
	plt.show()
	for t in range(1000):
		f_1=dot(AQ2, f_1)
		for y in range(2*Nv):
			for l in range(2*Nv):
				if ((pal_v[l]**2+per_v[y]**2)**0.5)<0.1:
					f_1[y*2*Nv+l]=10**(-10)

					
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC87190>

Warning (from warnings module):
  File "<pyshell#200>", line 16
MatplotlibDeprecationWarning: 
The set_smart_bounds function was deprecated in Matplotlib 3.2 and will be removed two minor releases later.

Warning (from warnings module):
  File "<pyshell#200>", line 18
MatplotlibDeprecationWarning: 
The set_smart_bounds function was deprecated in Matplotlib 3.2 and will be removed two minor releases later.
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A9F1BB0>, <matplotlib.axis.YTick object at 0x0000018E2AAC8BE0>, <matplotlib.axis.YTick object at 0x0000018E2AB7B280>, <matplotlib.axis.YTick object at 0x0000018E2AB7BB80>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AD76B20>, <matplotlib.axis.XTick object at 0x0000018E2AAC53D0>, <matplotlib.axis.XTick object at 0x0000018E2AB7B1F0>, <matplotlib.axis.XTick object at 0x0000018E2AC005E0>, <matplotlib.axis.XTick object at 0x0000018E2AC00910>, <matplotlib.axis.XTick object at 0x0000018E2AC00070>, <matplotlib.axis.XTick object at 0x0000018E2B31C940>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB8FC70>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC6A670>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AD5EB80>, <matplotlib.axis.YTick object at 0x0000018E2B324A60>, <matplotlib.axis.YTick object at 0x0000018E2A99D0A0>, <matplotlib.axis.YTick object at 0x0000018E2A99D5B0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AA46610>, <matplotlib.axis.XTick object at 0x0000018E2A9F32E0>, <matplotlib.axis.XTick object at 0x0000018E2A99D820>, <matplotlib.axis.XTick object at 0x0000018E2AC98250>, <matplotlib.axis.XTick object at 0x0000018E2AC98640>, <matplotlib.axis.XTick object at 0x0000018E2AC98B50>, <matplotlib.axis.XTick object at 0x0000018E2ACC72B0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB81850>
<matplotlib.contour.QuadContourSet object at 0x0000018E2BCA7610>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB5F400>, <matplotlib.axis.YTick object at 0x0000018E2B3B68B0>, <matplotlib.axis.YTick object at 0x0000018E2B324F70>, <matplotlib.axis.YTick object at 0x0000018E2B3241C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB7B850>, <matplotlib.axis.XTick object at 0x0000018E2AC413D0>, <matplotlib.axis.XTick object at 0x0000018E2B324EB0>, <matplotlib.axis.XTick object at 0x0000018E2A990490>, <matplotlib.axis.XTick object at 0x0000018E2A990D00>, <matplotlib.axis.XTick object at 0x0000018E2A97A4F0>, <matplotlib.axis.XTick object at 0x0000018E2A97A790>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AA9B7C0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2B34C880>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B47F7C0>, <matplotlib.axis.YTick object at 0x0000018E2AA0E040>, <matplotlib.axis.YTick object at 0x0000018E2B45C400>, <matplotlib.axis.YTick object at 0x0000018E2B45CDF0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B47F580>, <matplotlib.axis.XTick object at 0x0000018E2B2AFBB0>, <matplotlib.axis.XTick object at 0x0000018E2B45C520>, <matplotlib.axis.XTick object at 0x0000018E2A9F1400>, <matplotlib.axis.XTick object at 0x0000018E2A9F1EB0>, <matplotlib.axis.XTick object at 0x0000018E2AD5B400>, <matplotlib.axis.XTick object at 0x0000018E2AD5B5E0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ABA9D60>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB30F10>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAC8FD0>, <matplotlib.axis.YTick object at 0x0000018E2B3187C0>, <matplotlib.axis.YTick object at 0x0000018E2A9E5A60>, <matplotlib.axis.YTick object at 0x0000018E2AC4C490>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB8F160>, <matplotlib.axis.XTick object at 0x0000018E2AC61370>, <matplotlib.axis.XTick object at 0x0000018E2AC4C220>, <matplotlib.axis.XTick object at 0x0000018E2AC36340>, <matplotlib.axis.XTick object at 0x0000018E2AC367F0>, <matplotlib.axis.XTick object at 0x0000018E2AC36CA0>, <matplotlib.axis.XTick object at 0x0000018E2AA16730>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9FA850>
<matplotlib.contour.QuadContourSet object at 0x0000018E2BCC7550>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC70C70>, <matplotlib.axis.YTick object at 0x0000018E2B2BFA90>, <matplotlib.axis.YTick object at 0x0000018E2ABD69A0>, <matplotlib.axis.YTick object at 0x0000018E2ABE5040>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B3BA550>, <matplotlib.axis.XTick object at 0x0000018E2BC99460>, <matplotlib.axis.XTick object at 0x0000018E2ABE5A30>, <matplotlib.axis.XTick object at 0x0000018E2ABE5F40>, <matplotlib.axis.XTick object at 0x0000018E2ABD6D90>, <matplotlib.axis.XTick object at 0x0000018E2ABE92E0>, <matplotlib.axis.XTick object at 0x0000018E2ABE97F0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB45940>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AD80850>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ABEE2E0>, <matplotlib.axis.YTick object at 0x0000018E2AD742B0>, <matplotlib.axis.YTick object at 0x0000018E2B38A6D0>, <matplotlib.axis.YTick object at 0x0000018E2B38A0D0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ABEE1C0>, <matplotlib.axis.XTick object at 0x0000018E2A988EE0>, <matplotlib.axis.XTick object at 0x0000018E2B38A3D0>, <matplotlib.axis.XTick object at 0x0000018E2AAE0F10>, <matplotlib.axis.XTick object at 0x0000018E2AAE09A0>, <matplotlib.axis.XTick object at 0x0000018E2AAE0DF0>, <matplotlib.axis.XTick object at 0x0000018E2AAEEDC0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AAE6CA0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2A97AAC0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB228B0>, <matplotlib.axis.YTick object at 0x0000018E2A987280>, <matplotlib.axis.YTick object at 0x0000018E2B2A14C0>, <matplotlib.axis.YTick object at 0x0000018E2AAB5B80>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ABEE4F0>, <matplotlib.axis.XTick object at 0x0000018E2B31E0D0>, <matplotlib.axis.XTick object at 0x0000018E2AAB5C40>, <matplotlib.axis.XTick object at 0x0000018E2AAB5F40>, <matplotlib.axis.XTick object at 0x0000018E2A964820>, <matplotlib.axis.XTick object at 0x0000018E2A964EE0>, <matplotlib.axis.XTick object at 0x0000018E2A964580>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B47F970>
>>> for k in range(8):
	ff_1=f_1+fc_1
	for j in range(Nv):
		for i in range(2*Nv):
			if abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>1:
				solu2[j,i]=0
			elif abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>10**(-4.8):
				solu2[j,i]=np.log10(abs(f_1[(j+Nv)*2*Nv+i])/Mf_1)
			else:
				solu2[j,i]=-10
	fig = plt.figure()
	fig.set_dpi(350)
	plt.contourf(X2, Y2,solu2, cont_lev,cmap='Blues');
	ax = plt.gca()
	ax.spines['left'].set_position('center')
	ax.spines['left'].set_smart_bounds(True)
	ax.spines['bottom'].set_position('zero')
	ax.spines['bottom'].set_smart_bounds(True)
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	ax.xaxis.set_ticks_position('bottom')
	plt.axis('equal')
	plt.yticks([2,4,6,8])
	plt.xticks([-6,-4,-2,0,2,4,6])
	plt.rc('font', size=9)
	plt.tick_params(labelsize=9)
	plt.text(-0.2,-1.6,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae}$', fontsize=9)
	plt.text(-0.2,8.3,r'$\mathcal{v}_\perp/\mathcal{v}_{Ae}$', fontsize=9)
	plt.colorbar(label=r'$Log(F/F_{MAX})$')
	plt.show()

	
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC3BAC0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2BCB9D30>, <matplotlib.axis.YTick object at 0x0000018E2A9B2610>, <matplotlib.axis.YTick object at 0x0000018E2ACC7BE0>, <matplotlib.axis.YTick object at 0x0000018E2ACC7340>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2BCB96A0>, <matplotlib.axis.XTick object at 0x0000018E2AC87400>, <matplotlib.axis.XTick object at 0x0000018E2ACC7520>, <matplotlib.axis.XTick object at 0x0000018E2A9DAC10>, <matplotlib.axis.XTick object at 0x0000018E2A9DA700>, <matplotlib.axis.XTick object at 0x0000018E2B2AF9D0>, <matplotlib.axis.XTick object at 0x0000018E2B2AF4F0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AAD68B0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2BCA8760>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A9F94F0>, <matplotlib.axis.YTick object at 0x0000018E2A9F9DC0>, <matplotlib.axis.YTick object at 0x0000018E2A9FABB0>, <matplotlib.axis.YTick object at 0x0000018E2A9643A0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A9DA130>, <matplotlib.axis.XTick object at 0x0000018E2A9643D0>, <matplotlib.axis.XTick object at 0x0000018E2A964C10>, <matplotlib.axis.XTick object at 0x0000018E2A964AC0>, <matplotlib.axis.XTick object at 0x0000018E2A9FA790>, <matplotlib.axis.XTick object at 0x0000018E2B2A1C70>, <matplotlib.axis.XTick object at 0x0000018E2B2A1760>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B2F6A00>
<matplotlib.contour.QuadContourSet object at 0x0000018E2A996760>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAEEC10>, <matplotlib.axis.YTick object at 0x0000018E2BCBEBE0>, <matplotlib.axis.YTick object at 0x0000018E2ABEECD0>, <matplotlib.axis.YTick object at 0x0000018E2ABEE910>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ACCED60>, <matplotlib.axis.XTick object at 0x0000018E2AD5BB50>, <matplotlib.axis.XTick object at 0x0000018E2ABEEE20>, <matplotlib.axis.XTick object at 0x0000018E2ABF0AF0>, <matplotlib.axis.XTick object at 0x0000018E2ABF05B0>, <matplotlib.axis.XTick object at 0x0000018E2ABF0E50>, <matplotlib.axis.XTick object at 0x0000018E2AB06D30>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9FB760>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB8B970>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AD74A90>, <matplotlib.axis.YTick object at 0x0000018E2A9DA9D0>, <matplotlib.axis.YTick object at 0x0000018E2A9874F0>, <matplotlib.axis.YTick object at 0x0000018E2A987520>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AD74730>, <matplotlib.axis.XTick object at 0x0000018E2AC08580>, <matplotlib.axis.XTick object at 0x0000018E2A987E20>, <matplotlib.axis.XTick object at 0x0000018E2ACC6E50>, <matplotlib.axis.XTick object at 0x0000018E2ACC6100>, <matplotlib.axis.XTick object at 0x0000018E2ABDE190>, <matplotlib.axis.XTick object at 0x0000018E2ABDE250>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AA16160>
<matplotlib.contour.QuadContourSet object at 0x0000018E2ABFB1C0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AA26130>, <matplotlib.axis.YTick object at 0x0000018E2AA47CD0>, <matplotlib.axis.YTick object at 0x0000018E2ACADB20>, <matplotlib.axis.YTick object at 0x0000018E2AC96070>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A9E3AF0>, <matplotlib.axis.XTick object at 0x0000018E2AAC2DC0>, <matplotlib.axis.XTick object at 0x0000018E2ACADA00>, <matplotlib.axis.XTick object at 0x0000018E2AC962E0>, <matplotlib.axis.XTick object at 0x0000018E2AC92100>, <matplotlib.axis.XTick object at 0x0000018E2AC92610>, <matplotlib.axis.XTick object at 0x0000018E2AC92B20>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB4E310>
<matplotlib.contour.QuadContourSet object at 0x0000018E2B4124F0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B31EAF0>, <matplotlib.axis.YTick object at 0x0000018E2BCA5760>, <matplotlib.axis.YTick object at 0x0000018E2A996F40>, <matplotlib.axis.YTick object at 0x0000018E2AB06310>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAB5070>, <matplotlib.axis.XTick object at 0x0000018E2BCA5EB0>, <matplotlib.axis.XTick object at 0x0000018E2A996250>, <matplotlib.axis.XTick object at 0x0000018E2AB06970>, <matplotlib.axis.XTick object at 0x0000018CFFD74940>, <matplotlib.axis.XTick object at 0x0000018CFFD74730>, <matplotlib.axis.XTick object at 0x0000018E2AAEEA90>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2BCA7F70>
<matplotlib.contour.QuadContourSet object at 0x0000018E2A97ACD0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B2A1670>, <matplotlib.axis.YTick object at 0x0000018E2BCBE0A0>, <matplotlib.axis.YTick object at 0x0000018E2AAE08B0>, <matplotlib.axis.YTick object at 0x0000018E2ABA9C10>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B2A1CA0>, <matplotlib.axis.XTick object at 0x0000018E2A971C10>, <matplotlib.axis.XTick object at 0x0000018E2ABA9F40>, <matplotlib.axis.XTick object at 0x0000018E2AB301F0>, <matplotlib.axis.XTick object at 0x0000018E2AB306D0>, <matplotlib.axis.XTick object at 0x0000018E2AB30D00>, <matplotlib.axis.XTick object at 0x0000018E2A990EB0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9F3D00>
<matplotlib.contour.QuadContourSet object at 0x0000018E2ABF0CD0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B318A30>, <matplotlib.axis.YTick object at 0x0000018E2AAB2820>, <matplotlib.axis.YTick object at 0x0000018E2AC4C700>, <matplotlib.axis.YTick object at 0x0000018E2AC4C250>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B318070>, <matplotlib.axis.XTick object at 0x0000018E2A96FBE0>, <matplotlib.axis.XTick object at 0x0000018E2AC4C8B0>, <matplotlib.axis.XTick object at 0x0000018E2AC3DB20>, <matplotlib.axis.XTick object at 0x0000018E2AC3DBE0>, <matplotlib.axis.XTick object at 0x0000018E2AC3D520>, <matplotlib.axis.XTick object at 0x0000018E2A9B1250>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B34C550>
>>> f_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> solu2=np.zeros(shape = (Nv, 2*Nv))
>>> fc_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> ff_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        f_1[j*2*Nv+i]=Kappa_Initial_Strahl(pal_v[i],per_v[j])

        
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        fc_1[j*2*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j])

        
>>> ff_1=f_1+fc_1
>>> Mf_1=np.max(ff_1)
>>> for t in range(400):
        f_1=dot(AQ, f_1)

        
>>> for k in range(21):
	ff_1=f_1+fc_1
	for j in range(Nv):
		for i in range(2*Nv):
			if abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>1:
				solu2[j,i]=0
			elif abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>10**(-5):
				solu2[j,i]=np.log10(abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1)
			else:
				solu2[j,i]=-10
	fig = plt.figure()
	fig.set_dpi(350)
	plt.contourf(X2, Y2,solu2, cont_lev,cmap='Blues');
	ax = plt.gca()
	ax.spines['left'].set_position('center')
	ax.spines['left'].set_smart_bounds(True)
	ax.spines['bottom'].set_position('zero')
	ax.spines['bottom'].set_smart_bounds(True)
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	ax.xaxis.set_ticks_position('bottom')
	plt.axis('equal')
	plt.yticks([2,4,6,8])
	plt.xticks([-6,-4,-2,0,2,4,6])
	plt.rc('font', size=9)
	plt.tick_params(labelsize=9)
	plt.text(-0.2,-1.6,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae}$', fontsize=9)
	plt.text(-0.2,8.3,r'$\mathcal{v}_\perp/\mathcal{v}_{Ae}$', fontsize=9)
	plt.colorbar(label=r'$Log(F/F_{MAX})$')
	plt.show()
	for t in range(350):
		f_1=dot(AQ2, f_1)
		for y in range(2*Nv):
			for l in range(2*Nv):
				if ((pal_v[l]**2+per_v[y]**2)**0.5)<0.1:
					f_1[y*2*Nv+l]=10**(-10)

					
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAFC760>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AD06130>, <matplotlib.axis.YTick object at 0x0000018E2AC39790>, <matplotlib.axis.YTick object at 0x0000018E2AB226A0>, <matplotlib.axis.YTick object at 0x0000018E2B2F6D60>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB231C0>, <matplotlib.axis.XTick object at 0x0000018E2A9F2D90>, <matplotlib.axis.XTick object at 0x0000018E2B45CC10>, <matplotlib.axis.XTick object at 0x0000018E2AB22C40>, <matplotlib.axis.XTick object at 0x0000018E2B45C940>, <matplotlib.axis.XTick object at 0x0000018E2AAE0910>, <matplotlib.axis.XTick object at 0x0000018E2AAE0F10>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B2A11F0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2A9CCD00>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ABE95E0>, <matplotlib.axis.YTick object at 0x0000018E2AAC8A90>, <matplotlib.axis.YTick object at 0x0000018E2AB8B3D0>, <matplotlib.axis.YTick object at 0x0000018E2ABAEA00>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A96F070>, <matplotlib.axis.XTick object at 0x0000018E2AAC86A0>, <matplotlib.axis.XTick object at 0x0000018E2AB8B8E0>, <matplotlib.axis.XTick object at 0x0000018E2ABAE640>, <matplotlib.axis.XTick object at 0x0000018E2AB06D30>, <matplotlib.axis.XTick object at 0x0000018E2AB061C0>, <matplotlib.axis.XTick object at 0x0000018E2AB068B0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A964970>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAC2730>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ACCEA00>, <matplotlib.axis.YTick object at 0x0000018E2AC41640>, <matplotlib.axis.YTick object at 0x0000018E2B412BB0>, <matplotlib.axis.YTick object at 0x0000018E2B412FA0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ACCE100>, <matplotlib.axis.XTick object at 0x0000018E2ABC4CA0>, <matplotlib.axis.XTick object at 0x0000018E2B4122E0>, <matplotlib.axis.XTick object at 0x0000018E2B38A070>, <matplotlib.axis.XTick object at 0x0000018E2B38A520>, <matplotlib.axis.XTick object at 0x0000018E2B38ACD0>, <matplotlib.axis.XTick object at 0x0000018E2AB3C070>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B47F3A0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAB72E0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AD923A0>, <matplotlib.axis.YTick object at 0x0000018E2AD05E50>, <matplotlib.axis.YTick object at 0x0000018E2AD5A9A0>, <matplotlib.axis.YTick object at 0x0000018E2AD7C040>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AD920A0>, <matplotlib.axis.XTick object at 0x0000018E2ACF9220>, <matplotlib.axis.XTick object at 0x0000018E2AD5A7C0>, <matplotlib.axis.XTick object at 0x0000018E2AD7C2E0>, <matplotlib.axis.XTick object at 0x0000018E2AD730D0>, <matplotlib.axis.XTick object at 0x0000018E2AD73490>, <matplotlib.axis.XTick object at 0x0000018E2AD739A0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ABAD190>
<matplotlib.contour.QuadContourSet object at 0x0000018E2ABC4040>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB3C940>, <matplotlib.axis.YTick object at 0x0000018E2AC6ACA0>, <matplotlib.axis.YTick object at 0x0000018E2AC41130>, <matplotlib.axis.YTick object at 0x0000018E2B44D370>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AD058E0>, <matplotlib.axis.XTick object at 0x0000018E2BCBE640>, <matplotlib.axis.XTick object at 0x0000018E2B44DEE0>, <matplotlib.axis.XTick object at 0x0000018E2A9CCA30>, <matplotlib.axis.XTick object at 0x0000018E2A9CC6A0>, <matplotlib.axis.XTick object at 0x0000018E2A9CC5B0>, <matplotlib.axis.XTick object at 0x0000018E2A9BE8E0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB650A0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC877F0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC24790>, <matplotlib.axis.YTick object at 0x0000018E2A9BE130>, <matplotlib.axis.YTick object at 0x0000018E2AA0C8B0>, <matplotlib.axis.YTick object at 0x0000018E2AA0C2B0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC24340>, <matplotlib.axis.XTick object at 0x0000018E2AC084F0>, <matplotlib.axis.XTick object at 0x0000018E2AA0CBB0>, <matplotlib.axis.XTick object at 0x0000018E2A964D30>, <matplotlib.axis.XTick object at 0x0000018E2A9642E0>, <matplotlib.axis.XTick object at 0x0000018E2A9F9730>, <matplotlib.axis.XTick object at 0x0000018E2A9F9760>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B27E910>
<matplotlib.contour.QuadContourSet object at 0x0000018E2B38AAF0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAFFFD0>, <matplotlib.axis.YTick object at 0x0000018E2ACCE880>, <matplotlib.axis.YTick object at 0x0000018E2A96FEB0>, <matplotlib.axis.YTick object at 0x0000018E2A96FFD0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AD943D0>, <matplotlib.axis.XTick object at 0x0000018E2AB43400>, <matplotlib.axis.XTick object at 0x0000018E2A96F7F0>, <matplotlib.axis.XTick object at 0x0000018E2AC2E430>, <matplotlib.axis.XTick object at 0x0000018E2ACAE730>, <matplotlib.axis.XTick object at 0x0000018E2ACAE850>, <matplotlib.axis.XTick object at 0x0000018E2ACAE580>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AAC89D0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2ABDE880>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC653D0>, <matplotlib.axis.YTick object at 0x0000018E2A990160>, <matplotlib.axis.YTick object at 0x0000018E2A9B2EE0>, <matplotlib.axis.YTick object at 0x0000018E2A9B2460>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AD75BE0>, <matplotlib.axis.XTick object at 0x0000018E2A9901C0>, <matplotlib.axis.XTick object at 0x0000018E2A9F34F0>, <matplotlib.axis.XTick object at 0x0000018E2A9B2C70>, <matplotlib.axis.XTick object at 0x0000018E2A9F3A30>, <matplotlib.axis.XTick object at 0x0000018E2A9F3340>, <matplotlib.axis.XTick object at 0x0000018E2A9F3310>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B30F7C0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2A9966D0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ABC46D0>, <matplotlib.axis.YTick object at 0x0000018E2AAE08B0>, <matplotlib.axis.YTick object at 0x0000018E2AB5FD00>, <matplotlib.axis.YTick object at 0x0000018E2AB5F280>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAEE700>, <matplotlib.axis.XTick object at 0x0000018E2AB3C790>, <matplotlib.axis.XTick object at 0x0000018E2AB5FF70>, <matplotlib.axis.XTick object at 0x0000018E2AAFF8E0>, <matplotlib.axis.XTick object at 0x0000018E2ACA7AF0>, <matplotlib.axis.XTick object at 0x0000018E2ACA72B0>, <matplotlib.axis.XTick object at 0x0000018E2ACA71C0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AA3CD30>
<matplotlib.contour.QuadContourSet object at 0x0000018E2BC9AEE0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC4FF70>, <matplotlib.axis.YTick object at 0x0000018E2ABF6F40>, <matplotlib.axis.YTick object at 0x0000018E2AB36D90>, <matplotlib.axis.YTick object at 0x0000018E2AC9F760>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC00E80>, <matplotlib.axis.XTick object at 0x0000018E2AC393A0>, <matplotlib.axis.XTick object at 0x0000018E2AB36F70>, <matplotlib.axis.XTick object at 0x0000018E2AC9F4C0>, <matplotlib.axis.XTick object at 0x0000018E2AD07820>, <matplotlib.axis.XTick object at 0x0000018E2AD07D30>, <matplotlib.axis.XTick object at 0x0000018E2AD07100>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AA26400>
<matplotlib.contour.QuadContourSet object at 0x0000018E2A975580>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AADB5B0>, <matplotlib.axis.YTick object at 0x0000018E2AD63520>, <matplotlib.axis.YTick object at 0x0000018E2AB75C40>, <matplotlib.axis.YTick object at 0x0000018E2A98E190>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AADB7F0>, <matplotlib.axis.XTick object at 0x0000018E2AC6A5B0>, <matplotlib.axis.XTick object at 0x0000018E2AB75A00>, <matplotlib.axis.XTick object at 0x0000018E2A98E040>, <matplotlib.axis.XTick object at 0x0000018E2B33D220>, <matplotlib.axis.XTick object at 0x0000018E2B33D730>, <matplotlib.axis.XTick object at 0x0000018E2B33DC40>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AC48430>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAA4610>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A97DB50>, <matplotlib.axis.YTick object at 0x0000018E2A96A520>, <matplotlib.axis.YTick object at 0x0000018E2B44DF70>, <matplotlib.axis.YTick object at 0x0000018E2B44D190>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B3255E0>, <matplotlib.axis.XTick object at 0x0000018E2ABAA0A0>, <matplotlib.axis.XTick object at 0x0000018E2B44D280>, <matplotlib.axis.XTick object at 0x0000018E2AB65CD0>, <matplotlib.axis.XTick object at 0x0000018E2AB65A60>, <matplotlib.axis.XTick object at 0x0000018E2B416D90>, <matplotlib.axis.XTick object at 0x0000018E2B416A90>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AC3AA00>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAB26D0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAB5C40>, <matplotlib.axis.YTick object at 0x0000018E2B34C700>, <matplotlib.axis.YTick object at 0x0000018E2AC39070>, <matplotlib.axis.YTick object at 0x0000018E2AC61E80>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A996AC0>, <matplotlib.axis.XTick object at 0x0000018E2AC48C40>, <matplotlib.axis.XTick object at 0x0000018E2AC39400>, <matplotlib.axis.XTick object at 0x0000018E2AC616D0>, <matplotlib.axis.XTick object at 0x0000018E2AB305E0>, <matplotlib.axis.XTick object at 0x0000018E2AB306A0>, <matplotlib.axis.XTick object at 0x0000018E2AB302B0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB6EE20>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAC6700>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A971A00>, <matplotlib.axis.YTick object at 0x0000018E2ABC4040>, <matplotlib.axis.YTick object at 0x0000018E2B45C4F0>, <matplotlib.axis.YTick object at 0x0000018E2AC2ED30>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B318970>, <matplotlib.axis.XTick object at 0x0000018E2AC87790>, <matplotlib.axis.XTick object at 0x0000018E2B45C040>, <matplotlib.axis.XTick object at 0x0000018E2AC2E400>, <matplotlib.axis.XTick object at 0x0000018E2AD75430>, <matplotlib.axis.XTick object at 0x0000018E2AD75B50>, <matplotlib.axis.XTick object at 0x0000018E2AD75670>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ACCE4C0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC00940>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B38A280>, <matplotlib.axis.YTick object at 0x0000018E2AC00880>, <matplotlib.axis.YTick object at 0x0000018E2AAEECD0>, <matplotlib.axis.YTick object at 0x0000018E2AD707C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B456760>, <matplotlib.axis.XTick object at 0x0000018E2AD70850>, <matplotlib.axis.XTick object at 0x0000018E2AD70880>, <matplotlib.axis.XTick object at 0x0000018E2AD705B0>, <matplotlib.axis.XTick object at 0x0000018E2B3247F0>, <matplotlib.axis.XTick object at 0x0000018E2B3246D0>, <matplotlib.axis.XTick object at 0x0000018E2B324580>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AD76400>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB65820>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAA4A60>, <matplotlib.axis.YTick object at 0x0000018E2ABAD7F0>, <matplotlib.axis.YTick object at 0x0000018E2ABB88B0>, <matplotlib.axis.YTick object at 0x0000018E2ABB8250>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ABC47C0>, <matplotlib.axis.XTick object at 0x0000018E2ABAD1C0>, <matplotlib.axis.XTick object at 0x0000018E2AC9FCA0>, <matplotlib.axis.XTick object at 0x0000018E2ABB8AF0>, <matplotlib.axis.XTick object at 0x0000018E2B2F69D0>, <matplotlib.axis.XTick object at 0x0000018E2B33D430>, <matplotlib.axis.XTick object at 0x0000018E2B33DCD0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A97DBE0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2A9C7430>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAD0100>, <matplotlib.axis.YTick object at 0x0000018E2AC45520>, <matplotlib.axis.YTick object at 0x0000018E2A971760>, <matplotlib.axis.YTick object at 0x0000018E2A971340>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAD0520>, <matplotlib.axis.XTick object at 0x0000018E2ABBDD90>, <matplotlib.axis.XTick object at 0x0000018E2A9714F0>, <matplotlib.axis.XTick object at 0x0000018E2B2A1EE0>, <matplotlib.axis.XTick object at 0x0000018E2A96F820>, <matplotlib.axis.XTick object at 0x0000018E2A96F790>, <matplotlib.axis.XTick object at 0x0000018E2A96F8B0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ABA92E0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC30730>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B334100>, <matplotlib.axis.YTick object at 0x0000018E2B317160>, <matplotlib.axis.YTick object at 0x0000018E2AAEAEE0>, <matplotlib.axis.YTick object at 0x0000018E2AB23040>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A9F1400>, <matplotlib.axis.XTick object at 0x0000018E2ABAE220>, <matplotlib.axis.XTick object at 0x0000018E2ABAEF10>, <matplotlib.axis.XTick object at 0x0000018E2AAEA940>, <matplotlib.axis.XTick object at 0x0000018E2ABAE700>, <matplotlib.axis.XTick object at 0x0000018E2AAB50A0>, <matplotlib.axis.XTick object at 0x0000018E2AAB5E80>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB65BE0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2B44D880>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B324610>, <matplotlib.axis.YTick object at 0x0000018E2ACFA910>, <matplotlib.axis.YTick object at 0x0000018E2AC39E20>, <matplotlib.axis.YTick object at 0x0000018E2AC39070>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AD70FA0>, <matplotlib.axis.XTick object at 0x0000018E2B345970>, <matplotlib.axis.XTick object at 0x0000018E2AC39280>, <matplotlib.axis.XTick object at 0x0000018E2AB7B490>, <matplotlib.axis.XTick object at 0x0000018E2AB7B640>, <matplotlib.axis.XTick object at 0x0000018E2AD5B6D0>, <matplotlib.axis.XTick object at 0x0000018E2AD5B280>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ACC34C0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC6AA60>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AD92D90>, <matplotlib.axis.YTick object at 0x0000018E2AC65730>, <matplotlib.axis.YTick object at 0x0000018E2AB5FA30>, <matplotlib.axis.YTick object at 0x0000018E2AB5F3D0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018CFFD97B50>, <matplotlib.axis.XTick object at 0x0000018E2AD5E280>, <matplotlib.axis.XTick object at 0x0000018E2AB5F970>, <matplotlib.axis.XTick object at 0x0000018E2A990430>, <matplotlib.axis.XTick object at 0x0000018E2AC3B0D0>, <matplotlib.axis.XTick object at 0x0000018E2B318820>, <matplotlib.axis.XTick object at 0x0000018E2B318370>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AA46D00>
<matplotlib.contour.QuadContourSet object at 0x0000018E2BCC4790>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAC7A30>, <matplotlib.axis.YTick object at 0x0000018E2B329A30>, <matplotlib.axis.YTick object at 0x0000018E2AB31E50>, <matplotlib.axis.YTick object at 0x0000018E2AB143A0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAC7EE0>, <matplotlib.axis.XTick object at 0x0000018E2B47B490>, <matplotlib.axis.XTick object at 0x0000018E2AB14F70>, <matplotlib.axis.XTick object at 0x0000018E2AB14340>, <matplotlib.axis.XTick object at 0x0000018E2AB46430>, <matplotlib.axis.XTick object at 0x0000018E2AB46940>, <matplotlib.axis.XTick object at 0x0000018E2AB46E50>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB5A640>
>>> f_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> solu2=np.zeros(shape = (Nv, 2*Nv))
>>> fc_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> ff_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        f_1[j*2*Nv+i]=Kappa_Initial_Strahl(pal_v[i],per_v[j])

        
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        fc_1[j*2*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j])

        
>>> ff_1=f_1+fc_1
>>> Mf_1=np.max(ff_1)
>>> for t in range(400):
        f_1=dot(AQ, f_1)
        
SyntaxError: multiple statements found while compiling a single statement
>>> f_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> solu2=np.zeros(shape = (Nv, 2*Nv))
>>> fc_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> ff_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        f_1[j*2*Nv+i]=Kappa_Initial_Strahl(pal_v[i],per_v[j])

        
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        fc_1[j*2*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j])

        
>>> ff_1=f_1+fc_1
>>> Mf_1=np.max(ff_1)
>>> for t in range(400):
        f_1=dot(AQ, f_1)

        
>>> for k in range(21):
	ff_1=f_1+fc_1
	for j in range(Nv):
		for i in range(2*Nv):
			if abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>1:
				solu2[j,i]=0
			elif abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>10**(-5):
				solu2[j,i]=np.log10(abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1)
			else:
				solu2[j,i]=-10
	fig = plt.figure()
	fig.set_dpi(350)
	plt.contourf(X2, Y2,solu2, cont_lev,cmap='Blues');
	ax = plt.gca()
	ax.spines['left'].set_position('center')
	ax.spines['left'].set_smart_bounds(True)
	ax.spines['bottom'].set_position('zero')
	ax.spines['bottom'].set_smart_bounds(True)
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	ax.xaxis.set_ticks_position('bottom')
	plt.axis('equal')
	plt.yticks([2,4,6,8])
	plt.xticks([-6,-4,-2,0,2,4,6])
	plt.rc('font', size=9)
	plt.tick_params(labelsize=9)
	plt.text(-0.2,-1.6,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae}$', fontsize=9)
	plt.text(-0.2,8.3,r'$\mathcal{v}_\perp/\mathcal{v}_{Ae}$', fontsize=9)
	plt.colorbar(label=r'$Log(F/F_{MAX})$')
	plt.show()
	for t in range(350):
		f_1=dot(AQ2, f_1)
		for y in range(2*Nv):
			for l in range(2*Nv):
				if ((pal_v[l]**2+per_v[y]**2)**0.5)<0.1:
					f_1[y*2*Nv+l]=10**(-10)

					
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB064F0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAC7790>, <matplotlib.axis.YTick object at 0x0000018E2AC6A4C0>, <matplotlib.axis.YTick object at 0x0000018E2AAFCBB0>, <matplotlib.axis.YTick object at 0x0000018E2AAFC970>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B456130>, <matplotlib.axis.XTick object at 0x0000018E2BCA8460>, <matplotlib.axis.XTick object at 0x0000018E2AAFCFA0>, <matplotlib.axis.XTick object at 0x0000018E2ACCEFA0>, <matplotlib.axis.XTick object at 0x0000018E2ACCE700>, <matplotlib.axis.XTick object at 0x0000018E2ACCEB80>, <matplotlib.axis.XTick object at 0x0000018E2AAEEFA0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AD92700>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AD76D60>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B416760>, <matplotlib.axis.YTick object at 0x0000018E2B289C10>, <matplotlib.axis.YTick object at 0x0000018E2AC3A520>, <matplotlib.axis.YTick object at 0x0000018E2AA3C3D0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC4FD60>, <matplotlib.axis.XTick object at 0x0000018E2B3297F0>, <matplotlib.axis.XTick object at 0x0000018E2AA3CDC0>, <matplotlib.axis.XTick object at 0x0000018E2AA3C7C0>, <matplotlib.axis.XTick object at 0x0000018E2AB65B80>, <matplotlib.axis.XTick object at 0x0000018E2AB65E80>, <matplotlib.axis.XTick object at 0x0000018E2AB657C0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B44D1C0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2B3122E0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B345AF0>, <matplotlib.axis.YTick object at 0x0000018E2B34CA60>, <matplotlib.axis.YTick object at 0x0000018E2ABABD30>, <matplotlib.axis.YTick object at 0x0000018E2ABAB100>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B345E50>, <matplotlib.axis.XTick object at 0x0000018E2B34C7F0>, <matplotlib.axis.XTick object at 0x0000018E2ABAB5B0>, <matplotlib.axis.XTick object at 0x0000018E2BCA72B0>, <matplotlib.axis.XTick object at 0x0000018E2BCA78B0>, <matplotlib.axis.XTick object at 0x0000018E2AAD0370>, <matplotlib.axis.XTick object at 0x0000018E2AAD0C70>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AAEA400>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAEA790>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB8F970>, <matplotlib.axis.YTick object at 0x0000018E2AB8F1C0>, <matplotlib.axis.YTick object at 0x0000018E2A9CC6A0>, <matplotlib.axis.YTick object at 0x0000018E2B44DB50>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A988DC0>, <matplotlib.axis.XTick object at 0x0000018E2B44D610>, <matplotlib.axis.XTick object at 0x0000018E2B44D970>, <matplotlib.axis.XTick object at 0x0000018E2B44D640>, <matplotlib.axis.XTick object at 0x0000018E2AAB2790>, <matplotlib.axis.XTick object at 0x0000018E2AAB2520>, <matplotlib.axis.XTick object at 0x0000018E2AAE0AF0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ABEE280>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AD75190>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAEEB80>, <matplotlib.axis.YTick object at 0x0000018E2AC61AC0>, <matplotlib.axis.YTick object at 0x0000018E2AB38EE0>, <matplotlib.axis.YTick object at 0x0000018E2AB389A0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B416C40>, <matplotlib.axis.XTick object at 0x0000018E2AD07910>, <matplotlib.axis.XTick object at 0x0000018E2AB387F0>, <matplotlib.axis.XTick object at 0x0000018E2AC00910>, <matplotlib.axis.XTick object at 0x0000018E2AC00A60>, <matplotlib.axis.XTick object at 0x0000018E2AB14B20>, <matplotlib.axis.XTick object at 0x0000018E2AB142E0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2BCA81F0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2A9F3340>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A97A400>, <matplotlib.axis.YTick object at 0x0000018E2AC459D0>, <matplotlib.axis.YTick object at 0x0000018E2AAA6D00>, <matplotlib.axis.YTick object at 0x0000018E2AAA6520>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A97A7F0>, <matplotlib.axis.XTick object at 0x0000018E2B324100>, <matplotlib.axis.XTick object at 0x0000018E2AAA66A0>, <matplotlib.axis.XTick object at 0x0000018E2AB60D30>, <matplotlib.axis.XTick object at 0x0000018E2AB605B0>, <matplotlib.axis.XTick object at 0x0000018E2AB60250>, <matplotlib.axis.XTick object at 0x0000018E2AC16DC0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9F1E50>
<matplotlib.contour.QuadContourSet object at 0x0000018E2B34D9D0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC4AF70>, <matplotlib.axis.YTick object at 0x0000018E2A9AEF10>, <matplotlib.axis.YTick object at 0x0000018E2ACED0D0>, <matplotlib.axis.YTick object at 0x0000018E2ACED5E0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC4A130>, <matplotlib.axis.XTick object at 0x0000018E2B3417F0>, <matplotlib.axis.XTick object at 0x0000018E2ACED850>, <matplotlib.axis.XTick object at 0x0000018E2AD7C1C0>, <matplotlib.axis.XTick object at 0x0000018E2AD7C670>, <matplotlib.axis.XTick object at 0x0000018E2AD7CB80>, <matplotlib.axis.XTick object at 0x0000018E2AD630D0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A975880>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC443A0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A9B1100>, <matplotlib.axis.YTick object at 0x0000018E2AAA63D0>, <matplotlib.axis.YTick object at 0x0000018E2B2AFD30>, <matplotlib.axis.YTick object at 0x0000018E2B2AF310>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ACE6700>, <matplotlib.axis.XTick object at 0x0000018E2B412B50>, <matplotlib.axis.XTick object at 0x0000018E2B2AFC70>, <matplotlib.axis.XTick object at 0x0000018E2AC4A160>, <matplotlib.axis.XTick object at 0x0000018E2AC4AE80>, <matplotlib.axis.XTick object at 0x0000018E2A9B2580>, <matplotlib.axis.XTick object at 0x0000018E2A9B2670>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB30B20>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAC7490>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B435220>, <matplotlib.axis.YTick object at 0x0000018E2ABAD550>, <matplotlib.axis.YTick object at 0x0000018E2AB602B0>, <matplotlib.axis.YTick object at 0x0000018E2AB60250>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ABAE730>, <matplotlib.axis.XTick object at 0x0000018E2AC415E0>, <matplotlib.axis.XTick object at 0x0000018E2AB600D0>, <matplotlib.axis.XTick object at 0x0000018E2A9CC070>, <matplotlib.axis.XTick object at 0x0000018E2AA3CD30>, <matplotlib.axis.XTick object at 0x0000018E2AA3CF70>, <matplotlib.axis.XTick object at 0x0000018E2AA3CFD0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB14910>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAD0730>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ABAA250>, <matplotlib.axis.YTick object at 0x0000018E2AAE0A60>, <matplotlib.axis.YTick object at 0x0000018E2AA16A90>, <matplotlib.axis.YTick object at 0x0000018E2AA167C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC87070>, <matplotlib.axis.XTick object at 0x0000018E2B416A30>, <matplotlib.axis.XTick object at 0x0000018E2AA16C70>, <matplotlib.axis.XTick object at 0x0000018E2AAFF5E0>, <matplotlib.axis.XTick object at 0x0000018E2AAFF280>, <matplotlib.axis.XTick object at 0x0000018E2AAFF070>, <matplotlib.axis.XTick object at 0x0000018E2AC243D0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB43100>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC3B430>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB23E50>, <matplotlib.axis.YTick object at 0x0000018E2AB7B340>, <matplotlib.axis.YTick object at 0x0000018E2AB7BFD0>, <matplotlib.axis.YTick object at 0x0000018E2AB7B880>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AD70E80>, <matplotlib.axis.XTick object at 0x0000018E2AD5EAC0>, <matplotlib.axis.XTick object at 0x0000018E2ABB6550>, <matplotlib.axis.XTick object at 0x0000018E2B2F6D60>, <matplotlib.axis.XTick object at 0x0000018E2ABB6C10>, <matplotlib.axis.XTick object at 0x0000018E2ABB6D30>, <matplotlib.axis.XTick object at 0x0000018E2AC7C160>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB14550>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AD76520>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC41D60>, <matplotlib.axis.YTick object at 0x0000018E2BCA8AF0>, <matplotlib.axis.YTick object at 0x0000018E2ACCAD90>, <matplotlib.axis.YTick object at 0x0000018E2ACCAD60>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B456190>, <matplotlib.axis.XTick object at 0x0000018E2BCA8880>, <matplotlib.axis.XTick object at 0x0000018E2ACCAA90>, <matplotlib.axis.XTick object at 0x0000018E2ACB0FD0>, <matplotlib.axis.XTick object at 0x0000018E2ACB0C10>, <matplotlib.axis.XTick object at 0x0000018E2ACB0610>, <matplotlib.axis.XTick object at 0x0000018E2A975790>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AD72BB0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAA6F70>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB06910>, <matplotlib.axis.YTick object at 0x0000018E2AC00EB0>, <matplotlib.axis.YTick object at 0x0000018E2B38ADC0>, <matplotlib.axis.YTick object at 0x0000018E2B38A730>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B416880>, <matplotlib.axis.XTick object at 0x0000018E2AA46550>, <matplotlib.axis.XTick object at 0x0000018E2B38A2B0>, <matplotlib.axis.XTick object at 0x0000018E2AAC8100>, <matplotlib.axis.XTick object at 0x0000018E2AAC8550>, <matplotlib.axis.XTick object at 0x0000018E2AAC8A90>, <matplotlib.axis.XTick object at 0x0000018E2AB651F0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9AD910>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB46BB0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AA4DB50>, <matplotlib.axis.YTick object at 0x0000018E2AAA4BB0>, <matplotlib.axis.YTick object at 0x0000018E2AAA22B0>, <matplotlib.axis.YTick object at 0x0000018E2AAA27C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AD58D30>, <matplotlib.axis.XTick object at 0x0000018E2B323D90>, <matplotlib.axis.XTick object at 0x0000018E2AAA2640>, <matplotlib.axis.XTick object at 0x0000018E2BCB9370>, <matplotlib.axis.XTick object at 0x0000018E2BCB9850>, <matplotlib.axis.XTick object at 0x0000018E2BCB9D60>, <matplotlib.axis.XTick object at 0x0000018E2BCB72B0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9ECA60>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAA6FD0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B38A0D0>, <matplotlib.axis.YTick object at 0x0000018E2ACC3910>, <matplotlib.axis.YTick object at 0x0000018E2A9ADEB0>, <matplotlib.axis.YTick object at 0x0000018E2A9AD6A0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC1DB50>, <matplotlib.axis.XTick object at 0x0000018E2AD5FD00>, <matplotlib.axis.XTick object at 0x0000018E2A9AD970>, <matplotlib.axis.XTick object at 0x0000018E2B337580>, <matplotlib.axis.XTick object at 0x0000018E2B337700>, <matplotlib.axis.XTick object at 0x0000018E2AAA4DC0>, <matplotlib.axis.XTick object at 0x0000018E2AAA4640>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB4E7C0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2B2F6FA0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A975A00>, <matplotlib.axis.YTick object at 0x0000018E2B26CCA0>, <matplotlib.axis.YTick object at 0x0000018E2AD5E2B0>, <matplotlib.axis.YTick object at 0x0000018E2AD5E2E0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A975970>, <matplotlib.axis.XTick object at 0x0000018E2A9F10D0>, <matplotlib.axis.XTick object at 0x0000018E2AD5E280>, <matplotlib.axis.XTick object at 0x0000018E2A9CC250>, <matplotlib.axis.XTick object at 0x0000018E2A9CCAF0>, <matplotlib.axis.XTick object at 0x0000018E2AC393A0>, <matplotlib.axis.XTick object at 0x0000018E2AC39670>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ACC6B20>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAB5F10>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B44D700>, <matplotlib.axis.YTick object at 0x0000018E2AB43D00>, <matplotlib.axis.YTick object at 0x0000018E2AC87490>, <matplotlib.axis.YTick object at 0x0000018E2AC876A0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAC8430>, <matplotlib.axis.XTick object at 0x0000018E2AD074C0>, <matplotlib.axis.XTick object at 0x0000018E2AC87C70>, <matplotlib.axis.XTick object at 0x0000018E2AC61370>, <matplotlib.axis.XTick object at 0x0000018E2AC61D90>, <matplotlib.axis.XTick object at 0x0000018E2AC610D0>, <matplotlib.axis.XTick object at 0x0000018E2A97AAF0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ABC4D00>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAC8F10>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB8FB20>, <matplotlib.axis.YTick object at 0x0000018E2AB8F250>, <matplotlib.axis.YTick object at 0x0000018E2AC08370>, <matplotlib.axis.YTick object at 0x0000018E2AC08430>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB8FC70>, <matplotlib.axis.XTick object at 0x0000018E2AB8F460>, <matplotlib.axis.XTick object at 0x0000018E2AB7B5B0>, <matplotlib.axis.XTick object at 0x0000018E2AC082B0>, <matplotlib.axis.XTick object at 0x0000018E2AB7BA90>, <matplotlib.axis.XTick object at 0x0000018E2AB7BB20>, <matplotlib.axis.XTick object at 0x0000018E2B324AF0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AD5EF40>
<matplotlib.contour.QuadContourSet object at 0x0000018E2A9F1C10>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC00EB0>, <matplotlib.axis.YTick object at 0x0000018E2AC30A60>, <matplotlib.axis.YTick object at 0x0000018E2AC4ADC0>, <matplotlib.axis.YTick object at 0x0000018E2AB06DC0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AA46BE0>, <matplotlib.axis.XTick object at 0x0000018E2A971310>, <matplotlib.axis.XTick object at 0x0000018E2AB060D0>, <matplotlib.axis.XTick object at 0x0000018E2AC4ACD0>, <matplotlib.axis.XTick object at 0x0000018E2BCBEBB0>, <matplotlib.axis.XTick object at 0x0000018E2BCBE880>, <matplotlib.axis.XTick object at 0x0000018E2BCBEF10>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AD58E80>
<matplotlib.contour.QuadContourSet object at 0x0000018E2ACC3DF0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AD75B20>, <matplotlib.axis.YTick object at 0x0000018E2AAEEEE0>, <matplotlib.axis.YTick object at 0x0000018E2B456DF0>, <matplotlib.axis.YTick object at 0x0000018E2AAB3D60>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ACC9F10>, <matplotlib.axis.XTick object at 0x0000018E2AC76460>, <matplotlib.axis.XTick object at 0x0000018E2AAB31F0>, <matplotlib.axis.XTick object at 0x0000018E2AAB3670>, <matplotlib.axis.XTick object at 0x0000018E2AC6AEE0>, <matplotlib.axis.XTick object at 0x0000018E2AC6AE50>, <matplotlib.axis.XTick object at 0x0000018E2AC6A610>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B2A1910>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AA5CF70>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2BC95F10>, <matplotlib.axis.YTick object at 0x0000018E2B346310>, <matplotlib.axis.YTick object at 0x0000018E2A98F670>, <matplotlib.axis.YTick object at 0x0000018E2A98FB80>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AA5B820>, <matplotlib.axis.XTick object at 0x0000018E2AAF3880>, <matplotlib.axis.XTick object at 0x0000018E2A98FA60>, <matplotlib.axis.XTick object at 0x0000018E2A98E6D0>, <matplotlib.axis.XTick object at 0x0000018E2A98EC10>, <matplotlib.axis.XTick object at 0x0000018E2A973160>, <matplotlib.axis.XTick object at 0x0000018E2A973670>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ACEFE20>
>>> Us
2.5204181337734752
>>> Nv=60
>>> Mv=7
>>> pal_v = np.linspace(-Mv, Mv, 2*Nv)
>>> per_v = np.linspace(-Mv, Mv, 2*Nv)
>>> Nt=3000
>>> Mt=3000
>>> t=np.linspace(0, Mt, Nt-1)
>>> F=(t[1]-t[0])/(2*(pal_v[1]-pal_v[0]))**2
>>> n=1
>>> n2=0
>>> n3=-1
>>> omega=-1
>>> fre=0.08
>>> fre=0.07
>>> k_pal0=0.245
>>> k_per0=k_pal0*tan((55*np.pi)/180)
>>> k_per0
mpf('0.34989626165181803')
>>> k_pal_max=0.28
>>> k_pal_min=0.21
>>> k_per_max=k_pal_max*tan((55*np.pi)/180)
>>> k_per_min=k_pal_min*tan((55*np.pi)/180)
>>> a_pal=0.035
>>> a_per=a_pal*tan((55*np.pi)/180)
>>> a_per
mpf('0.049985180235974008')
>>> GV=0.86
>>> B_B0=0.001
>>> def k(b):
    f = lambda x: ((besselj(0, (b*x)/(omega), 0))**2)*np.exp(-(((x-0.35)**2)/(0.05**2)))*x
    I=integrate.quad(f, k_per_min, k_per_max)
    return I[0]

>>> def coefficient_a(a,b):
    return ((0.58*np.pi**2)/(0.035*0.05**2))*(B_B0**2)*(((b)**2)/abs(a-GV))*(fre/k_pal0)**2*k(b)*(np.exp(-(((fre-a*k_pal0-n*omega)/(a-GV))**2)/(0.035**2)))

>>> def coefficient_a2(a,b):
    return ((0.16*np.pi**2)/(0.035*0.05**2))*(B_B0**2)*(((a)**2)/abs(a-GV))*(fre/k_pal0)**2*k(b)*(np.exp(-(((fre-a*k_pal0-n2*omega)/(a-GV))**2)/(0.035**2)))

>>> def coefficient_a3(a,b):
    return ((0.58*np.pi**2)/(0.035*0.05**2))*(B_B0**2)*(((b)**2)/abs(a-GV))*(fre/k_pal0)**2*k(b)*(np.exp(-(((fre-a*k_pal0-n3*omega)/(a-GV))**2)/(0.035**2)))

>>> AA=np.zeros(((2*Nv)*(2*Nv),(2*Nv)*(2*Nv)))
>>> delv=2*abs(pal_v[1]-pal_v[0])
>>> def Matrix_A(b):
    A=np.zeros(((2*Nv),(2*Nv)))
    for i in range(2*Nv):
        for j in range(2*Nv):
            if i==0:
                A[i,j] =1+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==0 else 0 if j==1 else -(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==2 else 0
            elif i==1:
                A[i,j] =0 if j==0 else 1+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==1 else 0 if j==2 else -(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==3 else 0
            elif i==2*Nv-1:
                A[i,j] =-(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv-3 else 0 if j==2*Nv-2 else 1+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv-1 else 0 if j==2*Nv else 0
            elif i==2*Nv:
                A[i,j] =-(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv-2 else 0 if j==2*Nv-1 else 1+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv else 0
            else:
                A[i,j] =-(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==i-2 else 0 if j==i-1 else 1+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==i else 0 if j==i+1 else -(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==i+2 else 0
    return A

>>> def Matrix_B1(b):
    B=np.zeros(((2*Nv),(2*Nv)))
    for i in range(2*Nv):
        for j in range(2*Nv):
            if i==0:
                B[i,j] =0 if j==0 else (F/2)*((pal_v[i]*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i],per_v[b]-delv/2)+(F/2)*(((pal_v[i]+delv/2)*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i]+delv/2,per_v[b])+ (F/2)*((pal_v[i]*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(F/2)*(((pal_v[i]+delv/2)*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i]+delv/2,per_v[b])+ (F/2)*((pal_v[i]*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(F/2)*(((pal_v[i]+delv/2)*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==1 else 0
            elif i==2*Nv:
                B[i,j] =0 if j==2*Nx else -(F/2)*((pal_v[i]*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i],per_v[b]-delv/2)-(F/2)*(((pal_v[i]-delv/2)*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i]-delv/2,per_v[b])- (F/2)*((pal_v[i]*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i],per_v[b]-delv/2)-(F/2)*(((pal_v[i]-delv/2)*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i]-delv/2,per_v[b])- (F/2)*((pal_v[i]*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i],per_v[b]-delv/2)-(F/2)*(((pal_v[i]-delv/2)*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv-1 else 0
            else:
                B[i,j] =0 if j==i else -(F/2)*((pal_v[i]*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i],per_v[b]-delv/2)-(F/2)*(((pal_v[i]-delv/2)*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i]-delv/2,per_v[b])- (F/2)*((pal_v[i]*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i],per_v[b]-delv/2)-(F/2)*(((pal_v[i]-delv/2)*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i]-delv/2,per_v[b])- (F/2)*((pal_v[i]*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i],per_v[b]-delv/2)-(F/2)*(((pal_v[i]-delv/2)*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==i-1 else (F/2)*((pal_v[i]*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i],per_v[b]-delv/2)+(F/2)*(((pal_v[i]+delv/2)*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i]+delv/2,per_v[b])+ (F/2)*((pal_v[i]*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(F/2)*(((pal_v[i]+delv/2)*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i]+delv/2,per_v[b])+ (F/2)*((pal_v[i]*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(F/2)*(((pal_v[i]+delv/2)*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==i+1 else 0
    return B

>>> def Matrix_B2(b):
    B=np.zeros(((2*Nv),(2*Nv)))
    for i in range(2*Nv):
        for j in range(2*Nv):
            if i==0:
                B[i,j] =0 if j==0 else -(F/2)*((pal_v[i]*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i],per_v[b]+delv/2)-(F/2)*(((pal_v[i]+delv/2)*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i]+delv/2,per_v[b])- (F/2)*((pal_v[i]*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i],per_v[b]+delv/2)-(F/2)*(((pal_v[i]+delv/2)*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i]+delv/2,per_v[b])- (F/2)*((pal_v[i]*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i],per_v[b]+delv/2)-(F/2)*(((pal_v[i]+delv/2)*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==1 else 0
            elif i==2*Nv:
                B[i,j] =0 if j==2*Nx else (F/2)*((pal_v[i]*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i],per_v[b]+delv/2)+(F/2)*(((pal_v[i]-delv/2)*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i]-delv/2,per_v[b])+ (F/2)*((pal_v[i]*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(F/2)*(((pal_v[i]-delv/2)*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i]-delv/2,per_v[b])+ (F/2)*((pal_v[i]*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(F/2)*(((pal_v[i]-delv/2)*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv-1 else 0
            else:
                B[i,j] =0 if j==i else (F/2)*((pal_v[i]*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i],per_v[b]+delv/2)+(F/2)*(((pal_v[i]-delv/2)*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i]-delv/2,per_v[b])+ (F/2)*((pal_v[i]*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(F/2)*(((pal_v[i]-delv/2)*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i]-delv/2,per_v[b])+ (F/2)*((pal_v[i]*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(F/2)*(((pal_v[i]-delv/2)*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==i-1 else -(F/2)*((pal_v[i]*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i],per_v[b]+delv/2)-(F/2)*(((pal_v[i]+delv/2)*n*omega-GV*n*omega)*(fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega)**2)*(1/per_v[b])*coefficient_a(pal_v[i]+delv/2,per_v[b])- (F/2)*((pal_v[i]*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i],per_v[b]+delv/2)-(F/2)*(((pal_v[i]+delv/2)*n2*omega-GV*n2*omega)*(fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega)**2)*(1/per_v[b])*coefficient_a2(pal_v[i]+delv/2,per_v[b])- (F/2)*((pal_v[i]*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i],per_v[b]+delv/2)-(F/2)*(((pal_v[i]+delv/2)*n3*omega-GV*n3*omega)*(fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega)**2)*(1/per_v[b])*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==i+1 else 0
    return B

>>> def Matrix_C1(b):
    C=np.zeros(((2*Nv),(2*Nv)))
    for i in range(2*Nv):
        for j in range(2*Nv):
            C[i,j] =-(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)-(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)-(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2) if j==i else 0
    return C

>>> def Matrix_C2(b):
    C=np.zeros(((2*Nv),(2*Nv)))
    for i in range(2*Nv):
        for j in range(2*Nv):
            C[i,j] =-(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)-(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)-(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2) if j==i else 0
    return C

>>> def Matrix_A_1(b):
    A_1=np.zeros(((2*Nv),(2*Nv)))
    for i in range(2*Nv):
        for j in range(2*Nv):
            if i==0:
                A_1[i,j] =1+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==0 else 0 if j==1 else -(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==2 else 0
            elif i==1:
                A_1[i,j] =0 if j==0 else 1+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==1 else 0 if j==2 else -(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==3 else 0
            elif i==2*Nv-1:
                A_1[i,j] =-(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv-3 else 0 if j==2*Nv-2 else 1+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv-1 else 0 if j==2*Nv else 0
            elif i==2*Nv:
                A_1[i,j] =-(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv-2 else 0 if j==2*Nv-1 else 1+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==2*Nv else 0
            else:
                A_1[i,j] =-(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==i-2 else 0 if j==i-1 else 1+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(-F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==i else 0 if j==i+1 else -(-F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])-(-F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==i+2 else 0
    return A_1

>>> for a in range(2*Nv):
    for b in range(2*Nv):
        if a==b:
            AA[a*2*Nv:(a+1)*2*Nv,b*2*Nv:(b+1)*2*Nv]=Matrix_A(a)

            
>>> for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            AA[(a+1)*2*Nv:(a+2)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=Matrix_B1(a+1)

>>> for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            AA[a*2*Nv:(a+1)*2*Nv,(b+1)*2*Nv:(b+2)*2*Nv]=Matrix_B2(a)

>>> for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            AA[(a+2)*2*Nv:(a+3)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=Matrix_C1(a+2)

>>> for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            AA[a*2*Nv:(a+1)*2*Nv,(b+2)*2*Nv:(b+3)*2*Nv]=Matrix_C2(a)
            
SyntaxError: invalid syntax
>>> 
>>> for a in range(2*Nv):
    for b in range(2*Nv):
        if a==b:
            AA[a*2*Nv:(a+1)*2*Nv,b*2*Nv:(b+1)*2*Nv]=Matrix_A(a)

            
>>> for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            AA[(a+1)*2*Nv:(a+2)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=Matrix_B1(a+1)

            
>>> for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            AA[a*2*Nv:(a+1)*2*Nv,(b+1)*2*Nv:(b+2)*2*Nv]=Matrix_B2(a)

            
>>> for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            AA[(a+2)*2*Nv:(a+3)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=Matrix_C1(a+2)

            
>>> for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            AA[a*2*Nv:(a+1)*2*Nv,(b+2)*2*Nv:(b+3)*2*Nv]=Matrix_C2(a)

            
>>> AA_1 = inv(AA)
>>> QQ=np.zeros(((2*Nv)*(2*Nv),(2*Nv)*(2*Nv)))
>>> for a in range(2*Nv):
    for b in range(2*Nv):
        if a==b:
            QQ[a*2*Nv:(a+1)*2*Nv,b*2*Nv:(b+1)*2*Nv]=Matrix_A_1(a)

            
>>> for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            QQ[(a+1)*2*Nv:(a+2)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=-Matrix_B1(a+1)

            
>>> for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            QQ[a*2*Nv:(a+1)*2*Nv,(b+1)*2*Nv:(b+2)*2*Nv]=-Matrix_B2(a)

            
>>> for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            QQ[(a+2)*2*Nv:(a+3)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=-Matrix_C1(a+2)

            
>>> for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            QQ[a*2*Nv:(a+1)*2*Nv,(b+2)*2*Nv:(b+3)*2*Nv]=-Matrix_C2(a)

            
>>> AQ=dot(AA_1,QQ)
SyntaxError: multiple statements found while compiling a single statement
>>> AA_1 = inv(AA)
>>> QQ=np.zeros(((2*Nv)*(2*Nv),(2*Nv)*(2*Nv)))
>>> for a in range(2*Nv):
    for b in range(2*Nv):
        if a==b:
            QQ[a*2*Nv:(a+1)*2*Nv,b*2*Nv:(b+1)*2*Nv]=Matrix_A_1(a)

            
>>> for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            QQ[(a+1)*2*Nv:(a+2)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=-Matrix_B1(a+1)

            
>>> for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            QQ[a*2*Nv:(a+1)*2*Nv,(b+1)*2*Nv:(b+2)*2*Nv]=-Matrix_B2(a)

            
>>> for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            QQ[(a+2)*2*Nv:(a+3)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=-Matrix_C1(a+2)

            
>>> for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            QQ[a*2*Nv:(a+1)*2*Nv,(b+2)*2*Nv:(b+3)*2*Nv]=-Matrix_C2(a)

            
>>> AQ=dot(AA_1,QQ)
>>> def Kappa_Initial_Strahl(a,b):
    kappa=150
    return (2.175)**(-1.5)*0.08*np.exp(-((b)**2)/2.175)*np.exp(-((a-Us)**2)/2.175)

>>> def Kappa_Initial_Core(a,b):
    kappa=150
    return (1.087)**(-1.5)*0.92*np.exp(-((b)**2)/1.087)*np.exp(-((a-Uc)**2)/1.087)

>>> Me=9.1094*(10**(-28))
>>> Mp=1.6726*(10**(-24))
>>> ratio=Me/Mp
>>> Us=108*ratio**(0.5)
>>> Uc=-9.3913*ratio**(0.5)
>>> cont_lev = np.linspace(-8,0,25)
>>> f_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> solu2=np.zeros(shape = (Nv, 2*Nv))
>>> fc_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> ff_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        f_1[j*2*Nv+i]=Kappa_Initial_Strahl(pal_v[i],per_v[j])

        
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        fc_1[j*2*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j])

        
>>> ff_1=f_1+fc_1
>>> Mf_1=np.max(ff_1)
>>> per_v2 = np.linspace(0, Mv, Nv)
>>> X2,Y2 = np.meshgrid(pal_v,per_v2)
SyntaxError: invalid syntax
>>> def Kappa_Initial_Strahl(a,b):
    kappa=150
    return (2.175)**(-1.5)*0.08*np.exp(-((b)**2)/2.175)*np.exp(-((a-Us)**2)/2.175)

>>> def Kappa_Initial_Core(a,b):
    kappa=150
    return (1.087)**(-1.5)*0.92*np.exp(-((b)**2)/1.087)*np.exp(-((a-Uc)**2)/1.087)

>>> cont_lev = np.linspace(-8,0,25)
>>> f_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> solu2=np.zeros(shape = (Nv, 2*Nv))
>>> fc_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> ff_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        f_1[j*2*Nv+i]=Kappa_Initial_Strahl(pal_v[i],per_v[j])

        
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        fc_1[j*2*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j])

        
>>> for k in range(5): #Numer in range indicates the minute.
    print(k)
    #ff_1=f_1+fc_1
    for j in range(Nv):
        for i in range(2*Nv):
        #solu[j,i]=(abs(f_1[j*2*Nv+i])/Mf_1)
            if abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>1:
                solu2[j,i]=0
            elif abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>10**(-5):
                solu2[j,i]=np.log10(abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1)#np.log10
            else:
                solu2[j,i]=-10
    #Mf_1=np.max(f_1)
    fig = plt.figure()
    fig.set_dpi(350)
    plt.contourf(X2, Y2,solu2, cont_lev,cmap='Blues');
    ax = plt.gca()
    ax.spines['left'].set_position('center')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_smart_bounds(True)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    #ax.yaxis.set_ticks_position('left')
    #ax.set_xlim(-Mv, Mv); ax.set_ylim(0, 0.6);
    #plt.xlim(-Mv, Mv)
    #plt.ylim(0, 0.6)
    plt.axis('equal')
    plt.yticks([2,4,6,8])
    plt.xticks([-6,-4,-2,0,2,4,6])
    plt.rc('font', size=9)
    plt.tick_params(labelsize=9)
    plt.text(-0.2,-1.6,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae}$', fontsize=9)
    plt.text(-0.2,8.3,r'$\mathcal{v}_\perp/\mathcal{v}_{Ae}$', fontsize=9)
    plt.colorbar(label=r'$Log(F/F_{MAX})$')
    #plt.savefig(QLD/Collision/qld/{k}.png)
    #plt.clf()
    plt.show()
    for t in range(100):
        ff_1=dot(AQ, ff_1)

        
0
<matplotlib.contour.QuadContourSet object at 0x0000018E2ACC3040>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A979FA0>, <matplotlib.axis.YTick object at 0x0000018E2A9882B0>, <matplotlib.axis.YTick object at 0x0000018E2AAF32B0>, <matplotlib.axis.YTick object at 0x0000018E2AA97070>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A979A90>, <matplotlib.axis.XTick object at 0x0000018E2AAA4D90>, <matplotlib.axis.XTick object at 0x0000018E2ABDEF10>, <matplotlib.axis.XTick object at 0x0000018E2AA974C0>, <matplotlib.axis.XTick object at 0x0000018E2ABDE2E0>, <matplotlib.axis.XTick object at 0x0000018E2ABDE340>, <matplotlib.axis.XTick object at 0x0000018E2AC45FD0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AD89790>
1
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC085B0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B28EB20>, <matplotlib.axis.YTick object at 0x0000018E2AC65A00>, <matplotlib.axis.YTick object at 0x0000018E2ABADFA0>, <matplotlib.axis.YTick object at 0x0000018E2ABADEE0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A9AD940>, <matplotlib.axis.XTick object at 0x0000018E2AC26EB0>, <matplotlib.axis.XTick object at 0x0000018E2A9F9550>, <matplotlib.axis.XTick object at 0x0000018E2ABADC10>, <matplotlib.axis.XTick object at 0x0000018E2A9F99A0>, <matplotlib.axis.XTick object at 0x0000018E2B324940>, <matplotlib.axis.XTick object at 0x0000018E2B324F40>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB300D0>
2
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB22970>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAFFE20>, <matplotlib.axis.YTick object at 0x0000018E2ABC4A30>, <matplotlib.axis.YTick object at 0x0000018E2A971F40>, <matplotlib.axis.YTick object at 0x0000018E2A971070>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAFFD90>, <matplotlib.axis.XTick object at 0x0000018E2AAE00D0>, <matplotlib.axis.XTick object at 0x0000018E2ABAEBE0>, <matplotlib.axis.XTick object at 0x0000018E2A971160>, <matplotlib.axis.XTick object at 0x0000018E2ABAEBB0>, <matplotlib.axis.XTick object at 0x0000018E2B31EF70>, <matplotlib.axis.XTick object at 0x0000018E2B31EE50>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B44DE50>
3
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAB5280>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ACCE040>, <matplotlib.axis.YTick object at 0x0000018E2AAB5C70>, <matplotlib.axis.YTick object at 0x0000018E2AA46CA0>, <matplotlib.axis.YTick object at 0x0000018E2AA464C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B3187F0>, <matplotlib.axis.XTick object at 0x0000018E2AA98E20>, <matplotlib.axis.XTick object at 0x0000018E2B47FD30>, <matplotlib.axis.XTick object at 0x0000018E2AA46610>, <matplotlib.axis.XTick object at 0x0000018E2B47F8E0>, <matplotlib.axis.XTick object at 0x0000018E2B47F9D0>, <matplotlib.axis.XTick object at 0x0000018E2ABEEE80>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B2C03A0>
4
<matplotlib.contour.QuadContourSet object at 0x0000018E2ACC3AF0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B456700>, <matplotlib.axis.YTick object at 0x0000018E2AC9FD00>, <matplotlib.axis.YTick object at 0x0000018E2A9F1730>, <matplotlib.axis.YTick object at 0x0000018E2A9F12B0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A9B27C0>, <matplotlib.axis.XTick object at 0x0000018E2B34CD00>, <matplotlib.axis.XTick object at 0x0000018E2B2A13A0>, <matplotlib.axis.XTick object at 0x0000018E2A9F19D0>, <matplotlib.axis.XTick object at 0x0000018E2B2A16D0>, <matplotlib.axis.XTick object at 0x0000018E2B2A1940>, <matplotlib.axis.XTick object at 0x0000018E2A9F0CA0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A97A100>
>>> ff_1=f_1+fc_1
>>> Mf_1=np.max(ff_1)
>>> for k in range(5): #Numer in range indicates the minute.
    print(k)
    #ff_1=f_1+fc_1
    for j in range(Nv):
        for i in range(2*Nv):
        #solu[j,i]=(abs(f_1[j*2*Nv+i])/Mf_1)
            if abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>1:
                solu2[j,i]=0
            elif abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>10**(-5):
                solu2[j,i]=np.log10(abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1)#np.log10
            else:
                solu2[j,i]=-10
    #Mf_1=np.max(f_1)
    fig = plt.figure()
    fig.set_dpi(350)
    plt.contourf(X2, Y2,solu2, cont_lev,cmap='Blues');
    ax = plt.gca()
    ax.spines['left'].set_position('center')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_smart_bounds(True)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    #ax.yaxis.set_ticks_position('left')
    #ax.set_xlim(-Mv, Mv); ax.set_ylim(0, 0.6);
    #plt.xlim(-Mv, Mv)
    #plt.ylim(0, 0.6)
    plt.axis('equal')
    plt.yticks([2,4,6,8])
    plt.xticks([-6,-4,-2,0,2,4,6])
    plt.rc('font', size=9)
    plt.tick_params(labelsize=9)
    plt.text(-0.2,-1.6,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae}$', fontsize=9)
    plt.text(-0.2,8.3,r'$\mathcal{v}_\perp/\mathcal{v}_{Ae}$', fontsize=9)
    plt.colorbar(label=r'$Log(F/F_{MAX})$')
    #plt.savefig(QLD/Collision/qld/{k}.png)
    #plt.clf()
    plt.show()
    for t in range(100):
        ff_1=dot(AQ, ff_1)

        
0
<matplotlib.contour.QuadContourSet object at 0x0000018E2A98FAF0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB01DC0>, <matplotlib.axis.YTick object at 0x0000018E2BC9EC10>, <matplotlib.axis.YTick object at 0x0000018E2A9CCF10>, <matplotlib.axis.YTick object at 0x0000018E2A9CC280>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB01730>, <matplotlib.axis.XTick object at 0x0000018E2A9B1280>, <matplotlib.axis.XTick object at 0x0000018E2A9CCCA0>, <matplotlib.axis.XTick object at 0x0000018E2BCA89D0>, <matplotlib.axis.XTick object at 0x0000018E2BCA8EB0>, <matplotlib.axis.XTick object at 0x0000018E2BCA8EE0>, <matplotlib.axis.XTick object at 0x0000018E2B2F6850>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AA3C1F0>
1
<matplotlib.contour.QuadContourSet object at 0x0000018E2AA01A60>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ABA9490>, <matplotlib.axis.YTick object at 0x0000018E2AB65C40>, <matplotlib.axis.YTick object at 0x0000018E2A9C3160>, <matplotlib.axis.YTick object at 0x0000018E2A9C3670>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ABA9280>, <matplotlib.axis.XTick object at 0x0000018E2ABDE9A0>, <matplotlib.axis.XTick object at 0x0000018E2A9C38E0>, <matplotlib.axis.XTick object at 0x0000018E2AC86250>, <matplotlib.axis.XTick object at 0x0000018E2AC86700>, <matplotlib.axis.XTick object at 0x0000018E2AC86C10>, <matplotlib.axis.XTick object at 0x0000018E2AC6E160>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ACA6910>
2
<matplotlib.contour.QuadContourSet object at 0x0000018E2ACC6400>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC1F8E0>, <matplotlib.axis.YTick object at 0x0000018E2A9880D0>, <matplotlib.axis.YTick object at 0x0000018E2ACEFEE0>, <matplotlib.axis.YTick object at 0x0000018E2B26C160>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC30A30>, <matplotlib.axis.XTick object at 0x0000018E2AAB23D0>, <matplotlib.axis.XTick object at 0x0000018E2ACEFDF0>, <matplotlib.axis.XTick object at 0x0000018E2B346A00>, <matplotlib.axis.XTick object at 0x0000018E2B346D30>, <matplotlib.axis.XTick object at 0x0000018E2B2A1700>, <matplotlib.axis.XTick object at 0x0000018E2B2A1280>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A973FA0>
3
<matplotlib.contour.QuadContourSet object at 0x0000018E2AA46490>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B4567C0>, <matplotlib.axis.YTick object at 0x0000018E2ABB6EB0>, <matplotlib.axis.YTick object at 0x0000018E2BCB8AF0>, <matplotlib.axis.YTick object at 0x0000018E2BCB8940>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB65AC0>, <matplotlib.axis.XTick object at 0x0000018E2AC6A850>, <matplotlib.axis.XTick object at 0x0000018E2BCB87C0>, <matplotlib.axis.XTick object at 0x0000018E2A9F3310>, <matplotlib.axis.XTick object at 0x0000018E2B3240A0>, <matplotlib.axis.XTick object at 0x0000018E2B324190>, <matplotlib.axis.XTick object at 0x0000018E2B324580>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB58250>
4
<matplotlib.contour.QuadContourSet object at 0x0000018E2B31E6D0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AD073D0>, <matplotlib.axis.YTick object at 0x0000018E2ACC9310>, <matplotlib.axis.YTick object at 0x0000018E2BCA8EB0>, <matplotlib.axis.YTick object at 0x0000018E2B44D160>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC614C0>, <matplotlib.axis.XTick object at 0x0000018E2AC412B0>, <matplotlib.axis.XTick object at 0x0000018E2B44D640>, <matplotlib.axis.XTick object at 0x0000018E2B44D310>, <matplotlib.axis.XTick object at 0x0000018E2AB43550>, <matplotlib.axis.XTick object at 0x0000018E2AB43790>, <matplotlib.axis.XTick object at 0x0000018E2AB43BE0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A971F40>
>>> for k in range(5): #Numer in range indicates the minute.
    print(k)
    #ff_1=f_1+fc_1
    for j in range(Nv):
        for i in range(2*Nv):
        #solu[j,i]=(abs(f_1[j*2*Nv+i])/Mf_1)
            if abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>1:
                solu2[j,i]=0
            elif abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>10**(-5):
                solu2[j,i]=np.log10(abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1)#np.log10
            else:
                solu2[j,i]=-10
    #Mf_1=np.max(f_1)
    fig = plt.figure()
    fig.set_dpi(350)
    plt.contourf(X2, Y2,solu2, cont_lev,cmap='Blues');
    ax = plt.gca()
    ax.spines['left'].set_position('center')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_smart_bounds(True)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    #ax.yaxis.set_ticks_position('left')
    #ax.set_xlim(-Mv, Mv); ax.set_ylim(0, 0.6);
    #plt.xlim(-Mv, Mv)
    #plt.ylim(0, 0.6)
    plt.axis('equal')
    plt.yticks([2,4,6,8])
    plt.xticks([-6,-4,-2,0,2,4,6])
    plt.rc('font', size=9)
    plt.tick_params(labelsize=9)
    plt.text(-0.2,-1.6,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae}$', fontsize=9)
    plt.text(-0.2,8.3,r'$\mathcal{v}_\perp/\mathcal{v}_{Ae}$', fontsize=9)
    plt.colorbar(label=r'$Log(F/F_{MAX})$')
    #plt.savefig(QLD/Collision/qld/{k}.png)
    #plt.clf()
    plt.show()

    
0
<matplotlib.contour.QuadContourSet object at 0x0000018E2B318EE0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAB5310>, <matplotlib.axis.YTick object at 0x0000018E2B38A160>, <matplotlib.axis.YTick object at 0x0000018E2AB23580>, <matplotlib.axis.YTick object at 0x0000018E2AB23220>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC65FD0>, <matplotlib.axis.XTick object at 0x0000018E2AC61250>, <matplotlib.axis.XTick object at 0x0000018E2ACCE820>, <matplotlib.axis.XTick object at 0x0000018E2AB23520>, <matplotlib.axis.XTick object at 0x0000018E2ACCEEE0>, <matplotlib.axis.XTick object at 0x0000018E2B2F6D30>, <matplotlib.axis.XTick object at 0x0000018E2AC87820>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AD769A0>
1
<matplotlib.contour.QuadContourSet object at 0x0000018E2ACC3730>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A9F0850>, <matplotlib.axis.YTick object at 0x0000018E2AD5B100>, <matplotlib.axis.YTick object at 0x0000018E2AC19700>, <matplotlib.axis.YTick object at 0x0000018E2AC19C70>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AADA3A0>, <matplotlib.axis.XTick object at 0x0000018E2AD5B5E0>, <matplotlib.axis.XTick object at 0x0000018E2AC19D30>, <matplotlib.axis.XTick object at 0x0000018E2AB14A60>, <matplotlib.axis.XTick object at 0x0000018E2AB14340>, <matplotlib.axis.XTick object at 0x0000018E2ACCB1C0>, <matplotlib.axis.XTick object at 0x0000018E2ACCB460>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB1D130>
2
<matplotlib.contour.QuadContourSet object at 0x0000018E2ACC9F40>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB65940>, <matplotlib.axis.YTick object at 0x0000018E2B456BE0>, <matplotlib.axis.YTick object at 0x0000018E2A9CCEB0>, <matplotlib.axis.YTick object at 0x0000018E2A9CC400>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB65E80>, <matplotlib.axis.XTick object at 0x0000018E2AD75A00>, <matplotlib.axis.XTick object at 0x0000018E2A9CCC10>, <matplotlib.axis.XTick object at 0x0000018E2B45C880>, <matplotlib.axis.XTick object at 0x0000018E2B45CD90>, <matplotlib.axis.XTick object at 0x0000018E2AAEE640>, <matplotlib.axis.XTick object at 0x0000018E2AAEE3A0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB30EB0>
3
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB6DC10>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A9DEBB0>, <matplotlib.axis.YTick object at 0x0000018E2AC4F1C0>, <matplotlib.axis.YTick object at 0x0000018E2AC13310>, <matplotlib.axis.YTick object at 0x0000018E2AC13820>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB53550>, <matplotlib.axis.XTick object at 0x0000018E2ABBBDC0>, <matplotlib.axis.XTick object at 0x0000018E2AC137C0>, <matplotlib.axis.XTick object at 0x0000018E2AAA23D0>, <matplotlib.axis.XTick object at 0x0000018E2AAA28B0>, <matplotlib.axis.XTick object at 0x0000018E2AAA2DC0>, <matplotlib.axis.XTick object at 0x0000018E2AAB7310>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AC84AC0>
4
<matplotlib.contour.QuadContourSet object at 0x0000018E2AD898B0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B346970>, <matplotlib.axis.YTick object at 0x0000018E2ABBB580>, <matplotlib.axis.YTick object at 0x0000018E2ACCB940>, <matplotlib.axis.YTick object at 0x0000018E2ACCB9A0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ACC68E0>, <matplotlib.axis.XTick object at 0x0000018E2AC21B80>, <matplotlib.axis.XTick object at 0x0000018E2ACCB2B0>, <matplotlib.axis.XTick object at 0x0000018E2AC197C0>, <matplotlib.axis.XTick object at 0x0000018E2AC19B50>, <matplotlib.axis.XTick object at 0x0000018E2AC19250>, <matplotlib.axis.XTick object at 0x0000018E2ACC3970>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AD75F70>
>>> f_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> solu2=np.zeros(shape = (Nv, 2*Nv))
>>> fc_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> ff_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        f_1[j*2*Nv+i]=Kappa_Initial_Strahl(pal_v[i],per_v[j])

        
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        fc_1[j*2*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j])

        
>>> ff_1=f_1+fc_1
>>> Mf_1=np.max(ff_1)
SyntaxError: multiple statements found while compiling a single statement
>>> f_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> solu2=np.zeros(shape = (Nv, 2*Nv))
>>> fc_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> ff_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        f_1[j*2*Nv+i]=Kappa_Initial_Strahl(pal_v[i],per_v[j])

        
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        fc_1[j*2*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j])

        
>>> ff_1=f_1+fc_1
>>> Mf_1=np.max(ff_1)
>>> 
KeyboardInterrupt
>>> for k in range(51): #Numer in range indicates the minute.
    print(k)
    #ff_1=f_1+fc_1
    for j in range(Nv):
        for i in range(2*Nv):
        #solu[j,i]=(abs(f_1[j*2*Nv+i])/Mf_1)
            if abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>1:
                solu2[j,i]=0
            elif abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>10**(-5):
                solu2[j,i]=np.log10(abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1)#np.log10
            else:
                solu2[j,i]=-10
    #Mf_1=np.max(f_1)
    fig = plt.figure()
    fig.set_dpi(350)
    plt.contourf(X2, Y2,solu2, cont_lev,cmap='Blues');
    ax = plt.gca()
    ax.spines['left'].set_position('center')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_smart_bounds(True)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    #ax.yaxis.set_ticks_position('left')
    #ax.set_xlim(-Mv, Mv); ax.set_ylim(0, 0.6);
    #plt.xlim(-Mv, Mv)
    #plt.ylim(0, 0.6)
    plt.axis('equal')
    plt.yticks([2,4,6,8])
    plt.xticks([-6,-4,-2,0,2,4,6])
    plt.rc('font', size=9)
    plt.tick_params(labelsize=9)
    plt.text(-0.2,-1.6,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae}$', fontsize=9)
    plt.text(-0.2,8.3,r'$\mathcal{v}_\perp/\mathcal{v}_{Ae}$', fontsize=9)
    plt.colorbar(label=r'$Log(F/F_{MAX})$')
    #plt.savefig(QLD/Collision/qld/{k}.png)
    #plt.clf()
    plt.show()
    for t in range(10):
        ff_1=dot(AQ, ff_1)

        
0
<matplotlib.contour.QuadContourSet object at 0x0000018E2B47FFA0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A98EEE0>, <matplotlib.axis.YTick object at 0x0000018E2AB5FFD0>, <matplotlib.axis.YTick object at 0x0000018E2ACEF400>, <matplotlib.axis.YTick object at 0x0000018E2ACEFC10>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ABAD070>, <matplotlib.axis.XTick object at 0x0000018E2A990370>, <matplotlib.axis.XTick object at 0x0000018E2ACEF760>, <matplotlib.axis.XTick object at 0x0000018E2B3B6220>, <matplotlib.axis.XTick object at 0x0000018E2B3B66A0>, <matplotlib.axis.XTick object at 0x0000018E2B3B6820>, <matplotlib.axis.XTick object at 0x0000018E2AB14730>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AAA42E0>
1
<matplotlib.contour.QuadContourSet object at 0x0000018E2B2F6850>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC45160>, <matplotlib.axis.YTick object at 0x0000018E2AA98C40>, <matplotlib.axis.YTick object at 0x0000018E2AD76C70>, <matplotlib.axis.YTick object at 0x0000018E2AAB5A30>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A9797C0>, <matplotlib.axis.XTick object at 0x0000018E2AC95790>, <matplotlib.axis.XTick object at 0x0000018E2AD763A0>, <matplotlib.axis.XTick object at 0x0000018E2AAB5E20>, <matplotlib.axis.XTick object at 0x0000018E2A9F98E0>, <matplotlib.axis.XTick object at 0x0000018E2A9F9D00>, <matplotlib.axis.XTick object at 0x0000018E2A9F99A0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AA16460>
2
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC00760>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018D62FE2670>, <matplotlib.axis.YTick object at 0x0000018E2AAFFD90>, <matplotlib.axis.YTick object at 0x0000018E2B3B6880>, <matplotlib.axis.YTick object at 0x0000018E2B47F8E0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAE0A30>, <matplotlib.axis.XTick object at 0x0000018E2B47F310>, <matplotlib.axis.XTick object at 0x0000018E2B47F910>, <matplotlib.axis.XTick object at 0x0000018E2B47F9A0>, <matplotlib.axis.XTick object at 0x0000018E2B3B6CA0>, <matplotlib.axis.XTick object at 0x0000018E2AC269D0>, <matplotlib.axis.XTick object at 0x0000018CFFD97E50>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A98EAC0>
3
<matplotlib.contour.QuadContourSet object at 0x0000018E2AD5EDC0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC7CD30>, <matplotlib.axis.YTick object at 0x0000018E2B346430>, <matplotlib.axis.YTick object at 0x0000018E2AAA61F0>, <matplotlib.axis.YTick object at 0x0000018E2AAA6160>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AA5B730>, <matplotlib.axis.XTick object at 0x0000018E2B346A90>, <matplotlib.axis.XTick object at 0x0000018E2AAA6730>, <matplotlib.axis.XTick object at 0x0000018E2AAC8C10>, <matplotlib.axis.XTick object at 0x0000018E2AAC8040>, <matplotlib.axis.XTick object at 0x0000018E2A9886A0>, <matplotlib.axis.XTick object at 0x0000018E2A988D00>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A97AA90>
4
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC8CBB0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2BCBEC10>, <matplotlib.axis.YTick object at 0x0000018E2AD048E0>, <matplotlib.axis.YTick object at 0x0000018E2AB5F6A0>, <matplotlib.axis.YTick object at 0x0000018E2AB5F910>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A9CC7F0>, <matplotlib.axis.XTick object at 0x0000018E2AB301F0>, <matplotlib.axis.XTick object at 0x0000018E2AB5F8E0>, <matplotlib.axis.XTick object at 0x0000018E2AB537F0>, <matplotlib.axis.XTick object at 0x0000018E2AB53CA0>, <matplotlib.axis.XTick object at 0x0000018E2AB36280>, <matplotlib.axis.XTick object at 0x0000018E2AB36940>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AC45820>
5
<matplotlib.contour.QuadContourSet object at 0x0000018E2A96FEB0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AABEE20>, <matplotlib.axis.YTick object at 0x0000018E2A9B1610>, <matplotlib.axis.YTick object at 0x0000018E2ABD4400>, <matplotlib.axis.YTick object at 0x0000018E2ABD4C10>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC3B760>, <matplotlib.axis.XTick object at 0x0000018E2AC30040>, <matplotlib.axis.XTick object at 0x0000018E2ABD4E20>, <matplotlib.axis.XTick object at 0x0000018E2AC0EDC0>, <matplotlib.axis.XTick object at 0x0000018E2AC0E7C0>, <matplotlib.axis.XTick object at 0x0000018D62FFC040>, <matplotlib.axis.XTick object at 0x0000018D62FFC5B0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB70D60>
6
<matplotlib.contour.QuadContourSet object at 0x0000018E2AADA2E0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A990F70>, <matplotlib.axis.YTick object at 0x0000018E2AB067C0>, <matplotlib.axis.YTick object at 0x0000018E2AB22B50>, <matplotlib.axis.YTick object at 0x0000018E2AAB2EB0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A990CA0>, <matplotlib.axis.XTick object at 0x0000018E2B2C0760>, <matplotlib.axis.XTick object at 0x0000018E2AB22DF0>, <matplotlib.axis.XTick object at 0x0000018E2AAB21F0>, <matplotlib.axis.XTick object at 0x0000018CFFD74C70>, <matplotlib.axis.XTick object at 0x0000018CFFD74D60>, <matplotlib.axis.XTick object at 0x0000018E2AAB3A00>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ACA0490>
7
<matplotlib.contour.QuadContourSet object at 0x0000018E2ACC3BB0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B26CBE0>, <matplotlib.axis.YTick object at 0x0000018E2A9B2B50>, <matplotlib.axis.YTick object at 0x0000018E2A9F08E0>, <matplotlib.axis.YTick object at 0x0000018E2A9F0130>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ABEE580>, <matplotlib.axis.XTick object at 0x0000018E2AABA7C0>, <matplotlib.axis.XTick object at 0x0000018E2A9F0550>, <matplotlib.axis.XTick object at 0x0000018E2B2A1D90>, <matplotlib.axis.XTick object at 0x0000018E2B2A1B80>, <matplotlib.axis.XTick object at 0x0000018E2B2A1A00>, <matplotlib.axis.XTick object at 0x0000018E2AD89D90>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B346BB0>
8
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC24160>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AA46E50>, <matplotlib.axis.YTick object at 0x0000018E2B31E160>, <matplotlib.axis.YTick object at 0x0000018E2AAFCA90>, <matplotlib.axis.YTick object at 0x0000018E2AC45DC0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB65EE0>, <matplotlib.axis.XTick object at 0x0000018E2AC651F0>, <matplotlib.axis.XTick object at 0x0000018E2AAFCB50>, <matplotlib.axis.XTick object at 0x0000018E2AC45F40>, <matplotlib.axis.XTick object at 0x0000018E2AAFF250>, <matplotlib.axis.XTick object at 0x0000018E2AAFFC70>, <matplotlib.axis.XTick object at 0x0000018E2AAFFBE0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A979EB0>
9
<matplotlib.contour.QuadContourSet object at 0x0000018E2AA985B0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC45D90>, <matplotlib.axis.YTick object at 0x0000018E2AD76EB0>, <matplotlib.axis.YTick object at 0x0000018E2B34CC40>, <matplotlib.axis.YTick object at 0x0000018E2B34CBE0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A964460>, <matplotlib.axis.XTick object at 0x0000018E2B2A1F10>, <matplotlib.axis.XTick object at 0x0000018E2B2A1E50>, <matplotlib.axis.XTick object at 0x0000018E2B2A1220>, <matplotlib.axis.XTick object at 0x0000018E2B34CD00>, <matplotlib.axis.XTick object at 0x0000018E2A9F01F0>, <matplotlib.axis.XTick object at 0x0000018E2A9F0F40>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B416D90>
10
<matplotlib.contour.QuadContourSet object at 0x0000018E2AABAB50>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ACC3130>, <matplotlib.axis.YTick object at 0x0000018E2A971250>, <matplotlib.axis.YTick object at 0x0000018E2AA97B20>, <matplotlib.axis.YTick object at 0x0000018E2AA97100>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ABEE670>, <matplotlib.axis.XTick object at 0x0000018CFFD74D90>, <matplotlib.axis.XTick object at 0x0000018E2AA977C0>, <matplotlib.axis.XTick object at 0x0000018E2AD89670>, <matplotlib.axis.XTick object at 0x0000018E2AD89E20>, <matplotlib.axis.XTick object at 0x0000018E2AD04CD0>, <matplotlib.axis.XTick object at 0x0000018E2AD04730>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018D62FFC160>
11
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC306D0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A9B14C0>, <matplotlib.axis.YTick object at 0x0000018E2AC1DEB0>, <matplotlib.axis.YTick object at 0x0000018E2AC39760>, <matplotlib.axis.YTick object at 0x0000018E2AC392E0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018CFFD74EE0>, <matplotlib.axis.XTick object at 0x0000018E2AB14A30>, <matplotlib.axis.XTick object at 0x0000018E2AC39310>, <matplotlib.axis.XTick object at 0x0000018E2AAB3EB0>, <matplotlib.axis.XTick object at 0x0000018E2AAB3250>, <matplotlib.axis.XTick object at 0x0000018E2AC19EE0>, <matplotlib.axis.XTick object at 0x0000018E2AC190D0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B45CB80>
12
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB57C10>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC01FD0>, <matplotlib.axis.YTick object at 0x0000018E2B44DA60>, <matplotlib.axis.YTick object at 0x0000018E2AB44730>, <matplotlib.axis.YTick object at 0x0000018E2AB44C40>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B30F250>, <matplotlib.axis.XTick object at 0x0000018E2AC87040>, <matplotlib.axis.XTick object at 0x0000018E2AB445B0>, <matplotlib.axis.XTick object at 0x0000018E2A97F040>, <matplotlib.axis.XTick object at 0x0000018E2A97FCD0>, <matplotlib.axis.XTick object at 0x0000018E2A987220>, <matplotlib.axis.XTick object at 0x0000018E2A987730>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AA5CEE0>
13
<matplotlib.contour.QuadContourSet object at 0x0000018E2A9CD940>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B44DA90>, <matplotlib.axis.YTick object at 0x0000018E2B30F5B0>, <matplotlib.axis.YTick object at 0x0000018E2AB5FCD0>, <matplotlib.axis.YTick object at 0x0000018E2AB5F820>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC014F0>, <matplotlib.axis.XTick object at 0x0000018E2B4563A0>, <matplotlib.axis.XTick object at 0x0000018E2AB5FA90>, <matplotlib.axis.XTick object at 0x0000018E2AD89BB0>, <matplotlib.axis.XTick object at 0x0000018E2AD894C0>, <matplotlib.axis.XTick object at 0x0000018E2B2C0A00>, <matplotlib.axis.XTick object at 0x0000018E2B2C0670>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A96F6D0>
14
<matplotlib.contour.QuadContourSet object at 0x0000018E2A9F0E50>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAA6610>, <matplotlib.axis.YTick object at 0x0000018E2AB361F0>, <matplotlib.axis.YTick object at 0x0000018E2B2AF0A0>, <matplotlib.axis.YTick object at 0x0000018E2ABB64F0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB14B20>, <matplotlib.axis.XTick object at 0x0000018E2AAA4F40>, <matplotlib.axis.XTick object at 0x0000018E2B2AF6A0>, <matplotlib.axis.XTick object at 0x0000018E2ABB6FA0>, <matplotlib.axis.XTick object at 0x0000018E2AB8F610>, <matplotlib.axis.XTick object at 0x0000018E2AB8FC10>, <matplotlib.axis.XTick object at 0x0000018E2AB8FD60>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB229A0>
15
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC451F0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAB58B0>, <matplotlib.axis.YTick object at 0x0000018E2B289AC0>, <matplotlib.axis.YTick object at 0x0000018E2AC951C0>, <matplotlib.axis.YTick object at 0x0000018E2AC26760>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAB5580>, <matplotlib.axis.XTick object at 0x0000018E2A96F2B0>, <matplotlib.axis.XTick object at 0x0000018E2AC26460>, <matplotlib.axis.XTick object at 0x0000018E2AC267F0>, <matplotlib.axis.XTick object at 0x0000018E2AA5B280>, <matplotlib.axis.XTick object at 0x0000018E2AA5B4F0>, <matplotlib.axis.XTick object at 0x0000018E2AA5B670>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB065B0>
16
<matplotlib.contour.QuadContourSet object at 0x0000018E2A9649D0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018D62FF43D0>, <matplotlib.axis.YTick object at 0x0000018E2B2A13A0>, <matplotlib.axis.YTick object at 0x0000018E2B324220>, <matplotlib.axis.YTick object at 0x0000018E2B324880>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AD04C70>, <matplotlib.axis.XTick object at 0x0000018E2B2A1850>, <matplotlib.axis.XTick object at 0x0000018E2AAB2130>, <matplotlib.axis.XTick object at 0x0000018E2AAB2310>, <matplotlib.axis.XTick object at 0x0000018E2B324640>, <matplotlib.axis.XTick object at 0x0000018E2AB22070>, <matplotlib.axis.XTick object at 0x0000018E2AB22550>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9F0070>
17
<matplotlib.contour.QuadContourSet object at 0x0000018E2B45CA30>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B2C0D00>, <matplotlib.axis.YTick object at 0x0000018E2AAA4B20>, <matplotlib.axis.YTick object at 0x0000018E2AB43AC0>, <matplotlib.axis.YTick object at 0x0000018E2AC3BAF0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AD89130>, <matplotlib.axis.XTick object at 0x0000018E2B47FDF0>, <matplotlib.axis.XTick object at 0x0000018E2AB43160>, <matplotlib.axis.XTick object at 0x0000018E2AC3BBB0>, <matplotlib.axis.XTick object at 0x0000018CFFD97400>, <matplotlib.axis.XTick object at 0x0000018E2ABC77C0>, <matplotlib.axis.XTick object at 0x0000018E2ABC7970>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A990A90>
18
<matplotlib.contour.QuadContourSet object at 0x0000018E2AD753A0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A9ADDC0>, <matplotlib.axis.YTick object at 0x0000018E2AB53F70>, <matplotlib.axis.YTick object at 0x0000018E2AB341C0>, <matplotlib.axis.YTick object at 0x0000018E2AB34940>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A9F96A0>, <matplotlib.axis.XTick object at 0x0000018E2A9877F0>, <matplotlib.axis.XTick object at 0x0000018E2AB34400>, <matplotlib.axis.XTick object at 0x0000018E2ABDB730>, <matplotlib.axis.XTick object at 0x0000018E2AB36D90>, <matplotlib.axis.XTick object at 0x0000018E2AB36F10>, <matplotlib.axis.XTick object at 0x0000018E2AB36370>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ACF0D90>
19
<matplotlib.contour.QuadContourSet object at 0x0000018E2BCCA340>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A971B80>, <matplotlib.axis.YTick object at 0x0000018E2AC6A1F0>, <matplotlib.axis.YTick object at 0x0000018E2AC0BA00>, <matplotlib.axis.YTick object at 0x0000018E2ABF10A0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A971E80>, <matplotlib.axis.XTick object at 0x0000018E2B31E580>, <matplotlib.axis.XTick object at 0x0000018E2AC0B8E0>, <matplotlib.axis.XTick object at 0x0000018E2ABF1340>, <matplotlib.axis.XTick object at 0x0000018E2ABD4100>, <matplotlib.axis.XTick object at 0x0000018E2ABD44F0>, <matplotlib.axis.XTick object at 0x0000018E2ABD4A00>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ACC71F0>
20
<matplotlib.contour.QuadContourSet object at 0x0000018E2B38A5B0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AD76A60>, <matplotlib.axis.YTick object at 0x0000018E2ACC3430>, <matplotlib.axis.YTick object at 0x0000018E2ABAEB50>, <matplotlib.axis.YTick object at 0x0000018E2ABAE760>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB65CA0>, <matplotlib.axis.XTick object at 0x0000018E2AB510A0>, <matplotlib.axis.XTick object at 0x0000018E2ABAEEE0>, <matplotlib.axis.XTick object at 0x0000018E2B456A90>, <matplotlib.axis.XTick object at 0x0000018E2B4560D0>, <matplotlib.axis.XTick object at 0x0000018E2B28EB20>, <matplotlib.axis.XTick object at 0x0000018E2ABDEA30>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2BCBE4C0>
21
<matplotlib.contour.QuadContourSet object at 0x0000018E2AA3CAF0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAA6CD0>, <matplotlib.axis.YTick object at 0x0000018E2B26C220>, <matplotlib.axis.YTick object at 0x0000018E2ABADD30>, <matplotlib.axis.YTick object at 0x0000018E2ABAD4C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B2C0610>, <matplotlib.axis.XTick object at 0x0000018E2AB53EB0>, <matplotlib.axis.XTick object at 0x0000018E2ABADA60>, <matplotlib.axis.XTick object at 0x0000018E2AC87820>, <matplotlib.axis.XTick object at 0x0000018E2AC87070>, <matplotlib.axis.XTick object at 0x0000018E2AC61730>, <matplotlib.axis.XTick object at 0x0000018E2AC61E20>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB36C40>
22
<matplotlib.contour.QuadContourSet object at 0x0000018E2AA97910>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB34FA0>, <matplotlib.axis.YTick object at 0x0000018E2A9CCA30>, <matplotlib.axis.YTick object at 0x0000018E2AC96EE0>, <matplotlib.axis.YTick object at 0x0000018E2AC96E80>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B289970>, <matplotlib.axis.XTick object at 0x0000018E2ABC4B80>, <matplotlib.axis.XTick object at 0x0000018E2AAEE430>, <matplotlib.axis.XTick object at 0x0000018E2AC96C40>, <matplotlib.axis.XTick object at 0x0000018E2AAEEAF0>, <matplotlib.axis.XTick object at 0x0000018E2AAEEC40>, <matplotlib.axis.XTick object at 0x0000018E2B28EA90>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB53C10>
23
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC3B490>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ABD4EB0>, <matplotlib.axis.YTick object at 0x0000018E2B30F250>, <matplotlib.axis.YTick object at 0x0000018E2ACCE6A0>, <matplotlib.axis.YTick object at 0x0000018E2ACCE820>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AADA100>, <matplotlib.axis.XTick object at 0x0000018E2A987AC0>, <matplotlib.axis.XTick object at 0x0000018E2ACCEEB0>, <matplotlib.axis.XTick object at 0x0000018E2A9F3400>, <matplotlib.axis.XTick object at 0x0000018E2A9F3970>, <matplotlib.axis.XTick object at 0x0000018E2A971940>, <matplotlib.axis.XTick object at 0x0000018E2A971EB0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2BCA7DF0>
24
<matplotlib.contour.QuadContourSet object at 0x0000018E2AA5BEE0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAB2F40>, <matplotlib.axis.YTick object at 0x0000018E2ABA9B20>, <matplotlib.axis.YTick object at 0x0000018E2AD044F0>, <matplotlib.axis.YTick object at 0x0000018E2AAB5D90>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC7CF40>, <matplotlib.axis.XTick object at 0x0000018E2AC6A610>, <matplotlib.axis.XTick object at 0x0000018E2AD04190>, <matplotlib.axis.XTick object at 0x0000018E2AAB5C10>, <matplotlib.axis.XTick object at 0x0000018E2ABB6A90>, <matplotlib.axis.XTick object at 0x0000018E2ABB6D30>, <matplotlib.axis.XTick object at 0x0000018E2ABB6F10>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AC45640>
25
<matplotlib.contour.QuadContourSet object at 0x0000018E2ABEEF10>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A9F9340>, <matplotlib.axis.YTick object at 0x0000018E2AB7BB20>, <matplotlib.axis.YTick object at 0x0000018E2AAF2940>, <matplotlib.axis.YTick object at 0x0000018E2AAF2E50>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B318760>, <matplotlib.axis.XTick object at 0x0000018E2AAA4EE0>, <matplotlib.axis.XTick object at 0x0000018E2AAF2760>, <matplotlib.axis.XTick object at 0x0000018E2AAD2280>, <matplotlib.axis.XTick object at 0x0000018E2AAD2E80>, <matplotlib.axis.XTick object at 0x0000018E2AB0A430>, <matplotlib.axis.XTick object at 0x0000018E2AB0A940>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ACF0130>
26
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAA4B20>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AA98730>, <matplotlib.axis.YTick object at 0x0000018E2ABF9040>, <matplotlib.axis.YTick object at 0x0000018CFFD74370>, <matplotlib.axis.YTick object at 0x0000018E2AC26910>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC95760>, <matplotlib.axis.XTick object at 0x0000018E2AC9F1F0>, <matplotlib.axis.XTick object at 0x0000018CFFD74B50>, <matplotlib.axis.XTick object at 0x0000018E2AC263D0>, <matplotlib.axis.XTick object at 0x0000018E2B346970>, <matplotlib.axis.XTick object at 0x0000018E2B346940>, <matplotlib.axis.XTick object at 0x0000018E2B346700>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9F3DC0>
27
<matplotlib.contour.QuadContourSet object at 0x0000018E2A97AC70>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ABDC1F0>, <matplotlib.axis.YTick object at 0x0000018E2AC3B5E0>, <matplotlib.axis.YTick object at 0x0000018E2B324D90>, <matplotlib.axis.YTick object at 0x0000018E2B324790>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ACC3F70>, <matplotlib.axis.XTick object at 0x0000018E2AC6A670>, <matplotlib.axis.XTick object at 0x0000018E2B324F40>, <matplotlib.axis.XTick object at 0x0000018E2AA5B250>, <matplotlib.axis.XTick object at 0x0000018E2AA5B9A0>, <matplotlib.axis.XTick object at 0x0000018E2B2F6D90>, <matplotlib.axis.XTick object at 0x0000018E2AB340D0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AC19580>
28
<matplotlib.contour.QuadContourSet object at 0x0000018E2B44D130>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC9E340>, <matplotlib.axis.YTick object at 0x0000018E2AC96BE0>, <matplotlib.axis.YTick object at 0x0000018E2AA08310>, <matplotlib.axis.YTick object at 0x0000018E2AA081F0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC30F10>, <matplotlib.axis.XTick object at 0x0000018E2B416B20>, <matplotlib.axis.XTick object at 0x0000018E2AA08760>, <matplotlib.axis.XTick object at 0x0000018E2AD766D0>, <matplotlib.axis.XTick object at 0x0000018E2AD764C0>, <matplotlib.axis.XTick object at 0x0000018E2AD76490>, <matplotlib.axis.XTick object at 0x0000018E2AAD06D0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AD5B6A0>
29
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB06A00>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ABA92B0>, <matplotlib.axis.YTick object at 0x0000018E2ABA9250>, <matplotlib.axis.YTick object at 0x0000018E2ACC5760>, <matplotlib.axis.YTick object at 0x0000018E2AB366A0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018CFFD97D90>, <matplotlib.axis.XTick object at 0x0000018E2AB36D30>, <matplotlib.axis.XTick object at 0x0000018E2AB36AF0>, <matplotlib.axis.XTick object at 0x0000018E2AB36130>, <matplotlib.axis.XTick object at 0x0000018E2ACC5070>, <matplotlib.axis.XTick object at 0x0000018E2AC300A0>, <matplotlib.axis.XTick object at 0x0000018E2AC30BE0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B324490>
30
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC3B0A0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A9B15E0>, <matplotlib.axis.YTick object at 0x0000018CFF1FAEE0>, <matplotlib.axis.YTick object at 0x0000018E2AB65400>, <matplotlib.axis.YTick object at 0x0000018E2AB30610>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ACCE970>, <matplotlib.axis.XTick object at 0x0000018E2ACA05B0>, <matplotlib.axis.XTick object at 0x0000018E2AB30AC0>, <matplotlib.axis.XTick object at 0x0000018E2ABADDF0>, <matplotlib.axis.XTick object at 0x0000018E2ABAD100>, <matplotlib.axis.XTick object at 0x0000018E2B416790>, <matplotlib.axis.XTick object at 0x0000018E2B4165B0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A979040>
31
<matplotlib.contour.QuadContourSet object at 0x0000018E2AADA280>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AD034F0>, <matplotlib.axis.YTick object at 0x0000018E2AAB3C40>, <matplotlib.axis.YTick object at 0x0000018E2AAEE7F0>, <matplotlib.axis.YTick object at 0x0000018E2AAEE520>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B2A1DC0>, <matplotlib.axis.XTick object at 0x0000018E2A9641C0>, <matplotlib.axis.XTick object at 0x0000018E2AAEE850>, <matplotlib.axis.XTick object at 0x0000018E2AD07F70>, <matplotlib.axis.XTick object at 0x0000018E2AD07190>, <matplotlib.axis.XTick object at 0x0000018E2A96FB20>, <matplotlib.axis.XTick object at 0x0000018E2A96FEB0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AA98D60>
32
<matplotlib.contour.QuadContourSet object at 0x0000018E2ACAB4F0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC41EB0>, <matplotlib.axis.YTick object at 0x0000018E2AA16790>, <matplotlib.axis.YTick object at 0x0000018E2AC48BB0>, <matplotlib.axis.YTick object at 0x0000018E2AC2F100>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAFF550>, <matplotlib.axis.XTick object at 0x0000018E2B30FFA0>, <matplotlib.axis.XTick object at 0x0000018E2AC48820>, <matplotlib.axis.XTick object at 0x0000018E2AC2F2B0>, <matplotlib.axis.XTick object at 0x0000018E2A96C190>, <matplotlib.axis.XTick object at 0x0000018E2A96C6A0>, <matplotlib.axis.XTick object at 0x0000018E2A96CBB0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB1B3A0>
33
<matplotlib.contour.QuadContourSet object at 0x0000018E2AA16250>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAE9340>, <matplotlib.axis.YTick object at 0x0000018E2AA4ED60>, <matplotlib.axis.YTick object at 0x0000018E2AAC2B80>, <matplotlib.axis.YTick object at 0x0000018E2ABEEAF0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAE91C0>, <matplotlib.axis.XTick object at 0x0000018E2AAEE280>, <matplotlib.axis.XTick object at 0x0000018E2AAC2160>, <matplotlib.axis.XTick object at 0x0000018E2ABEEA00>, <matplotlib.axis.XTick object at 0x0000018E2AD034C0>, <matplotlib.axis.XTick object at 0x0000018E2AD03760>, <matplotlib.axis.XTick object at 0x0000018E2AD03160>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB30190>
34
<matplotlib.contour.QuadContourSet object at 0x0000018E2AA3C430>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B26C190>, <matplotlib.axis.YTick object at 0x0000018E2AC9FD90>, <matplotlib.axis.YTick object at 0x0000018E2B324220>, <matplotlib.axis.YTick object at 0x0000018E2A9B2DC0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ACA0760>, <matplotlib.axis.XTick object at 0x0000018E2AC9FD30>, <matplotlib.axis.XTick object at 0x0000018E2A9B2790>, <matplotlib.axis.XTick object at 0x0000018E2AB36850>, <matplotlib.axis.XTick object at 0x0000018E2AB362E0>, <matplotlib.axis.XTick object at 0x0000018E2AB36E80>, <matplotlib.axis.XTick object at 0x0000018E2AB36760>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A97AA60>
35
<matplotlib.contour.QuadContourSet object at 0x0000018E2B416430>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAA62E0>, <matplotlib.axis.YTick object at 0x0000018E2AD894C0>, <matplotlib.axis.YTick object at 0x0000018E2B289B20>, <matplotlib.axis.YTick object at 0x0000018E2BCBE130>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B34CB20>, <matplotlib.axis.XTick object at 0x0000018E2AC00910>, <matplotlib.axis.XTick object at 0x0000018E2AA5BD00>, <matplotlib.axis.XTick object at 0x0000018E2BCBEBB0>, <matplotlib.axis.XTick object at 0x0000018E2AB22790>, <matplotlib.axis.XTick object at 0x0000018E2AB229A0>, <matplotlib.axis.XTick object at 0x0000018E2AB22850>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018CFFD97D00>
36
<matplotlib.contour.QuadContourSet object at 0x0000018E2B456FD0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B318940>, <matplotlib.axis.YTick object at 0x0000018E2ACC6EB0>, <matplotlib.axis.YTick object at 0x0000018E2ACC6610>, <matplotlib.axis.YTick object at 0x0000018E2ACC6640>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AD76B50>, <matplotlib.axis.XTick object at 0x0000018E2AC61CA0>, <matplotlib.axis.XTick object at 0x0000018E2AC266D0>, <matplotlib.axis.XTick object at 0x0000018E2ACC6400>, <matplotlib.axis.XTick object at 0x0000018E2AC26E50>, <matplotlib.axis.XTick object at 0x0000018E2B27EB20>, <matplotlib.axis.XTick object at 0x0000018E2AD75310>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9B1610>
37
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC3B760>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAE0070>, <matplotlib.axis.YTick object at 0x0000018CFFD74700>, <matplotlib.axis.YTick object at 0x0000018E2AAC23D0>, <matplotlib.axis.YTick object at 0x0000018E2AA3CCD0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AD70910>, <matplotlib.axis.XTick object at 0x0000018E2B318A00>, <matplotlib.axis.XTick object at 0x0000018E2AAC2F40>, <matplotlib.axis.XTick object at 0x0000018E2AA3C9D0>, <matplotlib.axis.XTick object at 0x0000018E2AC4FA90>, <matplotlib.axis.XTick object at 0x0000018E2AC4F700>, <matplotlib.axis.XTick object at 0x0000018E2AC4FF40>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ACCEE50>
38
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAEE4F0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB347C0>, <matplotlib.axis.YTick object at 0x0000018E2AA16F70>, <matplotlib.axis.YTick object at 0x0000018E2AC9F880>, <matplotlib.axis.YTick object at 0x0000018E2AC951C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB0AC10>, <matplotlib.axis.XTick object at 0x0000018E2AA276A0>, <matplotlib.axis.XTick object at 0x0000018E2AC9F0D0>, <matplotlib.axis.XTick object at 0x0000018E2AC95400>, <matplotlib.axis.XTick object at 0x0000018E2A99D3D0>, <matplotlib.axis.XTick object at 0x0000018E2A99D580>, <matplotlib.axis.XTick object at 0x0000018E2A99DFD0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9F9EE0>
39
<matplotlib.contour.QuadContourSet object at 0x0000018E2ACEE670>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ABA96A0>, <matplotlib.axis.YTick object at 0x0000018E2AB32A90>, <matplotlib.axis.YTick object at 0x0000018E2A994D30>, <matplotlib.axis.YTick object at 0x0000018E2AC3C280>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB8F520>, <matplotlib.axis.XTick object at 0x0000018E2AAE9250>, <matplotlib.axis.XTick object at 0x0000018E2AC3CA90>, <matplotlib.axis.XTick object at 0x0000018E2AC3C220>, <matplotlib.axis.XTick object at 0x0000018E2ACAA310>, <matplotlib.axis.XTick object at 0x0000018E2ACAA820>, <matplotlib.axis.XTick object at 0x0000018E2ACAAD30>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9DA520>
40
<matplotlib.contour.QuadContourSet object at 0x0000018E2A9F1940>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB30EE0>, <matplotlib.axis.YTick object at 0x0000018E2AB111F0>, <matplotlib.axis.YTick object at 0x0000018E2A9F31F0>, <matplotlib.axis.YTick object at 0x0000018E2A9F3850>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB8F9A0>, <matplotlib.axis.XTick object at 0x0000018E2AB117F0>, <matplotlib.axis.XTick object at 0x0000018E2A9F3550>, <matplotlib.axis.XTick object at 0x0000018E2B30F9A0>, <matplotlib.axis.XTick object at 0x0000018E2B30F7C0>, <matplotlib.axis.XTick object at 0x0000018E2ACA0040>, <matplotlib.axis.XTick object at 0x0000018E2ACA0760>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AA16FD0>
41
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC3BCA0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAC2820>, <matplotlib.axis.YTick object at 0x0000018E2ABF6220>, <matplotlib.axis.YTick object at 0x0000018E2AB06910>, <matplotlib.axis.YTick object at 0x0000018E2AB06E80>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A964280>, <matplotlib.axis.XTick object at 0x0000018E2AB36670>, <matplotlib.axis.XTick object at 0x0000018E2AB06A00>, <matplotlib.axis.XTick object at 0x0000018E2B45C940>, <matplotlib.axis.XTick object at 0x0000018E2B45C190>, <matplotlib.axis.XTick object at 0x0000018CFFD97C40>, <matplotlib.axis.XTick object at 0x0000018CFFD976D0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9B1490>
42
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAA6A30>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2BCA8070>, <matplotlib.axis.YTick object at 0x0000018E2AC45C10>, <matplotlib.axis.YTick object at 0x0000018E2B34CCA0>, <matplotlib.axis.YTick object at 0x0000018E2AA5B430>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A9909D0>, <matplotlib.axis.XTick object at 0x0000018E2AC61460>, <matplotlib.axis.XTick object at 0x0000018E2AA5B490>, <matplotlib.axis.XTick object at 0x0000018E2AAE0A30>, <matplotlib.axis.XTick object at 0x0000018E2AAE0C10>, <matplotlib.axis.XTick object at 0x0000018E2AAE0EB0>, <matplotlib.axis.XTick object at 0x0000018E2AD5B2E0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AAC8370>
43
<matplotlib.contour.QuadContourSet object at 0x0000018E2BCA87C0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AA5B670>, <matplotlib.axis.YTick object at 0x0000018E2AC84FD0>, <matplotlib.axis.YTick object at 0x0000018E2AB7B1C0>, <matplotlib.axis.YTick object at 0x0000018E2AB7B910>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB65700>, <matplotlib.axis.XTick object at 0x0000018E2AC3BB50>, <matplotlib.axis.XTick object at 0x0000018E2AC3BCA0>, <matplotlib.axis.XTick object at 0x0000018E2AB7BAF0>, <matplotlib.axis.XTick object at 0x0000018E2AC3BDC0>, <matplotlib.axis.XTick object at 0x0000018E2AC3B9A0>, <matplotlib.axis.XTick object at 0x0000018E2AB227F0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B456E20>
44
<matplotlib.contour.QuadContourSet object at 0x0000018E2A96FEE0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AADA310>, <matplotlib.axis.YTick object at 0x0000018E2ACEF550>, <matplotlib.axis.YTick object at 0x0000018E2AC24610>, <matplotlib.axis.YTick object at 0x0000018E2AC957F0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ABC4D00>, <matplotlib.axis.XTick object at 0x0000018E2ACEFBE0>, <matplotlib.axis.XTick object at 0x0000018E2AC24400>, <matplotlib.axis.XTick object at 0x0000018E2AC952B0>, <matplotlib.axis.XTick object at 0x0000018E2A988EE0>, <matplotlib.axis.XTick object at 0x0000018E2A988CA0>, <matplotlib.axis.XTick object at 0x0000018E2A9888E0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AC19280>
45
<matplotlib.contour.QuadContourSet object at 0x0000018D62FF4B20>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AA46610>, <matplotlib.axis.YTick object at 0x0000018E2AC7C190>, <matplotlib.axis.YTick object at 0x0000018E2AB8F340>, <matplotlib.axis.YTick object at 0x0000018E2AB8F490>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B346F10>, <matplotlib.axis.XTick object at 0x0000018E2AC7C0D0>, <matplotlib.axis.XTick object at 0x0000018E2AB8F910>, <matplotlib.axis.XTick object at 0x0000018E2B2AFCD0>, <matplotlib.axis.XTick object at 0x0000018E2A9AD130>, <matplotlib.axis.XTick object at 0x0000018E2A9ADD00>, <matplotlib.axis.XTick object at 0x0000018E2A9ADDC0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AD07400>
46
<matplotlib.contour.QuadContourSet object at 0x0000018E2B31E340>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAE9DC0>, <matplotlib.axis.YTick object at 0x0000018E2AD5EA90>, <matplotlib.axis.YTick object at 0x0000018E2AB1AE20>, <matplotlib.axis.YTick object at 0x0000018E2AC8E370>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAE9430>, <matplotlib.axis.XTick object at 0x0000018E2AD76850>, <matplotlib.axis.XTick object at 0x0000018E2AC8EB50>, <matplotlib.axis.XTick object at 0x0000018E2AC8E2B0>, <matplotlib.axis.XTick object at 0x0000018E2AB11400>, <matplotlib.axis.XTick object at 0x0000018E2AB11910>, <matplotlib.axis.XTick object at 0x0000018E2AB11E20>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AA40610>
47
<matplotlib.contour.QuadContourSet object at 0x0000018E2A9F9190>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ABB6F10>, <matplotlib.axis.YTick object at 0x0000018E2AC41BB0>, <matplotlib.axis.YTick object at 0x0000018E2A9796A0>, <matplotlib.axis.YTick object at 0x0000018E2B30F8B0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B38A760>, <matplotlib.axis.XTick object at 0x0000018E2ABE81F0>, <matplotlib.axis.XTick object at 0x0000018E2A979A00>, <matplotlib.axis.XTick object at 0x0000018E2B30F6D0>, <matplotlib.axis.XTick object at 0x0000018E2AC19E50>, <matplotlib.axis.XTick object at 0x0000018E2AC194C0>, <matplotlib.axis.XTick object at 0x0000018E2AC19D00>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AD75970>
48
<matplotlib.contour.QuadContourSet object at 0x0000018E2A9F1970>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB43A60>, <matplotlib.axis.YTick object at 0x0000018E2AC4F8E0>, <matplotlib.axis.YTick object at 0x0000018E2AB30910>, <matplotlib.axis.YTick object at 0x0000018E2AB304C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB43220>, <matplotlib.axis.XTick object at 0x0000018E2ACC1520>, <matplotlib.axis.XTick object at 0x0000018E2AB306D0>, <matplotlib.axis.XTick object at 0x0000018E2B2F6E80>, <matplotlib.axis.XTick object at 0x0000018E2AA982E0>, <matplotlib.axis.XTick object at 0x0000018E2AA986A0>, <matplotlib.axis.XTick object at 0x0000018E2AC39FD0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B3243A0>
49
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC1D1C0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB22F70>, <matplotlib.axis.YTick object at 0x0000018E2AB067F0>, <matplotlib.axis.YTick object at 0x0000018E2A97A940>, <matplotlib.axis.YTick object at 0x0000018E2A97AAF0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AADAB80>, <matplotlib.axis.XTick object at 0x0000018E2B2F6910>, <matplotlib.axis.XTick object at 0x0000018E2A97A3A0>, <matplotlib.axis.XTick object at 0x0000018E2ABC4040>, <matplotlib.axis.XTick object at 0x0000018E2ABC4A00>, <matplotlib.axis.XTick object at 0x0000018E2B34C8E0>, <matplotlib.axis.XTick object at 0x0000018E2B34C610>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2BCA7A60>
50
<matplotlib.contour.QuadContourSet object at 0x0000018E2B2A1280>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAB2A60>, <matplotlib.axis.YTick object at 0x0000018E2AAFCB20>, <matplotlib.axis.YTick object at 0x0000018E2A9F04C0>, <matplotlib.axis.YTick object at 0x0000018E2A9F07F0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ABC4160>, <matplotlib.axis.XTick object at 0x0000018E2ABC4730>, <matplotlib.axis.XTick object at 0x0000018E2AC6A670>, <matplotlib.axis.XTick object at 0x0000018E2A9F0070>, <matplotlib.axis.XTick object at 0x0000018E2AC6AFA0>, <matplotlib.axis.XTick object at 0x0000018E2AC6ADF0>, <matplotlib.axis.XTick object at 0x0000018E2ACEF040>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B324760>
>>> f_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> solu2=np.zeros(shape = (Nv, 2*Nv))
>>> fc_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> ff_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        f_1[j*2*Nv+i]=Kappa_Initial_Strahl(pal_v[i],per_v[j])

        
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        fc_1[j*2*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j])

        
>>> ff_1=f_1+fc_1
>>> Mf_1=np.max(ff_1)
>>> for t in range(400):
        f_1=dot(AQ, f_1)

        
>>> for k in range(21):
	ff_1=f_1+fc_1
	for j in range(Nv):
		for i in range(2*Nv):
			if abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>1:
				solu2[j,i]=0
			elif abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>10**(-5):
				solu2[j,i]=np.log10(abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1)
			else:
				solu2[j,i]=-10
	fig = plt.figure()
	fig.set_dpi(350)
	plt.contourf(X2, Y2,solu2, cont_lev,cmap='Blues');
	ax = plt.gca()
	ax.spines['left'].set_position('center')
	ax.spines['left'].set_smart_bounds(True)
	ax.spines['bottom'].set_position('zero')
	ax.spines['bottom'].set_smart_bounds(True)
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	ax.xaxis.set_ticks_position('bottom')
	plt.axis('equal')
	plt.yticks([2,4,6,8])
	plt.xticks([-6,-4,-2,0,2,4,6])
	plt.rc('font', size=9)
	plt.tick_params(labelsize=9)
	plt.text(-0.2,-1.6,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae}$', fontsize=9)
	plt.text(-0.2,8.3,r'$\mathcal{v}_\perp/\mathcal{v}_{Ae}$', fontsize=9)
	plt.colorbar(label=r'$Log(F/F_{MAX})$')
	plt.show()
	for t in range(350):
		f_1=dot(AQ2, f_1)
		for y in range(2*Nv):
			for l in range(2*Nv):
				if ((pal_v[l]**2+per_v[y]**2)**0.5)<0.1:
					f_1[y*2*Nv+l]=10**(-10)
					
SyntaxError: multiple statements found while compiling a single statement
>>> 
>>> f_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> solu2=np.zeros(shape = (Nv, 2*Nv))
>>> fc_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> ff_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        f_1[j*2*Nv+i]=Kappa_Initial_Strahl(pal_v[i],per_v[j])

        
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        fc_1[j*2*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j])

        
>>> ff_1=f_1+fc_1
>>> Mf_1=np.max(ff_1)
>>> for t in range(500):
        f_1=dot(AQ, f_1)

        
>>> for t in range(500):
        fc_1=dot(AQ, fc_1)

        
>>> for k in range(51):
	ff_1=f_1+fc_1
	for j in range(Nv):
		for i in range(2*Nv):
			if abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>1:
				solu2[j,i]=0
			elif abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>10**(-5):
				solu2[j,i]=np.log10(abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1)
			else:
				solu2[j,i]=-10
	fig = plt.figure()
	fig.set_dpi(350)
	plt.contourf(X2, Y2,solu2, cont_lev,cmap='Blues');
	ax = plt.gca()
	ax.spines['left'].set_position('center')
	ax.spines['left'].set_smart_bounds(True)
	ax.spines['bottom'].set_position('zero')
	ax.spines['bottom'].set_smart_bounds(True)
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	ax.xaxis.set_ticks_position('bottom')
	plt.axis('equal')
	plt.yticks([2,4,6,8])
	plt.xticks([-6,-4,-2,0,2,4,6])
	plt.rc('font', size=9)
	plt.tick_params(labelsize=9)
	plt.text(-0.2,-1.6,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae}$', fontsize=9)
	plt.text(-0.2,8.3,r'$\mathcal{v}_\perp/\mathcal{v}_{Ae}$', fontsize=9)
	plt.colorbar(label=r'$Log(F/F_{MAX})$')
	plt.show()
	for t in range(140):
		f_1=dot(AQ2, f_1)
		for y in range(2*Nv):
			for l in range(2*Nv):
				if ((pal_v[l]**2+per_v[y]**2)**0.5)<0.1:
					f_1[y*2*Nv+l]=10**(-10)

					
<matplotlib.contour.QuadContourSet object at 0x0000018E2AD89B80>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ACCE940>, <matplotlib.axis.YTick object at 0x0000018E2AB5FEB0>, <matplotlib.axis.YTick object at 0x0000018E2ABB6790>, <matplotlib.axis.YTick object at 0x0000018E2ABB6A30>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ACCE970>, <matplotlib.axis.XTick object at 0x0000018E2AB06F70>, <matplotlib.axis.XTick object at 0x0000018E2ABB6340>, <matplotlib.axis.XTick object at 0x0000018E2AB23820>, <matplotlib.axis.XTick object at 0x0000018E2AB23550>, <matplotlib.axis.XTick object at 0x0000018E2B2AFEB0>, <matplotlib.axis.XTick object at 0x0000018E2B2AF280>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AD755E0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC45280>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC96B50>, <matplotlib.axis.YTick object at 0x0000018E2AC65970>, <matplotlib.axis.YTick object at 0x0000018E2B38A460>, <matplotlib.axis.YTick object at 0x0000018E2B38A040>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC965B0>, <matplotlib.axis.XTick object at 0x0000018E2AAB5A00>, <matplotlib.axis.XTick object at 0x0000018E2B38A1C0>, <matplotlib.axis.XTick object at 0x0000018E2ABE8370>, <matplotlib.axis.XTick object at 0x0000018E2ABE84C0>, <matplotlib.axis.XTick object at 0x0000018E2ABE8940>, <matplotlib.axis.XTick object at 0x0000018E2AC212E0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB8FEE0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2ABDE940>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB30070>, <matplotlib.axis.YTick object at 0x0000018E2A9B16D0>, <matplotlib.axis.YTick object at 0x0000018E2A9DF370>, <matplotlib.axis.YTick object at 0x0000018E2ABAD910>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AA088E0>, <matplotlib.axis.XTick object at 0x0000018E2AC95850>, <matplotlib.axis.XTick object at 0x0000018E2ABAD8E0>, <matplotlib.axis.XTick object at 0x0000018E2A98DBE0>, <matplotlib.axis.XTick object at 0x0000018E2A98DA00>, <matplotlib.axis.XTick object at 0x0000018E2A98D370>, <matplotlib.axis.XTick object at 0x0000018E2ACDA340>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ABE5AF0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB8FC40>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB7BD60>, <matplotlib.axis.YTick object at 0x0000018E2B31E160>, <matplotlib.axis.YTick object at 0x0000018E2A9F3DF0>, <matplotlib.axis.YTick object at 0x0000018E2A9F32E0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB7BCA0>, <matplotlib.axis.XTick object at 0x0000018E2AB30BE0>, <matplotlib.axis.XTick object at 0x0000018E2A9F3EE0>, <matplotlib.axis.XTick object at 0x0000018E2B2AF520>, <matplotlib.axis.XTick object at 0x0000018E2B2AFD30>, <matplotlib.axis.XTick object at 0x0000018E2A9B2610>, <matplotlib.axis.XTick object at 0x0000018E2A9B23A0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AAB5DF0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2ACC3370>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A96F100>, <matplotlib.axis.YTick object at 0x0000018E2AB368E0>, <matplotlib.axis.YTick object at 0x0000018E2AAFC940>, <matplotlib.axis.YTick object at 0x0000018E2B47F400>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A96F7F0>, <matplotlib.axis.XTick object at 0x0000018E2AC4F070>, <matplotlib.axis.XTick object at 0x0000018E2AAFC0D0>, <matplotlib.axis.XTick object at 0x0000018E2B47F580>, <matplotlib.axis.XTick object at 0x0000018E2AD89F10>, <matplotlib.axis.XTick object at 0x0000018E2AD896D0>, <matplotlib.axis.XTick object at 0x0000018E2AD89EE0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ACC6610>
<matplotlib.contour.QuadContourSet object at 0x0000018E2ABB6B20>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B3B6E80>, <matplotlib.axis.YTick object at 0x0000018E2ABA9760>, <matplotlib.axis.YTick object at 0x0000018E2ACF8820>, <matplotlib.axis.YTick object at 0x0000018E2AD04910>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B3182B0>, <matplotlib.axis.XTick object at 0x0000018E2AAB2430>, <matplotlib.axis.XTick object at 0x0000018E2ACF88B0>, <matplotlib.axis.XTick object at 0x0000018E2AD04F70>, <matplotlib.axis.XTick object at 0x0000018E2A97AA60>, <matplotlib.axis.XTick object at 0x0000018E2A97A9D0>, <matplotlib.axis.XTick object at 0x0000018E2A97A4C0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AA5BA00>
<matplotlib.contour.QuadContourSet object at 0x0000018E2ABEEF40>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB43EB0>, <matplotlib.axis.YTick object at 0x0000018E2B324730>, <matplotlib.axis.YTick object at 0x0000018E2B2A12E0>, <matplotlib.axis.YTick object at 0x0000018E2B2A1250>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AD04190>, <matplotlib.axis.XTick object at 0x0000018E2B2A1A00>, <matplotlib.axis.XTick object at 0x0000018E2BCA7760>, <matplotlib.axis.XTick object at 0x0000018E2B2A1670>, <matplotlib.axis.XTick object at 0x0000018E2BCA7CD0>, <matplotlib.axis.XTick object at 0x0000018E2BCA7700>, <matplotlib.axis.XTick object at 0x0000018E2A990CD0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B45C2B0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2B289880>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB8FC40>, <matplotlib.axis.YTick object at 0x0000018E2A9B2DC0>, <matplotlib.axis.YTick object at 0x0000018E2AB7B970>, <matplotlib.axis.YTick object at 0x0000018E2AB7B760>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC24F40>, <matplotlib.axis.XTick object at 0x0000018E2AD07100>, <matplotlib.axis.XTick object at 0x0000018E2AB7BAF0>, <matplotlib.axis.XTick object at 0x0000018E2B44D730>, <matplotlib.axis.XTick object at 0x0000018E2AB83070>, <matplotlib.axis.XTick object at 0x0000018E2AB83D30>, <matplotlib.axis.XTick object at 0x0000018E2AB83B20>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AD5E190>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAFF5B0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B47F9A0>, <matplotlib.axis.YTick object at 0x0000018E2B4168E0>, <matplotlib.axis.YTick object at 0x0000018E2AA08460>, <matplotlib.axis.YTick object at 0x0000018E2B38AEE0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B30F940>, <matplotlib.axis.XTick object at 0x0000018E2ACDAE80>, <matplotlib.axis.XTick object at 0x0000018E2B38A160>, <matplotlib.axis.XTick object at 0x0000018E2AB30940>, <matplotlib.axis.XTick object at 0x0000018E2AB30850>, <matplotlib.axis.XTick object at 0x0000018E2AB30670>, <matplotlib.axis.XTick object at 0x0000018CFFD97370>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9F1A60>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAC21C0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ABDE910>, <matplotlib.axis.YTick object at 0x0000018E2AD5BB20>, <matplotlib.axis.YTick object at 0x0000018E2AB7C340>, <matplotlib.axis.YTick object at 0x0000018E2AB7C850>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB65F10>, <matplotlib.axis.XTick object at 0x0000018E2A9F08B0>, <matplotlib.axis.XTick object at 0x0000018E2AB7C730>, <matplotlib.axis.XTick object at 0x0000018E2A9E53A0>, <matplotlib.axis.XTick object at 0x0000018E2A9E58E0>, <matplotlib.axis.XTick object at 0x0000018E2A9E5DF0>, <matplotlib.axis.XTick object at 0x0000018E2A9FD340>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB44AF0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AA16430>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB5F5B0>, <matplotlib.axis.YTick object at 0x0000018E2AC414C0>, <matplotlib.axis.YTick object at 0x0000018E2AC7C460>, <matplotlib.axis.YTick object at 0x0000018E2B28EB50>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ABAD190>, <matplotlib.axis.XTick object at 0x0000018E2AAA6460>, <matplotlib.axis.XTick object at 0x0000018E2AC7CBE0>, <matplotlib.axis.XTick object at 0x0000018D62FF4550>, <matplotlib.axis.XTick object at 0x0000018D62FF46D0>, <matplotlib.axis.XTick object at 0x0000018E2AC4F370>, <matplotlib.axis.XTick object at 0x0000018E2AC4F6D0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB8F550>
<matplotlib.contour.QuadContourSet object at 0x0000018E2B324A90>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ACC3280>, <matplotlib.axis.YTick object at 0x0000018E2AAE0DF0>, <matplotlib.axis.YTick object at 0x0000018E2ACC61C0>, <matplotlib.axis.YTick object at 0x0000018E2ACC6EB0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAFC9A0>, <matplotlib.axis.XTick object at 0x0000018E2AA3C880>, <matplotlib.axis.XTick object at 0x0000018E2ACC6D00>, <matplotlib.axis.XTick object at 0x0000018E2AB36970>, <matplotlib.axis.XTick object at 0x0000018E2AB36790>, <matplotlib.axis.XTick object at 0x0000018E2AC3B580>, <matplotlib.axis.XTick object at 0x0000018E2AC3B940>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AC87D00>
<matplotlib.contour.QuadContourSet object at 0x0000018E2B30F790>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2BCA7F40>, <matplotlib.axis.YTick object at 0x0000018E2B416970>, <matplotlib.axis.YTick object at 0x0000018E2AD750A0>, <matplotlib.axis.YTick object at 0x0000018CFFD97400>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2BCA7F10>, <matplotlib.axis.XTick object at 0x0000018E2BCA8610>, <matplotlib.axis.XTick object at 0x0000018CFFD977F0>, <matplotlib.axis.XTick object at 0x0000018E2B47F040>, <matplotlib.axis.XTick object at 0x0000018E2B47F580>, <matplotlib.axis.XTick object at 0x0000018E2B47FCD0>, <matplotlib.axis.XTick object at 0x0000018E2A971880>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ABA9D60>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAB2340>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A96F700>, <matplotlib.axis.YTick object at 0x0000018E2AA98CD0>, <matplotlib.axis.YTick object at 0x0000018E2B4168B0>, <matplotlib.axis.YTick object at 0x0000018E2B416070>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B3B6850>, <matplotlib.axis.XTick object at 0x0000018E2AA975E0>, <matplotlib.axis.XTick object at 0x0000018E2AB36C70>, <matplotlib.axis.XTick object at 0x0000018E2B416130>, <matplotlib.axis.XTick object at 0x0000018E2AB369A0>, <matplotlib.axis.XTick object at 0x0000018E2AB36100>, <matplotlib.axis.XTick object at 0x0000018E2AC3B8E0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2BCA74F0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC7C7C0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018D62FF4400>, <matplotlib.axis.YTick object at 0x0000018E2A9F1AC0>, <matplotlib.axis.YTick object at 0x0000018E2AB77E80>, <matplotlib.axis.YTick object at 0x0000018E2AB77D60>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC4FEB0>, <matplotlib.axis.XTick object at 0x0000018E2A9F1880>, <matplotlib.axis.XTick object at 0x0000018E2AB77C10>, <matplotlib.axis.XTick object at 0x0000018E2AD76F70>, <matplotlib.axis.XTick object at 0x0000018E2AD76130>, <matplotlib.axis.XTick object at 0x0000018E2AD76FD0>, <matplotlib.axis.XTick object at 0x0000018E2AAFC8B0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AAB36D0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB8FA00>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A9FDA90>, <matplotlib.axis.YTick object at 0x0000018E2AAA6340>, <matplotlib.axis.YTick object at 0x0000018E2AC968E0>, <matplotlib.axis.YTick object at 0x0000018E2AB83EB0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A9FD9D0>, <matplotlib.axis.XTick object at 0x0000018E2AAB2310>, <matplotlib.axis.XTick object at 0x0000018E2AB83C70>, <matplotlib.axis.XTick object at 0x0000018E2AC96730>, <matplotlib.axis.XTick object at 0x0000018E2AA086A0>, <matplotlib.axis.XTick object at 0x0000018E2AA08B20>, <matplotlib.axis.XTick object at 0x0000018E2AA08790>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ABB6DC0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AA01B80>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AD89550>, <matplotlib.axis.YTick object at 0x0000018E2A9DAF10>, <matplotlib.axis.YTick object at 0x0000018E2AA11280>, <matplotlib.axis.YTick object at 0x0000018E2AA11790>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AD89F10>, <matplotlib.axis.XTick object at 0x0000018E2AB06730>, <matplotlib.axis.XTick object at 0x0000018E2AA11A60>, <matplotlib.axis.XTick object at 0x0000018E2AA20340>, <matplotlib.axis.XTick object at 0x0000018E2AA20820>, <matplotlib.axis.XTick object at 0x0000018E2AA20D30>, <matplotlib.axis.XTick object at 0x0000018E2AA5C280>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AADCA30>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC39FD0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AA20130>, <matplotlib.axis.YTick object at 0x0000018E2A9B2B20>, <matplotlib.axis.YTick object at 0x0000018E2AD07FD0>, <matplotlib.axis.YTick object at 0x0000018E2AA16E20>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AD89160>, <matplotlib.axis.XTick object at 0x0000018E2B4564F0>, <matplotlib.axis.XTick object at 0x0000018E2AD07F10>, <matplotlib.axis.XTick object at 0x0000018E2AA16520>, <matplotlib.axis.XTick object at 0x0000018E2B346AC0>, <matplotlib.axis.XTick object at 0x0000018E2B346BE0>, <matplotlib.axis.XTick object at 0x0000018E2B346F70>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9F0550>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC26550>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B31EA00>, <matplotlib.axis.YTick object at 0x0000018E2AB7BCA0>, <matplotlib.axis.YTick object at 0x0000018E2A9ADB80>, <matplotlib.axis.YTick object at 0x0000018CFFD74820>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B31EF10>, <matplotlib.axis.XTick object at 0x0000018E2B3100A0>, <matplotlib.axis.XTick object at 0x0000018CFFD74730>, <matplotlib.axis.XTick object at 0x0000018E2AB347F0>, <matplotlib.axis.XTick object at 0x0000018E2AB34EE0>, <matplotlib.axis.XTick object at 0x0000018E2AC5B5E0>, <matplotlib.axis.XTick object at 0x0000018E2AC5B1C0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AAFFAF0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2B2A1490>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ABEE190>, <matplotlib.axis.YTick object at 0x0000018E2AC45D00>, <matplotlib.axis.YTick object at 0x0000018E2AB36A00>, <matplotlib.axis.YTick object at 0x0000018E2AB36670>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ABEECD0>, <matplotlib.axis.XTick object at 0x0000018E2ACC3BB0>, <matplotlib.axis.XTick object at 0x0000018E2AB36B50>, <matplotlib.axis.XTick object at 0x0000018E2AD5B880>, <matplotlib.axis.XTick object at 0x0000018E2AD5B7F0>, <matplotlib.axis.XTick object at 0x0000018E2ACF8490>, <matplotlib.axis.XTick object at 0x0000018E2ACF8430>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ACEF580>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC009D0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ACEFB50>, <matplotlib.axis.YTick object at 0x0000018E2AC41220>, <matplotlib.axis.YTick object at 0x0000018E2AC872B0>, <matplotlib.axis.YTick object at 0x0000018E2AC87BE0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2BCA80D0>, <matplotlib.axis.XTick object at 0x0000018E2AC87100>, <matplotlib.axis.XTick object at 0x0000018E2B44D700>, <matplotlib.axis.XTick object at 0x0000018E2AC87610>, <matplotlib.axis.XTick object at 0x0000018E2B44D820>, <matplotlib.axis.XTick object at 0x0000018E2B44D310>, <matplotlib.axis.XTick object at 0x0000018E2B28E9A0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AC95820>
<matplotlib.contour.QuadContourSet object at 0x0000018E2ABADEE0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AA16B80>, <matplotlib.axis.YTick object at 0x0000018E2AAC89A0>, <matplotlib.axis.YTick object at 0x0000018E2A9F02E0>, <matplotlib.axis.YTick object at 0x0000018E2A9F0EB0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B3242E0>, <matplotlib.axis.XTick object at 0x0000018E2AAFF460>, <matplotlib.axis.XTick object at 0x0000018E2A9F0C10>, <matplotlib.axis.XTick object at 0x0000018E2A9CC580>, <matplotlib.axis.XTick object at 0x0000018E2A9CC310>, <matplotlib.axis.XTick object at 0x0000018E2B318460>, <matplotlib.axis.XTick object at 0x0000018E2B318880>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B456B50>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAD08E0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC19D60>, <matplotlib.axis.YTick object at 0x0000018E2A9F9FA0>, <matplotlib.axis.YTick object at 0x0000018E2B2C0580>, <matplotlib.axis.YTick object at 0x0000018E2B2C0670>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC194F0>, <matplotlib.axis.XTick object at 0x0000018E2AB36700>, <matplotlib.axis.XTick object at 0x0000018E2B2C0F70>, <matplotlib.axis.XTick object at 0x0000018E2AB8F340>, <matplotlib.axis.XTick object at 0x0000018E2AB8FDC0>, <matplotlib.axis.XTick object at 0x0000018E2AB65B20>, <matplotlib.axis.XTick object at 0x0000018E2AB659D0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9DACD0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AA98430>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2BCBE130>, <matplotlib.axis.YTick object at 0x0000018E2A9B1FA0>, <matplotlib.axis.YTick object at 0x0000018E2AA03700>, <matplotlib.axis.YTick object at 0x0000018E2AA03C10>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ABDE670>, <matplotlib.axis.XTick object at 0x0000018E2A9963A0>, <matplotlib.axis.XTick object at 0x0000018E2AA03970>, <matplotlib.axis.XTick object at 0x0000018E2A9CE730>, <matplotlib.axis.XTick object at 0x0000018E2A9CECA0>, <matplotlib.axis.XTick object at 0x0000018E2A9CD1F0>, <matplotlib.axis.XTick object at 0x0000018E2A9CD700>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB98EB0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AAFCE20>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B2C0B80>, <matplotlib.axis.YTick object at 0x0000018E2AB9B880>, <matplotlib.axis.YTick object at 0x0000018E2ABAE7C0>, <matplotlib.axis.YTick object at 0x0000018E2AAE9070>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B2C0820>, <matplotlib.axis.XTick object at 0x0000018E2A9B1670>, <matplotlib.axis.XTick object at 0x0000018E2ABAE160>, <matplotlib.axis.XTick object at 0x0000018E2AAE9610>, <matplotlib.axis.XTick object at 0x0000018E2AD075B0>, <matplotlib.axis.XTick object at 0x0000018E2AD07D60>, <matplotlib.axis.XTick object at 0x0000018E2AD07C10>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A990760>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC96310>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC6D1F0>, <matplotlib.axis.YTick object at 0x0000018E2AC21CD0>, <matplotlib.axis.YTick object at 0x0000018E2AC269D0>, <matplotlib.axis.YTick object at 0x0000018E2AAA4B20>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC6DF10>, <matplotlib.axis.XTick object at 0x0000018E2AA08070>, <matplotlib.axis.XTick object at 0x0000018E2AC26B50>, <matplotlib.axis.XTick object at 0x0000018E2AAA4D30>, <matplotlib.axis.XTick object at 0x0000018E2A964C70>, <matplotlib.axis.XTick object at 0x0000018E2A964D00>, <matplotlib.axis.XTick object at 0x0000018E2A964160>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9AD310>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC00FD0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AA5B1F0>, <matplotlib.axis.YTick object at 0x0000018E2AC9F220>, <matplotlib.axis.YTick object at 0x0000018E2AD76220>, <matplotlib.axis.YTick object at 0x0000018E2AD76730>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC3B520>, <matplotlib.axis.XTick object at 0x0000018E2AB7B7C0>, <matplotlib.axis.XTick object at 0x0000018E2AD767F0>, <matplotlib.axis.XTick object at 0x0000018E2B3B6AC0>, <matplotlib.axis.XTick object at 0x0000018E2B3B6220>, <matplotlib.axis.XTick object at 0x0000018E2AD5B7C0>, <matplotlib.axis.XTick object at 0x0000018E2AD5B340>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B34CBB0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB23850>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B34C9D0>, <matplotlib.axis.YTick object at 0x0000018E2AA3C370>, <matplotlib.axis.YTick object at 0x0000018E2AC87880>, <matplotlib.axis.YTick object at 0x0000018E2AC87910>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B26C100>, <matplotlib.axis.XTick object at 0x0000018E2A97A8E0>, <matplotlib.axis.XTick object at 0x0000018CFFD74A00>, <matplotlib.axis.XTick object at 0x0000018E2AC87370>, <matplotlib.axis.XTick object at 0x0000018CFFD74730>, <matplotlib.axis.XTick object at 0x0000018CFFD74E50>, <matplotlib.axis.XTick object at 0x0000018E2AA5BBE0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AAA4850>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC21730>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B318BE0>, <matplotlib.axis.YTick object at 0x0000018E2ABB6F70>, <matplotlib.axis.YTick object at 0x0000018E2AC19700>, <matplotlib.axis.YTick object at 0x0000018E2AC39190>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AD89DC0>, <matplotlib.axis.XTick object at 0x0000018E2AAD07C0>, <matplotlib.axis.XTick object at 0x0000018E2AC39A90>, <matplotlib.axis.XTick object at 0x0000018E2AC19F70>, <matplotlib.axis.XTick object at 0x0000018E2A964A00>, <matplotlib.axis.XTick object at 0x0000018E2A964610>, <matplotlib.axis.XTick object at 0x0000018E2A964910>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AC96220>
<matplotlib.contour.QuadContourSet object at 0x0000018E2BCA7970>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAA60A0>, <matplotlib.axis.YTick object at 0x0000018E2ABB60A0>, <matplotlib.axis.YTick object at 0x0000018E2AC6DF70>, <matplotlib.axis.YTick object at 0x0000018E2AC6DBB0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AB77160>, <matplotlib.axis.XTick object at 0x0000018E2B2C0A90>, <matplotlib.axis.XTick object at 0x0000018E2AC6D4F0>, <matplotlib.axis.XTick object at 0x0000018E2AC88E80>, <matplotlib.axis.XTick object at 0x0000018E2AC88460>, <matplotlib.axis.XTick object at 0x0000018E2AC88F40>, <matplotlib.axis.XTick object at 0x0000018E2AA16CD0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9CCDF0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB83DF0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A99D910>, <matplotlib.axis.YTick object at 0x0000018E2AB30DC0>, <matplotlib.axis.YTick object at 0x0000018E2AD737F0>, <matplotlib.axis.YTick object at 0x0000018E2AD73D00>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018D62FF46D0>, <matplotlib.axis.XTick object at 0x0000018E2ACC3B50>, <matplotlib.axis.XTick object at 0x0000018E2AD73670>, <matplotlib.axis.XTick object at 0x0000018E2AD5F190>, <matplotlib.axis.XTick object at 0x0000018E2AD5FD90>, <matplotlib.axis.XTick object at 0x0000018E2AD5D2E0>, <matplotlib.axis.XTick object at 0x0000018E2AD5D7F0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AC48FA0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB30190>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAB2760>, <matplotlib.axis.YTick object at 0x0000018D62FF4EE0>, <matplotlib.axis.YTick object at 0x0000018E2B2C0430>, <matplotlib.axis.YTick object at 0x0000018E2B2C0730>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAB2910>, <matplotlib.axis.XTick object at 0x0000018E2AB9B400>, <matplotlib.axis.XTick object at 0x0000018E2B2C0940>, <matplotlib.axis.XTick object at 0x0000018E2ACCE580>, <matplotlib.axis.XTick object at 0x0000018E2ACCEF40>, <matplotlib.axis.XTick object at 0x0000018E2ABCD6A0>, <matplotlib.axis.XTick object at 0x0000018E2ABCDF40>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A964FD0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2B38A1C0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAA6C10>, <matplotlib.axis.YTick object at 0x0000018E2A9CD820>, <matplotlib.axis.YTick object at 0x0000018E2AAA4220>, <matplotlib.axis.YTick object at 0x0000018E2AD89AC0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAA6AC0>, <matplotlib.axis.XTick object at 0x0000018E2AC7C760>, <matplotlib.axis.XTick object at 0x0000018E2AAA48E0>, <matplotlib.axis.XTick object at 0x0000018E2AD899A0>, <matplotlib.axis.XTick object at 0x0000018E2ABB61C0>, <matplotlib.axis.XTick object at 0x0000018E2ABB6430>, <matplotlib.axis.XTick object at 0x0000018E2ABB6910>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AD07F10>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC6A550>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC41AC0>, <matplotlib.axis.YTick object at 0x0000018E2A9DA280>, <matplotlib.axis.YTick object at 0x0000018E2AD5BF40>, <matplotlib.axis.YTick object at 0x0000018E2AD5B640>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ABAE0A0>, <matplotlib.axis.XTick object at 0x0000018E2B31E2E0>, <matplotlib.axis.XTick object at 0x0000018E2AD5BD60>, <matplotlib.axis.XTick object at 0x0000018E2AAE9DF0>, <matplotlib.axis.XTick object at 0x0000018E2AAE9940>, <matplotlib.axis.XTick object at 0x0000018E2AAE9790>, <matplotlib.axis.XTick object at 0x0000018E2AD76790>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018CFFD973D0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2BCA74F0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AD76520>, <matplotlib.axis.YTick object at 0x0000018E2AAC8640>, <matplotlib.axis.YTick object at 0x0000018E2AB36E50>, <matplotlib.axis.YTick object at 0x0000018E2AB36040>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A965580>, <matplotlib.axis.XTick object at 0x0000018E2ACEFA30>, <matplotlib.axis.XTick object at 0x0000018E2AA3CD00>, <matplotlib.axis.XTick object at 0x0000018E2AB360A0>, <matplotlib.axis.XTick object at 0x0000018E2AA3C6A0>, <matplotlib.axis.XTick object at 0x0000018E2AA3CA60>, <matplotlib.axis.XTick object at 0x0000018E2AC5BF70>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B31EE80>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB8FD60>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC12A60>, <matplotlib.axis.YTick object at 0x0000018E2AAF5D30>, <matplotlib.axis.YTick object at 0x0000018E2A9B1A90>, <matplotlib.axis.YTick object at 0x0000018E2A9B1730>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAA6880>, <matplotlib.axis.XTick object at 0x0000018E2AAF5BB0>, <matplotlib.axis.XTick object at 0x0000018E2A9B1910>, <matplotlib.axis.XTick object at 0x0000018E2B2C08E0>, <matplotlib.axis.XTick object at 0x0000018E2B2C0AF0>, <matplotlib.axis.XTick object at 0x0000018E2AAC2250>, <matplotlib.axis.XTick object at 0x0000018E2AAC2700>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB11490>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AD7A310>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AAB2AF0>, <matplotlib.axis.YTick object at 0x0000018E2B346F40>, <matplotlib.axis.YTick object at 0x0000018E2B2A1A60>, <matplotlib.axis.YTick object at 0x0000018E2A996D90>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ACC3CA0>, <matplotlib.axis.XTick object at 0x0000018E2AB83C10>, <matplotlib.axis.XTick object at 0x0000018E2A996DC0>, <matplotlib.axis.XTick object at 0x0000018E2AAD0B20>, <matplotlib.axis.XTick object at 0x0000018E2AAD0CD0>, <matplotlib.axis.XTick object at 0x0000018E2AAD0310>, <matplotlib.axis.XTick object at 0x0000018E2AC96D60>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AD70220>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC4F820>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B30F0D0>, <matplotlib.axis.YTick object at 0x0000018E2AADA910>, <matplotlib.axis.YTick object at 0x0000018E2AA3A9A0>, <matplotlib.axis.YTick object at 0x0000018E2BCC0040>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC61E20>, <matplotlib.axis.XTick object at 0x0000018E2AAA6EE0>, <matplotlib.axis.XTick object at 0x0000018E2AA3A7C0>, <matplotlib.axis.XTick object at 0x0000018E2BCC02E0>, <matplotlib.axis.XTick object at 0x0000018E2AA3F0D0>, <matplotlib.axis.XTick object at 0x0000018E2AA3F490>, <matplotlib.axis.XTick object at 0x0000018E2AA3F9A0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AC66190>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC96670>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB5FA90>, <matplotlib.axis.YTick object at 0x0000018E2B2A16D0>, <matplotlib.axis.YTick object at 0x0000018E2A9F1520>, <matplotlib.axis.YTick object at 0x0000018E2A9F1C40>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAFF580>, <matplotlib.axis.XTick object at 0x0000018E2ACC67C0>, <matplotlib.axis.XTick object at 0x0000018E2A9F1550>, <matplotlib.axis.XTick object at 0x0000018E2AC247C0>, <matplotlib.axis.XTick object at 0x0000018E2AC24340>, <matplotlib.axis.XTick object at 0x0000018E2B316A90>, <matplotlib.axis.XTick object at 0x0000018E2B316610>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A9B13A0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC95B20>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2B38A3A0>, <matplotlib.axis.YTick object at 0x0000018E2A9F0CD0>, <matplotlib.axis.YTick object at 0x0000018E2AA3C310>, <matplotlib.axis.YTick object at 0x0000018E2AA3CD00>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B38AB20>, <matplotlib.axis.XTick object at 0x0000018E2AAA49A0>, <matplotlib.axis.XTick object at 0x0000018E2AA3CCD0>, <matplotlib.axis.XTick object at 0x0000018E2AB36190>, <matplotlib.axis.XTick object at 0x0000018E2AB369D0>, <matplotlib.axis.XTick object at 0x0000018E2AC5BDF0>, <matplotlib.axis.XTick object at 0x0000018E2AC5B970>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AAEE7C0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2B324760>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A965B80>, <matplotlib.axis.YTick object at 0x0000018E2B2AFEB0>, <matplotlib.axis.YTick object at 0x0000018E2ACF80D0>, <matplotlib.axis.YTick object at 0x0000018E2ACF8EB0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A965E20>, <matplotlib.axis.XTick object at 0x0000018E2AA08B80>, <matplotlib.axis.XTick object at 0x0000018E2ACF8490>, <matplotlib.axis.XTick object at 0x0000018E2A990FD0>, <matplotlib.axis.XTick object at 0x0000018E2A990910>, <matplotlib.axis.XTick object at 0x0000018E2A9905E0>, <matplotlib.axis.XTick object at 0x0000018E2A9DA8B0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AC19550>
<matplotlib.contour.QuadContourSet object at 0x0000018E2B346EE0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AB65220>, <matplotlib.axis.YTick object at 0x0000018E2AB65340>, <matplotlib.axis.YTick object at 0x0000018E2AAB5580>, <matplotlib.axis.YTick object at 0x0000018E2AAB58B0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B416610>, <matplotlib.axis.XTick object at 0x0000018E2B4166D0>, <matplotlib.axis.XTick object at 0x0000018E2AC392B0>, <matplotlib.axis.XTick object at 0x0000018E2AAB5070>, <matplotlib.axis.XTick object at 0x0000018E2AC39430>, <matplotlib.axis.XTick object at 0x0000018E2B2F6A30>, <matplotlib.axis.XTick object at 0x0000018E2B3B6C40>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AC00340>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB32310>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AD5EB80>, <matplotlib.axis.YTick object at 0x0000018E2A9ADB80>, <matplotlib.axis.YTick object at 0x0000018E2AAA4100>, <matplotlib.axis.YTick object at 0x0000018E2AAA4CD0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ABEE520>, <matplotlib.axis.XTick object at 0x0000018E2AC21700>, <matplotlib.axis.XTick object at 0x0000018E2AAA40A0>, <matplotlib.axis.XTick object at 0x0000018CFFD97940>, <matplotlib.axis.XTick object at 0x0000018E2AB5FE80>, <matplotlib.axis.XTick object at 0x0000018E2AB5F220>, <matplotlib.axis.XTick object at 0x0000018E2AB5F6D0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AA16880>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AADA820>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A9CDAF0>, <matplotlib.axis.YTick object at 0x0000018E2AC65CA0>, <matplotlib.axis.YTick object at 0x0000018E2AAE0880>, <matplotlib.axis.YTick object at 0x0000018E2ABC4FD0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2A9CDF70>, <matplotlib.axis.XTick object at 0x0000018E2AC4FCA0>, <matplotlib.axis.XTick object at 0x0000018E2ABC46D0>, <matplotlib.axis.XTick object at 0x0000018E2A964220>, <matplotlib.axis.XTick object at 0x0000018E2A9649D0>, <matplotlib.axis.XTick object at 0x0000018E2A964340>, <matplotlib.axis.XTick object at 0x0000018E2AC30640>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AB30BE0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AB43F70>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AD70250>, <matplotlib.axis.YTick object at 0x0000018E2ABB5C70>, <matplotlib.axis.YTick object at 0x0000018E2ACA2040>, <matplotlib.axis.YTick object at 0x0000018E2ACA2400>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AD70BE0>, <matplotlib.axis.XTick object at 0x0000018E2B45CA00>, <matplotlib.axis.XTick object at 0x0000018E2ACA2F40>, <matplotlib.axis.XTick object at 0x0000018E2AC330D0>, <matplotlib.axis.XTick object at 0x0000018E2AC33490>, <matplotlib.axis.XTick object at 0x0000018E2AC339A0>, <matplotlib.axis.XTick object at 0x0000018E2AC33E20>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AADC6A0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC7C580>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2A99DCA0>, <matplotlib.axis.YTick object at 0x0000018E2AC6D490>, <matplotlib.axis.YTick object at 0x0000018E2B45CD30>, <matplotlib.axis.YTick object at 0x0000018E2B45CD00>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B456E20>, <matplotlib.axis.XTick object at 0x0000018E2BCA8B20>, <matplotlib.axis.XTick object at 0x0000018E2B45CDC0>, <matplotlib.axis.XTick object at 0x0000018E2A9F13D0>, <matplotlib.axis.XTick object at 0x0000018E2A9F1E20>, <matplotlib.axis.XTick object at 0x0000018E2AB5FB50>, <matplotlib.axis.XTick object at 0x0000018E2AB5F490>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2B30FD00>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AD768E0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AC21430>, <matplotlib.axis.YTick object at 0x0000018E2A96FD60>, <matplotlib.axis.YTick object at 0x0000018E2AD89340>, <matplotlib.axis.YTick object at 0x0000018E2AD89550>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AC21640>, <matplotlib.axis.XTick object at 0x0000018E2AC863D0>, <matplotlib.axis.XTick object at 0x0000018E2AD899A0>, <matplotlib.axis.XTick object at 0x0000018E2B316880>, <matplotlib.axis.XTick object at 0x0000018E2B316FD0>, <matplotlib.axis.XTick object at 0x0000018E2ACC3E50>, <matplotlib.axis.XTick object at 0x0000018E2ACC3910>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AC616D0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2B34C880>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ACEF3A0>, <matplotlib.axis.YTick object at 0x0000018E2AAE96D0>, <matplotlib.axis.YTick object at 0x0000018E2A9F9D60>, <matplotlib.axis.YTick object at 0x0000018E2B416E80>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ACEF5E0>, <matplotlib.axis.XTick object at 0x0000018E2AD07460>, <matplotlib.axis.XTick object at 0x0000018E2A9F9C10>, <matplotlib.axis.XTick object at 0x0000018E2B416670>, <matplotlib.axis.XTick object at 0x0000018E2A9900D0>, <matplotlib.axis.XTick object at 0x0000018E2A990160>, <matplotlib.axis.XTick object at 0x0000018E2A990FD0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2ABAE0D0>
<matplotlib.contour.QuadContourSet object at 0x0000018E2A965850>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018CFFD74820>, <matplotlib.axis.YTick object at 0x0000018E2AB11700>, <matplotlib.axis.YTick object at 0x0000018E2AA46BE0>, <matplotlib.axis.YTick object at 0x0000018E2AA46E80>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2B44D280>, <matplotlib.axis.XTick object at 0x0000018E2B44D160>, <matplotlib.axis.XTick object at 0x0000018E2AADA040>, <matplotlib.axis.XTick object at 0x0000018E2AA46EE0>, <matplotlib.axis.XTick object at 0x0000018E2AADAEE0>, <matplotlib.axis.XTick object at 0x0000018E2AADAA30>, <matplotlib.axis.XTick object at 0x0000018E2AB833D0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2BCA7C70>
<matplotlib.contour.QuadContourSet object at 0x0000018E2AC21CD0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2AA16F40>, <matplotlib.axis.YTick object at 0x0000018E2AB06E50>, <matplotlib.axis.YTick object at 0x0000018E2B30FEB0>, <matplotlib.axis.YTick object at 0x0000018E2B30FE80>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2AAD00A0>, <matplotlib.axis.XTick object at 0x0000018E2AD5B9A0>, <matplotlib.axis.XTick object at 0x0000018E2B30F910>, <matplotlib.axis.XTick object at 0x0000018E2A99DFA0>, <matplotlib.axis.XTick object at 0x0000018E2A99D220>, <matplotlib.axis.XTick object at 0x0000018E2AC4F580>, <matplotlib.axis.XTick object at 0x0000018E2AC4F1C0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2AA2BA90>
<matplotlib.contour.QuadContourSet object at 0x0000018E2BCBE100>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x0000018E2ABB6CD0>, <matplotlib.axis.YTick object at 0x0000018E2AC33E50>, <matplotlib.axis.YTick object at 0x0000018E2B45C7C0>, <matplotlib.axis.YTick object at 0x0000018E2B45CBE0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x0000018E2ABB6FD0>, <matplotlib.axis.XTick object at 0x0000018E2ACC6D00>, <matplotlib.axis.XTick object at 0x0000018E2B45C9D0>, <matplotlib.axis.XTick object at 0x0000018E2AC1A340>, <matplotlib.axis.XTick object at 0x0000018E2AC1A700>, <matplotlib.axis.XTick object at 0x0000018E2AAFC850>, <matplotlib.axis.XTick object at 0x0000018E2AAFCA60>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x0000018E2A96FE50>
>>> 
