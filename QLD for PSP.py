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
>>> Nv=45
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
>>> fre=0.067
>>> B=5*10**(-4)
>>> Me=9.1094*(10**(-28))
>>> Mp=1.6726*(10**(-24))
>>> ratio=Me/Mp
>>> q=4.8032*(10**(-10))
>>> c=2.9979*10**(10)
>>> k_pal0=0.2185
>>> k_per0=k_pal0*tan((55*np.pi)/180)
>>> k_per0
mpf('0.31205033947315197')
>>> k_pal_max=0.25
>>> k_pal_min=0.187
>>> k_per_max=k_pal_max*tan((55*np.pi)/180)
>>> k_per_min=k_pal_min*tan((55*np.pi)/180)
>>> a_pal=0.0315
>>> a_per=a_pal*tan((55*np.pi)/180)
>>> a_per
mpf('0.044986662212376606')
>>> GV=0.75
>>> B_B0=0.001
>>> def k(b):
    f = lambda x: ((besselj(0, (b*x)/(omega), 0))**2)*np.exp(-(((x-0.312)**2)/(0.045**2)))*x
    I=integrate.quad(f, k_per_min, k_per_max)
    return I[0]

>>> def coefficient_a(a,b):
    return ((0.6*np.pi**2)/(0.0315*0.045**2))*(B_B0**2)*(((b)**2)/abs(a-GV))*(fre/k_pal0)**2*k(b)*(np.exp(-(((fre-a*k_pal0-n*omega)/(a-GV))**2)/(0.0315**2)))

>>> def coefficient_a2(a,b):
    return ((0.023*np.pi**2)/(0.0315*0.045**2))*(B_B0**2)*(((a)**2)/abs(a-GV))*(fre/k_pal0)**2*k(b)*(np.exp(-(((fre-a*k_pal0-n2*omega)/(a-GV))**2)/(0.0315**2)))

>>> def coefficient_a3(a,b):
    return ((0.6*np.pi**2)/(0.0315*0.045**2))*(B_B0**2)*(((b)**2)/abs(a-GV))*(fre/k_pal0)**2*k(b)*(np.exp(-(((fre-a*k_pal0-n3*omega)/(a-GV))**2)/(0.0315**2)))

>>> AA=np.zeros(((2*Nv)*(2*Nv),(2*Nv)*(2*Nv)))
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

>>> delv=2*abs(pal_v[1]-pal_v[0])
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
    return ((1.34)**(-0.5))*((0.2055)**(-1))*0.0262*np.exp(-((b)**2)/0.2055)*np.exp(-((a-Us)**2)/1.34)

>>> Us=150*ratio**(0.5)
>>> Uc=-4.034*ratio**(0.5)
>>> def Kappa_Initial_Core(a,b):
    kappa=150
    return ((0.296)**(-0.5))*((0.289)**(-1))*0.9738*np.exp(-((b)**2)/0.289)*np.exp(-((a-Uc)**2)/0.296)

>>> 
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
    fig.set_dpi(500)
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
    plt.rc('font', size=8)
    plt.tick_params(labelsize=8)
    plt.text(-0.2,-1.6,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae}$', fontsize=8)
    plt.text(-0.2,8.3,r'$\mathcal{v}_\perp/\mathcal{v}_{Ae}$', fontsize=8)
    plt.colorbar(label=r'$Log(F/F_{MAX})$')
    #plt.savefig(QLD/Collision/qld/{k}.png)
    #plt.clf()
    plt.show()
    for t in range(300):
        ff_1=dot(AQ, ff_1)

        
0
<matplotlib.contour.QuadContourSet object at 0x00000212AE285C10>

Warning (from warnings module):
  File "<pyshell#104>", line 19
MatplotlibDeprecationWarning: 
The set_smart_bounds function was deprecated in Matplotlib 3.2 and will be removed two minor releases later.

Warning (from warnings module):
  File "<pyshell#104>", line 21
MatplotlibDeprecationWarning: 
The set_smart_bounds function was deprecated in Matplotlib 3.2 and will be removed two minor releases later.
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE279BB0>, <matplotlib.axis.YTick object at 0x000002126CD5FB80>, <matplotlib.axis.YTick object at 0x00000212AE27B460>, <matplotlib.axis.YTick object at 0x00000212AE27B970>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE26DB80>, <matplotlib.axis.XTick object at 0x000002126CD5F400>, <matplotlib.axis.XTick object at 0x00000212AE2EB190>, <matplotlib.axis.XTick object at 0x00000212AE2EB6A0>, <matplotlib.axis.XTick object at 0x00000212AE2EBBB0>, <matplotlib.axis.XTick object at 0x00000212AE2F1100>, <matplotlib.axis.XTick object at 0x00000212AE2F1610>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE26D130>
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
    fig.set_dpi(500)
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
    plt.rc('font', size=8)
    plt.tick_params(labelsize=8)
    plt.text(-0.2,-1.6,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae}$', fontsize=8)
    plt.text(-0.2,8.3,r'$\mathcal{v}_\perp/\mathcal{v}_{Ae}$', fontsize=8)
    plt.colorbar(label=r'$Log(F/F_{MAX})$')
    #plt.savefig(QLD/Collision/qld/{k}.png)
    #plt.clf()
    plt.show()

    
0
<matplotlib.contour.QuadContourSet object at 0x00000212AE1D0730>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AEB6F6D0>, <matplotlib.axis.YTick object at 0x00000212AE1EEF40>, <matplotlib.axis.YTick object at 0x00000212AE18D040>, <matplotlib.axis.YTick object at 0x00000212AE18D400>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE1509D0>, <matplotlib.axis.XTick object at 0x00000212AE1F2370>, <matplotlib.axis.XTick object at 0x00000212AE4A40D0>, <matplotlib.axis.XTick object at 0x00000212AE4A4490>, <matplotlib.axis.XTick object at 0x00000212AE18D730>, <matplotlib.axis.XTick object at 0x00000212AE4A4610>, <matplotlib.axis.XTick object at 0x00000212AE4A4B50>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE4C86D0>
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
    fig.set_dpi(500)
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
    plt.rc('font', size=8)
    plt.tick_params(labelsize=8)
    plt.text(-0.2,-1.6,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae}$', fontsize=8)
    plt.text(-0.2,8.3,r'$\mathcal{v}_\perp/\mathcal{v}_{Ae}$', fontsize=8)
    plt.colorbar(label=r'$Log(F/F_{MAX})$')
    #plt.savefig(QLD/Collision/qld/{k}.png)
    #plt.clf()
    plt.show()
    for t in range(6):
        ff_1=dot(AQ, ff_1)

        
0
<matplotlib.contour.QuadContourSet object at 0x00000212AE17B970>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE2D5250>, <matplotlib.axis.YTick object at 0x00000212AEB5D250>, <matplotlib.axis.YTick object at 0x00000212AE2F7BE0>, <matplotlib.axis.YTick object at 0x00000212AE259430>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE26DBB0>, <matplotlib.axis.XTick object at 0x00000212AE4E27C0>, <matplotlib.axis.XTick object at 0x00000212AE259E80>, <matplotlib.axis.XTick object at 0x00000212AE259460>, <matplotlib.axis.XTick object at 0x00000212AE4F7220>, <matplotlib.axis.XTick object at 0x00000212AE4F7940>, <matplotlib.axis.XTick object at 0x00000212AE15B8E0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE519820>
1
<matplotlib.contour.QuadContourSet object at 0x00000212AE38E880>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE39BFD0>, <matplotlib.axis.YTick object at 0x00000212AE2BD2B0>, <matplotlib.axis.YTick object at 0x00000212AE37E070>, <matplotlib.axis.YTick object at 0x00000212AE37E520>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE39BE20>, <matplotlib.axis.XTick object at 0x00000212AE363A00>, <matplotlib.axis.XTick object at 0x00000212AE37E7F0>, <matplotlib.axis.XTick object at 0x00000212AE673070>, <matplotlib.axis.XTick object at 0x00000212AE6735B0>, <matplotlib.axis.XTick object at 0x00000212AE673AC0>, <matplotlib.axis.XTick object at 0x00000212AE6750A0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE699850>
2
<matplotlib.contour.QuadContourSet object at 0x00000212AE5869A0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE53AA00>, <matplotlib.axis.YTick object at 0x00000212AE5611C0>, <matplotlib.axis.YTick object at 0x00000212AE567130>, <matplotlib.axis.YTick object at 0x00000212AE567640>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE53A8E0>, <matplotlib.axis.XTick object at 0x00000212AE5439A0>, <matplotlib.axis.XTick object at 0x00000212AE5674C0>, <matplotlib.axis.XTick object at 0x00000212AE5B3130>, <matplotlib.axis.XTick object at 0x00000212AE5B36D0>, <matplotlib.axis.XTick object at 0x00000212AE5B3BE0>, <matplotlib.axis.XTick object at 0x00000212AE5B5130>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE5DB8E0>
3
<matplotlib.contour.QuadContourSet object at 0x00000212AE6F6A30>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE567FA0>, <matplotlib.axis.YTick object at 0x00000212AE6040D0>, <matplotlib.axis.YTick object at 0x00000212AE5617F0>, <matplotlib.axis.YTick object at 0x00000212AE561D30>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE5B5EB0>, <matplotlib.axis.XTick object at 0x00000212AE63AD30>, <matplotlib.axis.XTick object at 0x00000212AE37E760>, <matplotlib.axis.XTick object at 0x00000212AE561BE0>, <matplotlib.axis.XTick object at 0x00000212AE37E700>, <matplotlib.axis.XTick object at 0x00000212AE37E9D0>, <matplotlib.axis.XTick object at 0x00000212AE6B1760>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE675F70>
4
<matplotlib.contour.QuadContourSet object at 0x00000212AE2E83A0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE308790>, <matplotlib.axis.YTick object at 0x00000212AE2F7BE0>, <matplotlib.axis.YTick object at 0x00000212AEB86910>, <matplotlib.axis.YTick object at 0x00000212AEB868E0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE3088E0>, <matplotlib.axis.XTick object at 0x00000212AE39FB80>, <matplotlib.axis.XTick object at 0x00000212AEB869A0>, <matplotlib.axis.XTick object at 0x00000212AE15B850>, <matplotlib.axis.XTick object at 0x00000212AE15BDC0>, <matplotlib.axis.XTick object at 0x00000212AE15BD60>, <matplotlib.axis.XTick object at 0x00000212AE4DA8B0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE31B9D0>
5
<matplotlib.contour.QuadContourSet object at 0x00000212AE1D65B0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE1700D0>, <matplotlib.axis.YTick object at 0x00000212AEB5D910>, <matplotlib.axis.YTick object at 0x00000212AE710D30>, <matplotlib.axis.YTick object at 0x00000212AE165280>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE39D2E0>, <matplotlib.axis.XTick object at 0x000002126CD474C0>, <matplotlib.axis.XTick object at 0x00000212AE165A90>, <matplotlib.axis.XTick object at 0x00000212AE1650A0>, <matplotlib.axis.XTick object at 0x00000212AE715310>, <matplotlib.axis.XTick object at 0x00000212AE715820>, <matplotlib.axis.XTick object at 0x00000212AE715D30>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE4FA550>
6
<matplotlib.contour.QuadContourSet object at 0x00000212AE666550>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AEB8CD90>, <matplotlib.axis.YTick object at 0x00000212AEB6F9D0>, <matplotlib.axis.YTick object at 0x00000212AE674CA0>, <matplotlib.axis.YTick object at 0x00000212AE6681F0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE4B1160>, <matplotlib.axis.XTick object at 0x00000212AEB803D0>, <matplotlib.axis.XTick object at 0x00000212AE668A90>, <matplotlib.axis.XTick object at 0x00000212AE6680D0>, <matplotlib.axis.XTick object at 0x00000212AE9302B0>, <matplotlib.axis.XTick object at 0x00000212AE9307C0>, <matplotlib.axis.XTick object at 0x00000212AE930CD0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE95A4C0>
7
<matplotlib.contour.QuadContourSet object at 0x00000212AE14EDC0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE1B8E80>, <matplotlib.axis.YTick object at 0x00000212AEB59160>, <matplotlib.axis.YTick object at 0x00000212AE15B490>, <matplotlib.axis.YTick object at 0x00000212AE1EE940>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE3C4610>, <matplotlib.axis.XTick object at 0x00000212AE4AB310>, <matplotlib.axis.XTick object at 0x00000212AE1EEA90>, <matplotlib.axis.XTick object at 0x00000212AE15B7F0>, <matplotlib.axis.XTick object at 0x000002126CD47850>, <matplotlib.axis.XTick object at 0x000002126CD47940>, <matplotlib.axis.XTick object at 0x00000212AEB5DC70>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE5198B0>
8
<matplotlib.contour.QuadContourSet object at 0x00000212AE31B250>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE6BA1C0>, <matplotlib.axis.YTick object at 0x00000212AE675EB0>, <matplotlib.axis.YTick object at 0x00000212AE5E8D00>, <matplotlib.axis.YTick object at 0x00000212AE5E8F70>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE3635E0>, <matplotlib.axis.XTick object at 0x00000212AE4F10A0>, <matplotlib.axis.XTick object at 0x00000212AE5E8940>, <matplotlib.axis.XTick object at 0x00000212AE1F2250>, <matplotlib.axis.XTick object at 0x00000212AE1F28E0>, <matplotlib.axis.XTick object at 0x00000212AE1F2D30>, <matplotlib.axis.XTick object at 0x00000212AE2C27C0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE570A00>
9
<matplotlib.contour.QuadContourSet object at 0x00000212AE573790>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE6F6CA0>, <matplotlib.axis.YTick object at 0x00000212AE5B5AC0>, <matplotlib.axis.YTick object at 0x00000212AE6DE7C0>, <matplotlib.axis.YTick object at 0x00000212AE6DECD0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE561FA0>, <matplotlib.axis.XTick object at 0x00000212AE3385B0>, <matplotlib.axis.XTick object at 0x00000212AE6DE700>, <matplotlib.axis.XTick object at 0x00000212AE6DA1C0>, <matplotlib.axis.XTick object at 0x00000212AE6DAD60>, <matplotlib.axis.XTick object at 0x00000212AE6AF2B0>, <matplotlib.axis.XTick object at 0x00000212AE6AF7C0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE545F70>
10
<matplotlib.contour.QuadContourSet object at 0x00000212AE5BE0A0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE747A90>, <matplotlib.axis.YTick object at 0x00000212AE5BEBB0>, <matplotlib.axis.YTick object at 0x00000212AE5B59A0>, <matplotlib.axis.YTick object at 0x00000212AE5B5220>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE6CA580>, <matplotlib.axis.XTick object at 0x00000212AE5BF940>, <matplotlib.axis.XTick object at 0x00000212AE338A30>, <matplotlib.axis.XTick object at 0x00000212AE5B5CD0>, <matplotlib.axis.XTick object at 0x00000212AE338CA0>, <matplotlib.axis.XTick object at 0x00000212AE5E8520>, <matplotlib.axis.XTick object at 0x00000212AE5E8160>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE4E81F0>
11
<matplotlib.contour.QuadContourSet object at 0x00000212AE5BBC40>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE31B640>, <matplotlib.axis.YTick object at 0x00000212AE201940>, <matplotlib.axis.YTick object at 0x00000212AE71FEB0>, <matplotlib.axis.YTick object at 0x00000212AE71F2E0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE6F2280>, <matplotlib.axis.XTick object at 0x00000212AE6A5F70>, <matplotlib.axis.XTick object at 0x00000212AE71FD60>, <matplotlib.axis.XTick object at 0x00000212AE4FA5B0>, <matplotlib.axis.XTick object at 0x00000212AE4FA280>, <matplotlib.axis.XTick object at 0x00000212AE4FA610>, <matplotlib.axis.XTick object at 0x00000212AE552250>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE39DE80>
12
<matplotlib.contour.QuadContourSet object at 0x00000212AEB8CFD0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE3C4F40>, <matplotlib.axis.YTick object at 0x00000212AE4C1C70>, <matplotlib.axis.YTick object at 0x00000212AE96A3D0>, <matplotlib.axis.YTick object at 0x00000212AE96A640>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AEB6FD00>, <matplotlib.axis.XTick object at 0x00000212AE668790>, <matplotlib.axis.XTick object at 0x00000212AE96ABB0>, <matplotlib.axis.XTick object at 0x00000212AE561D90>, <matplotlib.axis.XTick object at 0x00000212AE308E20>, <matplotlib.axis.XTick object at 0x00000212AE308520>, <matplotlib.axis.XTick object at 0x00000212AE3088B0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE4E2220>
13
<matplotlib.contour.QuadContourSet object at 0x00000212AE1FD880>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE72A880>, <matplotlib.axis.YTick object at 0x00000212AE74BAF0>, <matplotlib.axis.YTick object at 0x00000212AE73B070>, <matplotlib.axis.YTick object at 0x00000212AE73B520>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE72A610>, <matplotlib.axis.XTick object at 0x00000212AE71CDF0>, <matplotlib.axis.XTick object at 0x00000212AE73B3A0>, <matplotlib.axis.XTick object at 0x00000212AE4EA100>, <matplotlib.axis.XTick object at 0x00000212AE4EA5B0>, <matplotlib.axis.XTick object at 0x00000212AE4EAAC0>, <matplotlib.axis.XTick object at 0x00000212AE4E40A0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE50CC10>
14
<matplotlib.contour.QuadContourSet object at 0x00000212AE552130>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AEBA0EB0>, <matplotlib.axis.YTick object at 0x000002126CD47D60>, <matplotlib.axis.YTick object at 0x00000212AE6A5460>, <matplotlib.axis.YTick object at 0x00000212AE2BD070>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE2F14C0>, <matplotlib.axis.XTick object at 0x00000212AE72A340>, <matplotlib.axis.XTick object at 0x00000212AE6A53A0>, <matplotlib.axis.XTick object at 0x00000212AE2BD4F0>, <matplotlib.axis.XTick object at 0x00000212AE285C40>, <matplotlib.axis.XTick object at 0x00000212AE285B80>, <matplotlib.axis.XTick object at 0x00000212AE285D00>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE6EF4C0>
15
<matplotlib.contour.QuadContourSet object at 0x00000212AE56E1C0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE5BF3D0>, <matplotlib.axis.YTick object at 0x00000212AE675790>, <matplotlib.axis.YTick object at 0x00000212AE5D4700>, <matplotlib.axis.YTick object at 0x00000212AE165940>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE36CEE0>, <matplotlib.axis.XTick object at 0x00000212AE2C22B0>, <matplotlib.axis.XTick object at 0x00000212AE5D4B80>, <matplotlib.axis.XTick object at 0x00000212AE165850>, <matplotlib.axis.XTick object at 0x00000212AE5FC0D0>, <matplotlib.axis.XTick object at 0x00000212AE5FCCA0>, <matplotlib.axis.XTick object at 0x00000212AE5FC340>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE642E20>
16
<matplotlib.contour.QuadContourSet object at 0x00000212AE72DF10>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE65E0D0>, <matplotlib.axis.YTick object at 0x00000212AE593940>, <matplotlib.axis.YTick object at 0x00000212AE674100>, <matplotlib.axis.YTick object at 0x00000212AE674790>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE65E790>, <matplotlib.axis.XTick object at 0x00000212AE4F1AC0>, <matplotlib.axis.XTick object at 0x00000212AE674220>, <matplotlib.axis.XTick object at 0x00000212AE618280>, <matplotlib.axis.XTick object at 0x00000212AE618E20>, <matplotlib.axis.XTick object at 0x00000212AE5E3400>, <matplotlib.axis.XTick object at 0x00000212AE5E3910>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE554100>
17
<matplotlib.contour.QuadContourSet object at 0x00000212AE943EE0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE6D0B20>, <matplotlib.axis.YTick object at 0x00000212AE943A60>, <matplotlib.axis.YTick object at 0x00000212AE1F2BE0>, <matplotlib.axis.YTick object at 0x00000212AE65E250>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE61EC40>, <matplotlib.axis.XTick object at 0x00000212AE573310>, <matplotlib.axis.XTick object at 0x00000212AE56EF40>, <matplotlib.axis.XTick object at 0x00000212AE65E820>, <matplotlib.axis.XTick object at 0x00000212AE56E670>, <matplotlib.axis.XTick object at 0x00000212AE56E970>, <matplotlib.axis.XTick object at 0x00000212AE3635B0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE140100>
18
<matplotlib.contour.QuadContourSet object at 0x00000212AE5BF2E0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE3C4880>, <matplotlib.axis.YTick object at 0x00000212AEB80BB0>, <matplotlib.axis.YTick object at 0x00000212AE4F1A60>, <matplotlib.axis.YTick object at 0x00000212AE39D2E0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE5935B0>, <matplotlib.axis.XTick object at 0x00000212AE552790>, <matplotlib.axis.XTick object at 0x00000212AE39DE50>, <matplotlib.axis.XTick object at 0x00000212AE6685E0>, <matplotlib.axis.XTick object at 0x00000212AE668A00>, <matplotlib.axis.XTick object at 0x00000212AE668FA0>, <matplotlib.axis.XTick object at 0x00000212AE668E80>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AEB5DFA0>
19
<matplotlib.contour.QuadContourSet object at 0x00000212AE4A0880>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE73B520>, <matplotlib.axis.YTick object at 0x00000212AE38C220>, <matplotlib.axis.YTick object at 0x00000212AE3062B0>, <matplotlib.axis.YTick object at 0x00000212AE306490>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE2F1550>, <matplotlib.axis.XTick object at 0x00000212AE1D2160>, <matplotlib.axis.XTick object at 0x00000212AE3061C0>, <matplotlib.axis.XTick object at 0x00000212AE4E4EE0>, <matplotlib.axis.XTick object at 0x00000212AE4E47F0>, <matplotlib.axis.XTick object at 0x00000212AE4E4580>, <matplotlib.axis.XTick object at 0x00000212AEB863A0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002126CD24BE0>
20
<matplotlib.contour.QuadContourSet object at 0x00000212AE620A60>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE963D60>, <matplotlib.axis.YTick object at 0x00000212AE15BCA0>, <matplotlib.axis.YTick object at 0x00000212AE9661F0>, <matplotlib.axis.YTick object at 0x00000212AE966700>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE963C70>, <matplotlib.axis.XTick object at 0x00000212AE93F640>, <matplotlib.axis.XTick object at 0x00000212AE9669D0>, <matplotlib.axis.XTick object at 0x00000212AE1FC250>, <matplotlib.axis.XTick object at 0x00000212AE1FC790>, <matplotlib.axis.XTick object at 0x00000212AE1FCCA0>, <matplotlib.axis.XTick object at 0x00000212AE1E11F0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE4C59A0>
21
<matplotlib.contour.QuadContourSet object at 0x00000212AE113520>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE33D9D0>, <matplotlib.axis.YTick object at 0x00000212AE51CA90>, <matplotlib.axis.YTick object at 0x00000212AE2F1D90>, <matplotlib.axis.YTick object at 0x00000212AEB8C0D0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE38CAC0>, <matplotlib.axis.XTick object at 0x00000212AE6A5FD0>, <matplotlib.axis.XTick object at 0x00000212AE2F1790>, <matplotlib.axis.XTick object at 0x00000212AEB8CD60>, <matplotlib.axis.XTick object at 0x00000212AE1D26A0>, <matplotlib.axis.XTick object at 0x00000212AE1D2910>, <matplotlib.axis.XTick object at 0x00000212AE6B17F0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE53AB80>
22
<matplotlib.contour.QuadContourSet object at 0x00000212AE2C2A60>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE6D0E50>, <matplotlib.axis.YTick object at 0x00000212AE31B490>, <matplotlib.axis.YTick object at 0x00000212AE5ED820>, <matplotlib.axis.YTick object at 0x00000212AE5ED2E0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE2BD7C0>, <matplotlib.axis.XTick object at 0x00000212AE6AC8E0>, <matplotlib.axis.XTick object at 0x00000212AE5ED310>, <matplotlib.axis.XTick object at 0x00000212AE618C70>, <matplotlib.axis.XTick object at 0x00000212AE6187F0>, <matplotlib.axis.XTick object at 0x00000212AE5E31F0>, <matplotlib.axis.XTick object at 0x00000212AE5E3E50>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE963FD0>
23
<matplotlib.contour.QuadContourSet object at 0x00000212AE5735B0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE2F72E0>, <matplotlib.axis.YTick object at 0x00000212AE675A00>, <matplotlib.axis.YTick object at 0x00000212AE66F8E0>, <matplotlib.axis.YTick object at 0x00000212AE66FDF0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE65EBE0>, <matplotlib.axis.XTick object at 0x00000212AE61E370>, <matplotlib.axis.XTick object at 0x00000212AE66F7C0>, <matplotlib.axis.XTick object at 0x00000212AEBA1250>, <matplotlib.axis.XTick object at 0x00000212AEBA1E80>, <matplotlib.axis.XTick object at 0x00000212AE69E3D0>, <matplotlib.axis.XTick object at 0x00000212AE69E8E0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE5C50D0>
24
<matplotlib.contour.QuadContourSet object at 0x00000212AE4F93D0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE56C340>, <matplotlib.axis.YTick object at 0x00000212AE554430>, <matplotlib.axis.YTick object at 0x00000212AE5540A0>, <matplotlib.axis.YTick object at 0x00000212AE554940>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE67FF70>, <matplotlib.axis.XTick object at 0x00000212AE5260D0>, <matplotlib.axis.XTick object at 0x00000212AE6BADF0>, <matplotlib.axis.XTick object at 0x00000212AE554B80>, <matplotlib.axis.XTick object at 0x00000212AE6BA520>, <matplotlib.axis.XTick object at 0x00000212AE2E8CD0>, <matplotlib.axis.XTick object at 0x00000212AE2E80A0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE1B8CD0>
25
<matplotlib.contour.QuadContourSet object at 0x00000212AE642280>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE675B50>, <matplotlib.axis.YTick object at 0x00000212AE53A190>, <matplotlib.axis.YTick object at 0x00000212AE2D5FD0>, <matplotlib.axis.YTick object at 0x00000212AE5BF6A0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE2A9400>, <matplotlib.axis.XTick object at 0x00000212AE53A070>, <matplotlib.axis.XTick object at 0x00000212AE2D5040>, <matplotlib.axis.XTick object at 0x00000212AE5BF760>, <matplotlib.axis.XTick object at 0x00000212AE4A0790>, <matplotlib.axis.XTick object at 0x00000212AE4A0E20>, <matplotlib.axis.XTick object at 0x00000212AE4A05E0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE6B1910>
26
<matplotlib.contour.QuadContourSet object at 0x00000212AE1E1700>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE3CDCD0>, <matplotlib.axis.YTick object at 0x00000212AE519D00>, <matplotlib.axis.YTick object at 0x00000212AE4C3400>, <matplotlib.axis.YTick object at 0x00000212AE4C35B0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE362670>, <matplotlib.axis.XTick object at 0x00000212AE50C550>, <matplotlib.axis.XTick object at 0x00000212AE4C3AF0>, <matplotlib.axis.XTick object at 0x00000212AE306A00>, <matplotlib.axis.XTick object at 0x00000212AE306550>, <matplotlib.axis.XTick object at 0x00000212AE306F40>, <matplotlib.axis.XTick object at 0x00000212AE1134C0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE4AD580>
27
<matplotlib.contour.QuadContourSet object at 0x00000212AE737B20>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AEB73AF0>, <matplotlib.axis.YTick object at 0x00000212AE6C5790>, <matplotlib.axis.YTick object at 0x00000212AEBA42B0>, <matplotlib.axis.YTick object at 0x00000212AEBA47C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AEB73BE0>, <matplotlib.axis.XTick object at 0x00000212AE749340>, <matplotlib.axis.XTick object at 0x00000212AEBA4700>, <matplotlib.axis.XTick object at 0x00000212AE95A310>, <matplotlib.axis.XTick object at 0x00000212AE95A850>, <matplotlib.axis.XTick object at 0x00000212AE95AD60>, <matplotlib.axis.XTick object at 0x00000212AE9472B0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE6F0A60>
28
<matplotlib.contour.QuadContourSet object at 0x00000212AE71F190>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE4A0100>, <matplotlib.axis.YTick object at 0x00000212AE749F10>, <matplotlib.axis.YTick object at 0x00000212AE71C610>, <matplotlib.axis.YTick object at 0x00000212AE71CDF0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE2F13D0>, <matplotlib.axis.XTick object at 0x00000212AE51CCD0>, <matplotlib.axis.XTick object at 0x00000212AE71C430>, <matplotlib.axis.XTick object at 0x00000212AE5575B0>, <matplotlib.axis.XTick object at 0x00000212AE557C10>, <matplotlib.axis.XTick object at 0x00000212AE557D00>, <matplotlib.axis.XTick object at 0x00000212AE149D30>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE4E85E0>
29
<matplotlib.contour.QuadContourSet object at 0x00000212AE6B17C0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE74D460>, <matplotlib.axis.YTick object at 0x00000212AE170CD0>, <matplotlib.axis.YTick object at 0x00000212AE756B50>, <matplotlib.axis.YTick object at 0x00000212AEB731C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE4B4E50>, <matplotlib.axis.XTick object at 0x00000212AE6AFB50>, <matplotlib.axis.XTick object at 0x00000212AE756DC0>, <matplotlib.axis.XTick object at 0x00000212AEB90DC0>, <matplotlib.axis.XTick object at 0x00000212AEB905E0>, <matplotlib.axis.XTick object at 0x00000212AEB90280>, <matplotlib.axis.XTick object at 0x00000212AE362B50>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE4E41C0>
30
<matplotlib.contour.QuadContourSet object at 0x00000212AE65E910>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AEB86370>, <matplotlib.axis.YTick object at 0x00000212AE67FA00>, <matplotlib.axis.YTick object at 0x00000212AE398340>, <matplotlib.axis.YTick object at 0x00000212AE3BF6D0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AEB86640>, <matplotlib.axis.XTick object at 0x00000212AEBC39D0>, <matplotlib.axis.XTick object at 0x00000212AE3985E0>, <matplotlib.axis.XTick object at 0x00000212AE3BF790>, <matplotlib.axis.XTick object at 0x00000212AE3B72B0>, <matplotlib.axis.XTick object at 0x00000212AE3B77C0>, <matplotlib.axis.XTick object at 0x00000212AE3B7CD0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE5664C0>
31
<matplotlib.contour.QuadContourSet object at 0x00000212AE357340>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE5F84F0>, <matplotlib.axis.YTick object at 0x00000212AE37CAF0>, <matplotlib.axis.YTick object at 0x00000212AE31B8B0>, <matplotlib.axis.YTick object at 0x00000212AE31B310>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE5F81F0>, <matplotlib.axis.XTick object at 0x00000212AE357160>, <matplotlib.axis.XTick object at 0x00000212AE2F7970>, <matplotlib.axis.XTick object at 0x00000212AE31BB80>, <matplotlib.axis.XTick object at 0x00000212AE2F73D0>, <matplotlib.axis.XTick object at 0x00000212AE5C5340>, <matplotlib.axis.XTick object at 0x00000212AE5C5B80>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE67F580>
32
<matplotlib.contour.QuadContourSet object at 0x00000212AE4E46A0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002126CD47640>, <matplotlib.axis.YTick object at 0x00000212AE941F10>, <matplotlib.axis.YTick object at 0x00000212AEB90880>, <matplotlib.axis.YTick object at 0x00000212AEB90C40>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002126CD47D90>, <matplotlib.axis.XTick object at 0x00000212AE4A0190>, <matplotlib.axis.XTick object at 0x00000212AEB90430>, <matplotlib.axis.XTick object at 0x00000212AE96AB80>, <matplotlib.axis.XTick object at 0x00000212AE96A940>, <matplotlib.axis.XTick object at 0x00000212AE1FBC40>, <matplotlib.axis.XTick object at 0x00000212AE1FB7F0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE113760>
33
<matplotlib.contour.QuadContourSet object at 0x00000212AE6C9160>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AEBA4790>, <matplotlib.axis.YTick object at 0x00000212AE739700>, <matplotlib.axis.YTick object at 0x00000212AE6EAE20>, <matplotlib.axis.YTick object at 0x00000212AE4C11C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE96ADC0>, <matplotlib.axis.XTick object at 0x00000212AE4DA6D0>, <matplotlib.axis.XTick object at 0x00000212AE4C1340>, <matplotlib.axis.XTick object at 0x00000212AE4C1EE0>, <matplotlib.axis.XTick object at 0x00000212AE6A5850>, <matplotlib.axis.XTick object at 0x00000212AE6A56D0>, <matplotlib.axis.XTick object at 0x00000212AE6A5F10>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE3B10A0>
34
<matplotlib.contour.QuadContourSet object at 0x00000212AE745B80>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE6C5C40>, <matplotlib.axis.YTick object at 0x00000212AE675070>, <matplotlib.axis.YTick object at 0x00000212AE659310>, <matplotlib.axis.YTick object at 0x00000212AE659820>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE6203A0>, <matplotlib.axis.XTick object at 0x00000212AE253040>, <matplotlib.axis.XTick object at 0x00000212AE659640>, <matplotlib.axis.XTick object at 0x00000212AE4EA370>, <matplotlib.axis.XTick object at 0x00000212AE4EA8B0>, <matplotlib.axis.XTick object at 0x00000212AE4EADC0>, <matplotlib.axis.XTick object at 0x00000212AE4F4310>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AEB65AC0>
35
<matplotlib.contour.QuadContourSet object at 0x00000212AEBC4250>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE4E4100>, <matplotlib.axis.YTick object at 0x00000212AE4C8850>, <matplotlib.axis.YTick object at 0x00000212AE4A0460>, <matplotlib.axis.YTick object at 0x00000212AE4A0D60>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE2857F0>, <matplotlib.axis.XTick object at 0x00000212AE6C5520>, <matplotlib.axis.XTick object at 0x00000212AE4A03A0>, <matplotlib.axis.XTick object at 0x00000212AE4B4850>, <matplotlib.axis.XTick object at 0x00000212AE4B4250>, <matplotlib.axis.XTick object at 0x00000212AE5AAEB0>, <matplotlib.axis.XTick object at 0x00000212AE5AA370>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002125D1D0250>
36
<matplotlib.contour.QuadContourSet object at 0x00000212AE6AC970>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE2F7E50>, <matplotlib.axis.YTick object at 0x00000212AE6F29A0>, <matplotlib.axis.YTick object at 0x00000212AE59F2B0>, <matplotlib.axis.YTick object at 0x00000212AE2D5D90>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE285550>, <matplotlib.axis.XTick object at 0x00000212AE62ED90>, <matplotlib.axis.XTick object at 0x00000212AE2D5EB0>, <matplotlib.axis.XTick object at 0x00000212AE2D53A0>, <matplotlib.axis.XTick object at 0x00000212AE36C760>, <matplotlib.axis.XTick object at 0x00000212AE36C6D0>, <matplotlib.axis.XTick object at 0x00000212AEBA4B80>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE59AA00>
37
<matplotlib.contour.QuadContourSet object at 0x00000212AE363250>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE2BF040>, <matplotlib.axis.YTick object at 0x00000212AE5F8430>, <matplotlib.axis.YTick object at 0x00000212AE658C70>, <matplotlib.axis.YTick object at 0x00000212AE6471C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE1FBA00>, <matplotlib.axis.XTick object at 0x00000212AE5FC220>, <matplotlib.axis.XTick object at 0x00000212AE647AC0>, <matplotlib.axis.XTick object at 0x00000212AE647100>, <matplotlib.axis.XTick object at 0x00000212AE64B250>, <matplotlib.axis.XTick object at 0x00000212AE64B760>, <matplotlib.axis.XTick object at 0x00000212AE64BC70>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AEBA2460>
38
<matplotlib.contour.QuadContourSet object at 0x00000212AE71A1F0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE6F1D90>, <matplotlib.axis.YTick object at 0x00000212AE397760>, <matplotlib.axis.YTick object at 0x00000212AE72DCA0>, <matplotlib.axis.YTick object at 0x00000212AE29D160>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE62DB20>, <matplotlib.axis.XTick object at 0x00000212AE201940>, <matplotlib.axis.XTick object at 0x00000212AE29D640>, <matplotlib.axis.XTick object at 0x00000212AE29DA90>, <matplotlib.axis.XTick object at 0x00000212AE2D5220>, <matplotlib.axis.XTick object at 0x00000212AE2D5E20>, <matplotlib.axis.XTick object at 0x00000212AE2D55E0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE1FB760>
39
<matplotlib.contour.QuadContourSet object at 0x00000212AEBC3370>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE6487C0>, <matplotlib.axis.YTick object at 0x00000212AE362F10>, <matplotlib.axis.YTick object at 0x00000212AE6C9F10>, <matplotlib.axis.YTick object at 0x00000212AE6ACDF0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE739730>, <matplotlib.axis.XTick object at 0x00000212AE362940>, <matplotlib.axis.XTick object at 0x00000212AE6C9F70>, <matplotlib.axis.XTick object at 0x00000212AE6ACD00>, <matplotlib.axis.XTick object at 0x00000212AE36C2E0>, <matplotlib.axis.XTick object at 0x00000212AE36CCD0>, <matplotlib.axis.XTick object at 0x00000212AE36C520>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE71FAF0>
40
<matplotlib.contour.QuadContourSet object at 0x00000212AE4DB640>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE4DA940>, <matplotlib.axis.YTick object at 0x00000212AE5AA610>, <matplotlib.axis.YTick object at 0x00000212AE4C17F0>, <matplotlib.axis.YTick object at 0x00000212AE4C1AC0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE4DA1C0>, <matplotlib.axis.XTick object at 0x00000212AE6AF130>, <matplotlib.axis.XTick object at 0x00000212AE4C1220>, <matplotlib.axis.XTick object at 0x00000212AEB8C430>, <matplotlib.axis.XTick object at 0x00000212AEB8C640>, <matplotlib.axis.XTick object at 0x00000212AE2BDC70>, <matplotlib.axis.XTick object at 0x00000212AE2BD520>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE1EE610>
41
<matplotlib.contour.QuadContourSet object at 0x00000212AE6A1B50>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE170CA0>, <matplotlib.axis.YTick object at 0x00000212AE6F23D0>, <matplotlib.axis.YTick object at 0x00000212AE6D22E0>, <matplotlib.axis.YTick object at 0x00000212AE6D27F0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE50B460>, <matplotlib.axis.XTick object at 0x00000212AE62C700>, <matplotlib.axis.XTick object at 0x00000212AE6D2790>, <matplotlib.axis.XTick object at 0x00000212AE52C340>, <matplotlib.axis.XTick object at 0x00000212AE52C880>, <matplotlib.axis.XTick object at 0x00000212AE52CD90>, <matplotlib.axis.XTick object at 0x00000212AE53B2E0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE50AA90>
42
<matplotlib.contour.QuadContourSet object at 0x00000212AE59AD00>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE4AAB20>, <matplotlib.axis.YTick object at 0x00000212AE38CEE0>, <matplotlib.axis.YTick object at 0x00000212AE71FC70>, <matplotlib.axis.YTick object at 0x00000212AE71FD00>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE4C7AF0>, <matplotlib.axis.XTick object at 0x00000212AE5E3550>, <matplotlib.axis.XTick object at 0x00000212AE71F700>, <matplotlib.axis.XTick object at 0x00000212AE6756A0>, <matplotlib.axis.XTick object at 0x00000212AE6753D0>, <matplotlib.axis.XTick object at 0x00000212AE6752E0>, <matplotlib.axis.XTick object at 0x00000212AE648970>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE259CA0>
43
<matplotlib.contour.QuadContourSet object at 0x00000212AE4C8820>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE1EE9A0>, <matplotlib.axis.YTick object at 0x00000212AE31BAC0>, <matplotlib.axis.YTick object at 0x00000212AE1F2220>, <matplotlib.axis.YTick object at 0x00000212AE1F2EE0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE947F70>, <matplotlib.axis.XTick object at 0x00000212AE50BAC0>, <matplotlib.axis.XTick object at 0x00000212AE1F2B20>, <matplotlib.axis.XTick object at 0x00000212AE2F7D90>, <matplotlib.axis.XTick object at 0x00000212AE2F7E50>, <matplotlib.axis.XTick object at 0x00000212AEB9C490>, <matplotlib.axis.XTick object at 0x00000212AEB9CAF0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE2D5880>
44
<matplotlib.contour.QuadContourSet object at 0x00000212AE960400>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE59FD90>, <matplotlib.axis.YTick object at 0x00000212AE5C5D90>, <matplotlib.axis.YTick object at 0x00000212AE636A90>, <matplotlib.axis.YTick object at 0x00000212AEBBF040>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE62DC70>, <matplotlib.axis.XTick object at 0x00000212AE6B13D0>, <matplotlib.axis.XTick object at 0x00000212AE636970>, <matplotlib.axis.XTick object at 0x00000212AEBBF400>, <matplotlib.axis.XTick object at 0x00000212AE65F0A0>, <matplotlib.axis.XTick object at 0x00000212AE65F580>, <matplotlib.axis.XTick object at 0x00000212AE65FA90>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE1D4280>
45
<matplotlib.contour.QuadContourSet object at 0x00000212AEB775B0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE5F3430>, <matplotlib.axis.YTick object at 0x00000212AEBA28E0>, <matplotlib.axis.YTick object at 0x00000212AEBBD9A0>, <matplotlib.axis.YTick object at 0x00000212AEBBD7C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE5F3F70>, <matplotlib.axis.XTick object at 0x00000212AEB77640>, <matplotlib.axis.XTick object at 0x00000212AE201A00>, <matplotlib.axis.XTick object at 0x00000212AE201B20>, <matplotlib.axis.XTick object at 0x00000212AEBBDB20>, <matplotlib.axis.XTick object at 0x00000212AE4E4A30>, <matplotlib.axis.XTick object at 0x00000212AE4E4130>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE62D3D0>
46
<matplotlib.contour.QuadContourSet object at 0x00000212AE53A580>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE64B5B0>, <matplotlib.axis.YTick object at 0x00000212AE2014F0>, <matplotlib.axis.YTick object at 0x00000212AE4DB190>, <matplotlib.axis.YTick object at 0x00000212AE4DB880>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE5174C0>, <matplotlib.axis.XTick object at 0x00000212AE2B9310>, <matplotlib.axis.XTick object at 0x00000212AE4DBDC0>, <matplotlib.axis.XTick object at 0x00000212AEB8C280>, <matplotlib.axis.XTick object at 0x00000212AEB8C490>, <matplotlib.axis.XTick object at 0x00000212AEB8C730>, <matplotlib.axis.XTick object at 0x00000212AEB9CF10>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE5738B0>
47
<matplotlib.contour.QuadContourSet object at 0x00000212AE50B340>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE4C79A0>, <matplotlib.axis.YTick object at 0x00000212AE170400>, <matplotlib.axis.YTick object at 0x00000212AE53BF40>, <matplotlib.axis.YTick object at 0x00000212AE53B820>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE6AF190>, <matplotlib.axis.XTick object at 0x00000212AE4C1DF0>, <matplotlib.axis.XTick object at 0x00000212AE53B910>, <matplotlib.axis.XTick object at 0x00000212AE96A5E0>, <matplotlib.axis.XTick object at 0x00000212AE96AE50>, <matplotlib.axis.XTick object at 0x00000212AE96AA30>, <matplotlib.axis.XTick object at 0x00000212AE2BD7C0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE31B6A0>
48
<matplotlib.contour.QuadContourSet object at 0x00000212AE930BE0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE5527C0>, <matplotlib.axis.YTick object at 0x00000212AE307D00>, <matplotlib.axis.YTick object at 0x00000212AE1C3370>, <matplotlib.axis.YTick object at 0x00000212AE1C3880>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE36E8B0>, <matplotlib.axis.XTick object at 0x00000212AE6B1D90>, <matplotlib.axis.XTick object at 0x00000212AE1C3760>, <matplotlib.axis.XTick object at 0x00000212AE5153D0>, <matplotlib.axis.XTick object at 0x00000212AE515910>, <matplotlib.axis.XTick object at 0x00000212AE515E20>, <matplotlib.axis.XTick object at 0x00000212AE50AF40>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE4C0B20>
49
<matplotlib.contour.QuadContourSet object at 0x00000212AE56E580>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE6C5850>, <matplotlib.axis.YTick object at 0x00000212AE2E5760>, <matplotlib.axis.YTick object at 0x00000212AE2F1AC0>, <matplotlib.axis.YTick object at 0x00000212AE2F1C10>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE4C3FA0>, <matplotlib.axis.XTick object at 0x00000212AE36E130>, <matplotlib.axis.XTick object at 0x00000212AE2F1370>, <matplotlib.axis.XTick object at 0x00000212AE64B2B0>, <matplotlib.axis.XTick object at 0x00000212AE64B3D0>, <matplotlib.axis.XTick object at 0x00000212AE5179D0>, <matplotlib.axis.XTick object at 0x00000212AE517280>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE4BA670>
50
<matplotlib.contour.QuadContourSet object at 0x00000212AEBBDAF0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE4AA310>, <matplotlib.axis.YTick object at 0x00000212AE4C7700>, <matplotlib.axis.YTick object at 0x00000212AE57DE50>, <matplotlib.axis.YTick object at 0x00000212AE57D0D0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE4AA7F0>, <matplotlib.axis.XTick object at 0x00000212AE38C820>, <matplotlib.axis.XTick object at 0x00000212AE57D3D0>, <matplotlib.axis.XTick object at 0x00000212AE1CB9A0>, <matplotlib.axis.XTick object at 0x00000212AE2A9D00>, <matplotlib.axis.XTick object at 0x00000212AE2A9940>, <matplotlib.axis.XTick object at 0x00000212AE2A92E0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE2C9E50>
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
<matplotlib.contour.QuadContourSet object at 0x00000212AE5B48B0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE9604F0>, <matplotlib.axis.YTick object at 0x00000212AE630D60>, <matplotlib.axis.YTick object at 0x00000212AE6A5700>, <matplotlib.axis.YTick object at 0x00000212AE4C63D0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE5FC730>, <matplotlib.axis.XTick object at 0x00000212AE72D970>, <matplotlib.axis.XTick object at 0x00000212AE6A5580>, <matplotlib.axis.XTick object at 0x00000212AE4C6C70>, <matplotlib.axis.XTick object at 0x00000212AE5238E0>, <matplotlib.axis.XTick object at 0x00000212AE5237C0>, <matplotlib.axis.XTick object at 0x00000212AE5233D0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE546670>
1
<matplotlib.contour.QuadContourSet object at 0x00000212AEBBA340>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE370460>, <matplotlib.axis.YTick object at 0x00000212AE37E070>, <matplotlib.axis.YTick object at 0x00000212AE65EB20>, <matplotlib.axis.YTick object at 0x00000212AE65E370>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE562DC0>, <matplotlib.axis.XTick object at 0x00000212AE1F4BB0>, <matplotlib.axis.XTick object at 0x00000212AE53A970>, <matplotlib.axis.XTick object at 0x00000212AE65E4F0>, <matplotlib.axis.XTick object at 0x00000212AE53A9D0>, <matplotlib.axis.XTick object at 0x00000212AE53A580>, <matplotlib.axis.XTick object at 0x00000212AE1FFC10>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE6C57F0>
2
<matplotlib.contour.QuadContourSet object at 0x00000212AE2BD7F0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE668C10>, <matplotlib.axis.YTick object at 0x00000212AE947C40>, <matplotlib.axis.YTick object at 0x00000212AE4A0A30>, <matplotlib.axis.YTick object at 0x00000212AE4C6490>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE5A8070>, <matplotlib.axis.XTick object at 0x00000212AE4C7160>, <matplotlib.axis.XTick object at 0x00000212AE523B20>, <matplotlib.axis.XTick object at 0x00000212AE675190>, <matplotlib.axis.XTick object at 0x00000212AE675AC0>, <matplotlib.axis.XTick object at 0x00000212AE31B970>, <matplotlib.axis.XTick object at 0x00000212AE31BA30>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE6E03D0>
3
<matplotlib.contour.QuadContourSet object at 0x00000212AE307040>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE6F29A0>, <matplotlib.axis.YTick object at 0x00000212AE6791C0>, <matplotlib.axis.YTick object at 0x00000212AE113BB0>, <matplotlib.axis.YTick object at 0x00000212AE62D340>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE6F2A00>, <matplotlib.axis.XTick object at 0x00000212AE94B400>, <matplotlib.axis.XTick object at 0x00000212AE62DB80>, <matplotlib.axis.XTick object at 0x000002126CD24790>, <matplotlib.axis.XTick object at 0x000002126CD24CD0>, <matplotlib.axis.XTick object at 0x00000212AE71F0D0>, <matplotlib.axis.XTick object at 0x00000212AE71F2E0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE37A550>
4
<matplotlib.contour.QuadContourSet object at 0x00000212AE95E070>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE50FE50>, <matplotlib.axis.YTick object at 0x00000212AE2F10A0>, <matplotlib.axis.YTick object at 0x00000212AE36EBE0>, <matplotlib.axis.YTick object at 0x00000212AE96AA00>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE50A0A0>, <matplotlib.axis.XTick object at 0x00000212AE65CEE0>, <matplotlib.axis.XTick object at 0x00000212AE96A400>, <matplotlib.axis.XTick object at 0x00000212AE96A820>, <matplotlib.axis.XTick object at 0x00000212AE2BDCA0>, <matplotlib.axis.XTick object at 0x00000212AE2BD070>, <matplotlib.axis.XTick object at 0x00000212AE2BD4C0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE170220>
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
    for t in range(6):
        ff_1=dot(AQ, ff_1)

        
0
<matplotlib.contour.QuadContourSet object at 0x00000212AE4AD520>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE285220>, <matplotlib.axis.YTick object at 0x00000212AE64EE20>, <matplotlib.axis.YTick object at 0x00000212AE668CD0>, <matplotlib.axis.YTick object at 0x00000212AE5D4C40>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE201370>, <matplotlib.axis.XTick object at 0x00000212AE4B4A90>, <matplotlib.axis.XTick object at 0x00000212AE668DF0>, <matplotlib.axis.XTick object at 0x00000212AE5D44C0>, <matplotlib.axis.XTick object at 0x00000212AE581910>, <matplotlib.axis.XTick object at 0x00000212AE581610>, <matplotlib.axis.XTick object at 0x00000212AE5815B0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE960BE0>
1
<matplotlib.contour.QuadContourSet object at 0x00000212AE6AFCD0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE562A30>, <matplotlib.axis.YTick object at 0x00000212AE71A5B0>, <matplotlib.axis.YTick object at 0x00000212AE630850>, <matplotlib.axis.YTick object at 0x00000212AE6302B0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE56EE50>, <matplotlib.axis.XTick object at 0x00000212AE6A54C0>, <matplotlib.axis.XTick object at 0x00000212AE6306A0>, <matplotlib.axis.XTick object at 0x00000212AE187E50>, <matplotlib.axis.XTick object at 0x00000212AE187910>, <matplotlib.axis.XTick object at 0x00000212AE633460>, <matplotlib.axis.XTick object at 0x00000212AE633C70>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AEB95640>
2
<matplotlib.contour.QuadContourSet object at 0x00000212AE67C6A0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AEB61FD0>, <matplotlib.axis.YTick object at 0x00000212AE62C910>, <matplotlib.axis.YTick object at 0x00000212AE708DF0>, <matplotlib.axis.YTick object at 0x00000212AEB67340>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AEB61700>, <matplotlib.axis.XTick object at 0x00000212AE62C4C0>, <matplotlib.axis.XTick object at 0x00000212AEB67EE0>, <matplotlib.axis.XTick object at 0x00000212AEB67250>, <matplotlib.axis.XTick object at 0x00000212AE7013D0>, <matplotlib.axis.XTick object at 0x00000212AE7018E0>, <matplotlib.axis.XTick object at 0x00000212AE701DF0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE5545E0>
3
<matplotlib.contour.QuadContourSet object at 0x00000212AE155820>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE16DD00>, <matplotlib.axis.YTick object at 0x00000212AE4B35E0>, <matplotlib.axis.YTick object at 0x00000212AE285D60>, <matplotlib.axis.YTick object at 0x00000212AE1FF910>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE172460>, <matplotlib.axis.XTick object at 0x00000212AE2598E0>, <matplotlib.axis.XTick object at 0x00000212AE285C70>, <matplotlib.axis.XTick object at 0x00000212AE1FFAF0>, <matplotlib.axis.XTick object at 0x00000212AE56EF70>, <matplotlib.axis.XTick object at 0x00000212AE56ED00>, <matplotlib.axis.XTick object at 0x00000212AE56E820>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE201610>
4
<matplotlib.contour.QuadContourSet object at 0x00000212AE36E130>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE63A8E0>, <matplotlib.axis.YTick object at 0x00000212AEB8C6D0>, <matplotlib.axis.YTick object at 0x00000212AE4B7790>, <matplotlib.axis.YTick object at 0x00000212AE4B77F0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE63A490>, <matplotlib.axis.XTick object at 0x00000212AE6EF5B0>, <matplotlib.axis.XTick object at 0x00000212AE4B7B20>, <matplotlib.axis.XTick object at 0x00000212AE4E45E0>, <matplotlib.axis.XTick object at 0x00000212AE4E4370>, <matplotlib.axis.XTick object at 0x00000212AEB86FD0>, <matplotlib.axis.XTick object at 0x00000212AEB869D0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE573DF0>
5
<matplotlib.contour.QuadContourSet object at 0x00000212AE4ADEE0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE94B580>, <matplotlib.axis.YTick object at 0x00000212AEBAAEB0>, <matplotlib.axis.YTick object at 0x00000212AE552E80>, <matplotlib.axis.YTick object at 0x00000212AE552850>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE62D580>, <matplotlib.axis.XTick object at 0x00000212AE5FC8B0>, <matplotlib.axis.XTick object at 0x00000212AE552A90>, <matplotlib.axis.XTick object at 0x00000212AE1FBC70>, <matplotlib.axis.XTick object at 0x00000212AE1FBE20>, <matplotlib.axis.XTick object at 0x00000212AE38C550>, <matplotlib.axis.XTick object at 0x00000212AE38C3D0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE6BA820>
6
<matplotlib.contour.QuadContourSet object at 0x00000212AE5FC5B0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE96A910>, <matplotlib.axis.YTick object at 0x00000212AE2B5F10>, <matplotlib.axis.YTick object at 0x00000212AE363280>, <matplotlib.axis.YTick object at 0x00000212AE363580>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE2D38B0>, <matplotlib.axis.XTick object at 0x00000212AE2D3040>, <matplotlib.axis.XTick object at 0x00000212AE4DA250>, <matplotlib.axis.XTick object at 0x00000212AE3634F0>, <matplotlib.axis.XTick object at 0x00000212AE4DA550>, <matplotlib.axis.XTick object at 0x00000212AE4DA100>, <matplotlib.axis.XTick object at 0x00000212AE679220>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE675640>
7
<matplotlib.contour.QuadContourSet object at 0x00000212AE6480D0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE6EF220>, <matplotlib.axis.YTick object at 0x00000212AE2A9AC0>, <matplotlib.axis.YTick object at 0x00000212AE71FBB0>, <matplotlib.axis.YTick object at 0x00000212AE71F3A0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE1F4D90>, <matplotlib.axis.XTick object at 0x00000212AE4A0F40>, <matplotlib.axis.XTick object at 0x00000212AE71F550>, <matplotlib.axis.XTick object at 0x00000212AE59FE50>, <matplotlib.axis.XTick object at 0x00000212AE59F730>, <matplotlib.axis.XTick object at 0x00000212AE4AC130>, <matplotlib.axis.XTick object at 0x00000212AE500C40>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AEB67B80>
8
<matplotlib.contour.QuadContourSet object at 0x00000212AE285310>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE2010D0>, <matplotlib.axis.YTick object at 0x00000212AE67F580>, <matplotlib.axis.YTick object at 0x00000212AE960C10>, <matplotlib.axis.YTick object at 0x00000212AE960820>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE1F4340>, <matplotlib.axis.XTick object at 0x00000212AE4C73D0>, <matplotlib.axis.XTick object at 0x00000212AE960460>, <matplotlib.axis.XTick object at 0x00000212AE6C5C40>, <matplotlib.axis.XTick object at 0x00000212AE6C5AC0>, <matplotlib.axis.XTick object at 0x00000212AE6C5C70>, <matplotlib.axis.XTick object at 0x00000212AE67EB20>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE62D6A0>
9
<matplotlib.contour.QuadContourSet object at 0x00000212AE6E4DC0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE39FD60>, <matplotlib.axis.YTick object at 0x00000212AE6E0E80>, <matplotlib.axis.YTick object at 0x00000212AE4C8550>, <matplotlib.axis.YTick object at 0x00000212AE4C8A60>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE38FEB0>, <matplotlib.axis.XTick object at 0x00000212AE50B3A0>, <matplotlib.axis.XTick object at 0x00000212AE4C8940>, <matplotlib.axis.XTick object at 0x00000212AEBBD5E0>, <matplotlib.axis.XTick object at 0x00000212AEBBDAF0>, <matplotlib.axis.XTick object at 0x00000212AEBC80A0>, <matplotlib.axis.XTick object at 0x00000212AEBC8550>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE536D00>
10
<matplotlib.contour.QuadContourSet object at 0x00000212AE4E5100>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE51DA30>, <matplotlib.axis.YTick object at 0x00000212AE4A73D0>, <matplotlib.axis.YTick object at 0x00000212AE6EFA00>, <matplotlib.axis.YTick object at 0x00000212AE6EF9A0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE51DD30>, <matplotlib.axis.XTick object at 0x00000212AE5D1D00>, <matplotlib.axis.XTick object at 0x00000212AE6EF6A0>, <matplotlib.axis.XTick object at 0x00000212AE4E8FD0>, <matplotlib.axis.XTick object at 0x00000212AE4E8C10>, <matplotlib.axis.XTick object at 0x00000212AE36EE50>, <matplotlib.axis.XTick object at 0x00000212AE36E310>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE65EEE0>
11
<matplotlib.contour.QuadContourSet object at 0x00000212AE633B80>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE4BA070>, <matplotlib.axis.YTick object at 0x00000212AE363730>, <matplotlib.axis.YTick object at 0x00000212AE56ED30>, <matplotlib.axis.YTick object at 0x00000212AE56ED90>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE338070>, <matplotlib.axis.XTick object at 0x00000212AE4DAEB0>, <matplotlib.axis.XTick object at 0x00000212AE56E250>, <matplotlib.axis.XTick object at 0x00000212AE12A1F0>, <matplotlib.axis.XTick object at 0x00000212AE5EE130>, <matplotlib.axis.XTick object at 0x00000212AE5EEF70>, <matplotlib.axis.XTick object at 0x00000212AE5EE6A0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE7394F0>
12
<matplotlib.contour.QuadContourSet object at 0x000002126CD24820>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE519CD0>, <matplotlib.axis.YTick object at 0x00000212AE38CA60>, <matplotlib.axis.YTick object at 0x00000212AE4AD9D0>, <matplotlib.axis.YTick object at 0x00000212AE4C17C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE1FBB80>, <matplotlib.axis.XTick object at 0x00000212AE4B45E0>, <matplotlib.axis.XTick object at 0x00000212AE4C11C0>, <matplotlib.axis.XTick object at 0x00000212AE4C1C70>, <matplotlib.axis.XTick object at 0x00000212AE4B3610>, <matplotlib.axis.XTick object at 0x00000212AE4B3EE0>, <matplotlib.axis.XTick object at 0x00000212AE4B3D00>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE4E49A0>
13
<matplotlib.contour.QuadContourSet object at 0x00000212AE64CF10>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE2FFB20>, <matplotlib.axis.YTick object at 0x00000212AE1F3E20>, <matplotlib.axis.YTick object at 0x00000212AE62D370>, <matplotlib.axis.YTick object at 0x00000212AE62D8B0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE3531F0>, <matplotlib.axis.XTick object at 0x00000212AE338EB0>, <matplotlib.axis.XTick object at 0x00000212AE338160>, <matplotlib.axis.XTick object at 0x00000212AE62D100>, <matplotlib.axis.XTick object at 0x00000212AE338A00>, <matplotlib.axis.XTick object at 0x00000212AE633CD0>, <matplotlib.axis.XTick object at 0x00000212AE633EE0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AEB68370>
14
<matplotlib.contour.QuadContourSet object at 0x00000212AE67E130>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE259BE0>, <matplotlib.axis.YTick object at 0x00000212AE2F1B20>, <matplotlib.axis.YTick object at 0x00000212AE5BE460>, <matplotlib.axis.YTick object at 0x00000212AEB866D0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE65EC70>, <matplotlib.axis.XTick object at 0x00000212AE2F1250>, <matplotlib.axis.XTick object at 0x00000212AE5BE1F0>, <matplotlib.axis.XTick object at 0x00000212AEB86580>, <matplotlib.axis.XTick object at 0x00000212AE675250>, <matplotlib.axis.XTick object at 0x00000212AE6752E0>, <matplotlib.axis.XTick object at 0x00000212AE6751C0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AEBC8040>
15
<matplotlib.contour.QuadContourSet object at 0x00000212AE31B0A0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE61D130>, <matplotlib.axis.YTick object at 0x00000212AE51DCA0>, <matplotlib.axis.YTick object at 0x00000212AE4E5D30>, <matplotlib.axis.YTick object at 0x00000212AE4E54C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE61D7F0>, <matplotlib.axis.XTick object at 0x00000212AE285850>, <matplotlib.axis.XTick object at 0x00000212AE4E5E50>, <matplotlib.axis.XTick object at 0x00000212AE701F40>, <matplotlib.axis.XTick object at 0x00000212AE701CA0>, <matplotlib.axis.XTick object at 0x00000212AEB6F6D0>, <matplotlib.axis.XTick object at 0x00000212AEB6F340>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE4BC070>
16
<matplotlib.contour.QuadContourSet object at 0x00000212AE52E0A0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE54AFD0>, <matplotlib.axis.YTick object at 0x00000212AE954BE0>, <matplotlib.axis.YTick object at 0x00000212AEBC97F0>, <matplotlib.axis.YTick object at 0x00000212AEBC9D00>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE948550>, <matplotlib.axis.XTick object at 0x00000212AE6EAF40>, <matplotlib.axis.XTick object at 0x00000212AEBC9730>, <matplotlib.axis.XTick object at 0x00000212AE3AC1F0>, <matplotlib.axis.XTick object at 0x00000212AE3ACD90>, <matplotlib.axis.XTick object at 0x00000212AEBBA2E0>, <matplotlib.axis.XTick object at 0x00000212AEBBA7F0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE640FA0>
17
<matplotlib.contour.QuadContourSet object at 0x00000212AE948D60>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE4E58B0>, <matplotlib.axis.YTick object at 0x00000212AE4BC970>, <matplotlib.axis.YTick object at 0x00000212AE96ABE0>, <matplotlib.axis.YTick object at 0x00000212AE96A4C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE4E5E80>, <matplotlib.axis.XTick object at 0x00000212AE6DA4C0>, <matplotlib.axis.XTick object at 0x00000212AE96A490>, <matplotlib.axis.XTick object at 0x00000212AE4E23A0>, <matplotlib.axis.XTick object at 0x00000212AE675A00>, <matplotlib.axis.XTick object at 0x00000212AE675A90>, <matplotlib.axis.XTick object at 0x00000212AE675CD0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE61DBE0>
18
<matplotlib.contour.QuadContourSet object at 0x00000212AE36E490>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE6BAEE0>, <matplotlib.axis.YTick object at 0x00000212AE201490>, <matplotlib.axis.YTick object at 0x00000212AE2A9EB0>, <matplotlib.axis.YTick object at 0x00000212AE2A9A30>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE6BA4F0>, <matplotlib.axis.XTick object at 0x00000212AEBBDDF0>, <matplotlib.axis.XTick object at 0x00000212AE2A9FD0>, <matplotlib.axis.XTick object at 0x00000212AEB8C2B0>, <matplotlib.axis.XTick object at 0x00000212AEB8C340>, <matplotlib.axis.XTick object at 0x00000212AEBC85E0>, <matplotlib.axis.XTick object at 0x00000212AEBC8E80>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE2F7E80>
19
<matplotlib.contour.QuadContourSet object at 0x00000212AE4B47F0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE38C9D0>, <matplotlib.axis.YTick object at 0x00000212AE2C9B20>, <matplotlib.axis.YTick object at 0x000002126CD24CA0>, <matplotlib.axis.YTick object at 0x00000212AE31B490>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE1FB3D0>, <matplotlib.axis.XTick object at 0x00000212AE5839D0>, <matplotlib.axis.XTick object at 0x000002126CD24460>, <matplotlib.axis.XTick object at 0x00000212AE31B0A0>, <matplotlib.axis.XTick object at 0x00000212AE172F40>, <matplotlib.axis.XTick object at 0x00000212AE172DF0>, <matplotlib.axis.XTick object at 0x00000212AE172E50>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE739E80>
20
<matplotlib.contour.QuadContourSet object at 0x00000212AE155850>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE4D80A0>, <matplotlib.axis.YTick object at 0x00000212AE53A2B0>, <matplotlib.axis.YTick object at 0x00000212AE53A6D0>, <matplotlib.axis.YTick object at 0x00000212AE53A790>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE62D640>, <matplotlib.axis.XTick object at 0x00000212AE4B3160>, <matplotlib.axis.XTick object at 0x00000212AE4B3190>, <matplotlib.axis.XTick object at 0x00000212AE4B30D0>, <matplotlib.axis.XTick object at 0x00000212AE53AE50>, <matplotlib.axis.XTick object at 0x00000212AEBC8B80>, <matplotlib.axis.XTick object at 0x00000212AEBC84C0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE4E8640>
21
<matplotlib.contour.QuadContourSet object at 0x00000212AEBB61F0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE2F1790>, <matplotlib.axis.YTick object at 0x00000212AE36CB80>, <matplotlib.axis.YTick object at 0x00000212AE4E5640>, <matplotlib.axis.YTick object at 0x00000212AE38CC70>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE1F2C10>, <matplotlib.axis.XTick object at 0x00000212AE6A5EB0>, <matplotlib.axis.XTick object at 0x00000212AE4E5550>, <matplotlib.axis.XTick object at 0x00000212AE54A190>, <matplotlib.axis.XTick object at 0x00000212AE54ABB0>, <matplotlib.axis.XTick object at 0x00000212AE1EEB20>, <matplotlib.axis.XTick object at 0x00000212AE1EEA60>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE712CD0>
22
<matplotlib.contour.QuadContourSet object at 0x00000212AE573220>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE65E520>, <matplotlib.axis.YTick object at 0x00000212AE2BD2E0>, <matplotlib.axis.YTick object at 0x00000212AE93EBB0>, <matplotlib.axis.YTick object at 0x00000212AE93E0A0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE65EA00>, <matplotlib.axis.XTick object at 0x00000212AE72D760>, <matplotlib.axis.XTick object at 0x00000212AE93ED30>, <matplotlib.axis.XTick object at 0x00000212AE4A7310>, <matplotlib.axis.XTick object at 0x00000212AE4A79D0>, <matplotlib.axis.XTick object at 0x00000212AE4A72E0>, <matplotlib.axis.XTick object at 0x00000212AE4E4A60>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE957160>
23
<matplotlib.contour.QuadContourSet object at 0x00000212AEBAB430>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AEBC5FD0>, <matplotlib.axis.YTick object at 0x00000212AEB5C4C0>, <matplotlib.axis.YTick object at 0x00000212AE5238E0>, <matplotlib.axis.YTick object at 0x00000212AE523DF0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE4E2DC0>, <matplotlib.axis.XTick object at 0x00000212AEB5EE20>, <matplotlib.axis.XTick object at 0x00000212AE5237C0>, <matplotlib.axis.XTick object at 0x00000212AE1F3250>, <matplotlib.axis.XTick object at 0x00000212AE1F3E80>, <matplotlib.axis.XTick object at 0x00000212AE5433D0>, <matplotlib.axis.XTick object at 0x00000212AE5438E0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE5D30D0>
24
<matplotlib.contour.QuadContourSet object at 0x00000212AE581E80>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE2B4F40>, <matplotlib.axis.YTick object at 0x00000212AE5FCC10>, <matplotlib.axis.YTick object at 0x00000212AE5D1970>, <matplotlib.axis.YTick object at 0x00000212AE5D1A00>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE2B48E0>, <matplotlib.axis.XTick object at 0x00000212AEBABB20>, <matplotlib.axis.XTick object at 0x00000212AE5D1BE0>, <matplotlib.axis.XTick object at 0x00000212AE4E2130>, <matplotlib.axis.XTick object at 0x00000212AE4E2CA0>, <matplotlib.axis.XTick object at 0x00000212AE2BD0D0>, <matplotlib.axis.XTick object at 0x00000212AE2BD340>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE6DAFA0>
25
<matplotlib.contour.QuadContourSet object at 0x00000212AE165C40>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE113FA0>, <matplotlib.axis.YTick object at 0x00000212AE285FA0>, <matplotlib.axis.YTick object at 0x00000212AE4ADEE0>, <matplotlib.axis.YTick object at 0x00000212AE4ADC70>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE1139D0>, <matplotlib.axis.XTick object at 0x00000212AEB684C0>, <matplotlib.axis.XTick object at 0x00000212AE61DBB0>, <matplotlib.axis.XTick object at 0x00000212AE4AD0D0>, <matplotlib.axis.XTick object at 0x00000212AE61DE80>, <matplotlib.axis.XTick object at 0x00000212AE61D0A0>, <matplotlib.axis.XTick object at 0x00000212AE61D250>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE33DEE0>
26
<matplotlib.contour.QuadContourSet object at 0x00000212AEBC8C70>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE62D3A0>, <matplotlib.axis.YTick object at 0x00000212AE4D8130>, <matplotlib.axis.YTick object at 0x00000212AE3638B0>, <matplotlib.axis.YTick object at 0x00000212AE363D00>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE619220>, <matplotlib.axis.XTick object at 0x00000212AE71F2B0>, <matplotlib.axis.XTick object at 0x00000212AE36C7F0>, <matplotlib.axis.XTick object at 0x00000212AE363220>, <matplotlib.axis.XTick object at 0x00000212AE36CB20>, <matplotlib.axis.XTick object at 0x00000212AE583340>, <matplotlib.axis.XTick object at 0x00000212AE583430>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE2AFC70>
27
<matplotlib.contour.QuadContourSet object at 0x00000212AE207460>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE62D640>, <matplotlib.axis.YTick object at 0x00000212AE4DA940>, <matplotlib.axis.YTick object at 0x00000212AE33D340>, <matplotlib.axis.YTick object at 0x00000212AE33D160>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE681CD0>, <matplotlib.axis.XTick object at 0x00000212AE4E4E50>, <matplotlib.axis.XTick object at 0x00000212AE33D3D0>, <matplotlib.axis.XTick object at 0x00000212AE31BFD0>, <matplotlib.axis.XTick object at 0x00000212AE31B220>, <matplotlib.axis.XTick object at 0x00000212AE1136D0>, <matplotlib.axis.XTick object at 0x00000212AE113670>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE72DF10>
28
<matplotlib.contour.QuadContourSet object at 0x00000212AE6B1A90>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE6983A0>, <matplotlib.axis.YTick object at 0x00000212AE12A790>, <matplotlib.axis.YTick object at 0x00000212AE5FCBB0>, <matplotlib.axis.YTick object at 0x00000212AE5FC3D0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE698070>, <matplotlib.axis.XTick object at 0x00000212AE172940>, <matplotlib.axis.XTick object at 0x00000212AE5FC280>, <matplotlib.axis.XTick object at 0x00000212AE65E700>, <matplotlib.axis.XTick object at 0x00000212AE65E160>, <matplotlib.axis.XTick object at 0x00000212AE1F3BE0>, <matplotlib.axis.XTick object at 0x00000212AE1F3040>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE6A5B50>
29
<matplotlib.contour.QuadContourSet object at 0x00000212AE633A30>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE6DA6A0>, <matplotlib.axis.YTick object at 0x00000212AEB6CB80>, <matplotlib.axis.YTick object at 0x00000212AE561040>, <matplotlib.axis.YTick object at 0x00000212AE561550>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE2BD9D0>, <matplotlib.axis.XTick object at 0x00000212AE54AA00>, <matplotlib.axis.XTick object at 0x00000212AE561820>, <matplotlib.axis.XTick object at 0x00000212AE5990A0>, <matplotlib.axis.XTick object at 0x00000212AE5995E0>, <matplotlib.axis.XTick object at 0x00000212AE599AF0>, <matplotlib.axis.XTick object at 0x00000212AE5650A0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE1D07F0>
30
<matplotlib.contour.QuadContourSet object at 0x00000212AE67D130>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AEBC50D0>, <matplotlib.axis.YTick object at 0x00000212AE37AFD0>, <matplotlib.axis.YTick object at 0x00000212AE4E2CD0>, <matplotlib.axis.YTick object at 0x00000212AE4E8C40>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE6C3790>, <matplotlib.axis.XTick object at 0x00000212AE6A2160>, <matplotlib.axis.XTick object at 0x00000212AE643640>, <matplotlib.axis.XTick object at 0x00000212AE4E8B50>, <matplotlib.axis.XTick object at 0x00000212AE643280>, <matplotlib.axis.XTick object at 0x00000212AE643C70>, <matplotlib.axis.XTick object at 0x00000212AE5FC100>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE149D60>
31
<matplotlib.contour.QuadContourSet object at 0x00000212AE4B4340>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AEB86610>, <matplotlib.axis.YTick object at 0x00000212AE72DEE0>, <matplotlib.axis.YTick object at 0x00000212AE675C70>, <matplotlib.axis.YTick object at 0x00000212AE675F10>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE6EFBB0>, <matplotlib.axis.XTick object at 0x00000212AE72DCA0>, <matplotlib.axis.XTick object at 0x00000212AE675130>, <matplotlib.axis.XTick object at 0x00000212AE554280>, <matplotlib.axis.XTick object at 0x00000212AE554730>, <matplotlib.axis.XTick object at 0x00000212AE201160>, <matplotlib.axis.XTick object at 0x00000212AE2019D0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE62D6D0>
32
<matplotlib.contour.QuadContourSet object at 0x00000212AE61D3A0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE363FA0>, <matplotlib.axis.YTick object at 0x00000212AE65E880>, <matplotlib.axis.YTick object at 0x00000212AE519940>, <matplotlib.axis.YTick object at 0x00000212AE609FA0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE4E5460>, <matplotlib.axis.XTick object at 0x00000212AE338700>, <matplotlib.axis.XTick object at 0x00000212AE609AF0>, <matplotlib.axis.XTick object at 0x00000212AE609790>, <matplotlib.axis.XTick object at 0x00000212AE53A640>, <matplotlib.axis.XTick object at 0x00000212AE53ABB0>, <matplotlib.axis.XTick object at 0x00000212AE53AB80>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE668520>
33
<matplotlib.contour.QuadContourSet object at 0x00000212AEBCF490>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE602A00>, <matplotlib.axis.YTick object at 0x00000212AEBB5820>, <matplotlib.axis.YTick object at 0x00000212AE559BE0>, <matplotlib.axis.YTick object at 0x00000212AE54F130>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE6027F0>, <matplotlib.axis.XTick object at 0x00000212AE368DC0>, <matplotlib.axis.XTick object at 0x00000212AE559970>, <matplotlib.axis.XTick object at 0x00000212AE54F400>, <matplotlib.axis.XTick object at 0x00000212AE5361C0>, <matplotlib.axis.XTick object at 0x00000212AE5366D0>, <matplotlib.axis.XTick object at 0x00000212AE536BE0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE6E83D0>
34
<matplotlib.contour.QuadContourSet object at 0x00000212AE602DC0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE4ADE80>, <matplotlib.axis.YTick object at 0x00000212AEBB5AF0>, <matplotlib.axis.YTick object at 0x00000212AE36E700>, <matplotlib.axis.YTick object at 0x00000212AE590250>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE1703A0>, <matplotlib.axis.XTick object at 0x00000212AE619280>, <matplotlib.axis.XTick object at 0x00000212AE5906D0>, <matplotlib.axis.XTick object at 0x00000212AE590EB0>, <matplotlib.axis.XTick object at 0x00000212AE2C9610>, <matplotlib.axis.XTick object at 0x00000212AE2C9970>, <matplotlib.axis.XTick object at 0x00000212AE2C9D60>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE675D00>
35
<matplotlib.contour.QuadContourSet object at 0x00000212AE33D160>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE14E580>, <matplotlib.axis.YTick object at 0x00000212AE65E6A0>, <matplotlib.axis.YTick object at 0x00000212AE4E2160>, <matplotlib.axis.YTick object at 0x00000212AE4B3490>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE1F3B20>, <matplotlib.axis.XTick object at 0x00000212AE679490>, <matplotlib.axis.XTick object at 0x00000212AE4E2A30>, <matplotlib.axis.XTick object at 0x00000212AE4B3BB0>, <matplotlib.axis.XTick object at 0x00000212AE3CF970>, <matplotlib.axis.XTick object at 0x00000212AE3CF640>, <matplotlib.axis.XTick object at 0x00000212AE3CFFA0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AEBB6C40>
36
<matplotlib.contour.QuadContourSet object at 0x00000212AE62DD30>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE4E82B0>, <matplotlib.axis.YTick object at 0x00000212AE5D30D0>, <matplotlib.axis.YTick object at 0x00000212AE38CDF0>, <matplotlib.axis.YTick object at 0x00000212AE155370>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE4F6FA0>, <matplotlib.axis.XTick object at 0x00000212AE53AD00>, <matplotlib.axis.XTick object at 0x00000212AE38C0D0>, <matplotlib.axis.XTick object at 0x00000212AE155790>, <matplotlib.axis.XTick object at 0x00000212AE2F1CD0>, <matplotlib.axis.XTick object at 0x00000212AE2F1A00>, <matplotlib.axis.XTick object at 0x00000212AE2F1D60>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE5655E0>
37
<matplotlib.contour.QuadContourSet object at 0x00000212AE6DDCA0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE172FD0>, <matplotlib.axis.YTick object at 0x00000212AE4E26A0>, <matplotlib.axis.YTick object at 0x00000212AE4E2B50>, <matplotlib.axis.YTick object at 0x00000212AE33DE20>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE96A130>, <matplotlib.axis.XTick object at 0x00000212AE33DB80>, <matplotlib.axis.XTick object at 0x00000212AE33D0A0>, <matplotlib.axis.XTick object at 0x00000212AE33DA00>, <matplotlib.axis.XTick object at 0x00000212AE4E25B0>, <matplotlib.axis.XTick object at 0x00000212AE53A760>, <matplotlib.axis.XTick object at 0x00000212AE53AF40>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AEBB6DF0>
38
<matplotlib.contour.QuadContourSet object at 0x000002126CD24C40>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE6BAA00>, <matplotlib.axis.YTick object at 0x00000212AE201190>, <matplotlib.axis.YTick object at 0x00000212AE6E86A0>, <matplotlib.axis.YTick object at 0x00000212AE4BA1F0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE4F6EB0>, <matplotlib.axis.XTick object at 0x00000212AE739A30>, <matplotlib.axis.XTick object at 0x00000212AE6E84C0>, <matplotlib.axis.XTick object at 0x00000212AE1FB760>, <matplotlib.axis.XTick object at 0x00000212AE1FB340>, <matplotlib.axis.XTick object at 0x00000212AE6C5BE0>, <matplotlib.axis.XTick object at 0x00000212AE6C5580>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE54F6D0>
39
<matplotlib.contour.QuadContourSet object at 0x00000212AE5FC280>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE4F63D0>, <matplotlib.axis.YTick object at 0x00000212AE619310>, <matplotlib.axis.YTick object at 0x00000212AE4A0C10>, <matplotlib.axis.YTick object at 0x00000212AE4A0820>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE937460>, <matplotlib.axis.XTick object at 0x00000212AE6A2160>, <matplotlib.axis.XTick object at 0x00000212AE4A0F40>, <matplotlib.axis.XTick object at 0x00000212AE72D7F0>, <matplotlib.axis.XTick object at 0x00000212AE95E670>, <matplotlib.axis.XTick object at 0x00000212AE95E430>, <matplotlib.axis.XTick object at 0x00000212AE95E610>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE9422E0>
40
<matplotlib.contour.QuadContourSet object at 0x00000212AEB82430>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE5D59A0>, <matplotlib.axis.YTick object at 0x00000212AE5BCCA0>, <matplotlib.axis.YTick object at 0x00000212AEB9CB80>, <matplotlib.axis.YTick object at 0x00000212AEBAF0D0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE5D5130>, <matplotlib.axis.XTick object at 0x00000212AE519070>, <matplotlib.axis.XTick object at 0x00000212AEB9CA90>, <matplotlib.axis.XTick object at 0x00000212AEBAF280>, <matplotlib.axis.XTick object at 0x00000212AEB9D160>, <matplotlib.axis.XTick object at 0x00000212AEB9D670>, <matplotlib.axis.XTick object at 0x00000212AEB9DB80>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE55D370>
41
<matplotlib.contour.QuadContourSet object at 0x00000212AE675BE0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE1FB6A0>, <matplotlib.axis.YTick object at 0x00000212AE942C70>, <matplotlib.axis.YTick object at 0x00000212AEB8CC10>, <matplotlib.axis.YTick object at 0x00000212AEB8CF70>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE654370>, <matplotlib.axis.XTick object at 0x00000212AE6530D0>, <matplotlib.axis.XTick object at 0x00000212AEB8C700>, <matplotlib.axis.XTick object at 0x00000212AE3383A0>, <matplotlib.axis.XTick object at 0x00000212AE739A00>, <matplotlib.axis.XTick object at 0x00000212AE739070>, <matplotlib.axis.XTick object at 0x00000212AE739B80>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE31BEE0>
42
<matplotlib.contour.QuadContourSet object at 0x00000212AE172B20>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AEB565B0>, <matplotlib.axis.YTick object at 0x00000212AE6B1C70>, <matplotlib.axis.YTick object at 0x00000212AE2BDCA0>, <matplotlib.axis.YTick object at 0x00000212AE2BDDF0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AEB56820>, <matplotlib.axis.XTick object at 0x00000212AE1B8B20>, <matplotlib.axis.XTick object at 0x00000212AE2BDEE0>, <matplotlib.axis.XTick object at 0x00000212AE4E4D00>, <matplotlib.axis.XTick object at 0x00000212AE4E4BB0>, <matplotlib.axis.XTick object at 0x00000212AE4E4CA0>, <matplotlib.axis.XTick object at 0x00000212AE2D5AC0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE2F14F0>
43
<matplotlib.contour.QuadContourSet object at 0x00000212AE561FA0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE96A310>, <matplotlib.axis.YTick object at 0x00000212AE5D3A30>, <matplotlib.axis.YTick object at 0x00000212AE679E80>, <matplotlib.axis.YTick object at 0x00000212AE2072B0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE573970>, <matplotlib.axis.XTick object at 0x00000212AEB68A90>, <matplotlib.axis.XTick object at 0x00000212AE207FA0>, <matplotlib.axis.XTick object at 0x00000212AE4BA310>, <matplotlib.axis.XTick object at 0x00000212AE4BACD0>, <matplotlib.axis.XTick object at 0x00000212AE4BAE80>, <matplotlib.axis.XTick object at 0x00000212AEBC3AC0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE1F6FA0>
44
<matplotlib.contour.QuadContourSet object at 0x00000212AE352610>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE38C820>, <matplotlib.axis.YTick object at 0x00000212AE72E1F0>, <matplotlib.axis.YTick object at 0x00000212AE53ABE0>, <matplotlib.axis.YTick object at 0x00000212AE53A820>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE353DF0>, <matplotlib.axis.XTick object at 0x00000212AE3B9850>, <matplotlib.axis.XTick object at 0x00000212AEBB6E20>, <matplotlib.axis.XTick object at 0x00000212AE53A1C0>, <matplotlib.axis.XTick object at 0x00000212AEBB6580>, <matplotlib.axis.XTick object at 0x00000212AE6EFB50>, <matplotlib.axis.XTick object at 0x00000212AE6EF6A0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE14EE20>
45
<matplotlib.contour.QuadContourSet object at 0x00000212AE4A0F10>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE937B50>, <matplotlib.axis.YTick object at 0x00000212AE2BD790>, <matplotlib.axis.YTick object at 0x00000212AE59F2B0>, <matplotlib.axis.YTick object at 0x00000212AE59FB50>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE4E8070>, <matplotlib.axis.XTick object at 0x00000212AE4E41F0>, <matplotlib.axis.XTick object at 0x00000212AE59F7F0>, <matplotlib.axis.XTick object at 0x00000212AE590A30>, <matplotlib.axis.XTick object at 0x00000212AE590D00>, <matplotlib.axis.XTick object at 0x00000212AE590F40>, <matplotlib.axis.XTick object at 0x00000212AE368760>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE739730>
46
<matplotlib.contour.QuadContourSet object at 0x00000212AE149F10>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE5B6A30>, <matplotlib.axis.YTick object at 0x00000212AE338370>, <matplotlib.axis.YTick object at 0x00000212AE1F2610>, <matplotlib.axis.YTick object at 0x00000212AE172550>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE4E2310>, <matplotlib.axis.XTick object at 0x00000212AEBAF370>, <matplotlib.axis.XTick object at 0x00000212AE1F2850>, <matplotlib.axis.XTick object at 0x00000212AE172A90>, <matplotlib.axis.XTick object at 0x00000212AE6C56A0>, <matplotlib.axis.XTick object at 0x00000212AE6C5C40>, <matplotlib.axis.XTick object at 0x00000212AE6C5670>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE4B35E0>
47
<matplotlib.contour.QuadContourSet object at 0x00000212AE71A580>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE2BC220>, <matplotlib.axis.YTick object at 0x00000212AE6D1970>, <matplotlib.axis.YTick object at 0x00000212AE20ACD0>, <matplotlib.axis.YTick object at 0x00000212AE4A7220>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE728E50>, <matplotlib.axis.XTick object at 0x00000212AE2C9340>, <matplotlib.axis.XTick object at 0x00000212AE4A7AC0>, <matplotlib.axis.XTick object at 0x00000212AE4A71C0>, <matplotlib.axis.XTick object at 0x00000212AE1F72B0>, <matplotlib.axis.XTick object at 0x00000212AE1F77C0>, <matplotlib.axis.XTick object at 0x00000212AE1F7CD0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE9504C0>
48
<matplotlib.contour.QuadContourSet object at 0x00000212AE4E5100>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE65FE80>, <matplotlib.axis.YTick object at 0x00000212AE6C4A90>, <matplotlib.axis.YTick object at 0x00000212AE590850>, <matplotlib.axis.YTick object at 0x00000212AE59FE50>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE706CA0>, <matplotlib.axis.XTick object at 0x00000212AE6AD7F0>, <matplotlib.axis.XTick object at 0x00000212AE5909D0>, <matplotlib.axis.XTick object at 0x00000212AE59FBB0>, <matplotlib.axis.XTick object at 0x00000212AE2C9E20>, <matplotlib.axis.XTick object at 0x00000212AE2C9310>, <matplotlib.axis.XTick object at 0x00000212AE338070>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE36EC70>
49
<matplotlib.contour.QuadContourSet object at 0x00000212AE4C1070>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE285D00>, <matplotlib.axis.YTick object at 0x00000212AE31B400>, <matplotlib.axis.YTick object at 0x00000212AE680A30>, <matplotlib.axis.YTick object at 0x00000212AE6801C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE1B8D60>, <matplotlib.axis.XTick object at 0x00000212AE4E24C0>, <matplotlib.axis.XTick object at 0x00000212AE680760>, <matplotlib.axis.XTick object at 0x00000212AE675E50>, <matplotlib.axis.XTick object at 0x00000212AE6759D0>, <matplotlib.axis.XTick object at 0x00000212AE675100>, <matplotlib.axis.XTick object at 0x00000212AE4AA280>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE35F670>
50
<matplotlib.contour.QuadContourSet object at 0x00000212AE1553A0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x00000212AE1F3AC0>, <matplotlib.axis.YTick object at 0x00000212AE20D6D0>, <matplotlib.axis.YTick object at 0x00000212AE583550>, <matplotlib.axis.YTick object at 0x00000212AE583EB0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x00000212AE6C5400>, <matplotlib.axis.XTick object at 0x00000212AEBB6190>, <matplotlib.axis.XTick object at 0x00000212AE583AC0>, <matplotlib.axis.XTick object at 0x00000212AE352280>, <matplotlib.axis.XTick object at 0x00000212AE3523A0>, <matplotlib.axis.XTick object at 0x00000212AE38CA90>, <matplotlib.axis.XTick object at 0x00000212AE38C4F0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x00000212AE4ADEE0>
>>> 