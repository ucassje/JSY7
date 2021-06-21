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
>>> fre=0.072
>>> Me=9.1094*(10**(-28))
>>> Mp=1.6726*(10**(-24))
>>> ratio=Me/Mp
>>> q=4.8032*(10**(-10))
>>> c=2.9979*10**(10)
>>> k_pal0=0.231
>>> k_per0=k_pal0*tan((55*np.pi)/180)
>>> k_per0
mpf('0.32990218955742845')
>>> k_pal_max=0.252
>>> k_pal_min=0.21
>>> k_per_max=k_pal_max*tan((55*np.pi)/180)
>>> k_per_min=k_pal_min*tan((55*np.pi)/180)
>>> a_pal=0.021
>>> a_per=a_pal*tan((55*np.pi)/180)
>>> a_per
mpf('0.029991108141584403')
>>> GV=0.75
>>> B_B0=0.001
>>> def k(b):
    f = lambda x: ((besselj(0, (b*x)/(omega), 0))**2)*np.exp(-(((x-0.33)**2)/(0.03**2)))*x
    I=integrate.quad(f, k_per_min, k_per_max)
    return I[0]

>>> def coefficient_a(a,b):
    return ((0.511*np.pi**2)/(0.021*0.03**2))*(B_B0**2)*(((b)**2)/abs(a-GV))*(fre/k_pal0)**2*k(b)*(np.exp(-(((fre-a*k_pal0-n*omega)/(a-GV))**2)/(0.021**2)))**2

>>> def coefficient_a2(a,b):
    return (((9.8*10**(-7))*np.pi**2)/(0.021*0.03**2))*(B_B0**2)*(((a)**2)/abs(a-GV))*(fre/k_pal0)**2*k(b)*(np.exp(-(((fre-a*k_pal0-n2*omega)/(a-GV))**2)/(0.021**2)))**2

>>> def coefficient_a3(a,b):
    return ((0.511*np.pi**2)/(0.021*0.03**2))*(B_B0**2)*(((b)**2)/abs(a-GV))*(fre/k_pal0)**2*k(b)*(np.exp(-(((fre-a*k_pal0-n3*omega)/(a-GV))**2)/(0.021**2)))**2

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

>>> for a in range(2*Nv):
    for b in range(2*Nv):
        if a==b:
            AA[a*2*Nv:(a+1)*2*Nv,b*2*Nv:(b+1)*2*Nv]=Matrix_A(a)

            
Traceback (most recent call last):
  File "<pyshell#51>", line 4, in <module>
    AA[a*2*Nv:(a+1)*2*Nv,b*2*Nv:(b+1)*2*Nv]=Matrix_A(a)
  File "<pyshell#49>", line 6, in Matrix_A
    A[i,j] =1+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]-delv/2,per_v[b])+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2)+(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2)+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b])+(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]-delv/2)-GV*k_pal0*(pal_v[i]-delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]-delv/2,per_v[b]) if j==0 else 0 if j==1 else -(F/2)*((fre-GV*k_pal0-n*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n*omega))**2*coefficient_a(pal_v[i]+delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n2*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n2*omega))**2*coefficient_a2(pal_v[i]+delv/2,per_v[b])-(F/2)*((fre-GV*k_pal0-n3*omega)/(fre*(pal_v[i]+delv/2)-GV*k_pal0*(pal_v[i]+delv/2)-GV*n3*omega))**2*coefficient_a3(pal_v[i]+delv/2,per_v[b]) if j==2 else 0
NameError: name 'delv' is not defined
>>> delv=2*abs(pal_v[1]-pal_v[0])
>>> for a in range(2*Nv):
    for b in range(2*Nv):
        if a==b:
            AA[a*2*Nv:(a+1)*2*Nv,b*2*Nv:(b+1)*2*Nv]=Matrix_A(a)

            
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

>>> for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            AA[(a+1)*2*Nv:(a+2)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=Matrix_B1(a+1)

            
>>> for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            AA[a*2*Nv:(a+1)*2*Nv,(b+1)*2*Nv:(b+2)*2*Nv]=Matrix_B2(a)

            
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
>>> Us=140*ratio**(0.5)
>>> Uc=-3.77*ratio**(0.5)
>>> def Kappa_Initial_Strahl(a,b):
    kappa=150
    return ((1.34)**(-0.5))*((0.2055)**(-1))*0.0262*np.exp(-((b)**2)/0.2055)*np.exp(-((a-Us)**2)/1.34)

>>> def Kappa_Initial_Core(a,b):
    kappa=150
    return ((0.296)**(-0.5))*((0.289)**(-1))*0.9738*np.exp(-((b)**2)/0.289)*np.exp(-((a-Uc)**2)/0.296)

>>> cont_lev = np.linspace(-8,0,25)
>>> f_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> solu2=np.zeros(shape = (Nv, 2*Nv))
>>> ff_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        f_1[j*2*Nv+i]=Kappa_Initial_Strahl(pal_v[i],per_v[j])

        
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        fc_1[j*2*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j])

        
Traceback (most recent call last):
  File "<pyshell#94>", line 3, in <module>
    fc_1[j*2*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j])
NameError: name 'fc_1' is not defined
>>> fc_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        fc_1[j*2*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j])

        
>>> ff_1=f_1+fc_1
>>> Mf_1=np.max(ff_1)
>>> per_v2 = np.linspace(0, Mv, Nv)
>>> X2,Y2 = np.meshgrid(pal_v,per_v2)
>>> for k in range(30): #Numer in range indicates the minute.
    print(k)
    #ff_1=f_1+fc_1
    for j in range(Nv):
        for i in range(2*Nv):
        #solu[j,i]=(abs(f_1[j*2*Nv+i])/Mf_1)
            if abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>1:
                solu2[j,i]=0
            elif abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>10**(-6):
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
    for t in range(10):
        ff_1=dot(AQ, ff_1)

        
0
<matplotlib.contour.QuadContourSet object at 0x000002389C654C40>

Warning (from warnings module):
  File "<pyshell#103>", line 19
MatplotlibDeprecationWarning: 
The set_smart_bounds function was deprecated in Matplotlib 3.2 and will be removed two minor releases later.

Warning (from warnings module):
  File "<pyshell#103>", line 21
MatplotlibDeprecationWarning: 
The set_smart_bounds function was deprecated in Matplotlib 3.2 and will be removed two minor releases later.
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C628C10>, <matplotlib.axis.YTick object at 0x000002389C61DBB0>, <matplotlib.axis.YTick object at 0x000002389C649490>, <matplotlib.axis.YTick object at 0x000002389C6499A0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C63ABB0>, <matplotlib.axis.XTick object at 0x000002389C61D430>, <matplotlib.axis.XTick object at 0x000002389C6C81C0>, <matplotlib.axis.XTick object at 0x000002389C6C86D0>, <matplotlib.axis.XTick object at 0x000002389C6C8BE0>, <matplotlib.axis.XTick object at 0x000002389C6CC130>, <matplotlib.axis.XTick object at 0x000002389C6CC640>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C63AD90>
1
<matplotlib.contour.QuadContourSet object at 0x000002389CF79A90>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C578A30>, <matplotlib.axis.YTick object at 0x000002389C51EBE0>, <matplotlib.axis.YTick object at 0x000002389C557250>, <matplotlib.axis.YTick object at 0x000002389C557760>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C577A00>, <matplotlib.axis.XTick object at 0x000002389C4DB280>, <matplotlib.axis.XTick object at 0x000002389BD792E0>, <matplotlib.axis.XTick object at 0x000002389BD797F0>, <matplotlib.axis.XTick object at 0x000002389C5575E0>, <matplotlib.axis.XTick object at 0x000002389BD79C10>, <matplotlib.axis.XTick object at 0x000002389BD79E20>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389BD99A30>
2
<matplotlib.contour.QuadContourSet object at 0x000002389C6B3E80>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C731FD0>, <matplotlib.axis.YTick object at 0x000002389C506220>, <matplotlib.axis.YTick object at 0x000002389C6D5BB0>, <matplotlib.axis.YTick object at 0x000002389C6D59D0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C4DBFA0>, <matplotlib.axis.XTick object at 0x000002389C506B50>, <matplotlib.axis.XTick object at 0x000002389C6D57F0>, <matplotlib.axis.XTick object at 0x000002389C628370>, <matplotlib.axis.XTick object at 0x000002389C628880>, <matplotlib.axis.XTick object at 0x000002389C6286D0>, <matplotlib.axis.XTick object at 0x000002389C023C10>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C6E0DC0>
3
<matplotlib.contour.QuadContourSet object at 0x000002389C0581C0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C5646A0>, <matplotlib.axis.YTick object at 0x000002389C4DB550>, <matplotlib.axis.YTick object at 0x000002389C8ED940>, <matplotlib.axis.YTick object at 0x000002389C8EDE50>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C06C9D0>, <matplotlib.axis.XTick object at 0x000002389BD6C3D0>, <matplotlib.axis.XTick object at 0x000002389C8ED820>, <matplotlib.axis.XTick object at 0x000002389C8F1100>, <matplotlib.axis.XTick object at 0x000002389C8F1E80>, <matplotlib.axis.XTick object at 0x000002389C8F3430>, <matplotlib.axis.XTick object at 0x000002389C8F3940>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C91D190>
4
<matplotlib.contour.QuadContourSet object at 0x000002389C81E160>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C80EFA0>, <matplotlib.axis.YTick object at 0x000002389C943160>, <matplotlib.axis.YTick object at 0x000002389C8408E0>, <matplotlib.axis.YTick object at 0x000002389C840DF0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C93D910>, <matplotlib.axis.XTick object at 0x000002389C681E20>, <matplotlib.axis.XTick object at 0x000002389C8407C0>, <matplotlib.axis.XTick object at 0x000002389C841040>, <matplotlib.axis.XTick object at 0x000002389C841E80>, <matplotlib.axis.XTick object at 0x000002389C8463D0>, <matplotlib.axis.XTick object at 0x000002389C8468E0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C8710D0>
5
<matplotlib.contour.QuadContourSet object at 0x000002389C96E3A0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C8972E0>, <matplotlib.axis.YTick object at 0x000002389C806610>, <matplotlib.axis.YTick object at 0x000002389CB11C70>, <matplotlib.axis.YTick object at 0x000002389C96F1C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C89DF40>, <matplotlib.axis.XTick object at 0x000002389C841850>, <matplotlib.axis.XTick object at 0x000002389C96FAC0>, <matplotlib.axis.XTick object at 0x000002389C96F0A0>, <matplotlib.axis.XTick object at 0x000002389CB15250>, <matplotlib.axis.XTick object at 0x000002389CB15760>, <matplotlib.axis.XTick object at 0x000002389CB15C70>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389CB41490>
6
<matplotlib.contour.QuadContourSet object at 0x000002389C87BA90>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C848D00>, <matplotlib.axis.YTick object at 0x000002389C9867C0>, <matplotlib.axis.YTick object at 0x000002389C93D220>, <matplotlib.axis.YTick object at 0x000002389C8F3190>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C87E6D0>, <matplotlib.axis.XTick object at 0x000002389C865A00>, <matplotlib.axis.XTick object at 0x000002389C8F3F70>, <matplotlib.axis.XTick object at 0x000002389C7D1C70>, <matplotlib.axis.XTick object at 0x000002389C7D1340>, <matplotlib.axis.XTick object at 0x000002389C7D1D30>, <matplotlib.axis.XTick object at 0x000002389C7D1910>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C6F3B50>
7
<matplotlib.contour.QuadContourSet object at 0x000002389BD632E0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C6CC790>, <matplotlib.axis.YTick object at 0x000002389C67A3D0>, <matplotlib.axis.YTick object at 0x000002389BD82460>, <matplotlib.axis.YTick object at 0x000002389C6F9880>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C77C610>, <matplotlib.axis.XTick object at 0x000002389C6B3730>, <matplotlib.axis.XTick object at 0x000002389C6F9A00>, <matplotlib.axis.XTick object at 0x000002389C6F98E0>, <matplotlib.axis.XTick object at 0x000002389C57FD90>, <matplotlib.axis.XTick object at 0x000002389C57F820>, <matplotlib.axis.XTick object at 0x000002389C57F2B0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002385C14E490>
8
<matplotlib.contour.QuadContourSet object at 0x000002389C7005B0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C5776D0>, <matplotlib.axis.YTick object at 0x000002385C114370>, <matplotlib.axis.YTick object at 0x000002389C918040>, <matplotlib.axis.YTick object at 0x000002389C918400>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C577FD0>, <matplotlib.axis.XTick object at 0x000002389C565E20>, <matplotlib.axis.XTick object at 0x000002389C918790>, <matplotlib.axis.XTick object at 0x000002389C9310D0>, <matplotlib.axis.XTick object at 0x000002389C931490>, <matplotlib.axis.XTick object at 0x000002389C9319A0>, <matplotlib.axis.XTick object at 0x000002389C931E20>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C00DA60>
9
<matplotlib.contour.QuadContourSet object at 0x000002389C85CAF0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C6B1910>, <matplotlib.axis.YTick object at 0x000002389C85C7C0>, <matplotlib.axis.YTick object at 0x000002389C012220>, <matplotlib.axis.YTick object at 0x000002389C012700>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C026E50>, <matplotlib.axis.XTick object at 0x000002389C8430D0>, <matplotlib.axis.XTick object at 0x000002389BD7D520>, <matplotlib.axis.XTick object at 0x000002389C012190>, <matplotlib.axis.XTick object at 0x000002389BD7D340>, <matplotlib.axis.XTick object at 0x000002389BD7DFA0>, <matplotlib.axis.XTick object at 0x000002389C575340>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C57F880>
10
<matplotlib.contour.QuadContourSet object at 0x000002389C6B3AF0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C6D50D0>, <matplotlib.axis.YTick object at 0x000002389C6CCC70>, <matplotlib.axis.YTick object at 0x000002389CB62550>, <matplotlib.axis.YTick object at 0x000002389C67A850>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C6D5E50>, <matplotlib.axis.XTick object at 0x000002389C927A90>, <matplotlib.axis.XTick object at 0x000002389CB625B0>, <matplotlib.axis.XTick object at 0x000002389C67AD90>, <matplotlib.axis.XTick object at 0x000002389CB68190>, <matplotlib.axis.XTick object at 0x000002389CB68BB0>, <matplotlib.axis.XTick object at 0x000002389CB68EE0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C8221C0>
11
<matplotlib.contour.QuadContourSet object at 0x000002389C0293A0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389CB18F10>, <matplotlib.axis.YTick object at 0x000002389C023190>, <matplotlib.axis.YTick object at 0x000002389C76B850>, <matplotlib.axis.YTick object at 0x000002389C76BC70>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389CB18490>, <matplotlib.axis.XTick object at 0x000002389C023A30>, <matplotlib.axis.XTick object at 0x000002389C76B100>, <matplotlib.axis.XTick object at 0x000002389CF75730>, <matplotlib.axis.XTick object at 0x000002389CF75670>, <matplotlib.axis.XTick object at 0x000002389C020550>, <matplotlib.axis.XTick object at 0x000002389C0205E0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C8165E0>
12
<matplotlib.contour.QuadContourSet object at 0x000002389BD9C280>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C841E50>, <matplotlib.axis.YTick object at 0x000002389C970550>, <matplotlib.axis.YTick object at 0x000002389C9A8B80>, <matplotlib.axis.YTick object at 0x000002389C9A20D0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C9832E0>, <matplotlib.axis.XTick object at 0x000002389CB26CD0>, <matplotlib.axis.XTick object at 0x000002389C9A8A90>, <matplotlib.axis.XTick object at 0x000002389C9A23A0>, <matplotlib.axis.XTick object at 0x000002389C998160>, <matplotlib.axis.XTick object at 0x000002389C998670>, <matplotlib.axis.XTick object at 0x000002389C998B80>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C9AF370>
13
<matplotlib.contour.QuadContourSet object at 0x000002389CB4C370>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C681EE0>, <matplotlib.axis.YTick object at 0x000002389CB185E0>, <matplotlib.axis.YTick object at 0x000002389C95F460>, <matplotlib.axis.YTick object at 0x000002389C9CA2B0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C99D220>, <matplotlib.axis.XTick object at 0x000002389CB223D0>, <matplotlib.axis.XTick object at 0x000002389C95FB50>, <matplotlib.axis.XTick object at 0x000002389C9CA7F0>, <matplotlib.axis.XTick object at 0x000002389C77E6D0>, <matplotlib.axis.XTick object at 0x000002389C77E910>, <matplotlib.axis.XTick object at 0x000002389C77EC10>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C822700>
14
<matplotlib.contour.QuadContourSet object at 0x000002389C57F910>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C6F39D0>, <matplotlib.axis.YTick object at 0x000002389C6F92E0>, <matplotlib.axis.YTick object at 0x000002389CF75160>, <matplotlib.axis.YTick object at 0x000002389CF75E50>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C6F3670>, <matplotlib.axis.XTick object at 0x000002389C01A1C0>, <matplotlib.axis.XTick object at 0x000002385C114A00>, <matplotlib.axis.XTick object at 0x000002389C715550>, <matplotlib.axis.XTick object at 0x000002389C06CEB0>, <matplotlib.axis.XTick object at 0x000002389C06CE80>, <matplotlib.axis.XTick object at 0x000002389C06CFA0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389CB626A0>
15
<matplotlib.contour.QuadContourSet object at 0x000002389C6D54F0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C6C80D0>, <matplotlib.axis.YTick object at 0x000002389C628070>, <matplotlib.axis.YTick object at 0x000002389C00C490>, <matplotlib.axis.YTick object at 0x000002389C00C9A0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C6C8DC0>, <matplotlib.axis.XTick object at 0x000002389C565C40>, <matplotlib.axis.XTick object at 0x000002389C00C880>, <matplotlib.axis.XTick object at 0x000002389C6F24C0>, <matplotlib.axis.XTick object at 0x000002389C6F2A30>, <matplotlib.axis.XTick object at 0x000002389C6F10D0>, <matplotlib.axis.XTick object at 0x000002389C6F1490>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C6DDC40>
16
<matplotlib.contour.QuadContourSet object at 0x000002389CA13400>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C6F1850>, <matplotlib.axis.YTick object at 0x000002389CA010D0>, <matplotlib.axis.YTick object at 0x000002389CB62D60>, <matplotlib.axis.YTick object at 0x000002389CB62820>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C6C7CD0>, <matplotlib.axis.XTick object at 0x000002389CB623A0>, <matplotlib.axis.XTick object at 0x000002389C565880>, <matplotlib.axis.XTick object at 0x000002389CB62F40>, <matplotlib.axis.XTick object at 0x000002389C565AF0>, <matplotlib.axis.XTick object at 0x000002389C565100>, <matplotlib.axis.XTick object at 0x000002389BD82130>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C012100>
17
<matplotlib.contour.QuadContourSet object at 0x000002389C023AF0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C8F6460>, <matplotlib.axis.YTick object at 0x000002389C806790>, <matplotlib.axis.YTick object at 0x000002389C846250>, <matplotlib.axis.YTick object at 0x000002389C846A00>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C00D370>, <matplotlib.axis.XTick object at 0x000002389C01A880>, <matplotlib.axis.XTick object at 0x000002389C846E80>, <matplotlib.axis.XTick object at 0x000002385C14E910>, <matplotlib.axis.XTick object at 0x000002385C14EAC0>, <matplotlib.axis.XTick object at 0x000002389C7E26D0>, <matplotlib.axis.XTick object at 0x000002389C7E24C0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C80C8E0>
18
<matplotlib.contour.QuadContourSet object at 0x000002389C67AEB0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C867FD0>, <matplotlib.axis.YTick object at 0x000002389C9C0310>, <matplotlib.axis.YTick object at 0x000002389CF7B400>, <matplotlib.axis.YTick object at 0x000002389CF7B880>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C867BE0>, <matplotlib.axis.XTick object at 0x000002389C57F4C0>, <matplotlib.axis.XTick object at 0x000002389CF7B730>, <matplotlib.axis.XTick object at 0x000002389CF6A0D0>, <matplotlib.axis.XTick object at 0x000002389C545D30>, <matplotlib.axis.XTick object at 0x000002389C8973D0>, <matplotlib.axis.XTick object at 0x000002389C897A60>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C936EB0>
19
<matplotlib.contour.QuadContourSet object at 0x000002389CB414C0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C90E250>, <matplotlib.axis.YTick object at 0x000002389C96C430>, <matplotlib.axis.YTick object at 0x000002389C81BDF0>, <matplotlib.axis.YTick object at 0x000002389C8E5340>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C90E910>, <matplotlib.axis.XTick object at 0x000002389C8DF8E0>, <matplotlib.axis.XTick object at 0x000002389C8E5E50>, <matplotlib.axis.XTick object at 0x000002389C8E50D0>, <matplotlib.axis.XTick object at 0x000002389C8093D0>, <matplotlib.axis.XTick object at 0x000002389C8098E0>, <matplotlib.axis.XTick object at 0x000002389C809DF0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C7B25E0>
20
<matplotlib.contour.QuadContourSet object at 0x000002389C67A610>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C07CEB0>, <matplotlib.axis.YTick object at 0x000002389CB2C5B0>, <matplotlib.axis.YTick object at 0x000002389C6B3E20>, <matplotlib.axis.YTick object at 0x000002389C747130>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C06E700>, <matplotlib.axis.XTick object at 0x000002389C897C10>, <matplotlib.axis.XTick object at 0x000002389C6B3DC0>, <matplotlib.axis.XTick object at 0x000002389CB22BE0>, <matplotlib.axis.XTick object at 0x000002389CB222E0>, <matplotlib.axis.XTick object at 0x000002389CB22BB0>, <matplotlib.axis.XTick object at 0x000002385C14E8B0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C9CAEE0>
21
<matplotlib.contour.QuadContourSet object at 0x000002389C012700>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389CB18910>, <matplotlib.axis.YTick object at 0x000002389CB62BE0>, <matplotlib.axis.YTick object at 0x000002389CB68310>, <matplotlib.axis.YTick object at 0x000002389CB686D0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C77EE50>, <matplotlib.axis.XTick object at 0x000002389C76B220>, <matplotlib.axis.XTick object at 0x000002389CB68F40>, <matplotlib.axis.XTick object at 0x000002389C506580>, <matplotlib.axis.XTick object at 0x000002389BD83AC0>, <matplotlib.axis.XTick object at 0x000002389BD831F0>, <matplotlib.axis.XTick object at 0x000002389BD83760>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389CA164C0>
22
<matplotlib.contour.QuadContourSet object at 0x000002389CF9A580>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C6F33D0>, <matplotlib.axis.YTick object at 0x000002389C9049A0>, <matplotlib.axis.YTick object at 0x000002389C838C40>, <matplotlib.axis.YTick object at 0x000002389C8381C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389CB55160>, <matplotlib.axis.XTick object at 0x000002389C506220>, <matplotlib.axis.XTick object at 0x000002389C838850>, <matplotlib.axis.XTick object at 0x000002389C8E32E0>, <matplotlib.axis.XTick object at 0x000002389CB196D0>, <matplotlib.axis.XTick object at 0x000002389CB19A90>, <matplotlib.axis.XTick object at 0x000002389CB19C40>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C942910>
23
<matplotlib.contour.QuadContourSet object at 0x000002389C6D5B20>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C9F7490>, <matplotlib.axis.YTick object at 0x000002389CB26550>, <matplotlib.axis.YTick object at 0x000002389CB26370>, <matplotlib.axis.YTick object at 0x000002389C8766A0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C7003D0>, <matplotlib.axis.XTick object at 0x000002389C6D5430>, <matplotlib.axis.XTick object at 0x000002389C876640>, <matplotlib.axis.XTick object at 0x000002389C876220>, <matplotlib.axis.XTick object at 0x000002389CB26B50>, <matplotlib.axis.XTick object at 0x000002389C681520>, <matplotlib.axis.XTick object at 0x000002389C681790>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C8F6520>
24
<matplotlib.contour.QuadContourSet object at 0x000002389CF7B940>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389CB222E0>, <matplotlib.axis.YTick object at 0x000002389C6EDA60>, <matplotlib.axis.YTick object at 0x000002389C5063A0>, <matplotlib.axis.YTick object at 0x000002389C506160>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389CB22820>, <matplotlib.axis.XTick object at 0x000002389C9E1160>, <matplotlib.axis.XTick object at 0x000002389C5069D0>, <matplotlib.axis.XTick object at 0x000002389C577D00>, <matplotlib.axis.XTick object at 0x000002389C5770A0>, <matplotlib.axis.XTick object at 0x000002389C577B50>, <matplotlib.axis.XTick object at 0x000002389C816520>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389CA169D0>
25
<matplotlib.contour.QuadContourSet object at 0x000002389C76B280>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C57FF40>, <matplotlib.axis.YTick object at 0x000002389C8970D0>, <matplotlib.axis.YTick object at 0x000002389C846BB0>, <matplotlib.axis.YTick object at 0x000002389C846880>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C57FAC0>, <matplotlib.axis.XTick object at 0x000002389C545D30>, <matplotlib.axis.XTick object at 0x000002389C846D30>, <matplotlib.axis.XTick object at 0x000002389C77A370>, <matplotlib.axis.XTick object at 0x000002389C77AA00>, <matplotlib.axis.XTick object at 0x000002389C77A3A0>, <matplotlib.axis.XTick object at 0x000002389C07BCA0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C896EE0>
26
<matplotlib.contour.QuadContourSet object at 0x000002389CB42400>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C034EE0>, <matplotlib.axis.YTick object at 0x000002389C9530D0>, <matplotlib.axis.YTick object at 0x000002389C7EFD30>, <matplotlib.axis.YTick object at 0x000002389C7EA280>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C686640>, <matplotlib.axis.XTick object at 0x000002389C0388B0>, <matplotlib.axis.XTick object at 0x000002389C7EAA90>, <matplotlib.axis.XTick object at 0x000002389C7EA190>, <matplotlib.axis.XTick object at 0x000002389C81A310>, <matplotlib.axis.XTick object at 0x000002389C81A820>, <matplotlib.axis.XTick object at 0x000002389C81AD30>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C7CD550>
27
<matplotlib.contour.QuadContourSet object at 0x000002389C77E7F0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C67AA90>, <matplotlib.axis.YTick object at 0x000002389C5820D0>, <matplotlib.axis.YTick object at 0x000002389C9659D0>, <matplotlib.axis.YTick object at 0x000002389C9653A0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C9CAE20>, <matplotlib.axis.XTick object at 0x000002389C545E20>, <matplotlib.axis.XTick object at 0x000002389C965D30>, <matplotlib.axis.XTick object at 0x000002389BD83310>, <matplotlib.axis.XTick object at 0x000002389BD830D0>, <matplotlib.axis.XTick object at 0x000002389BD83550>, <matplotlib.axis.XTick object at 0x000002389BD7D670>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C876C70>
28
<matplotlib.contour.QuadContourSet object at 0x000002389CA01C40>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389BD82520>, <matplotlib.axis.YTick object at 0x000002389C88BAF0>, <matplotlib.axis.YTick object at 0x000002389C8E3670>, <matplotlib.axis.YTick object at 0x000002389C8E34C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C8961F0>, <matplotlib.axis.XTick object at 0x000002389C870550>, <matplotlib.axis.XTick object at 0x000002389CA07880>, <matplotlib.axis.XTick object at 0x000002389CA07430>, <matplotlib.axis.XTick object at 0x000002389CA07C40>, <matplotlib.axis.XTick object at 0x000002389C6FBE20>, <matplotlib.axis.XTick object at 0x000002389C6FB250>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C541D90>
29
<matplotlib.contour.QuadContourSet object at 0x000002389C07CF70>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C506BE0>, <matplotlib.axis.YTick object at 0x000002389C88BBE0>, <matplotlib.axis.YTick object at 0x000002389CB72670>, <matplotlib.axis.YTick object at 0x000002389C968DF0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C506A00>, <matplotlib.axis.XTick object at 0x000002389C897460>, <matplotlib.axis.XTick object at 0x000002389CB72940>, <matplotlib.axis.XTick object at 0x000002389C56E970>, <matplotlib.axis.XTick object at 0x000002389CB5B5B0>, <matplotlib.axis.XTick object at 0x000002389CB5B3A0>, <matplotlib.axis.XTick object at 0x000002389CB5BE80>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C77C430>
>>> f_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> solu2=np.zeros(shape = (Nv, 2*Nv))
>>> ff_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        f_1[j*2*Nv+i]=Kappa_Initial_Strahl(pal_v[i],per_v[j])#+Kappa_Initial_Halo(pal_v[i],per_v[j])#+Kappa_Initial_Strahl(pal_v[i],per_v[j])#Initial_Core(pal_v[i],per_v[j])+Kappa_Initial_Halo(pal_v[i],per_v[j])+Kappa_Initial_Strahl2(pal_v[i],per_v[j])+Kappa_Initial_Strahl(pal_v[i],per_v[j])

>>> for j in range(2*Nv):
    for i in range(2*Nv):
        fc_1[j*2*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j])

>>> ff_1=f_1+fc_1
>>> Mf_1=np.max(ff_1)
>>> per_v2 = np.linspace(0, Mv, Nv)
>>> for k in range(20): #Numer in range indicates the minute.
    print(k)
    #ff_1=f_1+fc_1
    for j in range(Nv):
        for i in range(2*Nv):
        #solu[j,i]=(abs(f_1[j*2*Nv+i])/Mf_1)
            if abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>1:
                solu2[j,i]=0
            elif abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>10**(-6):
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
    for t in range(5):
        ff_1=dot(AQ, ff_1)

        
0
<matplotlib.contour.QuadContourSet object at 0x000002389C9CECA0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C9B2130>, <matplotlib.axis.YTick object at 0x000002389CF7BD30>, <matplotlib.axis.YTick object at 0x000002389CF7BF70>, <matplotlib.axis.YTick object at 0x000002389CF7B970>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C9C6F70>, <matplotlib.axis.XTick object at 0x000002389CA21AF0>, <matplotlib.axis.XTick object at 0x000002389C6CC400>, <matplotlib.axis.XTick object at 0x000002389CF7BFA0>, <matplotlib.axis.XTick object at 0x000002389C6CC820>, <matplotlib.axis.XTick object at 0x000002389C6CCA60>, <matplotlib.axis.XTick object at 0x000002389C577D60>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C91B580>
1
<matplotlib.contour.QuadContourSet object at 0x000002389C7B2700>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C8E54C0>, <matplotlib.axis.YTick object at 0x000002389C77E250>, <matplotlib.axis.YTick object at 0x000002389C6DCF70>, <matplotlib.axis.YTick object at 0x000002389C6DCA90>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C71A5E0>, <matplotlib.axis.XTick object at 0x000002389BD82BB0>, <matplotlib.axis.XTick object at 0x000002389C6DC880>, <matplotlib.axis.XTick object at 0x000002389C897790>, <matplotlib.axis.XTick object at 0x000002389C897DF0>, <matplotlib.axis.XTick object at 0x000002389CB11370>, <matplotlib.axis.XTick object at 0x000002389CB119A0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C88B340>
2
<matplotlib.contour.QuadContourSet object at 0x000002389C7CDC10>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C545E80>, <matplotlib.axis.YTick object at 0x000002389C541EE0>, <matplotlib.axis.YTick object at 0x000002389C715F40>, <matplotlib.axis.YTick object at 0x000002389C8F3610>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C6BFDF0>, <matplotlib.axis.XTick object at 0x000002389BD7D970>, <matplotlib.axis.XTick object at 0x000002389C8F3A60>, <matplotlib.axis.XTick object at 0x000002389C7158E0>, <matplotlib.axis.XTick object at 0x000002389CB6BF40>, <matplotlib.axis.XTick object at 0x000002389CB6B1C0>, <matplotlib.axis.XTick object at 0x000002389CB6BE50>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C0207C0>
3
<matplotlib.contour.QuadContourSet object at 0x000002389C6B7DC0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C9537C0>, <matplotlib.axis.YTick object at 0x000002389BD8C880>, <matplotlib.axis.YTick object at 0x000002389C6AB550>, <matplotlib.axis.YTick object at 0x000002389C6ABA60>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C6C9490>, <matplotlib.axis.XTick object at 0x000002389CB5F6A0>, <matplotlib.axis.XTick object at 0x000002389C6AB940>, <matplotlib.axis.XTick object at 0x000002389C81B5E0>, <matplotlib.axis.XTick object at 0x000002389C81BAF0>, <matplotlib.axis.XTick object at 0x000002389C8020A0>, <matplotlib.axis.XTick object at 0x000002389C802550>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C85CD00>
4
<matplotlib.contour.QuadContourSet object at 0x000002389CB8CCD0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C979310>, <matplotlib.axis.YTick object at 0x000002389C7F2850>, <matplotlib.axis.YTick object at 0x000002389C6DC6A0>, <matplotlib.axis.YTick object at 0x000002389C6DCB50>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C8F3460>, <matplotlib.axis.XTick object at 0x000002389C56D2B0>, <matplotlib.axis.XTick object at 0x000002389C6DCEE0>, <matplotlib.axis.XTick object at 0x000002389C8E30A0>, <matplotlib.axis.XTick object at 0x000002389C8E39D0>, <matplotlib.axis.XTick object at 0x000002389C8E3280>, <matplotlib.axis.XTick object at 0x000002389C9E9CD0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C8E5AC0>
5
<matplotlib.contour.QuadContourSet object at 0x000002389CB68BB0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C6B13A0>, <matplotlib.axis.YTick object at 0x000002389C80D760>, <matplotlib.axis.YTick object at 0x000002389C70BC40>, <matplotlib.axis.YTick object at 0x000002389C70B4C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C77ED00>, <matplotlib.axis.XTick object at 0x000002389C846910>, <matplotlib.axis.XTick object at 0x000002389C70BF10>, <matplotlib.axis.XTick object at 0x000002389CB18940>, <matplotlib.axis.XTick object at 0x000002389CB18970>, <matplotlib.axis.XTick object at 0x000002389CB18520>, <matplotlib.axis.XTick object at 0x000002389C681850>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C876C40>
6
<matplotlib.contour.QuadContourSet object at 0x000002389C577BE0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C88B5E0>, <matplotlib.axis.YTick object at 0x000002389CB55820>, <matplotlib.axis.YTick object at 0x000002389C012520>, <matplotlib.axis.YTick object at 0x000002389C012400>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C968A60>, <matplotlib.axis.XTick object at 0x000002389C6B1A30>, <matplotlib.axis.XTick object at 0x000002389C012C10>, <matplotlib.axis.XTick object at 0x000002389CB877F0>, <matplotlib.axis.XTick object at 0x000002389CB874F0>, <matplotlib.axis.XTick object at 0x000002389C877A60>, <matplotlib.axis.XTick object at 0x000002389C877940>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389CF9A3A0>
7
<matplotlib.contour.QuadContourSet object at 0x000002389C565BE0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389CF9A1C0>, <matplotlib.axis.YTick object at 0x000002389C581FA0>, <matplotlib.axis.YTick object at 0x000002389CF7BC10>, <matplotlib.axis.YTick object at 0x000002389C6CCCA0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C012D90>, <matplotlib.axis.XTick object at 0x000002389C565B50>, <matplotlib.axis.XTick object at 0x000002389C6CC0D0>, <matplotlib.axis.XTick object at 0x000002389C6CCA30>, <matplotlib.axis.XTick object at 0x000002389CF7B670>, <matplotlib.axis.XTick object at 0x000002389C8060D0>, <matplotlib.axis.XTick object at 0x000002389C8069D0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389CB551C0>
8
<matplotlib.contour.QuadContourSet object at 0x000002389C6B3400>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C840400>, <matplotlib.axis.YTick object at 0x000002389C6DCC40>, <matplotlib.axis.YTick object at 0x000002389C67A910>, <matplotlib.axis.YTick object at 0x000002389C67A2E0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C8401C0>, <matplotlib.axis.XTick object at 0x000002389C038490>, <matplotlib.axis.XTick object at 0x000002389C67A850>, <matplotlib.axis.XTick object at 0x000002389C68EEE0>, <matplotlib.axis.XTick object at 0x000002389C68E580>, <matplotlib.axis.XTick object at 0x000002389C7F2160>, <matplotlib.axis.XTick object at 0x000002389C7F2BE0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C8763A0>
9
<matplotlib.contour.QuadContourSet object at 0x000002389C979A90>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C6C9760>, <matplotlib.axis.YTick object at 0x000002389C9CA7C0>, <matplotlib.axis.YTick object at 0x000002389C768610>, <matplotlib.axis.YTick object at 0x000002389C768400>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C6C9370>, <matplotlib.axis.XTick object at 0x000002389CF6AFA0>, <matplotlib.axis.XTick object at 0x000002389C768040>, <matplotlib.axis.XTick object at 0x000002389C98CBE0>, <matplotlib.axis.XTick object at 0x000002389C98CA90>, <matplotlib.axis.XTick object at 0x000002389C6F9B50>, <matplotlib.axis.XTick object at 0x000002389C6F9700>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C580970>
10
<matplotlib.contour.QuadContourSet object at 0x000002389C07E0D0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C7EFFD0>, <matplotlib.axis.YTick object at 0x000002389C6F9370>, <matplotlib.axis.YTick object at 0x000002389C897EB0>, <matplotlib.axis.YTick object at 0x000002389C863070>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C958190>, <matplotlib.axis.XTick object at 0x000002389C976F40>, <matplotlib.axis.XTick object at 0x000002389C897550>, <matplotlib.axis.XTick object at 0x000002389C8630D0>, <matplotlib.axis.XTick object at 0x000002389C87D040>, <matplotlib.axis.XTick object at 0x000002389C87DD90>, <matplotlib.axis.XTick object at 0x000002389C87D9D0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C79C1C0>
11
<matplotlib.contour.QuadContourSet object at 0x000002389C9CA370>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C07B040>, <matplotlib.axis.YTick object at 0x000002389BD7C550>, <matplotlib.axis.YTick object at 0x000002389C876070>, <matplotlib.axis.YTick object at 0x000002389C876C10>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C7F4820>, <matplotlib.axis.XTick object at 0x000002389C9586D0>, <matplotlib.axis.XTick object at 0x000002389C8760A0>, <matplotlib.axis.XTick object at 0x000002389C8E1C10>, <matplotlib.axis.XTick object at 0x000002389C8E1910>, <matplotlib.axis.XTick object at 0x000002389C8E1340>, <matplotlib.axis.XTick object at 0x000002389C6E7FD0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C76AB50>
12
<matplotlib.contour.QuadContourSet object at 0x000002389CB5F610>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C6AB940>, <matplotlib.axis.YTick object at 0x000002389C6C9460>, <matplotlib.axis.YTick object at 0x000002389C9CF370>, <matplotlib.axis.YTick object at 0x000002389C9CFFA0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C6AB250>, <matplotlib.axis.XTick object at 0x000002389C069970>, <matplotlib.axis.XTick object at 0x000002389C02BFD0>, <matplotlib.axis.XTick object at 0x000002389C654B20>, <matplotlib.axis.XTick object at 0x000002389C654CD0>, <matplotlib.axis.XTick object at 0x000002389C834C40>, <matplotlib.axis.XTick object at 0x000002389C834730>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C5751C0>
13
<matplotlib.contour.QuadContourSet object at 0x000002389C7E25E0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C7F24C0>, <matplotlib.axis.YTick object at 0x000002389C6B1B20>, <matplotlib.axis.YTick object at 0x000002389C9C6700>, <matplotlib.axis.YTick object at 0x000002389CB551F0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C012700>, <matplotlib.axis.XTick object at 0x000002385C114400>, <matplotlib.axis.XTick object at 0x000002389CB556D0>, <matplotlib.axis.XTick object at 0x000002389CB55970>, <matplotlib.axis.XTick object at 0x000002389C989880>, <matplotlib.axis.XTick object at 0x000002389C989C10>, <matplotlib.axis.XTick object at 0x000002389C838C70>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C676FA0>
14
<matplotlib.contour.QuadContourSet object at 0x000002389CA0F820>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C7DC6A0>, <matplotlib.axis.YTick object at 0x000002389C7DC490>, <matplotlib.axis.YTick object at 0x000002389CF7A6D0>, <matplotlib.axis.YTick object at 0x000002389CF7A400>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C7D0FD0>, <matplotlib.axis.XTick object at 0x000002389C9B3FA0>, <matplotlib.axis.XTick object at 0x000002389C06CA60>, <matplotlib.axis.XTick object at 0x000002389CF7AAF0>, <matplotlib.axis.XTick object at 0x000002389C06C970>, <matplotlib.axis.XTick object at 0x000002389C06C730>, <matplotlib.axis.XTick object at 0x000002389C6F3E20>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C88B520>
15
<matplotlib.contour.QuadContourSet object at 0x000002389C6E48B0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C768B80>, <matplotlib.axis.YTick object at 0x000002389C07B730>, <matplotlib.axis.YTick object at 0x000002389C840EE0>, <matplotlib.axis.YTick object at 0x000002389C6BFA30>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C6DCD00>, <matplotlib.axis.XTick object at 0x000002389C8E13A0>, <matplotlib.axis.XTick object at 0x000002389C840070>, <matplotlib.axis.XTick object at 0x000002389C6BF490>, <matplotlib.axis.XTick object at 0x000002389C541EE0>, <matplotlib.axis.XTick object at 0x000002389CB26730>, <matplotlib.axis.XTick object at 0x000002389CB261C0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C870AF0>
16
<matplotlib.contour.QuadContourSet object at 0x000002389C953DC0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389CB22F40>, <matplotlib.axis.YTick object at 0x000002389C83C130>, <matplotlib.axis.YTick object at 0x000002389BD7C160>, <matplotlib.axis.YTick object at 0x000002389BD7C820>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389CB372E0>, <matplotlib.axis.XTick object at 0x000002389C57EB20>, <matplotlib.axis.XTick object at 0x000002389BD7C370>, <matplotlib.axis.XTick object at 0x000002389C968A60>, <matplotlib.axis.XTick object at 0x000002389C6AB4C0>, <matplotlib.axis.XTick object at 0x000002389C6ABD30>, <matplotlib.axis.XTick object at 0x000002389C6ABF70>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C575910>
17
<matplotlib.contour.QuadContourSet object at 0x000002389C6D1A60>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C00BD90>, <matplotlib.axis.YTick object at 0x000002389C7F4A30>, <matplotlib.axis.YTick object at 0x000002389C5664F0>, <matplotlib.axis.YTick object at 0x000002389C566A00>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389CB11940>, <matplotlib.axis.XTick object at 0x000002389C7C8100>, <matplotlib.axis.XTick object at 0x000002389C566370>, <matplotlib.axis.XTick object at 0x000002389C987580>, <matplotlib.axis.XTick object at 0x000002389C987A90>, <matplotlib.axis.XTick object at 0x000002389CB88100>, <matplotlib.axis.XTick object at 0x000002389CB884F0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389CB2EC10>
18
<matplotlib.contour.QuadContourSet object at 0x000002389C768CD0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C8F3B20>, <matplotlib.axis.YTick object at 0x000002389C628D30>, <matplotlib.axis.YTick object at 0x000002389C8713A0>, <matplotlib.axis.YTick object at 0x000002389C871EE0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C8F3EB0>, <matplotlib.axis.XTick object at 0x000002389CB116A0>, <matplotlib.axis.XTick object at 0x000002389C871580>, <matplotlib.axis.XTick object at 0x000002389C506370>, <matplotlib.axis.XTick object at 0x000002389C506D30>, <matplotlib.axis.XTick object at 0x000002389C6CCBE0>, <matplotlib.axis.XTick object at 0x000002389C8E1D90>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C7C8340>
19
<matplotlib.contour.QuadContourSet object at 0x000002389CB187C0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C6817F0>, <matplotlib.axis.YTick object at 0x000002389CB621F0>, <matplotlib.axis.YTick object at 0x000002389C8066A0>, <matplotlib.axis.YTick object at 0x000002389C807B80>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C681BE0>, <matplotlib.axis.XTick object at 0x000002389C79CEB0>, <matplotlib.axis.XTick object at 0x000002389C807970>, <matplotlib.axis.XTick object at 0x000002389CB22640>, <matplotlib.axis.XTick object at 0x000002389C7F28B0>, <matplotlib.axis.XTick object at 0x000002389C7F2F70>, <matplotlib.axis.XTick object at 0x000002389C7F2880>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C9AFFD0>
>>> f_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> solu2=np.zeros(shape = (Nv, 2*Nv))
>>> ff_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
>>> for j in range(2*Nv):
    for i in range(2*Nv):
        f_1[j*2*Nv+i]=Kappa_Initial_Strahl(pal_v[i],per_v[j])#+Kappa_Initial_Halo(pal_v[i],per_v[j])#+Kappa_Initial_Strahl(pal_v[i],per_v[j])#Initial_Core(pal_v[i],per_v[j])+Kappa_Initial_Halo(pal_v[i],per_v[j])+Kappa_Initial_Strahl2(pal_v[i],per_v[j])+Kappa_Initial_Strahl(pal_v[i],per_v[j])

>>> for j in range(2*Nv):
    for i in range(2*Nv):
        fc_1[j*2*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j])

>>> ff_1=f_1+fc_1
>>> Mf_1=np.max(ff_1)
>>> for k in range(30): #Numer in range indicates the minute.
    print(k)
    #ff_1=f_1+fc_1
    for j in range(Nv):
        for i in range(2*Nv):
        #solu[j,i]=(abs(f_1[j*2*Nv+i])/Mf_1)
            if abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>1:
                solu2[j,i]=0
            elif abs(ff_1[(j+Nv)*2*Nv+i])/Mf_1>10**(-6):
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
    for t in range(1):
        ff_1=dot(AQ, ff_1)

        
0
<matplotlib.contour.QuadContourSet object at 0x000002389CB5BD00>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C9AC370>, <matplotlib.axis.YTick object at 0x000002389C682E20>, <matplotlib.axis.YTick object at 0x000002389C6B3190>, <matplotlib.axis.YTick object at 0x000002389C67A940>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002385C114700>, <matplotlib.axis.XTick object at 0x000002389C7D0BE0>, <matplotlib.axis.XTick object at 0x000002389C67ADF0>, <matplotlib.axis.XTick object at 0x000002389C67A7C0>, <matplotlib.axis.XTick object at 0x000002389CA08B50>, <matplotlib.axis.XTick object at 0x000002389CA08D00>, <matplotlib.axis.XTick object at 0x000002389CA08940>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C6FFF40>
1
<matplotlib.contour.QuadContourSet object at 0x000002389C6E5B80>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C075190>, <matplotlib.axis.YTick object at 0x000002389C075CA0>, <matplotlib.axis.YTick object at 0x000002389C9CA580>, <matplotlib.axis.YTick object at 0x000002389C6DC550>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C87AA00>, <matplotlib.axis.XTick object at 0x000002389C6DCB80>, <matplotlib.axis.XTick object at 0x000002389C6DC430>, <matplotlib.axis.XTick object at 0x000002389C6DCA00>, <matplotlib.axis.XTick object at 0x000002389C9CA880>, <matplotlib.axis.XTick object at 0x000002389C806340>, <matplotlib.axis.XTick object at 0x000002389C806550>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389CB26940>
2
<matplotlib.contour.QuadContourSet object at 0x000002389C846640>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C953DC0>, <matplotlib.axis.YTick object at 0x000002389C70B100>, <matplotlib.axis.YTick object at 0x000002389C0693A0>, <matplotlib.axis.YTick object at 0x000002389C069E50>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C871DF0>, <matplotlib.axis.XTick object at 0x000002389C9AF5B0>, <matplotlib.axis.XTick object at 0x000002389C069490>, <matplotlib.axis.XTick object at 0x000002389C5753D0>, <matplotlib.axis.XTick object at 0x000002389C5756D0>, <matplotlib.axis.XTick object at 0x000002389BD9D310>, <matplotlib.axis.XTick object at 0x000002389BD9D760>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389CB32C40>
3
<matplotlib.contour.QuadContourSet object at 0x000002389CB116D0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389CB1DCA0>, <matplotlib.axis.YTick object at 0x000002389C00B7F0>, <matplotlib.axis.YTick object at 0x000002389C506970>, <matplotlib.axis.YTick object at 0x000002389C506460>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C70BB50>, <matplotlib.axis.XTick object at 0x000002389CA0FB80>, <matplotlib.axis.XTick object at 0x000002389C506070>, <matplotlib.axis.XTick object at 0x000002389C715FA0>, <matplotlib.axis.XTick object at 0x000002389C715190>, <matplotlib.axis.XTick object at 0x000002389C07B9D0>, <matplotlib.axis.XTick object at 0x000002389C07B4F0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C88B310>
4
<matplotlib.contour.QuadContourSet object at 0x000002389CA35B50>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C70B370>, <matplotlib.axis.YTick object at 0x000002389C054DF0>, <matplotlib.axis.YTick object at 0x000002389C7662E0>, <matplotlib.axis.YTick object at 0x000002389C7667F0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C76A3A0>, <matplotlib.axis.XTick object at 0x000002389CB6CCD0>, <matplotlib.axis.XTick object at 0x000002389C766790>, <matplotlib.axis.XTick object at 0x000002389CB42340>, <matplotlib.axis.XTick object at 0x000002389CB42880>, <matplotlib.axis.XTick object at 0x000002389CB42D90>, <matplotlib.axis.XTick object at 0x000002389CB412E0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389CB8BA90>
5
<matplotlib.contour.QuadContourSet object at 0x000002389C871BE0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389CB684C0>, <matplotlib.axis.YTick object at 0x000002389C6BF1C0>, <matplotlib.axis.YTick object at 0x000002389C545F10>, <matplotlib.axis.YTick object at 0x000002389CB557C0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C870940>, <matplotlib.axis.XTick object at 0x000002389C768820>, <matplotlib.axis.XTick object at 0x000002389CB551C0>, <matplotlib.axis.XTick object at 0x000002389CB55400>, <matplotlib.axis.XTick object at 0x000002389C838C40>, <matplotlib.axis.XTick object at 0x000002389C838BB0>, <matplotlib.axis.XTick object at 0x000002389C838D00>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389CF8FFA0>
6
<matplotlib.contour.QuadContourSet object at 0x000002389C628A60>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C57E460>, <matplotlib.axis.YTick object at 0x000002389C71A580>, <matplotlib.axis.YTick object at 0x000002389C6F0880>, <matplotlib.axis.YTick object at 0x000002389C6F0550>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C57E340>, <matplotlib.axis.XTick object at 0x000002389C76AF70>, <matplotlib.axis.XTick object at 0x000002389C6F0E50>, <matplotlib.axis.XTick object at 0x000002389C6FA0D0>, <matplotlib.axis.XTick object at 0x000002389C6FAF70>, <matplotlib.axis.XTick object at 0x000002389C6FA730>, <matplotlib.axis.XTick object at 0x000002389C8E38E0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C0543A0>
7
<matplotlib.contour.QuadContourSet object at 0x000002389C7DC1C0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389BD9DB80>, <matplotlib.axis.YTick object at 0x000002389C68E130>, <matplotlib.axis.YTick object at 0x000002389C70B5E0>, <matplotlib.axis.YTick object at 0x000002389C70B040>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389BD9DF70>, <matplotlib.axis.XTick object at 0x000002389C07B1F0>, <matplotlib.axis.XTick object at 0x000002389C70BFA0>, <matplotlib.axis.XTick object at 0x000002389CF9AD90>, <matplotlib.axis.XTick object at 0x000002389CF9ACD0>, <matplotlib.axis.XTick object at 0x000002389CF9A1C0>, <matplotlib.axis.XTick object at 0x000002389C682220>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C968130>
8
<matplotlib.contour.QuadContourSet object at 0x000002389C6FF4C0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C6785E0>, <matplotlib.axis.YTick object at 0x000002389CF7BF70>, <matplotlib.axis.YTick object at 0x000002389CF7B790>, <matplotlib.axis.YTick object at 0x000002389C816700>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C92ABE0>, <matplotlib.axis.XTick object at 0x000002389CB2ABB0>, <matplotlib.axis.XTick object at 0x000002389C8160D0>, <matplotlib.axis.XTick object at 0x000002389C816880>, <matplotlib.axis.XTick object at 0x000002389C8F3E50>, <matplotlib.axis.XTick object at 0x000002389C8F3460>, <matplotlib.axis.XTick object at 0x000002389C8F3250>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C628850>
9
<matplotlib.contour.QuadContourSet object at 0x000002389C57EC10>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C88AC40>, <matplotlib.axis.YTick object at 0x000002389C773EB0>, <matplotlib.axis.YTick object at 0x000002389BD7D9D0>, <matplotlib.axis.YTick object at 0x000002389BD7DE20>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389BD82220>, <matplotlib.axis.XTick object at 0x000002389C846D90>, <matplotlib.axis.XTick object at 0x000002389BD7D730>, <matplotlib.axis.XTick object at 0x000002389CB84FA0>, <matplotlib.axis.XTick object at 0x000002389CB840A0>, <matplotlib.axis.XTick object at 0x000002389CB84B20>, <matplotlib.axis.XTick object at 0x000002389C768E80>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C063490>
10
<matplotlib.contour.QuadContourSet object at 0x000002389C8F2730>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389CF6A8B0>, <matplotlib.axis.YTick object at 0x000002389CB66190>, <matplotlib.axis.YTick object at 0x000002389CB68250>, <matplotlib.axis.YTick object at 0x000002389CB681F0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C8E1520>, <matplotlib.axis.XTick object at 0x000002389C07B2B0>, <matplotlib.axis.XTick object at 0x000002389CB688B0>, <matplotlib.axis.XTick object at 0x000002389C7E2100>, <matplotlib.axis.XTick object at 0x000002389C7E2D60>, <matplotlib.axis.XTick object at 0x000002389CB18AC0>, <matplotlib.axis.XTick object at 0x000002389CB18640>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C80D880>
11
<matplotlib.contour.QuadContourSet object at 0x000002389CB1D040>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389CB43FA0>, <matplotlib.axis.YTick object at 0x000002389C8E3B80>, <matplotlib.axis.YTick object at 0x000002389CB897C0>, <matplotlib.axis.YTick object at 0x000002389CB89CD0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C07DE80>, <matplotlib.axis.XTick object at 0x000002389C02E4C0>, <matplotlib.axis.XTick object at 0x000002389CB89700>, <matplotlib.axis.XTick object at 0x000002389CB7E1C0>, <matplotlib.axis.XTick object at 0x000002389CB7ED60>, <matplotlib.axis.XTick object at 0x000002389CB532B0>, <matplotlib.axis.XTick object at 0x000002389CB537C0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C98CF70>
12
<matplotlib.contour.QuadContourSet object at 0x000002389C9E94F0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C7D0070>, <matplotlib.axis.YTick object at 0x000002389CB32CA0>, <matplotlib.axis.YTick object at 0x000002389C871DF0>, <matplotlib.axis.YTick object at 0x000002389C871100>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C982100>, <matplotlib.axis.XTick object at 0x000002389C80D520>, <matplotlib.axis.XTick object at 0x000002389C871FD0>, <matplotlib.axis.XTick object at 0x000002389CB84520>, <matplotlib.axis.XTick object at 0x000002389CB84580>, <matplotlib.axis.XTick object at 0x000002389BD7DD00>, <matplotlib.axis.XTick object at 0x000002389BD7DA00>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C8469A0>
13
<matplotlib.contour.QuadContourSet object at 0x000002389C816B20>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389CB11370>, <matplotlib.axis.YTick object at 0x000002389C069EE0>, <matplotlib.axis.YTick object at 0x000002389CF7BCD0>, <matplotlib.axis.YTick object at 0x000002389C51EC10>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C541BE0>, <matplotlib.axis.XTick object at 0x000002389C565580>, <matplotlib.axis.XTick object at 0x000002389CF7B940>, <matplotlib.axis.XTick object at 0x000002389C8E1160>, <matplotlib.axis.XTick object at 0x000002389C6F08E0>, <matplotlib.axis.XTick object at 0x000002389C6F0FA0>, <matplotlib.axis.XTick object at 0x000002389C6F0A30>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C7B25E0>
14
<matplotlib.contour.QuadContourSet object at 0x000002389C68E370>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389BD9D7C0>, <matplotlib.axis.YTick object at 0x000002389CB516A0>, <matplotlib.axis.YTick object at 0x000002389C02B6D0>, <matplotlib.axis.YTick object at 0x000002389C6B3820>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002385C114D90>, <matplotlib.axis.XTick object at 0x000002389BD83130>, <matplotlib.axis.XTick object at 0x000002389C02BD60>, <matplotlib.axis.XTick object at 0x000002389C6B34C0>, <matplotlib.axis.XTick object at 0x000002389C682D30>, <matplotlib.axis.XTick object at 0x000002389C6825E0>, <matplotlib.axis.XTick object at 0x000002389C682070>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C7732B0>
15
<matplotlib.contour.QuadContourSet object at 0x000002389C6DCF70>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C7157F0>, <matplotlib.axis.YTick object at 0x000002389C628730>, <matplotlib.axis.YTick object at 0x000002389C6288E0>, <matplotlib.axis.YTick object at 0x000002389CB22820>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C6AA580>, <matplotlib.axis.XTick object at 0x000002389CB22910>, <matplotlib.axis.XTick object at 0x000002389CB22B80>, <matplotlib.axis.XTick object at 0x000002389CB22940>, <matplotlib.axis.XTick object at 0x000002389C628EE0>, <matplotlib.axis.XTick object at 0x000002389C7F2F10>, <matplotlib.axis.XTick object at 0x000002389C7F20D0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C816730>
16
<matplotlib.contour.QuadContourSet object at 0x000002389C9E9FD0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C871C10>, <matplotlib.axis.YTick object at 0x000002389CB845E0>, <matplotlib.axis.YTick object at 0x000002389CB550A0>, <matplotlib.axis.YTick object at 0x000002389CB556D0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C6A32E0>, <matplotlib.axis.XTick object at 0x000002389BD83460>, <matplotlib.axis.XTick object at 0x000002389CB55850>, <matplotlib.axis.XTick object at 0x000002389C9AFAC0>, <matplotlib.axis.XTick object at 0x000002389C9AF340>, <matplotlib.axis.XTick object at 0x000002389C58D8E0>, <matplotlib.axis.XTick object at 0x000002389C58DFD0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389CF6A580>
17
<matplotlib.contour.QuadContourSet object at 0x000002389C577DF0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C71A8B0>, <matplotlib.axis.YTick object at 0x000002389C70B5B0>, <matplotlib.axis.YTick object at 0x000002389C7F15B0>, <matplotlib.axis.YTick object at 0x000002389C7F1F70>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C71AD30>, <matplotlib.axis.XTick object at 0x000002389C02EC40>, <matplotlib.axis.XTick object at 0x000002389C7F12E0>, <matplotlib.axis.XTick object at 0x000002389C059F40>, <matplotlib.axis.XTick object at 0x000002389C059760>, <matplotlib.axis.XTick object at 0x000002389CF90730>, <matplotlib.axis.XTick object at 0x000002389CF90580>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C809160>
18
<matplotlib.contour.QuadContourSet object at 0x000002389C940130>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C94BF70>, <matplotlib.axis.YTick object at 0x000002389CB81F10>, <matplotlib.axis.YTick object at 0x000002389C7908B0>, <matplotlib.axis.YTick object at 0x000002389C790DC0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C0590A0>, <matplotlib.axis.XTick object at 0x000002389CB29760>, <matplotlib.axis.XTick object at 0x000002389C7906D0>, <matplotlib.axis.XTick object at 0x000002389C76E1F0>, <matplotlib.axis.XTick object at 0x000002389C76EE50>, <matplotlib.axis.XTick object at 0x000002389C79E3A0>, <matplotlib.axis.XTick object at 0x000002389C79E8B0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389BD660A0>
19
<matplotlib.contour.QuadContourSet object at 0x000002389C0201C0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C069FD0>, <matplotlib.axis.YTick object at 0x000002389C80D280>, <matplotlib.axis.YTick object at 0x000002389C57E910>, <matplotlib.axis.YTick object at 0x000002389C9C6A30>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C069A60>, <matplotlib.axis.XTick object at 0x000002389CB817C0>, <matplotlib.axis.XTick object at 0x000002389C57E430>, <matplotlib.axis.XTick object at 0x000002389C9C62B0>, <matplotlib.axis.XTick object at 0x000002389C876670>, <matplotlib.axis.XTick object at 0x000002389C8764F0>, <matplotlib.axis.XTick object at 0x000002389C876550>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C846070>
20
<matplotlib.contour.QuadContourSet object at 0x000002389C9B3100>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C806190>, <matplotlib.axis.YTick object at 0x000002389C506FD0>, <matplotlib.axis.YTick object at 0x000002389C00C7C0>, <matplotlib.axis.YTick object at 0x000002389C00C910>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C8066D0>, <matplotlib.axis.XTick object at 0x000002389C514130>, <matplotlib.axis.XTick object at 0x000002389C00CB20>, <matplotlib.axis.XTick object at 0x000002389C02EF40>, <matplotlib.axis.XTick object at 0x000002389C02E4F0>, <matplotlib.axis.XTick object at 0x000002389C8164F0>, <matplotlib.axis.XTick object at 0x000002389C816EE0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389CB26A30>
21
<matplotlib.contour.QuadContourSet object at 0x000002389C99B160>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389CF9A5B0>, <matplotlib.axis.YTick object at 0x000002389C77C130>, <matplotlib.axis.YTick object at 0x000002389C012C40>, <matplotlib.axis.YTick object at 0x000002389C012490>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C7682E0>, <matplotlib.axis.XTick object at 0x000002389C577850>, <matplotlib.axis.XTick object at 0x000002389C012460>, <matplotlib.axis.XTick object at 0x000002389C6AA430>, <matplotlib.axis.XTick object at 0x000002389C6AA7C0>, <matplotlib.axis.XTick object at 0x000002389C6AAC10>, <matplotlib.axis.XTick object at 0x000002389C02B250>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389CF7A3D0>
22
<matplotlib.contour.QuadContourSet object at 0x000002389C986CA0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C7F26A0>, <matplotlib.axis.YTick object at 0x000002389C7F2B50>, <matplotlib.axis.YTick object at 0x000002389C88B550>, <matplotlib.axis.YTick object at 0x000002389C88B190>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002385C114A00>, <matplotlib.axis.XTick object at 0x000002389C986070>, <matplotlib.axis.XTick object at 0x000002389CB26970>, <matplotlib.axis.XTick object at 0x000002389C88BE20>, <matplotlib.axis.XTick object at 0x000002389CB26E80>, <matplotlib.axis.XTick object at 0x000002389CB26160>, <matplotlib.axis.XTick object at 0x000002389CB180A0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C71A220>
23
<matplotlib.contour.QuadContourSet object at 0x000002389C7B27F0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C80D820>, <matplotlib.axis.YTick object at 0x000002389C031520>, <matplotlib.axis.YTick object at 0x000002389C024460>, <matplotlib.axis.YTick object at 0x000002389C79E220>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C6A3310>, <matplotlib.axis.XTick object at 0x000002389C876760>, <matplotlib.axis.XTick object at 0x000002389C79E550>, <matplotlib.axis.XTick object at 0x000002389C79E850>, <matplotlib.axis.XTick object at 0x000002389BD8DCA0>, <matplotlib.axis.XTick object at 0x000002389BD8D100>, <matplotlib.axis.XTick object at 0x000002389BD8D9D0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C8163D0>
24
<matplotlib.contour.QuadContourSet object at 0x000002389BD74820>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C809610>, <matplotlib.axis.YTick object at 0x000002389C059D60>, <matplotlib.axis.YTick object at 0x000002389CB62640>, <matplotlib.axis.YTick object at 0x000002389CB84910>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C7708B0>, <matplotlib.axis.XTick object at 0x000002389C7E26D0>, <matplotlib.axis.XTick object at 0x000002389CB62DF0>, <matplotlib.axis.XTick object at 0x000002389CB77610>, <matplotlib.axis.XTick object at 0x000002389CB77040>, <matplotlib.axis.XTick object at 0x000002389CB77A90>, <matplotlib.axis.XTick object at 0x000002389C7EB670>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C5825E0>
25
<matplotlib.contour.QuadContourSet object at 0x000002389C688730>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C945670>, <matplotlib.axis.YTick object at 0x000002389C94BB20>, <matplotlib.axis.YTick object at 0x000002389C86E040>, <matplotlib.axis.YTick object at 0x000002389C86E400>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C945F10>, <matplotlib.axis.XTick object at 0x000002389C54D670>, <matplotlib.axis.XTick object at 0x000002389C86EF40>, <matplotlib.axis.XTick object at 0x000002389C8790D0>, <matplotlib.axis.XTick object at 0x000002389C879490>, <matplotlib.axis.XTick object at 0x000002389C8799A0>, <matplotlib.axis.XTick object at 0x000002389C879E20>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C7836A0>
26
<matplotlib.contour.QuadContourSet object at 0x000002389CB81850>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C7EA6D0>, <matplotlib.axis.YTick object at 0x000002389C54DF40>, <matplotlib.axis.YTick object at 0x000002389BD82370>, <matplotlib.axis.YTick object at 0x000002389BD821F0>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389BD8F5E0>, <matplotlib.axis.XTick object at 0x000002389C94B9A0>, <matplotlib.axis.XTick object at 0x000002389BD82A30>, <matplotlib.axis.XTick object at 0x000002389C80D9D0>, <matplotlib.axis.XTick object at 0x000002389C80DD90>, <matplotlib.axis.XTick object at 0x000002389C80DC40>, <matplotlib.axis.XTick object at 0x000002389C747370>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C7B2370>
27
<matplotlib.contour.QuadContourSet object at 0x000002389C6F6AC0>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C88BFA0>, <matplotlib.axis.YTick object at 0x000002389CF6A3D0>, <matplotlib.axis.YTick object at 0x000002389C71A7F0>, <matplotlib.axis.YTick object at 0x000002389C71A580>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C88B4F0>, <matplotlib.axis.XTick object at 0x000002389C059670>, <matplotlib.axis.XTick object at 0x000002389C71A2B0>, <matplotlib.axis.XTick object at 0x000002389BD7DB20>, <matplotlib.axis.XTick object at 0x000002389BD7DBB0>, <matplotlib.axis.XTick object at 0x000002389C986DF0>, <matplotlib.axis.XTick object at 0x000002389C986AF0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C68E340>
28
<matplotlib.contour.QuadContourSet object at 0x000002389C809D30>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C012160>, <matplotlib.axis.YTick object at 0x000002389CB59940>, <matplotlib.axis.YTick object at 0x000002389C768400>, <matplotlib.axis.YTick object at 0x000002389C768A00>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C012AC0>, <matplotlib.axis.XTick object at 0x000002389C9CA2E0>, <matplotlib.axis.XTick object at 0x000002389C768700>, <matplotlib.axis.XTick object at 0x000002389C806880>, <matplotlib.axis.XTick object at 0x000002389C8063D0>, <matplotlib.axis.XTick object at 0x000002389C806760>, <matplotlib.axis.XTick object at 0x000002389C51EC10>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C6DC220>
29
<matplotlib.contour.QuadContourSet object at 0x000002389C654850>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389BD92340>, <matplotlib.axis.YTick object at 0x000002389C7F2100>, <matplotlib.axis.YTick object at 0x000002389C81CEE0>, <matplotlib.axis.YTick object at 0x000002389C06C460>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C870A30>, <matplotlib.axis.XTick object at 0x000002389C06CA30>, <matplotlib.axis.XTick object at 0x000002389C06C5E0>, <matplotlib.axis.XTick object at 0x000002389C06CEB0>, <matplotlib.axis.XTick object at 0x000002389C81CE80>, <matplotlib.axis.XTick object at 0x000002389C68E430>, <matplotlib.axis.XTick object at 0x000002389C68E0A0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389C6CC730>
>>> Us=50.31*ratio**(0.5)
>>> Us=-1.35*ratio**(0.5)
>>> Us=50.31*ratio**(0.5)
>>> Uc=-1.35*ratio**(0.5)
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
<matplotlib.contour.QuadContourSet object at 0x000002389CB22B20>
(-7.0, 7.0, 0.0, 7.0)
([<matplotlib.axis.YTick object at 0x000002389C9E9550>, <matplotlib.axis.YTick object at 0x000002389C8F36D0>, <matplotlib.axis.YTick object at 0x000002389C71A9A0>, <matplotlib.axis.YTick object at 0x000002389C71AC40>], <a list of 4 Text major ticklabel objects>)
([<matplotlib.axis.XTick object at 0x000002389C029AC0>, <matplotlib.axis.XTick object at 0x000002389C8F3BE0>, <matplotlib.axis.XTick object at 0x000002389C71AD30>, <matplotlib.axis.XTick object at 0x000002389CF6AB80>, <matplotlib.axis.XTick object at 0x000002389CF6A6A0>, <matplotlib.axis.XTick object at 0x000002389CB59FA0>, <matplotlib.axis.XTick object at 0x000002389CB594F0>], <a list of 7 Text major ticklabel objects>)
Text(-0.2, -1.6, '$\\mathcal{v}_\\parallel/\\mathcal{v}_{Ae}$')
Text(-0.2, 8.3, '$\\mathcal{v}_\\perp/\\mathcal{v}_{Ae}$')
<matplotlib.colorbar.Colorbar object at 0x000002389CB772B0>
>>> 