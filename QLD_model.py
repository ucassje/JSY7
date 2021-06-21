from numpy.linalg import inv
from numpy import dot
import numpy as np
from numpy import pi,exp,sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpmath import *
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
from math import gamma
import math as math
from scipy import integrate

path_home="/Users/user/Desktop/JSY/"
path_lab="/disk/plasma4/syj2/Code/JSY/"
# path_current=path_home
path_current=path_lab
Nv=60
Mv=10
pal_v = np.linspace(-Mv, Mv, 2*Nv)
per_v = np.linspace(-Mv, Mv, 2*Nv)
Nt=3000
Mt=3000
t=np.linspace(0, Mt, Nt-1)
F=(t[1]-t[0])/(2*(pal_v[1]-pal_v[0]))**2
n=1
n2=0
n3=-1
omega=-1
fre=0.1
k_pal0=0.171
k_pal_max=0.193
k_pal_min=0.149
a_pal=0.022
GV=2
B_B0=0.001

k_per0=k_pal0*tan((68*np.pi)/180)
a_per=a_pal*tan((68*np.pi)/180)
k_per_max=k_pal_max*tan((68*np.pi)/180)
k_per_min=k_pal_min*tan((68*np.pi)/180)
print(k_per0)
print(a_per)
def k(b):
    f = lambda x: ((besselj(0, (b*x)/(omega), 0))**2)*np.exp(-(((x-0.42)**2)/(0.054**2)))*x
    I=integrate.quad(f, k_per_min, k_per_max)
    return I[0]

def coefficient_a(a,b):
    return ((0.52*np.pi**2)/(a_pal*a_per**2))*(B_B0**2)*(((b)**2)/abs(a-GV))*(fre/k_pal0)**2*k(b)*(np.exp(-(((fre-a*k_pal0-n*omega)/(a-GV))**2)/(a_pal**2)))**2

def coefficient_a2(a,b):
    return ((0.04*np.pi**2)/(a_pal*a_per**2))*(B_B0**2)*(((a)**2)/abs(a-GV))*(fre/k_pal0)**2*k(b)*(np.exp(-(((fre-a*k_pal0-n2*omega)/(a-GV))**2)/(a_pal**2)))**2

def coefficient_a3(a,b):
    return ((0.52*np.pi**2)/(a_pal*a_per**2))*(B_B0**2)*(((b)**2)/abs(a-GV))*(fre/k_pal0)**2*k(b)*(np.exp(-(((fre-a*k_pal0-n3*omega)/(a-GV))**2)/(a_pal**2)))**2

AA=np.zeros(((2*Nv)*(2*Nv),(2*Nv)*(2*Nv)))
delv=2*abs(pal_v[1]-pal_v[0])
def Matrix_A(b):
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

def Matrix_B1(b):
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

def Matrix_B2(b):
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

def Matrix_C1(b):
    C=np.zeros(((2*Nv),(2*Nv)))
    for i in range(2*Nv):
        for j in range(2*Nv):
            C[i,j] =-(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a(pal_v[i],per_v[b]-delv/2)-(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a2(pal_v[i],per_v[b]-delv/2)-(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]-delv/2)))*coefficient_a3(pal_v[i],per_v[b]-delv/2) if j==i else 0
    return C

def Matrix_C2(b):
    C=np.zeros(((2*Nv),(2*Nv)))
    for i in range(2*Nv):
        for j in range(2*Nv):
            C[i,j] =-(F/2)*((pal_v[i]*n*omega-GV*n*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a(pal_v[i],per_v[b]+delv/2)-(F/2)*((pal_v[i]*n2*omega-GV*n2*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n2*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a2(pal_v[i],per_v[b]+delv/2)-(F/2)*((pal_v[i]*n3*omega-GV*n3*omega)/(fre*pal_v[i]-GV*k_pal0*pal_v[i]-GV*n3*omega))**2*(1/(per_v[b]*(per_v[b]+delv/2)))*coefficient_a3(pal_v[i],per_v[b]+delv/2) if j==i else 0
    return C

def Matrix_A_1(b):
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

for a in range(2*Nv):
    for b in range(2*Nv):
        if a==b:
            AA[a*2*Nv:(a+1)*2*Nv,b*2*Nv:(b+1)*2*Nv]=Matrix_A(a)

            
for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            AA[(a+1)*2*Nv:(a+2)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=Matrix_B1(a+1)

            
for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            AA[a*2*Nv:(a+1)*2*Nv,(b+1)*2*Nv:(b+2)*2*Nv]=Matrix_B2(a)

            
for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            AA[(a+2)*2*Nv:(a+3)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=Matrix_C1(a+2)

            
for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            AA[a*2*Nv:(a+1)*2*Nv,(b+2)*2*Nv:(b+3)*2*Nv]=Matrix_C2(a)

AA_1 = inv(AA)
QQ=np.zeros(((2*Nv)*(2*Nv),(2*Nv)*(2*Nv)))
for a in range(2*Nv):
    for b in range(2*Nv):
        if a==b:
            QQ[a*2*Nv:(a+1)*2*Nv,b*2*Nv:(b+1)*2*Nv]=Matrix_A_1(a)

for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            QQ[(a+1)*2*Nv:(a+2)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=-Matrix_B1(a+1)

for a in range(2*Nv-1):
    for b in range(2*Nv-1):
        if a==b:
            QQ[a*2*Nv:(a+1)*2*Nv,(b+1)*2*Nv:(b+2)*2*Nv]=-Matrix_B2(a)

for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            QQ[(a+2)*2*Nv:(a+3)*2*Nv,(b)*2*Nv:(b+1)*2*Nv]=-Matrix_C1(a+2)

for a in range(2*Nv-2):
    for b in range(2*Nv-2):
        if a==b:
            QQ[a*2*Nv:(a+1)*2*Nv,(b+2)*2*Nv:(b+3)*2*Nv]=-Matrix_C2(a)

            
AQ=dot(AA_1,QQ)
def Kappa_Initial_Core(a,b):
    kappa=150
    return 4**(-1)*(3.58)**(-0.5)*0.978*np.exp(-((b)**2)/4)*np.exp(-((a-Uc)**2)/3.58)

def Kappa_Initial_Strahl(a,b):
    kappa=150
    return 1.136**(-1)*(2.27)**(-0.5)*0.022*np.exp(-((b)**2)/1.136)*np.exp(-((a-Us)**2)/2.27)

Me=9.1094*(10**(-28))
Mp=1.6726*(10**(-24))
ratio=Me/Mp
Us=202.804*ratio**(0.5)
Uc=-4.48*ratio**(0.5)
cont_lev = np.linspace(-8,0,25)
f_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
solu2=np.zeros(shape = (Nv, 2*Nv))
fc_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
ff_1=np.zeros(shape = ((2*Nv)*(2*Nv), 1))
for j in range(2*Nv):
    for i in range(2*Nv):
        f_1[j*2*Nv+i]=Kappa_Initial_Strahl(pal_v[i],per_v[j])

for j in range(2*Nv):
    for i in range(2*Nv):
        fc_1[j*2*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j])

ff_1=f_1+fc_1
Mf_1=np.max(ff_1)
per_v2 = np.linspace(0, Mv, Nv)
X2,Y2 = np.meshgrid(pal_v,per_v2)
for k in range(60): #Numer in range indicates the minute.
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
    plt.ylim([0, Mv])
    plt.xlim([-Mv, Mv])
    plt.rc('font', size=9)
    plt.tick_params(labelsize=9)
    plt.text(-0.2,-1.6,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae}$', fontsize=9)
    plt.text(-0.2,10.3,r'$\mathcal{v}_\perp/\mathcal{v}_{Ae}$', fontsize=9)
    plt.colorbar(label=r'$Log(F/F_{MAX})$')
    plt.savefig(f'{path_current}QLD/{k}.png')
    plt.clf()
    plt.close()
    for t in range(5):
        ff_1=dot(AQ, ff_1)
