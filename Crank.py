from numpy.linalg import inv
from numpy import dot
from numpy import pi,exp,sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpmath import *
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
from math import gamma
import math as math
import numpy as np
from scipy import integrate
Nv=37
i_solar_r=10
f_solar_r=40
path_home="/Users/user/Desktop/python-code/"
path_lab="/disk/plasma4/syj2/Code/python-code/"
# path_current=path_home
path_current=path_lab
def n_0(r):
        return 10*(215/r)**2

def B_0(r):
        return 10*(215/r)**2

v_Ae_0=(B_0(215)*10**(-9))/(4.*np.pi*10**(-7)*9.1094*10**(-31)*n_0(215)*10**6)**0.5
q=1.6022*(10**(-19))
Me=9.1094*(10**(-31))
Mp=1.6726*(10**(-27))
Mv=5*10**7/v_Ae_0
epsilon=8.8542*10**(-12)
pal_v = np.linspace(-Mv, Mv, Nv)
per_v = np.linspace(-Mv, Mv, Nv)
delv=pal_v[1]-pal_v[0]
Nr=30
r_s=696340000.
z=np.linspace(i_solar_r, f_solar_r, Nr)
delz=z[1]-z[0]
Mt=0.01
Nt=3
t=np.linspace(0, Mt, Nt-1)
delt=t[1]-t[0]
Fv=delt/delv
Fvv=delt/(delv)**2
Fz=delt/delz
print(Fv)
print(Fz)
U_f=800000./v_Ae_0
T_e=10*(10**(5));
T_e_back=10*(10**(5));
Bol_k=1.3807*(10**(-23));
kappa=2
v_th_e=((2.*kappa-3)*Bol_k*T_e/(kappa*Me))**0.5/v_Ae_0
v_th_p=((2.*kappa-3)*Bol_k*T_e/(kappa*Mp))**0.5/v_Ae_0
v_th_e_back=((2.*kappa-3)*Bol_k*T_e_back/(kappa*Me))**0.5/v_Ae_0
time_nor=r_s/v_Ae_0
Omega=2.7*10**(-6)*time_nor
G=6.6726*10**(-11)
M_s=1.989*10**(30)
print((f_solar_r-i_solar_r)/U_f)
print(((f_solar_r-i_solar_r)/U_f)/delt)

def n(r):
        return n_0(i_solar_r)*(i_solar_r/r)**2

def U_solar(r):
        return U_f*(np.exp(r/5.)-np.exp(-r/5.))/(np.exp(r/5.)+np.exp(-r/5.))

def dU_solar(x):
        return U_f*(1/5)*(2/(np.exp(x/5.)+np.exp(-x/5.)))**2

def cos(r):
        return (1/(1+(r*Omega/U_solar(r))**2)**0.5)

def temperature(r):
        return T_e*(r/z[0])**(-0.35)

def v_th_function(T):
        kappa=2
        return ((2.*kappa-3)*Bol_k*T/(kappa*Me))**0.5/v_Ae_0

def Kappa_Initial_Core(a,b,r):
   kappa=2
   return (U_f/U_solar(r))*(r_s**3)*(n(r)*10**6)*(2*np.pi*v_th_function(temperature(r))**3*kappa**1.5)**(-1)*(gamma(kappa+1)/(gamma(kappa-0.5)*gamma(1.5)))*(1.+((b/v_th_function(temperature(r)))**2)/kappa+((a/v_th_function(temperature(r)))**2)/kappa)**(-kappa-1.)#+(U_f/U_solar(r))*10**(-6)*(r_s**3)*(n(r)*10**6)*(np.pi**1.5*v_th_function(T_e*(z[0]/r))**3)**(-1)*(gamma(kappa+1)/(gamma(kappa-0.5)*kappa**1.5))*(1.+((b/(v_th_function(T_e*(z[0]/r))*100000))**2)/kappa+((a/(v_th_function(T_e*(z[0]/r))*100000))**2)/kappa)**(-kappa-1.) #(((7.5*10**9/r_s)/c)**2+0.05*np.exp(-(c-23)**2))*

       
f=np.zeros(shape = (Nr*Nv**2, 1))
for r in range(Nr):
       for j in range(Nv):
              for i in range(Nv):
                 f[r*(Nv)*(Nv)+j*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j],z[r])
                 
Mf=np.max(f)

f_1=np.zeros(shape = (Nr*Nv**2, 1))
for r in range(Nr):
        for j in range(Nv):
                for i in range(Nv):
                        f_1[r*(Nv)*(Nv)+j*Nv+i]=Kappa_Initial_Core(pal_v[i],per_v[j],z[r])

                
                        



X2,Y2 = np.meshgrid(pal_v,per_v)
solu1=np.zeros(shape = (Nv, Nv))
solu2=np.zeros(shape = (Nv))
solu3=np.zeros(shape = (Nv))
cont_lev = np.linspace(-10,0,25)
difference=np.zeros(shape = ((Nr)*(Nv)*(Nv), 1))


def sin(x):
        return (1/(1+(U_solar(x)/Omega*(1/x))**2)**0.5)

def dcos(x):
        return -(0.5*(Omega/U_solar(x))**2*x)/(1+(x*Omega/U_solar(x))**2)**1.5

def dsin(x):
        return ((U_solar(x)/Omega*(1/x))**2*(1/x**3))/(1+(U_solar(x)/Omega*(1/x))**2)**1.5

def dlnB(x):
        return (-(1/x)*(2+(x*Omega/U_solar(x))**2)/(1+(x*Omega/U_solar(x))**2))

def B(x):
        return B_0(x)*(i_solar_r/x)*((i_solar_r/x)**2+(i_solar_r*Omega/U_solar(x))**2)**0.5

def electric(x):
        return 0*(1/v_Ae_0**2)*(Bol_k*n_0(i_solar_r)*T_e)/(Me*n(x))*2.35*(i_solar_r/x)**(1.35)*(i_solar_r/x**2)

def Matrix_A(R,M):
    A=np.zeros(((Nv),(Nv)))
    for i in range(Nv):
            for j in range(Nv):
                    if R<1:
                            if i==0:
                                    A[i,j] =1+0*0.5*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*(t[1]-t[0])/z[R])+0*(delt/2)*sin(z[R])*(sin(z[R])*dU_solar(z[R])+U_solar(z[R])*dsin(z[R])) if j==0 else 0*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2)) if j==1 else 0
                            elif i==Nv-1:
                                    A[i,j] =-0*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2)) if j==Nv-2 else 1+0*0.5*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*(t[1]-t[0])/z[R])+0*(delt/2)*sin(z[R])*(sin(z[R])*dU_solar(z[R])+U_solar(z[R])*dsin(z[R])) if j==Nv-1 else 0
                            else:
                                    A[i,j] =-0*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2)) if j==i-1 else 1+0*0.5*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*(t[1]-t[0])/z[R])+0*(delt/2)*sin(z[R])*(sin(z[R])*dU_solar(z[R])+U_solar(z[R])*dsin(z[R])) if j==i else 0*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2)) if j==i+1 else 0
                    elif R==Nr-1:
                            if i==0:
                                    A[i,j] =1+0*0.5*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*(t[1]-t[0])/z[R])+(delt/2)*sin(z[R])*(sin(z[R])*dU_solar(z[R])+U_solar(z[R])*dsin(z[R])) if j==0 else (Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-(Fv/4)*cos(z[R])*electric(z[R]) if j==1 else 0
                            elif i==Nv-1:
                                    A[i,j] =-(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+(Fv/4)*cos(z[R])*electric(z[R]) if j==Nv-2 else 1+0*0.5*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*(t[1]-t[0])/z[R])+(delt/2)*sin(z[R])*(sin(z[R])*dU_solar(z[R])+U_solar(z[R])*dsin(z[R])) if j==Nv-1 else 0
                            else:
                                    A[i,j] =-(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+(Fv/4)*cos(z[R])*electric(z[R]) if j==i-1 else 1+0*0.5*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*(t[1]-t[0])/z[R])+(delt/2)*sin(z[R])*(sin(z[R])*dU_solar(z[R])+U_solar(z[R])*dsin(z[R])) if j==i else (Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-(Fv/4)*cos(z[R])*electric(z[R]) if j==i+1 else 0
                    else:
                            if i==0:
                                    A[i,j] =1+0.5*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*(t[1]-t[0])/z[R])+(delt/2)*sin(z[R])*(sin(z[R])*dU_solar(z[R])+U_solar(z[R])*dsin(z[R])) if j==0 else (Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-(Fv/4)*cos(z[R])*electric(z[R]) if j==1 else 0
                            elif i==Nv-1:
                                    A[i,j] =-(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+(Fv/4)*cos(z[R])*electric(z[R]) if j==Nv-2 else 1+0.5*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*(t[1]-t[0])/z[R])+(delt/2)*sin(z[R])*(sin(z[R])*dU_solar(z[R])+U_solar(z[R])*dsin(z[R])) if j==Nv-1 else 0
                            else:
                                    A[i,j] =-(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+(Fv/4)*cos(z[R])*electric(z[R]) if j==i-1 else 1+0.5*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*(t[1]-t[0])/z[R])+(delt/2)*sin(z[R])*(sin(z[R])*dU_solar(z[R])+U_solar(z[R])*dsin(z[R])) if j==i else (Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-(Fv/4)*cos(z[R])*electric(z[R]) if j==i+1 else 0

    return A

def Matrix_B(R,M):
    B=np.zeros(((Nv),(Nv)))
    for i in range(Nv):
        for j in range(Nv):
                if R<1:
                        if i==0:
                                B[i,j] =0*(Fv/4)*((pal_v[i]*cos(z[R]))*dlnB(z[R])*per_v[M]/2) if j==0 else 0
                        elif i==Nv-1:
                                B[i,j] =0*(Fv/4)*((pal_v[i]*cos(z[R]))*dlnB(z[R])*per_v[M]/2) if j==Nv-1 else 0
                        else:
                                B[i,j] =0*(Fv/4)*((pal_v[i]*cos(z[R]))*dlnB(z[R])*per_v[M]/2) if j==i else 0
                elif R==Nr-1:
                        if i==0:
                                B[i,j] =(Fv/4)*((pal_v[i]*cos(z[R]))*dlnB(z[R])*per_v[M]/2) if j==0 else 0
                        elif i==Nv-1:
                                B[i,j] =(Fv/4)*((pal_v[i]*cos(z[R]))*dlnB(z[R])*per_v[M]/2) if j==Nv-1 else 0
                        else:
                                B[i,j] =(Fv/4)*((pal_v[i]*cos(z[R]))*dlnB(z[R])*per_v[M]/2) if j==i else 0
                else:
                        if i==0:
                                B[i,j] =(Fv/4)*((pal_v[i]*cos(z[R]))*dlnB(z[R])*per_v[M]/2) if j==0 else 0
                        elif i==Nv-1:
                                B[i,j] =(Fv/4)*((pal_v[i]*cos(z[R]))*dlnB(z[R])*per_v[M]/2) if j==Nv-1 else 0
                        else:
                                B[i,j] =(Fv/4)*((pal_v[i]*cos(z[R]))*dlnB(z[R])*per_v[M]/2) if j==i else 0
    return B

def Matrix_alpha(R,M):
    alpha=np.zeros(((Nv),(Nv)))
    for i in range(Nv):
        for j in range(Nv):
           if R<1:
              alpha[i,j] =0*(Fz/4)*(U_solar(z[R])+pal_v[i]*cos(z[R])) if j==i else 0
           elif R==Nr-1:
              alpha[i,j] =0*(Fz/4)*(U_solar(z[R])+pal_v[i]*cos(z[R])) if j==i else 0
           else:
              alpha[i,j] =(Fz/4)*(U_solar(z[R])+pal_v[i]*cos(z[R])) if j==i else 0 #*rect((2.*(t[1]-t[0])/z[R])*f_1[M*Nv+i,R]/Mf[0]+Fz*0.5*(f_1[M*Nv+i,R+1]/Mf[0]-f_1[M*Nv+i,R-1]/Mf[0]))     return alpha
    return alpha

def Matrix_AA(R):
    AA=np.zeros(((Nv)*(Nv),(Nv)*(Nv)))
    for a in range(Nv-1):
	    for b in range(Nv-1):
		    if a==b:
			    AA[a*Nv:(a+1)*Nv,(b+1)*Nv:(b+2)*Nv]=Matrix_B(R,a)
    for a in range(Nv-1):
	    for b in range(Nv-1):
		    if a==b:
			    AA[(a+1)*Nv:(a+2)*Nv,(b)*Nv:(b+1)*Nv]=-Matrix_B(R,a+1)
    for a in range(Nv):
	    for b in range(Nv):
		    if a==b:
			    AA[a*Nv:(a+1)*Nv,b*Nv:(b+1)*Nv]=Matrix_A(R,a)
    return AA

def Matrix_alphaA(R):
    alphaA=np.zeros(((Nv)*(Nv),(Nv)*(Nv)))
    for a in range(Nv):
	    for b in range(Nv):
		    if a==b:
			    alphaA[a*Nv:(a+1)*Nv,b*Nv:(b+1)*Nv]=Matrix_alpha(R,a)
    return alphaA


def Matrix_Q(R,M):
    A=np.zeros(((Nv),(Nv)))
    for i in range(Nv):
        for j in range(Nv):
                if R<1:
                        if i==0:
                                A[i,j] =1-0*0.5*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*(t[1]-t[0])/z[R])-0*(delt/2)*sin(z[R])*(sin(z[R])*dU_solar(z[R])+U_solar(z[R])*dsin(z[R])) if j==0 else -0*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2)) if j==1 else 0
                        elif i==Nv-1:
                                A[i,j] =0*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2)) if j==Nv-2 else 1-0*0.5*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*(t[1]-t[0])/z[R])-0*(delt/2)*sin(z[R])*(sin(z[R])*dU_solar(z[R])+U_solar(z[R])*dsin(z[R])) if j==Nv-1 else 0
                        else:
                                A[i,j] =0*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2)) if j==i-1 else 1-0*0.5*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*(t[1]-t[0])/z[R])-0*(delt/2)*sin(z[R])*(sin(z[R])*dU_solar(z[R])+U_solar(z[R])*dsin(z[R])) if j==i else -0*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2)) if j==i+1 else 0
                elif R==Nr-1:
                        if i==0:
                                A[i,j] =1-0*0.5*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*(t[1]-t[0])/z[R])-(delt/2)*sin(z[R])*(sin(z[R])*dU_solar(z[R])-U_solar(z[R])*dsin(z[R])) if j==0 else -(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+(Fv/4)*cos(z[R])*electric(z[R]) if j==1 else 0
                        elif i==Nv-1:
                                A[i,j] =(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-(Fv/4)*cos(z[R])*electric(z[R]) if j==Nv-2 else 1-0*0.5*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*(t[1]-t[0])/z[R])-(delt/2)*sin(z[R])*(sin(z[R])*dU_solar(z[R])+U_solar(z[R])*dsin(z[R])) if j==Nv-1 else 0
                        else:
                                A[i,j] =(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-(Fv/4)*cos(z[R])*electric(z[R]) if j==i-1 else 1-0*0.5*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*(t[1]-t[0])/z[R])-(delt/2)*sin(z[R])*(sin(z[R])*dU_solar(z[R])+U_solar(z[R])*dsin(z[R])) if j==i else -(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+(Fv/4)*cos(z[R])*electric(z[R]) if j==i+1 else 0
                else:
                        if i==0:
                                A[i,j] =1-0.5*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*(t[1]-t[0])/z[R])-(delt/2)*sin(z[R])*(sin(z[R])*dU_solar(z[R])+U_solar(z[R])*dsin(z[R])) if j==0 else -(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+(Fv/4)*cos(z[R])*electric(z[R]) if j==1 else 0
                        elif i==Nv-1:
                                A[i,j] =(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-(Fv/4)*cos(z[R])*electric(z[R]) if j==Nv-2 else 1-0.5*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*(t[1]-t[0])/z[R])-(delt/2)*sin(z[R])*(sin(z[R])*dU_solar(z[R])+U_solar(z[R])*dsin(z[R])) if j==Nv-1 else 0
                        else:
                                A[i,j] =(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-(Fv/4)*cos(z[R])*electric(z[R]) if j==i-1 else 1-0.5*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*(t[1]-t[0])/z[R])-(delt/2)*sin(z[R])*(sin(z[R])*dU_solar(z[R])+U_solar(z[R])*dsin(z[R])) if j==i else -(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(cos(z[R])*dU_solar(z[R])+dcos(z[R])*U_solar(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+(Fv/4)*cos(z[R])*electric(z[R]) if j==i+1 else 0
    return A

def Matrix_QQ(R):
    AA=np.zeros(((Nv)*(Nv),(Nv)*(Nv)))
    for a in range(Nv-1):
	    for b in range(Nv-1):
		    if a==b:
			    AA[a*Nv:(a+1)*Nv,(b+1)*Nv:(b+2)*Nv]=-Matrix_B(R,a)
    for a in range(Nv-1):
	    for b in range(Nv-1):
		    if a==b:
			    AA[(a+1)*Nv:(a+2)*Nv,(b)*Nv:(b+1)*Nv]=Matrix_B(R,a+1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    for a in range(Nv):
	    for b in range(Nv):
		    if a==b:
			    AA[a*Nv:(a+1)*Nv,b*Nv:(b+1)*Nv]=Matrix_Q(R,a)
    return AA

AAA=np.zeros(((Nr)*(Nv)**2,(Nr)*(Nv)**2))
QQQ=np.zeros(((Nr)*(Nv)**2,(Nr)*(Nv)**2))
for a in range(Nr):
	for b in range(Nr):
		if a==b:
			AAA[a*(Nv*Nv):(a+1)*(Nv*Nv),b*(Nv*Nv):(b+1)*(Nv*Nv)]=Matrix_AA(a)

for a in range(Nr-1):
	for b in range(Nr-1):
		if a==b:
			AAA[(a+1)*(Nv*Nv):(a+2)*(Nv*Nv),(b)*(Nv*Nv):(b+1)*(Nv*Nv)]=-Matrix_alphaA(a+1)

for a in range(Nr-1):
	for b in range(Nr-1):
		if a==b:
			AAA[(a)*(Nv*Nv):(a+1)*(Nv*Nv),(b+1)*(Nv*Nv):(b+2)*(Nv*Nv)]=Matrix_alphaA(a)

for a in range(Nr):
	for b in range(Nr):
		if a==b:
			QQQ[a*(Nv*Nv):(a+1)*(Nv*Nv),b*(Nv*Nv):(b+1)*(Nv*Nv)]=Matrix_QQ(a)

for a in range(Nr-1):
	for b in range(Nr-1):
		if a==b:
			QQQ[a*(Nv*Nv):(a+1)*(Nv*Nv),(b+1)*(Nv*Nv):(b+2)*(Nv*Nv)]=-Matrix_alphaA(a)

for a in range(Nr-1):
	for b in range(Nr-1):
		if a==b:
			QQQ[(a+1)*(Nv*Nv):(a+2)*(Nv*Nv),(b)*(Nv*Nv):(b+1)*(Nv*Nv)]=Matrix_alphaA(a+1)

AAA_1 = inv(AAA)
AQ=dot(AAA_1,QQQ)
print(AQ)
#for k in range(300):
#    print(k)
#    norm=0
#    f_temp=np.zeros(shape = (Nr*Nv**2, 1))
#    f_temp[:,:]=f_1[:,:]
#    f_1=dot(AQ, f_1)
#    for r in range(Nr):
#          for j in range(Nv):
#             for i in range(Nv):
#                norm=norm+(f_1[r*(Nv)*(Nv)+j*Nv+i]/Mf-f_temp[r*(Nv)*(Nv)+j*Nv+i]/Mf)**2
#    print(norm**0.5)
f_temp=np.zeros(shape = (Nr*Nv**2, 1))
f_temp[:,:]=f_1[:,:]

timestep=500
Normvalue=np.zeros(shape = (timestep))
for k in range(timestep):
    print(k)
    f_pre=np.zeros(shape = (Nr*Nv**2, 1))
    f_next=np.zeros(shape = (Nr*Nv**2, 1))
    f_temp3=np.zeros(shape = (Nr*Nv**2, 1))
    f_temp2=np.zeros(shape = (Nr*Nv**2, 1))
    f_temp1=np.zeros(shape = (Nr*Nv**2, 1))
    f_pre[:,:]=f_1[:,:]
    
    tempDensity=np.zeros(shape = (Nr))
    Density=np.zeros(shape = (Nr))
    for q in range(Nr):
            for j in range(Nv):
                    for i in range(Nv):
                            tempDensity[q]=tempDensity[q]+2*np.pi*f_1[q*(Nv)*(Nv)+j*Nv+i]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
            Density[q]=tempDensity[q]

   
    f_temp2=np.zeros(shape = (Nr*Nv**2, 1))
    f_temp2[:,:]=f_1[:,:]
    for r in range(Nr-1):
            for j in range(Nv):
                    for i in range(Nv):
                            if r==Nr-2:
                                    f_temp2[(r+1)*(Nv)*(Nv)+j*Nv+i]=f_1[(r+1)*(Nv)*(Nv)+j*Nv+i]
                            else:
                                    f_temp2[(r+1)*(Nv)*(Nv)+j*Nv+i]=0.5*(f_1[r*(Nv)*(Nv)+j*Nv+i]*(Density[r+1]/Density[r])+f_1[(r+2)*(Nv)*(Nv)+j*Nv+i]*(Density[r+1]/Density[r+2]))
    f_1[:,:]=f_temp2[:,:]
    

    f_1=dot(AQ, f_1)

    for q in range(Nr):
            for j in range(Nv):
                    for i in range(Nv):
                            if q==Nr-1:
                                    f_temp3[q*(Nv)*(Nv)+j*Nv+i]=(1-delz*2/z[q-1])*f_1[(q-1)*(Nv)*(Nv)+j*Nv+i]
                            #elif q==0:
                            #        f_temp3[q*(Nv)*(Nv)+j*Nv+i]=(1+delz*2/z[q+1])*f_1[(q+1)*(Nv)*(Nv)+j*Nv+i]

    f_temp3=dot(AQ, f_temp3)

    for q in range(Nr):
            for j in range(Nv):
                    for i in range(Nv):
                            if q==Nr-1:
                                    f_1[q*(Nv)*(Nv)+j*Nv+i]=f_temp3[(q)*(Nv)*(Nv)+j*Nv+i]
                            #elif q==0:
                            #        f_1[q*(Nv)*(Nv)+j*Nv+i]=f_temp3[(q)*(Nv)*(Nv)+j*Nv+i]

    tempDensity_1=0
    Density_1=0
    for j in range(Nv):
            for i in range(Nv):
                      tempDensity_1=tempDensity_1+2*np.pi*f_1[1*(Nv)*(Nv)+j*Nv+i]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
    Density_1=tempDensity_1

    for q in range(1):
            for j in range(Nv):
                    for i in range(Nv):
                            f_1[q*(Nv)*(Nv)+j*Nv+i]=f_1[q*(Nv)*(Nv)+j*Nv+i]*(1/Density[q])*Density_1*(z[1]**2/z[q]**2)

    f_temp1[:,:]=f_1[:,:]
    for r in range(Nr):
            for j in range(Nv):
                    for i in range(Nv):
                            if i==0:
                                    f_temp1[(r)*(Nv)*(Nv)+j*Nv+i]=2*f_1[(r)*(Nv)*(Nv)+j*Nv+i+1]-f_1[(r)*(Nv)*(Nv)+j*Nv+i+2]
                            elif i==Nv-1:
                                    f_temp1[(r)*(Nv)*(Nv)+j*Nv+i]=2*f_1[(r)*(Nv)*(Nv)+j*Nv+i-1]-f_1[(r)*(Nv)*(Nv)+j*Nv+i-2]
                            elif j==0:
                                    f_temp1[(r)*(Nv)*(Nv)+j*Nv+i]=2*f_1[(r)*(Nv)*(Nv)+(j+1)*Nv+i]-f_1[(r)*(Nv)*(Nv)+(j+2)*Nv+i]
                            elif j==Nv-1:
                                    f_temp1[(r)*(Nv)*(Nv)+j*Nv+i]=2*f_1[(r)*(Nv)*(Nv)+(j-1)*Nv+i]-f_1[(r)*(Nv)*(Nv)+(j-2)*Nv+i]
    f_1[:,:]=f_temp1[:,:]


    

    f_next[:,:]=f_1[:,:]
    norm=0
    for R in range(Nr):
            for J in range(Nv):
                    for I in range(Nv):
                            if R==Nr-1:#and pal_v[I]>0 and (pal_v[I]**2+per_v[J]**2)**0.5<5
                                    norm=norm+abs((f_next[R*(Nv)*(Nv)+J*Nv+I]/np.max(f_next)-f_pre[R*(Nv)*(Nv)+J*Nv+I]/np.max(f_pre)))**2
    Normvalue[k]=norm**0.5
    print(norm**0.5)
    
o=np.linspace(1, timestep, timestep)

plt.figure(figsize=(20,15))
plt.grid()
ax = plt.gca()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlim([o[0],o[timestep-1]])
ax.set_ylim([10**(-5),10*10**(-5)])
ax.set_xlabel(r'$t$', fontsize=28)
ax.set_ylabel(r'$norm$', fontsize=28)
ax.plot(o,Normvalue,linewidth=3.0, color='k');
plt.savefig(f'{path_current}without_collisions/norm.png')
plt.clf()
plt.close()


num=k+1
for r in range(Nr):
   for j in range(Nv):
       for i in range(Nv):
          solu1[j,i]=np.log10(f_1[(r)*(Nv)*(Nv)+j*Nv+i]/np.max(f_1))
   fig = plt.figure()
   fig.set_dpi(500)
   plt.contourf(X2, Y2,solu1, cont_lev,cmap='Blues');
   ax = plt.gca()
   ax.spines['left'].set_position('center')
   ax.spines['left'].set_smart_bounds(True)
   ax.spines['bottom'].set_position('zero')
   ax.spines['bottom'].set_smart_bounds(True)
   ax.spines['right'].set_color('none')
   ax.spines['top'].set_color('none')
   ax.xaxis.set_ticks_position('bottom')
   plt.axis('equal')
   ax.xaxis.set_ticks_position('bottom')
   ax.yaxis.set_ticks_position('left')
   plt.rc('font', size=8)
   plt.tick_params(labelsize=8)
   plt.text(pal_v[Nv-1],-0.,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae}$', fontsize=8)
   plt.text(-0.,pal_v[Nv-1],r'$\mathcal{v}_\perp/\mathcal{v}_{Ae}$', fontsize=8)
   plt.text(pal_v[Nv-8],pal_v[Nv-4], r'$r/r_s=$' "%.2f" % z[r], fontsize=8)
   #plt.text(pal_v[Nv-14],pal_v[Nv-2], r'$Time \ Iteration \ Number:$' "%.0f" % num, fontsize=8)
   #plt.text(pal_v[Nv-8],pal_v[Nv-6], r'$Nv=$' "%.2f" % Nv, fontsize=8)
   #plt.text(pal_v[Nv-8],pal_v[Nv-7], r'$Nr=$' "%.2f" % Nr, fontsize=8)
   plt.colorbar(label=r'$Log(F/F_{MAX})$')
   plt.savefig(f'{path_current}without_collisions/{r}.png')
   plt.clf()
   plt.close()


for r in range(Nr):
   for i in range(Nv):
        solu2[i]=np.log10(f_1[(r)*(Nv)*(Nv)+19*Nv+i]/np.max(f_1))
   fig = plt.figure()
   fig.set_dpi(500)
   plt.plot(pal_v,solu2,color='k',label=r'$r/r_s=$' "%.2f" % z[r]);
   plt.legend(loc='upper right')
   plt.grid()
   ax = plt.gca()
   ax.spines['left'].set_position('center')
   ax.spines['right'].set_color('none')
   ax.spines['top'].set_color('none')
   ax.xaxis.set_ticks_position('bottom')
   ax.yaxis.set_ticks_position('left')
   ax.set_yticks([-8,-6,-4,-2,-0])
   plt.ylim([-8, 0])
   plt.rc('font', size=8)
   plt.tick_params(labelsize=8)
   plt.savefig(f'{path_current}without_collisions/0/1D{r}.png')
   plt.clf()
   plt.close()


for r in range(Nr):
   for i in range(Nv):
        solu2[i]=np.log10(f_1[(r)*(Nv)*(Nv)+20*Nv+i]/np.max(f_1))
   fig = plt.figure()
   fig.set_dpi(500)
   plt.plot(pal_v,solu2,color='k',label=r'$r/r_s=$' "%.2f" % z[r]);
   plt.legend(loc='upper right')
   plt.grid()
   ax = plt.gca()
   ax.spines['left'].set_position('center')
   ax.spines['right'].set_color('none')
   ax.spines['top'].set_color('none')
   ax.xaxis.set_ticks_position('bottom')
   ax.yaxis.set_ticks_position('left')
   ax.set_yticks([-8,-6,-4,-2,-0])
   plt.ylim([-8, 0])
   plt.rc('font', size=8)
   plt.tick_params(labelsize=8)
   plt.savefig(f'{path_current}without_collisions/1/1D{r}.png')
   plt.clf()
   plt.close()

for r in range(Nr):
   for i in range(Nv):
        solu2[i]=np.log10(f_1[(r)*(Nv)*(Nv)+21*Nv+i]/np.max(f_1))
   fig = plt.figure()
   fig.set_dpi(500)
   plt.plot(pal_v,solu2,color='k',label=r'$r/r_s=$' "%.2f" % z[r]);
   plt.legend(loc='upper right')
   plt.grid()
   ax = plt.gca()
   ax.spines['left'].set_position('center')
   ax.spines['right'].set_color('none')
   ax.spines['top'].set_color('none')
   ax.xaxis.set_ticks_position('bottom')
   ax.yaxis.set_ticks_position('left')
   ax.set_yticks([-8,-6,-4,-2,-0])
   plt.ylim([-8, 0])
   plt.rc('font', size=8)
   plt.tick_params(labelsize=8)
   plt.savefig(f'{path_current}without_collisions/2/1D{r}.png')
   plt.clf()
   plt.close()

for r in range(Nr):
   for i in range(Nv):
        solu2[i]=np.log10(f_1[(r)*(Nv)*(Nv)+22*Nv+i]/np.max(f_1))
   fig = plt.figure()
   fig.set_dpi(500)
   plt.plot(pal_v,solu2,color='k',label=r'$r/r_s=$' "%.2f" % z[r]);
   plt.legend(loc='upper right')
   plt.grid()
   ax = plt.gca()
   ax.spines['left'].set_position('center')
   ax.spines['right'].set_color('none')
   ax.spines['top'].set_color('none')
   ax.xaxis.set_ticks_position('bottom')
   ax.yaxis.set_ticks_position('left')
   ax.set_yticks([-8,-6,-4,-2,-0])
   plt.ylim([-8, 0])
   plt.rc('font', size=8)
   plt.tick_params(labelsize=8)
   plt.savefig(f'{path_current}without_collisions/3/1D{r}.png')
   plt.clf()
   plt.close()

Density=np.zeros(shape = (Nr))
for r in range(Nr):
   tempDensity=0
   for j in range(Nv):
      for i in range(Nv):
              tempDensity=tempDensity+2*np.pi*f_1[r*(Nv)*(Nv)+j*Nv+i]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
   Density[r]=tempDensity/((r_s**3)*2)


plt.figure(figsize=(20,15))
plt.grid()
ax = plt.gca()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlim([z[0],z[Nr-1]])
ax.set_ylim([min(Density),max(Density)])
ax.set_xlabel(r'$r/r_s$', fontsize=28)
ax.set_ylabel(r'$n_e (m^{-3})$', fontsize=28)
ax.plot(z,Density,linewidth=3.0, color='k',label=r'$Numerical \ Density$');
ax.plot(z,max(Density)*(z[0]/z)**2,linewidth=3.0, color='k',linestyle='--',label=r'$Anaytical \ 1/r^{2} \ Density$');
plt.legend(loc='upper right')
plt.savefig(f'{path_current}without_collisions/density.png')
plt.clf()
plt.close()

Bulk=np.zeros(shape = (Nr))
for r in range(Nr):
   tempBulk=0
   for j in range(Nv):
      for i in range(Nv):
              if (pal_v[i]**2+abs(per_v[j])**2)**0.5<=10:
                      tempBulk=tempBulk+2*np.pi*pal_v[i]*f_1[r*(Nv)*(Nv)+j*Nv+i]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
              else:
                      tempBulk=tempBulk
   Bulk[r]=tempBulk/((r_s**3)*Density[r])


plt.figure(figsize=(20,15))
plt.grid()
ax = plt.gca()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlim([z[0],z[Nr-1]])
ax.set_ylim([min(Bulk),max(Bulk)])
ax.set_xlabel(r'$r/r_s$', fontsize=28)
ax.set_ylabel(r'$U/v_{Ae}$', fontsize=28)
plt.plot(z,Bulk,linewidth=3.0, color='k');
plt.savefig(f'{path_current}without_collisions/bulk.png')
plt.clf()
plt.close()

Q=np.zeros(shape = (Nr))
for r in range(Nr):
   tempQ=0
   for j in range(Nv):
      for i in range(Nv):
              if (pal_v[i]**2+abs(per_v[j])**2)**0.5<=10:
                      tempQ=tempQ+2*np.pi*(pal_v[i]**2+abs(per_v[j])**2)*pal_v[i]*f_1[r*(Nv)*(Nv)+j*Nv+i]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
              else:
                      tempQ=tempQ
   Q[r]=(z[r]**2)*tempQ/((r_s**3)*2)

plt.figure(figsize=(20,15))
plt.grid()
ax = plt.gca()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlim([z[0],z[Nr-1]])
ax.set_ylim([min(Q),max(Q)])
ax.set_xlabel(r'$r/r_s$', fontsize=28)
ax.set_ylabel(r'$Q_\parallel$', fontsize=28)
ax.plot(z,Q,linewidth=3.0, color='k',label=r'$Numerical \ Q_\parallel$');
ax.plot(z,max(Q)-max(Q)*np.exp(0.5*np.log(B(z)/B(z[0]))),linewidth=3.0, color='k',linestyle='--',label=r'$Anaytical \ Q_\parallel$');
plt.legend(loc='upper right')
plt.savefig(f'{path_current}without_collisions/Q.png')
plt.clf()
plt.close()

P_flux=np.zeros(shape = (Nr))
for r in range(Nr):
        P_flux[r]=z[r]**2*U_solar(z[r])*Density[r]
plt.figure(figsize=(20,15))
plt.grid()
ax = plt.gca()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlim([z[0],z[Nr-1]])
ax.set_ylim([min(P_flux),max(P_flux)])
ax.set_xlabel(r'$r/r_s$', fontsize=28)
ax.set_ylabel(r'$Particle \ Flux$', fontsize=28)
ax.plot(z,P_flux,linewidth=3.0, color='k',label=r'$Particle \ Flux$');
plt.legend(loc='upper right')
plt.savefig(f'{path_current}without_collisions/particleflux.png')
plt.clf()
plt.close()
