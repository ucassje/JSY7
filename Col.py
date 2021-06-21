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
from scipy import special
Nv=34
i_solar_r=10
f_solar_r=20
def n_0(r):
        return 10*(215/r)**2

def B_0(r):
        return 10*(215/r)**2

v_Ae_0=(B_0(215)*10**(-9))/(4.*np.pi*10**(-7)*9.1094*10**(-31)*n_0(215)*10**6)**0.5
q=1.6022*(10**(-19))
Me=9.1094*(10**(-31))
Mp=1.6726*(10**(-27))
Mv=3*10**7/v_Ae_0
epsilon=8.8542*10**(-12)
pal_v = np.linspace(-Mv, Mv, Nv)
per_v = np.linspace(-Mv, Mv, Nv)
delv=pal_v[1]-pal_v[0]
Nr=20
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
T_e=3.*(10**(5));
Bol_k=1.3807*(10**(-23));
v_th_e=(2.*Bol_k*T_e/Me)**0.5/v_Ae_0
v_th_p=(2.*Bol_k*T_e/Mp)**0.5/v_Ae_0
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
        return 0

def cos(r):
        return 0

Col=4*np.pi/(r_s**2*v_Ae_0**4)*(q**2/(4*np.pi*epsilon*Me))**2*25

def Collision_Core(a,b,r):
    kappa=50.
    return (r_s**3)*(n(r)*10**6)*(np.pi**1.5*v_th_e**3)**(-1)*(gamma(kappa+1)/(gamma(kappa-0.5)*kappa**1.5))*(1.+((b/v_th_e)**2)/kappa+((a/v_th_e)**2)/kappa)**(-kappa-1.)

def Collision_Proton(a,b,r):
    kappa=50.
    return (r_s**3)*(n(r)*10**6)*(np.pi**1.5*v_th_p**3)**(-1)*(gamma(kappa+1)/(gamma(kappa-0.5)*kappa**1.5))*(1.+((b/v_th_p)**2)/kappa+((a/v_th_p)**2)/kappa)**(-kappa-1.)


def G_per_2e(a,b,r):
	return (r_s**3)*(n(r)*10**6)*v_th_e**2*(0.5/(a**2+b**2)**1.5-1.5*b**2/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_e)*np.exp(-(a**2+b**2)/v_th_e**2)+(2*(a**2+b**2)/v_th_e**2-1)*special.erf((a**2+b**2)**0.5/v_th_e))+2*(r_s**3)*(n(r)*10**6)*b**2/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_e)

def G_pal_2e(a,b,r):
	return (r_s**3)*(n(r)*10**6)*v_th_e**2*(0.5/(a**2+b**2)**1.5-1.5*a**2/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_e)*np.exp(-(a**2+b**2)/v_th_e**2)+(2*(a**2+b**2)/v_th_e**2-1)*special.erf((a**2+b**2)**0.5/v_th_e))+2*(r_s**3)*(n(r)*10**6)*a**2/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_e)

def G_pal_per_e(a,b,r):
	return (-(r_s**3)*(n(r)*10**6)*v_th_e**2*(1.5*a*b/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_e)*np.exp(-(a**2+b**2)/v_th_e**2)+(2*(a**2+b**2)/v_th_e**2-1)*special.erf((a**2+b**2)**0.5/v_th_e))+2*(r_s**3)*(n(r)*10**6)*a*b/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_e))



def H_per(a,b,r):
        return 0

def H_pal(a,b,r):
        return 0

def G_per_2p(a,b,r):
	return (r_s**3)*(n(r)*10**6)*v_th_p**2*(0.5/(a**2+b**2)**1.5-1.5*b**2/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_p)*np.exp(-(a**2+b**2)/v_th_p**2)+(2*(a**2+b**2)/v_th_p**2-1)*special.erf((a**2+b**2)**0.5/v_th_p))+2*(r_s**3)*(n(r)*10**6)*b**2/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_p)

def G_pal_2p(a,b,r):
	return (r_s**3)*(n(r)*10**6)*v_th_p**2*(0.5/(a**2+b**2)**1.5-1.5*a**2/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_p)*np.exp(-(a**2+b**2)/v_th_p**2)+(2*(a**2+b**2)/v_th_p**2-1)*special.erf((a**2+b**2)**0.5/v_th_p))+2*(r_s**3)*(n(r)*10**6)*a**2/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_p)

def G_pal_per_p(a,b,r):
	return (-(r_s**3)*(n(r)*10**6)*v_th_p**2*(1.5*a*b/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_p)*np.exp(-(a**2+b**2)/v_th_p**2)+(2*(a**2+b**2)/v_th_p**2-1)*special.erf((a**2+b**2)**0.5/v_th_p))+2*(r_s**3)*(n(r)*10**6)*a*b/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_p))

def H_palp(a,b,r):
    	return (r_s**3)*(n(r)*10**6)*(1/v_th_p)*((2/np.pi**0.5)*(a/(a**2+b**2))*np.exp(-(a**2+b**2)/v_th_p**2)-(a*v_th_p)/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_p))

def H_perp(a,b,r):
    	return (r_s**3)*(n(r)*10**6)*(1/v_th_p)*((2/np.pi**0.5)*(b/(a**2+b**2))*np.exp(-(a**2+b**2)/v_th_p**2)-(b*v_th_p)/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_p))


def Kappa_Initial_Core(a,b,r):
   kappa=2
   return n(r)*(np.pi**1.5*v_th_e**3)**(-1)*(gamma(kappa+1)/(gamma(kappa-0.5)*kappa**1.5))*(1.+((b/v_th_e)**2)/kappa+((a/v_th_e)**2)/kappa)**(-kappa-1.)

       
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
solu2=np.zeros(shape = (Nv, Nv))
cont_lev = np.linspace(-10,0,25)
difference=np.zeros(shape = ((Nr)*(Nv)*(Nv), 1))

def rect_v(x):
	return 1#0 if abs(x)>=abs(pal_v[0]) else 1


def sin(x):
        return 0

def dcos(x):
        return 0

def dsin(x):
        return 0

def dlnB(x):
        return 0

def Matrix_A(R,M):
    A=np.zeros(((Nv),(Nv)))
    for i in range(Nv):
            for j in range(Nv):
                    if R==0:
                            if i==0:
                                    A[i,j] =1-(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))-(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==0 else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==1 else (Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==2 else 0
                            elif i==1:
                                    A[i,j] =-(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==0 else 1-(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))-(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==1 else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==2 else (Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==3 else 0
                            elif i==Nv-1:
                                    A[i,j] =(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-3 else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-2 else 1-(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))-(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                            elif i==Nv-2:
                                    A[i,j] =(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-4 else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-3 else 1-(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))-(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==Nv-2 else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                            else:
                                    A[i,j] =(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i-2 else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i-1 else 1-(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))-(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==i else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i+1 else (Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i+2 else 0
                    elif R==Nr-1:
                            if i==0:
                                    A[i,j] =1-(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))-(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==0 else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==1 else (Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==2 else 0
                            elif i==1:
                                    A[i,j] =-(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==0 else 1-(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))-(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==1 else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==2 else (Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==3 else 0
                            elif i==Nv-1:
                                    A[i,j] =(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-3 else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-2 else 1-(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))-(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                            elif i==Nv-2:
                                    A[i,j] =(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-4 else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-3 else 1-(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))-(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==Nv-2 else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                            else:
                                    A[i,j] =(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i-2 else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i-1 else 1-(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))-(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==i else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i+1 else (Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i+2 else 0
                    else:
                            if i==0:
                                    A[i,j] =1-(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))-(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==0 else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==1 else (Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==2 else 0
                            elif i==1:
                                    A[i,j] =-(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==0 else 1-(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))-(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==1 else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==2 else (Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==3 else 0
                            elif i==Nv-1:
                                    A[i,j] =(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-3 else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-2 else 1-(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))-(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                            elif i==Nv-2:
                                    A[i,j] =(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-4 else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-3 else 1-(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))-(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==Nv-2 else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                            else:
                                    A[i,j] =(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i-2 else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i-1 else 1-(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))-(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==i else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i+1 else (Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i+2 else 0

    return A

def Matrix_B(R,M):
    B=np.zeros(((Nv),(Nv)))
    for i in range(Nv):
        for j in range(Nv):
                if R==0:
                        if i==0:
                                B[i,j] =(Fv/4)*(-Col*H_perp(pal_v[i],per_v[M],z[R])) if j==0 else (Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==1 else 0
                        elif i==Nv-1:
                                B[i,j] =-(Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==Nv-2 else (Fv/4)*(-Col*H_perp(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                        else:
                                B[i,j] =-(Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==i-1 else (Fv/4)*(-Col*H_perp(pal_v[i],per_v[M],z[R])) if j==i else (Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==i+1 else 0
                elif R==Nr-1:
                        if i==0:
                                B[i,j] =(Fv/4)*(-Col*H_perp(pal_v[i],per_v[M],z[R])) if j==0 else (Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==1 else 0
                        elif i==Nv-1:
                                B[i,j] =-(Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==Nv-2 else (Fv/4)*(-Col*H_perp(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                        else:
                                B[i,j] =-(Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==i-1 else (Fv/4)*(-Col*H_perp(pal_v[i],per_v[M],z[R])) if j==i else (Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==i+1 else 0
                else:
                        if i==0:
                                B[i,j] =(Fv/4)*(-Col*H_perp(pal_v[i],per_v[M],z[R])) if j==0 else (Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==1 else 0
                        elif i==Nv-1:
                                B[i,j] =-(Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==Nv-2 else (Fv/4)*(-Col*H_perp(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                        else:
                                B[i,j] =-(Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==i-1 else (Fv/4)*(-Col*H_perp(pal_v[i],per_v[M],z[R])) if j==i else (Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==i+1 else 0
    return B

def Matrix_C(R,M):
    C=np.zeros(((Nv),(Nv)))
    for i in range(Nv):
        for j in range(Nv):
            C[i,j] =(Fvv/8)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R]))) if j==i else 0 
    return C


def Matrix_AA(R):
    AA=np.zeros(((Nv)*(Nv),(Nv)*(Nv)))
    for a in range(Nv-1):
	    for b in range(Nv-1):
		    if a==b:
			    AA[a*Nv:(a+1)*Nv,(b+1)*Nv:(b+2)*Nv]=Matrix_B(R,a)
    for a in range(Nv-2):
	    for b in range(Nv-2):
		    if a==b:
			    AA[a*Nv:(a+1)*Nv,(b+2)*Nv:(b+3)*Nv]=Matrix_C(R,a)
    for a in range(Nv-1):
	    for b in range(Nv-1):
		    if a==b:
			    AA[(a+1)*Nv:(a+2)*Nv,(b)*Nv:(b+1)*Nv]=-Matrix_B(R,a+1)
    for a in range(Nv-2):
	    for b in range(Nv-2):
		    if a==b:
			    AA[(a+2)*Nv:(a+3)*Nv,(b)*Nv:(b+1)*Nv]=Matrix_C(R,a+2)
    for a in range(Nv):
	    for b in range(Nv):
		    if a==b:
			    AA[a*Nv:(a+1)*Nv,b*Nv:(b+1)*Nv]=Matrix_A(R,a)
    return AA



def Matrix_Q(R,M):
    A=np.zeros(((Nv),(Nv)))
    for i in range(Nv):
        for j in range(Nv):
                if R==0:
                        if i==0:
                                A[i,j] =1+(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==0 else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==1 else -(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==2 else 0
                        elif i==1:
                                A[i,j] =(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==0 else 1+(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==1 else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==2 else -(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==3 else 0
                        elif i==Nv-1:
                                A[i,j] =-(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-3 else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-2 else 1+(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                        elif i==Nv-2:
                                A[i,j] =-(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-4 else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-3 else 1+(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==Nv-2 else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                        else:
                                A[i,j] =-(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i-2 else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i-1 else 1+(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==i else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i+1 else -(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i+2 else 0
                elif R==Nr-1:
                        if i==0:
                                A[i,j] =1+(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==0 else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==1 else -(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==2 else 0
                        elif i==1:
                                A[i,j] =(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==0 else 1+(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==1 else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==2 else -(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==3 else 0
                        elif i==Nv-1:
                                A[i,j] =-(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-3 else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-2 else 1+(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                        elif i==Nv-2:
                                A[i,j] =-(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-4 else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-3 else 1+(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==Nv-2 else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                        else:
                                A[i,j] =-(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i-2 else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i-1 else 1+(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==i else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i+1 else -(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i+2 else 0
                else:
                        if i==0:
                                A[i,j] =1+(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==0 else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==1 else -(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==2 else 0
                        elif i==1:
                                A[i,j] =(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==0 else 1+(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==1 else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==2 else -(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==3 else 0
                        elif i==Nv-1:
                                A[i,j] =-(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-3 else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-2 else 1+(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                        elif i==Nv-2:
                                A[i,j] =-(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-4 else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-3 else 1+(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==Nv-2 else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                        else:
                                A[i,j] =-(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i-2 else (Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i-1 else 1+(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt/2)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R])) if j==i else -(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i+1 else -(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i+2 else 0
    return A

def Matrix_QQ(R):
    AA=np.zeros(((Nv)*(Nv),(Nv)*(Nv)))
    for a in range(Nv-1):
	    for b in range(Nv-1):
		    if a==b:
			    AA[a*Nv:(a+1)*Nv,(b+1)*Nv:(b+2)*Nv]=-Matrix_B(R,a)
    for a in range(Nv-2):
	    for b in range(Nv-2):
		    if a==b:
			    AA[a*Nv:(a+1)*Nv,(b+2)*Nv:(b+3)*Nv]=-Matrix_C(R,a)
    for a in range(Nv-1):
	    for b in range(Nv-1):
		    if a==b:
			    AA[(a+1)*Nv:(a+2)*Nv,(b)*Nv:(b+1)*Nv]=Matrix_B(R,a+1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    for a in range(Nv-2):
	    for b in range(Nv-2):
		    if a==b:
			    AA[(a+2)*Nv:(a+3)*Nv,(b)*Nv:(b+1)*Nv]=-Matrix_C(R,a+2)
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

for a in range(Nr):
	for b in range(Nr):
		if a==b:
			QQQ[a*(Nv*Nv):(a+1)*(Nv*Nv),b*(Nv*Nv):(b+1)*(Nv*Nv)]=Matrix_QQ(a)


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

for k in range(100):
    print(k)
    f_temp2=np.zeros(shape = (Nr*Nv**2, 1))
    f_temp1=np.zeros(shape = (Nr*Nv**2, 1))

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
    

    
    #for j in range(Nv):
    #           for i in range(Nv):
    #              solu2[j,i]=(f_1[(1)*(Nv)*(Nv)+j*Nv+i]-f_temp2[(1)*(Nv)*(Nv)+j*Nv+i])/f_1[(1)*(Nv)*(Nv)+j*Nv+i]
    #fig = plt.figure()
    #fig.set_dpi(500)
    #plt.contourf(X2, Y2,solu2, cont_lev,cmap='RdBu');
    #ax = plt.gca()
    #ax.spines['left'].set_position('center')
    #ax.spines['left'].set_smart_bounds(True)
    #ax.spines['bottom'].set_position('zero')
    #ax.spines['bottom'].set_smart_bounds(True)
    #ax.spines['right'].set_color('none')
    #ax.spines['top'].set_color('none')
    #ax.xaxis.set_ticks_position('bottom')
    #plt.axis('equal')
    #plt.yticks([per_v[0],per_v[Nv-1]])
    #plt.xticks([pal_v[0],pal_v[Nv-1]])
    #plt.rc('font', size=8)
    #plt.tick_params(labelsize=8)
    #plt.text(pal_v[Nv-1],-0.,r'$\mathcal{v}_\parallel$', fontsize=8)
    #plt.text(-0.,pal_v[Nv-1],r'$\mathcal{v}_\perp$', fontsize=8)
    #plt.text(pal_v[Nv-2],pal_v[Nv-2], "%.2f" % z[1], fontsize=8)
    #plt.text(pal_v[Nv-2],pal_v[Nv-3], k+1, fontsize=8)
    #plt.colorbar(label=r'$Log(F/F_{MAX})$')
    #plt.savefig(f'/Users/user/Desktop/radial/D1/{k}.png')
    #plt.clf()
    #plt.close()
    

num=k+1
for r in range(Nr):
   for j in range(Nv):
       for i in range(Nv):
          solu2[j,i]=np.log10(f_1[(r)*(Nv)*(Nv)+j*Nv+i]/np.max(f_1))
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
   plt.axis('equal')
   plt.yticks([per_v[0],per_v[Nv-1]])
   plt.xticks([pal_v[0],pal_v[Nv-1]])
   plt.rc('font', size=8)
   plt.tick_params(labelsize=8)
   plt.text(pal_v[Nv-1],-0.,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae}$', fontsize=8)
   plt.text(-0.,pal_v[Nv-1],r'$\mathcal{v}_\perp/\mathcal{v}_{Ae}$', fontsize=8)
   plt.text(pal_v[Nv-8],pal_v[Nv-4], r'$r/r_s=$' "%.2f" % z[r], fontsize=8)
   plt.text(pal_v[Nv-14],pal_v[Nv-2], r'$Time \ Iteration \ Number:$' "%.0f" % num, fontsize=8)
   plt.colorbar(label=r'$Log(F/F_{MAX})$')
   plt.savefig(f'/Users/user/Desktop/radial/{r}.png')
   plt.clf()
   plt.close()

#for r in range(1):
#   for j in range(Nv):
#       for i in range(Nv):
#          solu2[j,i]=(f_1[(r)*(Nv)*(Nv)+j*Nv+i]-f_temp[(r)*(Nv)*(Nv)+j*Nv+i])/f_1[(r)*(Nv)*(Nv)+j*Nv+i]
#   fig = plt.figure()
#   fig.set_dpi(500)
#   plt.contourf(X2, Y2,solu2, cont_lev,cmap='RdBu');
#   ax = plt.gca()
#   ax.spines['left'].set_position('center')
#   ax.spines['left'].set_smart_bounds(True)
#   ax.spines['bottom'].set_position('zero')
#   ax.spines['bottom'].set_smart_bounds(True)
#   ax.spines['right'].set_color('none')
#   ax.spines['top'].set_color('none')
#   ax.xaxis.set_ticks_position('bottom')
#   plt.axis('equal')
#   plt.yticks([per_v[0],per_v[Nv-1]])
#   plt.xticks([pal_v[0],pal_v[Nv-1]])
#   plt.rc('font', size=8)
#   plt.tick_params(labelsize=8)
#   plt.text(pal_v[Nv-1],-0.,r'$\mathcal{v}_\parallel$', fontsize=8)
#   plt.text(-0.,pal_v[Nv-1],r'$\mathcal{v}_\perp$', fontsize=8)
#   plt.text(pal_v[Nv-2],pal_v[Nv-2], "%.2f" % z[r], fontsize=8)
#   plt.text(pal_v[Nv-2],pal_v[Nv-3], k+1, fontsize=8)
#   plt.colorbar(label=r'$Log(F/F_{MAX})$')
#   plt.savefig(f'/Users/user/Desktop/radial/{r}.png')
#   plt.clf()
#   plt.close()

Density=np.zeros(shape = (Nr))
for r in range(Nr):
   tempDensity=0
   for j in range(Nv):
      for i in range(Nv):
              tempDensity=tempDensity+2*np.pi*f_1[r*(Nv)*(Nv)+j*Nv+i]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
   Density[r]=tempDensity


plt.figure(figsize=(20,15))
plt.grid()
ax = plt.gca()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlim([z[0],z[Nr-1]])
ax.set_ylim([min(Density),max(Density)])
ax.set_xlabel(r'$r/r_s$', fontsize=28)
ax.set_ylabel(r'$n_e$', fontsize=28)
ax.plot(z,Density,linewidth=3.0, color='k',label="numerical density");
ax.plot(z,max(Density)*(z[0]/z)**2,linewidth=3.0, color='k',linestyle='--',label="analytical 1/r^2");
plt.legend(loc='upper right')
plt.savefig(f'/Users/user/Desktop/radial/density.png')
plt.clf()
plt.close()

Bulk=np.zeros(shape = (Nr))
for r in range(Nr):
   tempBulk=0
   for j in range(Nv):
      for i in range(Nv):
              tempBulk=tempBulk+2*np.pi*pal_v[i]*f_1[r*(Nv)*(Nv)+j*Nv+i]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
   Bulk[r]=tempBulk/Density[r]

print(max(Density))
print(min(Density))
plt.figure(figsize=(20,15))
plt.grid()
ax = plt.gca()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlim([z[0],z[Nr-1]])
ax.set_ylim([min(Bulk),max(Bulk)])
ax.set_xlabel(r'$r/r_s$', fontsize=28)
ax.set_ylabel(r'$U$', fontsize=28)
plt.plot(z,Bulk,linewidth=3.0, color='k');
plt.savefig(f'/Users/user/Desktop/radial/bulk.png')
plt.clf()
plt.close()
