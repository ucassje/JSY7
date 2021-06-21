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
Nv=55
i_solar_r=10
f_solar_r=20
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
   return (r_s**3)*(n(r)*10**6)*(2*np.pi*v_th_function(temperature(r))**3*kappa**1.5)**(-1)*(gamma(kappa+1)/(gamma(kappa-0.5)*gamma(1.5)))*(1.+((b/v_th_function(temperature(r)))**2)/kappa+((a/v_th_function(temperature(r)))**2)/kappa)**(-kappa-1.)#+(U_f/U_solar(r))*10**(-6)*(r_s**3)*(n(r)*10**6)*(np.pi**1.5*v_th_function(T_e*(z[0]/r))**3)**(-1)*(gamma(kappa+1)/(gamma(kappa-0.5)*kappa**1.5))*(1.+((b/(v_th_function(T_e*(z[0]/r))*100000))**2)/kappa+((a/(v_th_function(T_e*(z[0]/r))*100000))**2)/kappa)**(-kappa-1.) #(((7.5*10**9/r_s)/c)**2+0.05*np.exp(-(c-23)**2))*

       
              
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
        return (1/v_Ae_0**2)*(Bol_k*n_0(i_solar_r)*T_e)/(Me*n(x))*2.35*(i_solar_r/x)**(1.35)*(i_solar_r/x**2)
          
def U_solar(x):
        U_0=400000./v_Ae_0
        tempU=0
        for q in range(Nr):
                if z[q]<x:
                        tempU=tempU+electric(z[q])*delz
        return (U_0**2+(2)*tempU)**0.5
num=1
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
   plt.savefig(f'{path_current}figure/{r}.png')
   plt.clf()
   plt.close()
