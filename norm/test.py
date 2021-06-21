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
from scipy import special
from math import e
from tempfile import TemporaryFile
from numpy import exp, loadtxt, pi, sqrt
from lmfit import Parameters, fit_report, minimize
#from lmfit import Model
import lmfit

Nv=31  #velocity step number
i_solar_r=5 #10
f_solar_r=20 #30
path_home="/Users/user/Desktop/test/"
path_lab="/disk/plasma4/syj2/Code/JSY2/"
path_current=path_home
#path_current=path_lab
def n_0(r):
        return 5*(215/r)**2

def B_0(r):
        return 5*(215/r)**2

v_Ae_0=(B_0(215)*10**(-9))/(4.*np.pi*10**(-7)*9.1094*10**(-31)*n_0(215)*10**6)**0.5
print(v_Ae_0)
q=1.6022*(10**(-19))
Me=9.1094*(10**(-31))
Mp=1.6726*(10**(-27))
ratio=(Me/Mp)**0.5
Mv=15*10**6/v_Ae_0  #5*10**7 #(2/3)*5*10**7 
epsilon=8.8542*10**(-12)
pal_v = np.linspace(-Mv, Mv, Nv)
per_v = np.linspace(-Mv, Mv, Nv)
delv=pal_v[1]-pal_v[0]
print(delv)
Nr=50      #radial step number
r_s=696340000.
z=np.linspace(i_solar_r, f_solar_r, Nr)
delz=z[1]-z[0]
print(delz)
Mt=3600*v_Ae_0/r_s
Nt=3600
t=np.linspace(0, Mt, Nt)
delt=(t[1]-t[0])            #time step
print("del t")
print(delt)
Fv=delt/delv
Fvv=delt/(delv)**2
Fz=delt/delz
print("Fv")
print(Fv)
print("Fz")
print(Fz)
print("Fv/Fz")
print(Fv/Fz)

U_f=800000./v_Ae_0
T_e=10*10**5; #5*(10**(5))
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

