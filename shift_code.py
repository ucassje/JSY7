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

Nv=30
Nv2=30+2*30-2  #velocity step number
i_solar_r=5 #10
f_solar_r=20 #30
path_home="/Users/user/Desktop/JSY2/"
path_lab="/disk/plasma4/syj2/Code/JSY2/"
# path_current=path_home
path_current=path_lab
def n_0(r):
        return 1*(215/r)**2

def B_0(r):
        return 10*(215/r)**2

v_Ae_0=(B_0(215)*10**(-9))/(4.*np.pi*10**(-7)*9.1094*10**(-31)*10*n_0(215)*10**6)**0.5
print(v_Ae_0)
q=1.6022*(10**(-19))
Me=9.1094*(10**(-31))
Mp=1.6726*(10**(-27))
ratio=(Me/Mp)**0.5
Mv=15*10**6/v_Ae_0  #5*10**7 #(2/3)*5*10**7 
epsilon=8.8542*10**(-12)
pal_v = np.linspace(-Mv, Mv, Nv2)
per_v = np.linspace(-Mv, Mv, Nv)
delv=pal_v[1]-pal_v[0]
print(delv)
Nr=30      #radial step number
r_s=696340000.
z=np.linspace(i_solar_r, f_solar_r, Nr)
delz=z[1]-z[0]
print(delz)
Mt=0.01
Nt=3
t=np.linspace(0, Mt, Nt-1)
delt=0.5*(t[1]-t[0])            #time step
print(delt)
Fv=delt/delv
Fvv=delt/(delv)**2
Fz=delt/delz
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
print((f_solar_r-i_solar_r)/U_f)
print(((f_solar_r-i_solar_r)/U_f)/delt)

f_1 = np.load('data_next.npy')

f_shift=np.zeros(shape = (Nr*Nv*Nv2, 1))
for r in range(Nr):
        for j in range(Nv):
                for i in range(Nv2):
                        if i%3==0:
                                f_shift[r*(Nv)*(Nv2)+j*Nv2+i]=f_1[r*(Nv)*(Nv)+j*Nv+i//3]
                        elif (i-1)%3==0:
                                f_shift[r*(Nv)*(Nv2)+j*Nv2+i]=f_1[r*(Nv)*(Nv)+j*Nv+(i-1)//3]+(1/3)*(f_1[r*(Nv)*(Nv)+j*Nv+(i+2)//3]-f_1[r*(Nv)*(Nv)+j*Nv+(i-1)//3])
                        elif (i-2)%3==0:
                                print(i)
                                f_shift[r*(Nv)*(Nv2)+j*Nv2+i]=f_1[r*(Nv)*(Nv)+j*Nv+(i-2)//3]+(2/3)*(f_1[r*(Nv)*(Nv)+j*Nv+(i+1)//3]-f_1[r*(Nv)*(Nv)+j*Nv+(i-2)//3])


solu2=np.zeros(shape = (Nv2))
for r in range(Nr):
   for i in range(Nv2):
        solu2[i]=np.log10(f_shift[(r)*(Nv)*(Nv2)+(15)*Nv2+i]/np.max(f_1))
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
   plt.text(-2*delv,-8.7,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae0}$', fontsize=12)
   plt.text(-2*delv,2*delv,r'$Log(F/F_{MAX})$', fontsize=12)
   plt.ylim([-8, 0])
   plt.xlim([-Mv, Mv])
   plt.rc('font', size=8)
   plt.tick_params(labelsize=8)
   plt.savefig(f'{path_current}17/shift/1D{r}.png')
   plt.clf()
   plt.close()
