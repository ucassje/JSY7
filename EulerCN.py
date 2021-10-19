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

Nv=51 #velocity step number
i_solar_r=5 #10
f_solar_r=20 #30
path_home="/Users/user/Desktop/JSY7/"
path_lab="/disk/plasma4/syj2/Code/JSY7/"
# path_current=path_home
path_current=path_lab
def n_0(r):
        return 5*(215/r)**2

def B_0(r):
        return 2*(215/r)**2

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

Nr=60      #radial step number
r_s=696340000.
z=np.linspace(i_solar_r, f_solar_r, Nr)
delz=z[1]-z[0]

Mt=3600*v_Ae_0/r_s
Nt=3600
t=np.linspace(0, Mt, Nt)
delt=(t[1]-t[0])            #time step

Fv=delt/delv
Fvv=delt/(delv)**2
Fz=delt/delz
print(Fv)
print(Fz)
U_f=400000./v_Ae_0
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
#print((f_solar_r-i_solar_r)/U_f)
#print(((f_solar_r-i_solar_r)/U_f)/delt)

Exp=0.5


updatetime=40
timestep=25 #700

#calculate Beta

def n(r):
        return n_0(i_solar_r)*(i_solar_r/r)**2*(U_solar(215)/U_solar(r))

def U_solar(r):
        return U_f*(np.exp(r/10.)-np.exp(-r/10.))/(np.exp(r/10.)+np.exp(-r/10.)) 

def dU_solar(x):
        return U_f*(1./10.)*(2./(np.exp(x/10.)+np.exp(-x/10.)))**2

def cos(r):
        return (1/(1+((r-0*i_solar_r)*Omega/U_solar(r))**2)**0.5)

def temperature(r):
        return T_e*(i_solar_r/r)**(0.8) #T_e*np.exp(-(r-i_solar_r)**2/600) #T_e*np.exp(2/(r-2.2)**0.7) #(0.1*T_e-T_e)/(f_solar_r-i_solar_r)*(r-i_solar_r)+T_e

def v_th_function(T):
        kappa=20
        return ((2)*Bol_k*T/(Me))**0.5/v_Ae_0

def v_th_function_p(T):
        kappa=20
        return ((2)*Bol_k*T/(Mp))**0.5/v_Ae_0


def Kappa_Initial_Core(a,b,r):
   kappac=8 #2
   return (r_s**3)*(n(r)*10**6)*(v_th_function(temperature(r))*v_th_function(temperature(r))**2)**(-1)*(2/(np.pi*(2*kappac-3)))**1.5*(gamma(kappac+1)/gamma(kappac-0.5))*(1.+(2/(2*kappac-3))*((b/v_th_function(temperature(r)))**2)+(2/(2*kappac-3))*((a/v_th_function(temperature(r)))**2))**(-kappac-1.) #(U_f/U_solar(r))*(r_s**3)*(n(r)*10**6)*(2*np.pi*kappa_v_th_function(temperature(r))**3*kappa**1.5)**(-1)*(gamma(kappa+1)/(gamma(kappa-0.5)*gamma(1.5)))*(1.+((b/kappa_v_th_function(temperature(r)))**2)/kappa+((a/kappa_v_th_function(temperature(r)))**2)/kappa)**(-kappa-1.)#+10**(-6)*(r_s**3)*(n(r)*10**6)*(np.pi**1.5*kappa_v_th_function(temperature(r))**3)**(-1)*(gamma(kappa+1)/(gamma(kappa-0.5)*kappa**1.5))*(1.+((b/(kappa_v_th_function(temperature(r))*100000))**2)/kappa+((a/(kappa_v_th_function(temperature(r))*100000))**2)/kappa)**(-kappa-1.) #(((7.5*10**9/r_s)/c)**2+0.05*np.exp(-(c-23)**2))*
#(r_s**3)*(n(r)*10**6)/(v_th_function(temperature(r))**3*np.pi**(3/2))*np.exp(-a**2/v_th_function(temperature(r))**2-b**2/v_th_function(temperature(r))**2)
         
f=np.zeros(shape = (Nv**2, Nr))
for r in range(Nr):
       for j in range(Nv):
              for i in range(Nv):
                 f[j*Nv+i,r]=Kappa_Initial_Core(pal_v[i],per_v[j],z[r])
                 
Mf=np.max(f)

f_1=np.zeros(shape = (Nv**2, Nr))
for r in range(Nr):
        for j in range(Nv):
                for i in range(Nv):
                        f_1[j*Nv+i,r]=Kappa_Initial_Core(pal_v[i],per_v[j],z[r])


Density_next=np.zeros(shape = (Nr))
for r in range(Nr):
        tempDensity=0
        for j in range(Nv):
            for i in range(Nv):
                    if per_v[j]<0:
                            tempDensity=tempDensity
                    else:
                            tempDensity=tempDensity+2*np.pi*f_1[j*Nv+i,r]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
        Density_next[r]=tempDensity/(r_s**3)

Bulk=np.zeros(shape = (Nr))
for r in range(Nr):
   tempBulk=0
   for j in range(Nv):
      for i in range(Nv):
              if per_v[j]>=0:
                      tempBulk=tempBulk+2*np.pi*pal_v[i]*f_1[j*Nv+i,r]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
              else:
                      tempBulk=tempBulk
   Bulk[r]=tempBulk/((r_s**3)*Density_next[r])

Temperature_pal=np.zeros(shape = (Nr))
for r in range(Nr):
        temptemp=0
        for j in range(Nv):
            for i in range(Nv):
                    if per_v[j]<0:
                            temptemp=temptemp
                    else:
                            temptemp=temptemp+2*np.pi*(pal_v[i]**2)*f_1[j*Nv+i,r]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
        Temperature_pal[r]=v_Ae_0**2*Me*temptemp/((r_s**3)*Density_next[r]*Bol_k)

Temperature_per=np.zeros(shape = (Nr))
for r in range(Nr):
        temptemp=0
        for j in range(Nv):
            for i in range(Nv):
                    if per_v[j]<0:
                            temptemp=temptemp
                    else:
                            temptemp=temptemp+2*np.pi*(per_v[j]**2)*f_1[j*Nv+i,r]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
        Temperature_per[r]=v_Ae_0**2*Me*temptemp/(2*(r_s**3)*Density_next[r]*Bol_k) 

Temperature_tol=np.zeros(shape = (Nr))
Temperature_tol=(1/3)*(Temperature_pal+2*Temperature_per)


ratio_r=np.zeros(shape = (Nv**2, Nr))
for r in range(Nr-1):
        for j in range(Nv):
                for i in range(Nv):
                        ratio_r[j*Nv+i,r]=abs(f_1[j*Nv+i,r]/f_1[j*Nv+i,r+1])


d_pal_ne=np.zeros(shape = (Nv, Nr))
for r in range(Nr):
        for j in range(Nv):
                d_pal_ne[j,r]=abs(f_1[j*Nv+0,r]/f_1[j*Nv+1,r])#abs(f_1[r*(Nv)*(Nv)+j*Nv]-f_1[r*(Nv)*(Nv)+j*Nv+1])

d_pal_po=np.zeros(shape = (Nv, Nr))
for r in range(Nr):
        for j in range(Nv):
                d_pal_po[j,r]=abs(f_1[j*Nv+Nv-1,r]/f_1[j*Nv+Nv-2,r])#abs(f_1[r*(Nv)*(Nv)+j*Nv+Nv-1]-f_1[r*(Nv)*(Nv)+j*Nv+Nv-2])

d_per_ne=np.zeros(shape = (Nv, Nr))
for r in range(Nr):
        for i in range(Nv):
                d_per_ne[i,r]=abs(f_1[i,r]/f_1[1*Nv+i,r])#abs(f_1[r*(Nv)*(Nv)+i]-f_1[r*(Nv)*(Nv)+1*Nv+i])

d_per_po=np.zeros(shape = (Nv, Nr))
for r in range(Nr):
        for i in range(Nv):
                d_per_po[i,r]=abs(f_1[(Nv-1)*Nv+i,r]/f_1[(Nv-2)*Nv+i,r])#abs(f_1[r*(Nv)*(Nv)+(Nv-1)*Nv+i]-f_1[r*(Nv)*(Nv)+(Nv-2)*Nv+i])

d_pal_ne_per_ne=np.zeros(shape = (Nr, 1))
for r in range(Nr):
        d_pal_ne_per_ne[r]=abs(f_1[0,r]/f_1[1*Nv+1,r])#abs(f_1[r*(Nv)*(Nv)]-f_1[r*(Nv)*(Nv)+1*Nv+1])

d_pal_ne_per_po=np.zeros(shape = (Nr, 1))
for r in range(Nr):
        d_pal_ne_per_po[r]=abs(f_1[(Nv-1)*Nv,r]/f_1[(Nv-2)*Nv+1,r])#abs(f_1[r*(Nv)*(Nv)+(Nv-1)*Nv]-f_1[r*(Nv)*(Nv)+(Nv-2)*Nv+1])          

d_pal_po_per_ne=np.zeros(shape = (Nr, 1))
for r in range(Nr):
        d_pal_po_per_ne[r]=abs(f_1[Nv-1,r]/f_1[1*Nv+Nv-2,r])#abs(f_1[r*(Nv)*(Nv)+Nv-1]-f_1[r*(Nv)*(Nv)+1*Nv+Nv-2])

d_pal_po_per_po=np.zeros(shape = (Nr, 1))
for r in range(Nr):
        d_pal_po_per_po[r]=abs(f_1[(Nv-1)*Nv+Nv-1,r]/f_1[(Nv-2)*Nv+Nv-2,r])#abs(f_1[r*(Nv)*(Nv)+(Nv-1)*Nv+Nv-1]-f_1[r*(Nv)*(Nv)+(Nv-2)*Nv+Nv-2])
         
Col=4*np.pi/(r_s**2*v_Ae_0**4)*(q**2/(4*np.pi*epsilon*Me))**2*25

def Collision_Core(a,b,x):
    for r in range(Nr):
        if abs(x-z[r])<0.5*delz:
                l=r
    kappa=50.
    d=0
    if (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])==0:
            d=(r_s**3)*(Density_next[l])/(v_th_function(Temperature_tol[l])**3*np.pi**(3/2))*np.exp(-a**2/v_th_function(Temperature_tol[l])**2-b**2/v_th_function(Temperature_tol[l])**2) #(r_s**3)*(n(r)*10**6)*(np.pi**1.5*v_th_function(temperature(r))**3)**(-1)*(gamma(kappa+1)/(gamma(kappa-0.5)*kappa**1.5))*(1.+((b/v_th_function(temperature(r)))**2)/kappa+((a/v_th_function(temperature(r)))**2)/kappa)**(-kappa-1.)            
    else:
            d=(r_s**3)*(Density_next[l])/(v_th_function(Temperature_tol[l])**3*np.pi**(3/2))*np.exp(-a**2/v_th_function(Temperature_tol[l])**2-b**2/v_th_function(Temperature_tol[l])**2) #(r_s**3)*(n(r)*10**6)*(np.pi**1.5*v_th_function(temperature(r))**3)**(-1)*(gamma(kappa+1)/(gamma(kappa-0.5)*kappa**1.5))*(1.+((b/v_th_function(temperature(r)))**2)/kappa+((a/v_th_function(temperature(r)))**2)/kappa)**(-kappa-1.)            
    return d

def Collision_Proton(a,b,x):
    for r in range(Nr):
        if abs(x-z[r])<0.5*delz:
                l=r
    kappa=50.
    d=0
    if (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])==0:
            d=(r_s**3)*(Density_next[l])/(v_th_function_p(Temperature_tol[l])**3*np.pi**(3/2))*np.exp(-a**2/v_th_function_p(Temperature_tol[l])**2-b**2/v_th_function_p(Temperature_tol[l])**2) #(r_s**3)*(n(r)*10**6)*(np.pi**1.5*v_th_function_p(temperature(r))**3)**(-1)*(gamma(kappa+1)/(gamma(kappa-0.5)*kappa**1.5))*(1.+((b/v_th_function_p(temperature(r)))**2)/kappa+((a/v_th_function_p(temperature(r)))**2)/kappa)**(-kappa-1.)
    else:
            d=(r_s**3)*(Density_next[l])/(v_th_function_p(Temperature_tol[l])**3*np.pi**(3/2))*np.exp(-a**2/v_th_function_p(Temperature_tol[l])**2-b**2/v_th_function_p(Temperature_tol[l])**2) #(r_s**3)*(n(r)*10**6)*(np.pi**1.5*v_th_function_p(temperature(r))**3)**(-1)*(gamma(kappa+1)/(gamma(kappa-0.5)*kappa**1.5))*(1.+((b/v_th_function_p(temperature(r)))**2)/kappa+((a/v_th_function_p(temperature(r)))**2)/kappa)**(-kappa-1.)
    return d

def G_per_2e(a,b,x):
    for r in range(Nr):
        if abs(x-z[r])<0.5*delz:
                l=r
    d=0
    if (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])<1 and (a**2+b**2)**0.5>0:
        d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*((2/(3*v_th_function(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function(Temperature_tol[l])**3)-(4/15)*(b**2/v_th_function(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function(Temperature_tol[l])**5)+(2/5)*(b**2*(a**2+b**2)/v_th_function(Temperature_tol[l])**5))
    elif (a**2+b**2)==0:
        d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*((2/(3*v_th_function(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function(Temperature_tol[l])**3)-(4/15)*(b**2/v_th_function(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function(Temperature_tol[l])**5)+(2/5)*(b**2*(a**2+b**2)/v_th_function(Temperature_tol[l])**5))
    else:
        d=(r_s**3)*(Density_next[l])*v_th_function(Temperature_tol[l])**2*(0.5/(a**2+b**2)**1.5-1.5*b**2/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function(Temperature_tol[l])))+2*(r_s**3)*(Density_next[l])*b**2/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function(Temperature_tol[l]))
    return d


def G_per_e(a,b,x):
    for r in range(Nr):
        if abs(x-z[r])<0.5*delz:
                l=r
    d=0
    if (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])<1 and abs(b)>0:
        d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*(1/b)*((2/(3*v_th_function(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function(Temperature_tol[l])**5))
    elif (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])<1 and abs(b)==0:
        d=0*(2*(2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*(1/(b+delv))*((2/(3*v_th_function(Temperature_tol[l])))-(2/15)*((a**2+(b+delv)**2)/v_th_function(Temperature_tol[l])**3)+(1/10)*((a**2+(b+delv)**2)**2/v_th_function(Temperature_tol[l])**5)))-0*(2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*(1/(b+2*delv))*((2/(3*v_th_function(Temperature_tol[l])))-(2/15)*((a**2+(b+2*delv)**2)/v_th_function(Temperature_tol[l])**3)+(1/10)*((a**2+(b+2*delv)**2)**2/v_th_function(Temperature_tol[l])**5))))
    elif (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])>=1 and abs(b)>0:
        d=(r_s**3)*(Density_next[l])*v_th_function(Temperature_tol[l])**2*(1/b)*(0.5/(a**2+b**2)**1.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function(Temperature_tol[l])))
    elif (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])>=1 and abs(b)==0:
        d=0*(2*((r_s**3)*(Density_next[l])*v_th_function(Temperature_tol[l])**2*(1/(b+delv))*(0.5/(a**2+(b+delv)**2)**1.5)*((2/np.pi**0.5)*((a**2+(b+delv)**2)**0.5/v_th_function(Temperature_tol[l]))*np.exp(-(a**2+(b+delv)**2)/v_th_function(Temperature_tol[l])**2)+(2*(a**2+(b+delv)**2)/v_th_function(Temperature_tol[l])**2-1)*special.erf((a**2+(b+delv)**2)**0.5/v_th_function(Temperature_tol[l]))))-0*((r_s**3)*(Density_next[l])*v_th_function(Temperature_tol[l])**2*(1/(b+2*delv))*(0.5/(a**2+(b+2*delv)**2)**1.5)*((2/np.pi**0.5)*((a**2+(b+2*delv)**2)**0.5/v_th_function(Temperature_tol[l]))*np.exp(-(a**2+(b+2*delv)**2)/v_th_function(Temperature_tol[l])**2)+(2*(a**2+(b+2*delv)**2)/v_th_function(Temperature_tol[l])**2-1)*special.erf((a**2+(b+2*delv)**2)**0.5/v_th_function(Temperature_tol[l])))))
    return d

def G_per_ee(a,b,x):
    for r in range(Nr):
        if abs(x-z[r])<0.5*delz:
                l=r
    d=0
    if (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])<1 and abs(b)==0:
        d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*((2/(3*v_th_function(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function(Temperature_tol[l])**5))#2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*((2/(3*v_th_function(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function(Temperature_tol[l])**3)-(4/15)*(b**2/v_th_function(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function(Temperature_tol[l])**5)+(2/5)*(b**2*(a**2+b**2)/v_th_function(Temperature_tol[l])**5)) #2*(r_s**3)*(n(r)*10**6)/(np.pi**0.5)*((2/(3*v_th_function(temperature(r))))-(2/15)*((a**2+b**2)/v_th_function(temperature(r))**3)+(1/10)*((a**2+b**2)**2/v_th_function(temperature(r))**5))
    elif (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])>=1 and abs(b)==0:
        d=(r_s**3)*(Density_next[l])*v_th_function(Temperature_tol[l])**2*(0.5/(a**2+b**2)**1.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function(Temperature_tol[l])))#(r_s**3)*(Density_next[l])*v_th_function(Temperature_tol[l])**2*(0.5/(a**2+b**2)**1.5-1.5*b**2/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function(Temperature_tol[l])))+2*(r_s**3)*(Density_next[l])*b**2/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function(Temperature_tol[l])) #(r_s**3)*(n(r)*10**6)*v_th_function(temperature(r))**2*(0.5/(a**2+b**2)**1.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function(temperature(r)))*np.exp(-(a**2+b**2)/v_th_function(temperature(r))**2)+(2*(a**2+b**2)/v_th_function(temperature(r))**2-1)*special.erf((a**2+b**2)**0.5/v_th_function(temperature(r))))
    return d

def G_pal_2e(a,b,x):
    for r in range(Nr):
        if abs(x-z[r])<0.5*delz:
                l=r
    d=0
    if (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])<1 and (a**2+b**2)**0.5>0:
        d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*((2/(3*v_th_function(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function(Temperature_tol[l])**3)-(4/15)*(a**2/v_th_function(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function(Temperature_tol[l])**5)+(2/5)*(a**2*(a**2+b**2)/v_th_function(Temperature_tol[l])**5))
    elif (a**2+b**2)==0:
        d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*((2/(3*v_th_function(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function(Temperature_tol[l])**3)-(4/15)*(a**2/v_th_function(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function(Temperature_tol[l])**5)+(2/5)*(a**2*(a**2+b**2)/v_th_function(Temperature_tol[l])**5))
    else:
        d=(r_s**3)*(Density_next[l])*v_th_function(Temperature_tol[l])**2*(0.5/(a**2+b**2)**1.5-1.5*a**2/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function(Temperature_tol[l])))+2*(r_s**3)*(Density_next[l])*a**2/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function(Temperature_tol[l]))
    return d


def G_pal_per_e(a,b,x):
    for r in range(Nr):
        if abs(x-z[r])<0.5*delz:
                l=r
    d=0
    if (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])<1 and (a**2+b**2)**0.5>0:
        d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*(-(4/15)*(a*b/v_th_function(Temperature_tol[l])**3)+(2/5)*(a*b*(a**2+b**2)/v_th_function(Temperature_tol[l])**5))
    elif (a**2+b**2)==0:
        d=0
    else:
        d=(-(r_s**3)*(Density_next[l])*v_th_function(Temperature_tol[l])**2*(1.5*a*b/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function(Temperature_tol[l])))+2*(r_s**3)*(Density_next[l])*a*b/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function(Temperature_tol[l])))
    return d




def H_per(a,b,x):
        return 0

def H_pal(a,b,x):
        return 0

def G_per_2p(a,b,x):
    for r in range(Nr):
        if abs(x-z[r])<0.5*delz:
                l=r
    d=0
    if (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])<1:
        d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*((2/(3*v_th_function_p(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function_p(Temperature_tol[l])**3)-(4/15)*(b**2/v_th_function_p(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function_p(Temperature_tol[l])**5)+(2/5)*(b**2*(a**2+b**2)/v_th_function_p(Temperature_tol[l])**5))
    else:
        d=(r_s**3)*(Density_next[l])*v_th_function_p(Temperature_tol[l])**2*(0.5/(a**2+b**2)**1.5-1.5*b**2/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])))+2*(r_s**3)*(Density_next[l])*b**2/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l]))
    return d

def G_per_p(a,b,x):
    for r in range(Nr):
        if abs(x-z[r])<0.5*delz:
                l=r
    d=0
    if (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])<1 and abs(b)>0:
        d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*(1/b)*((2/(3*v_th_function_p(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function_p(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function_p(Temperature_tol[l])**5))
    elif (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])<1 and abs(b)==0:
        d=0*(2*(2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*(1/(b+delv))*((2/(3*v_th_function_p(Temperature_tol[l])))-(2/15)*((a**2+(b+delv)**2)/v_th_function_p(Temperature_tol[l])**3)+(1/10)*((a**2+(b+delv)**2)**2/v_th_function_p(Temperature_tol[l])**5)))-0*(2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*(1/(b+2*delv))*((2/(3*v_th_function_p(Temperature_tol[l])))-(2/15)*((a**2+(b+2*delv)**2)/v_th_function_p(Temperature_tol[l])**3)+(1/10)*((a**2+(b+2*delv)**2)**2/v_th_function_p(Temperature_tol[l])**5))))
    elif (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])>=1 and abs(b)>0:
        d=(r_s**3)*(Density_next[l])*v_th_function_p(Temperature_tol[l])**2*(1/b)*(0.5/(a**2+b**2)**1.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])))
    elif (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])>=1 and abs(b)==0:
        d=0*(2*((r_s**3)*(Density_next[l])*v_th_function_p(Temperature_tol[l])**2*(1/(b+delv))*(0.5/(a**2+(b+delv)**2)**1.5)*((2/np.pi**0.5)*((a**2+(b+delv)**2)**0.5/v_th_function_p(Temperature_tol[l]))*np.exp(-(a**2+(b+delv)**2)/v_th_function_p(Temperature_tol[l])**2)+(2*(a**2+(b+delv)**2)/v_th_function_p(Temperature_tol[l])**2-1)*special.erf((a**2+(b+delv)**2)**0.5/v_th_function_p(Temperature_tol[l]))))-0*((r_s**3)*(Density_next[l])*v_th_function_p(Temperature_tol[l])**2*(1/(b+2*delv))*(0.5/(a**2+(b+2*delv)**2)**1.5)*((2/np.pi**0.5)*((a**2+(b+2*delv)**2)**0.5/v_th_function_p(Temperature_tol[l]))*np.exp(-(a**2+(b+2*delv)**2)/v_th_function_p(Temperature_tol[l])**2)+(2*(a**2+(b+2*delv)**2)/v_th_function_p(Temperature_tol[l])**2-1)*special.erf((a**2+(b+2*delv)**2)**0.5/v_th_function_p(Temperature_tol[l])))))    
    return d

def G_per_pp(a,b,x):
    for r in range(Nr):
        if abs(x-z[r])<0.5*delz:
                l=r
    d=0
    if (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])<1 and abs(b)==0:
        d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*((2/(3*v_th_function_p(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function_p(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function_p(Temperature_tol[l])**5))#2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*((2/(3*v_th_function_p(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function_p(Temperature_tol[l])**3)-(4/15)*(b**2/v_th_function_p(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function_p(Temperature_tol[l])**5)+(2/5)*(b**2*(a**2+b**2)/v_th_function_p(Temperature_tol[l])**5)) #2*(r_s**3)*(n(r)*10**6)/(np.pi**0.5)*((2/(3*v_th_function_p(temperature(r))))-(2/15)*((a**2+b**2)/v_th_function_p(temperature(r))**3)+(1/10)*((a**2+b**2)**2/v_th_function_p(temperature(r))**5))
    elif (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])>=1 and abs(b)==0:
        d=(r_s**3)*(Density_next[l])*v_th_function_p(Temperature_tol[l])**2*(0.5/(a**2+b**2)**1.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])))#(r_s**3)*(Density_next[l])*v_th_function_p(Temperature_tol[l])**2*(0.5/(a**2+b**2)**1.5-1.5*b**2/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])))+2*(r_s**3)*(Density_next[l])*b**2/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])) #(r_s**3)*(n(r)*10**6)*v_th_function_p(temperature(r))**2*(0.5/(a**2+b**2)**1.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function_p(temperature(r)))*np.exp(-(a**2+b**2)/v_th_function_p(temperature(r))**2)+(2*(a**2+b**2)/v_th_function_p(temperature(r))**2-1)*special.erf((a**2+b**2)**0.5/v_th_function_p(temperature(r))))
    return d


def G_pal_2p(a,b,x):
    for r in range(Nr):
        if abs(x-z[r])<0.5*delz:
                l=r
    d=0
    if (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])<1:
        d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*((2/(3*v_th_function_p(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function_p(Temperature_tol[l])**3)-(4/15)*(a**2/v_th_function_p(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function_p(Temperature_tol[l])**5)+(2/5)*(a**2*(a**2+b**2)/v_th_function_p(Temperature_tol[l])**5))
    else:
        d=(r_s**3)*(Density_next[l])*v_th_function_p(Temperature_tol[l])**2*(0.5/(a**2+b**2)**1.5-1.5*a**2/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])))+2*(r_s**3)*(Density_next[l])*a**2/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l]))
    return d

def G_pal_per_p(a,b,x):
    for r in range(Nr):
        if abs(x-z[r])<0.5*delz:
                l=r
    d=0
    if (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])<1:
        d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*(-(4/15)*(a*b/(v_th_function_p(Temperature_tol[l]))**3)+(2/5)*(a*b*(a**2+b**2)/(v_th_function_p(Temperature_tol[l]))**5))
    else:
        d=(-(r_s**3)*(Density_next[l])*v_th_function_p(Temperature_tol[l])**2*(1.5*a*b/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])))+2*(r_s**3)*(Density_next[l])*a*b/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])))
    return d

def H_palp(a,b,x):
    for r in range(Nr):
        if abs(x-z[r])<0.5*delz:
                l=r
    d=0
    if (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])<1:
            d=(r_s**3)*(Density_next[l])*(4/np.pi**0.5)*(a/v_th_function_p(Temperature_tol[l]))*(-(1/3)*(1/v_th_function_p(Temperature_tol[l])**2)+(1/5)*((a**2+b**2)/v_th_function_p(Temperature_tol[l])**4))
    else:
            d=(r_s**3)*(Density_next[l])*(1/v_th_function_p(Temperature_tol[l]))*((2/np.pi**0.5)*(a/(a**2+b**2))*np.exp(-(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2)-(a*v_th_function_p(Temperature_tol[l]))/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])))
    return d

def H_perp(a,b,x):
    for r in range(Nr):
        if abs(x-z[r])<0.5*delz:
                l=r
    d=0
    if (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])<1:
            d=(r_s**3)*(Density_next[l])*(4/np.pi**0.5)*(b/v_th_function_p(Temperature_tol[l]))*(-(1/3)*(1/v_th_function_p(Temperature_tol[l])**2)+(1/5)*((a**2+b**2)/v_th_function_p(Temperature_tol[l])**4))
    else:
            d=(r_s**3)*(Density_next[l])*(1/v_th_function_p(Temperature_tol[l]))*((2/np.pi**0.5)*(b/(a**2+b**2))*np.exp(-(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2)-(b*v_th_function_p(Temperature_tol[l]))/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])))
    return d





def rect_v(x):
	return 1#0 if abs(x)>=abs(pal_v[0]) else 1

def dcos(x):
        return -1/(1+(x*Omega/U_solar(x))**2)**1.5*(((Omega/U_solar(x))**2*(x-0*i_solar_r))-(x-0*i_solar_r)**2*(Omega**2/U_solar(x)**3)*dU_solar(x))


def B(x):
        return B_0(i_solar_r)*(i_solar_r/x)**2*(1+((x-0*i_solar_r)*Omega/U_solar(x))**2)**0.5

def dlnB(x):
        return (np.log(B(x+delz))-np.log(B(x-delz)))/(2*delz)

 



e_col=np.zeros(shape = (Nr))
for r in range(Nr):
        temp=0
        for j in range(Nv):
                for i in range(Nv):
                        if per_v[j]<0:
                                temp=temp
                        else:
                                temp=temp+2*np.pi*pal_v[i]*4*np.pi*Collision_Core(pal_v[i],per_v[j],z[r])*f_1[j*Nv+i,r]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
        e_col[r]=Col*temp



e_col_G1=np.zeros(shape = (Nr))
for r in range(Nr):
        temp=0
        for j in range(Nv):
                for i in range(Nv):
                        if per_v[j]<0:
                                temp=temp
                        elif per_v[j]>=0 and j!=0 and j!=Nv-1 and j!=1 and j!=Nv-2:
                                temp=temp+2*np.pi*pal_v[i]*0.5*G_per_2e(pal_v[i],per_v[j],z[r])*((f_1[(j+2)*Nv+i,r]-2*f_1[j*Nv+i,r]+f_1[(j-2)*Nv+i,r])/(4*delv**2))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
        e_col_G1[r]=Col*temp


e_col_G2=np.zeros(shape = (Nr))
for r in range(Nr):
        temp=0
        for j in range(Nv):
                for i in range(Nv):
                        if per_v[j]<0:
                                temp=temp
                        elif per_v[j]>=0 and i!=0 and i!=Nv-1 and i!=1 and i!=Nv-2:
                                temp=temp+2*np.pi*pal_v[i]*0.5*G_pal_2e(pal_v[i],per_v[j],z[r])*((f_1[j*Nv+i+2,r]-2*f_1[j*Nv+i,r]+f_1[j*Nv+i-2,r])/(4*delv**2))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
        e_col_G2[r]=Col*temp


e_col_G3=np.zeros(shape = (Nr))
for r in range(Nr):
        temp=0
        for j in range(Nv):
                for i in range(Nv):
                        if per_v[j]<0:
                                temp=temp
                        elif per_v[j]>=0 and i!=0 and i!=Nv-1 and j!=0 and j!=Nv-1:
                                temp=temp+2*np.pi*pal_v[i]*G_pal_per_e(pal_v[i],per_v[j],z[r])*((f_1[(j+1)*Nv+i+1,r]-f_1[(j+1)*Nv+i-1,r]-f_1[(j-1)*Nv+i+1,r]+f_1[(j-1)*Nv+i-1,r])/(4*delv**2))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
        e_col_G3[r]=Col*temp

e_col_G4=np.zeros(shape = (Nr))
for r in range(Nr):
        temp=0
        for j in range(Nv):
                for i in range(Nv):
                        if per_v[j]<0:
                                temp=temp
                        elif per_v[j]>0 and j!=0 and j!=Nv-1:
                                temp=temp+2*np.pi*pal_v[i]*0.5*G_per_e(pal_v[i],per_v[j],z[r])*((f_1[(j+1)*Nv+i,r]-f_1[(j-1)*Nv+i,r])/(2*delv))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
                        elif per_v[j]==0 and j!=0 and j!=Nv-1 and j!=1 and j!=Nv-2:
                                temp=temp+2*np.pi*pal_v[i]*0.5*G_per_ee(pal_v[i],per_v[j],z[r])*((f_1[(j+2)*Nv+i,r]-2*f_1[j*Nv+i,r]+f_1[(j-2)*Nv+i,r])/(4*delv**2))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
        e_col_G4[r]=Col*temp




p_col=np.zeros(shape = (Nr))
for r in range(Nr):
        temp=0
        for j in range(Nv):
                for i in range(Nv):
                        if per_v[j]<0:
                                temp=temp
                        else:
                                temp=temp+2*np.pi*pal_v[i]*4*np.pi*(Me/Mp)*Collision_Proton(pal_v[i],per_v[j],z[r])*f_1[j*Nv+i,r]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
        p_col[r]=Col*temp



p_col_G1=np.zeros(shape = (Nr))
for r in range(Nr):
        temp=0
        for j in range(Nv):
                for i in range(Nv):
                        if per_v[j]<0:
                                temp=temp
                        elif per_v[j]>=0 and j!=0 and j!=Nv-1 and j!=1 and j!=Nv-2:
                                temp=temp+2*np.pi*pal_v[i]*0.5*G_per_2p(pal_v[i],per_v[j],z[r])*((f_1[(j+2)*Nv+i,r]-2*f_1[j*Nv+i,r]+f_1[(j-2)*Nv+i,r])/(4*delv**2))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
        p_col_G1[r]=Col*temp


p_col_G2=np.zeros(shape = (Nr))
for r in range(Nr):
        temp=0
        for j in range(Nv):
                for i in range(Nv):
                        if per_v[j]<0:
                                temp=temp
                        elif per_v[j]>=0 and i!=0 and i!=Nv-1 and i!=1 and i!=Nv-2:
                                temp=temp+2*np.pi*pal_v[i]*0.5*G_pal_2p(pal_v[i],per_v[j],z[r])*((f_1[j*Nv+i+2,r]-2*f_1[j*Nv+i,r]+f_1[j*Nv+i-2,r])/(4*delv**2))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
        p_col_G2[r]=Col*temp


p_col_G3=np.zeros(shape = (Nr))
for r in range(Nr):
        temp=0
        for j in range(Nv):
                for i in range(Nv):
                        if per_v[j]<0:
                                temp=temp
                        elif per_v[j]>=0 and i!=0 and i!=Nv-1 and j!=0 and j!=Nv-1:
                                temp=temp+2*np.pi*pal_v[i]*G_pal_per_p(pal_v[i],per_v[j],z[r])*((f_1[(j+1)*Nv+i+1,r]-f_1[(j+1)*Nv+i-1,r]-f_1[(j-1)*Nv+i+1,r]+f_1[(j-1)*Nv+i-1,r])/(4*delv**2))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
        p_col_G3[r]=Col*temp

p_col_G4=np.zeros(shape = (Nr))
for r in range(Nr):
        temp=0
        for j in range(Nv):
                for i in range(Nv):
                        if per_v[j]<0:
                                temp=temp
                        elif per_v[j]>0 and j!=0 and j!=Nv-1:
                                temp=temp+2*np.pi*pal_v[i]*0.5*G_per_p(pal_v[i],per_v[j],z[r])*((f_1[(j+1)*Nv+i,r]-f_1[(j-1)*Nv+i,r])/(2*delv))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
                        elif per_v[j]==0 and j!=0 and j!=Nv-1 and j!=1 and j!=Nv-2:
                                temp=temp+2*np.pi*pal_v[i]*0.5*G_per_pp(pal_v[i],per_v[j],z[r])*((f_1[(j+2)*Nv+i,r]-2*f_1[j*Nv+i,r]+f_1[(j-2)*Nv+i,r])/(4*delv**2))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
        p_col_G4[r]=Col*temp



p_col_H1=np.zeros(shape = (Nr))
for r in range(Nr):
        temp=0
        for j in range(Nv):
                for i in range(Nv):
                        if per_v[j]<0:
                                temp=temp
                        elif per_v[j]>=0 and j!=0 and j!=Nv-1:
                                temp=temp+2*np.pi*pal_v[i]*H_perp(pal_v[i],per_v[j],z[r])*((f_1[(j+1)*Nv+i,r]-f_1[(j-1)*Nv+i,r])/(2*delv))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
        p_col_H1[r]=Col*temp

p_col_H2=np.zeros(shape = (Nr))
for r in range(Nr):
        temp=0
        for j in range(Nv):
                for i in range(Nv):
                        if per_v[j]<0:
                                temp=temp
                        elif per_v[j]>=0 and i!=0 and i!=Nv-1:
                                temp=temp+2*np.pi*pal_v[i]*H_palp(pal_v[i],per_v[j],z[r])*((f_1[j*Nv+i+1,r]-f_1[j*Nv+i-1,r])/(2*delv))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
        p_col_H2[r]=Col*temp



        
def electric(x):
        for r in range(Nr):
                if abs(x-z[r])<0.5*delz:
                        l=r
        if l!=0:
                E=-(Bulk[l]/(timestep*delt*cos(x)))-(1/((r_s**3)*Density_next[l]*cos(x)))*(e_col[l]+e_col_G1[l]+e_col_G2[l]+e_col_G3[l]+e_col_G4[l]+p_col[l]+p_col_G1[l]+p_col_G2[l]+p_col_G3[l]+p_col_G4[l]+p_col_H1[l]+p_col_H2[l])+U_solar(x)*dU_solar(x)/(cos(x)**2)+(1/v_Ae_0**2)*(Bol_k)/(Me*Density_next[l])*(Density_next[l]*Temperature_pal[l]-Density_next[l-1]*Temperature_pal[l-1])/delz+2*(1/v_Ae_0**2)*(Bol_k)/(Me)*dcos(x)/cos(x)*Temperature_pal[l]+(1/v_Ae_0**2)*(Bol_k)/(Me)*dlnB(x)*Temperature_per[l]+(1/v_Ae_0**2)*(2*Bol_k)/(Me*x)*Temperature_pal[l]#+(1/Density_next[l])*(Density_next[l]*Bulk_next[l]-Density_pre[l]*Bulk_pre[l])/(10*delt)+(Bulk_next[l]/cos(x))*dU_solar(x)+Bulk_next[l]*(dU_solar(x)/cos(x)+U_solar(x)*dcos_1(x))+(U_solar(x)/cos(x))*Bulk_next[l]/x+(U_solar(x)/(cos(x)*Density_next[l]))*(Density_next[l]*Bulk_next[l]-Density_next[l-1]*Bulk_next[l-1])/delz
        else:
                E=-(Bulk[l]/(timestep*delt*cos(x)))-(1/((r_s**3)*Density_next[l]*cos(x)))*(e_col[l]+e_col_G1[l]+e_col_G2[l]+e_col_G3[l]+e_col_G4[l]+p_col[l]+p_col_G1[l]+p_col_G2[l]+p_col_G3[l]+p_col_G4[l]+p_col_H1[l]+p_col_H2[l])+U_solar(x)*dU_solar(x)/(cos(x)**2)+(1/v_Ae_0**2)*(Bol_k)/(Me*Density_next[l])*(Density_next[l+1]*Temperature_pal[l+1]-Density_next[l]*Temperature_pal[l])/delz+2*(1/v_Ae_0**2)*(Bol_k)/(Me)*dcos(x)/cos(x)*Temperature_pal[l]+(1/v_Ae_0**2)*(Bol_k)/(Me)*dlnB(x)*Temperature_per[l]+(1/v_Ae_0**2)*(2*Bol_k)/(Me*x)*Temperature_pal[l]#+(1/Density_next[l])*(Density_next[l]*Bulk_next[l]-Density_pre[l]*Bulk_pre[l])/(10*delt)+(Bulk_next[l]/cos(x))*dU_solar(x)+Bulk_next[l]*(dU_solar(x)/cos(x)+U_solar(x)*dcos_1(x))+(U_solar(x)/cos(x))*Bulk_next[l]/x+(U_solar(x)/(cos(x)*Density_next[l]))*(Density_next[l]*Bulk_next[l]-Density_next[l-1]*Bulk_next[l-1])/delz
        return E


def Matrix_A(R,M):
    A=np.zeros(((Nv),(Nv)))
    for i in range(Nv):
            for j in range(Nv):
                    if R==0:
                            if i==0:
                                    A[i,j] =1+0*rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))-0*rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-0*rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==0 else 0*rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+0*rect_v(pal_v[i])*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+0*rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==1 else 0*rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==2 else 0
                            elif i==1:
                                    A[i,j] =-0*rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-0*rect_v(pal_v[i])*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-0*rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==0 else 1+0*rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))-0*rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-0*rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==1 else 0*rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+0*rect_v(pal_v[i])*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+0*rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==2 else 0*rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==3 else 0
                            elif i==Nv-1:
                                    A[i,j] =0*rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-3 else -0*rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-0*rect_v(pal_v[i])*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-0*rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-2 else 1+0*rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))-0*rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-0*rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-1 else 0
                            elif i==Nv-2:
                                    A[i,j] =0*rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-4 else -0*rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-0*rect_v(pal_v[i])*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-0*rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-3 else 1+0*rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))-0*rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-0*rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-2 else 0*rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+0*rect_v(pal_v[i])*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+0*rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                            else:
                                    A[i,j] =0*rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i-2 else -0*rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-0*rect_v(pal_v[i])*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-0*rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i-1 else 1+0*rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))-0*rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-0*rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i else 0*rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+0*rect_v(pal_v[i])*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+0*rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i+1 else 0*rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i+2 else 0
                    elif R==Nr-1:
                            if i==0:
                                    A[i,j] =1+Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])-Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R])-Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))+rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))-rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==0 else rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==1 else rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==2 else 0
                            elif i==1:
                                    A[i,j] =-rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==0 else 1+Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])-Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R])-Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))+rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))-rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==1 else rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==2 else rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==3 else 0
                            elif i==Nv-1:
                                    A[i,j] =rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-3 else -rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-2 else 1+Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])-Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R])-Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))+rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))-rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-1 else 0
                            elif i==Nv-2:
                                    A[i,j] =rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-4 else -rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-3 else 1+Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])-Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R])-Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))+rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))-rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-2 else rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                            else:
                                    A[i,j] =rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i-2 else -rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i-1 else 1+Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])-Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R])-Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))+rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))-rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i else rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i+1 else rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i+2 else 0
                    else:
                            if i==0:
                                    A[i,j] =1+Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])-Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R])-Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))+rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))-rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==0 else rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==1 else rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==2 else 0
                            elif i==1:
                                    A[i,j] =-rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==0 else 1+Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])-Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R])-Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))+rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))-rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==1 else rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==2 else rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==3 else 0
                            elif i==Nv-1:
                                    A[i,j] =rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-3 else -rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-2 else 1+Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])-Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R])-Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))+rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))-rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-1 else 0
                            elif i==Nv-2:
                                    A[i,j] =rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-4 else -rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-3 else 1+Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])-Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R])-Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))+rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))-rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-2 else rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                            else:
                                    A[i,j] =rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i-2 else -rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i-1 else 1+Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])-Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R])-Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))+rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))-rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))-rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i else rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i+1 else rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i+2 else 0

    return A

def Matrix_B(R,M):
    B=np.zeros(((Nv),(Nv)))
    for i in range(Nv):
        for j in range(Nv):
                if R==0:
                        if i==0:
                                B[i,j] =0*rect_v(per_v[M])*(Fv/4)*((U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R])*per_v[M]/2)+0*rect_v(per_v[M])*(Fv/4)*(-Col*H_perp(pal_v[i],per_v[M],z[R]))+0*rect_v(per_v[M])*(Fv/4)*(-Col/2*G_per_e(pal_v[i],per_v[M],z[R]))+0*rect_v(per_v[M])*(Fv/4)*(-Col/2*G_per_p(pal_v[i],per_v[M],z[R])) if j==0 else 0*rect_v(per_v[M])*rect_v(pal_v[i])*(Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==1 else 0
                        elif i==Nv-1:
                                B[i,j] =-0*rect_v(per_v[M])*rect_v(pal_v[i])*(Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==Nv-2 else 0*rect_v(per_v[M])*(Fv/4)*((U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R])*per_v[M]/2)+0*rect_v(per_v[M])*(Fv/4)*(-Col*H_perp(pal_v[i],per_v[M],z[R]))+0*rect_v(per_v[M])*(Fv/4)*(-Col/2*G_per_e(pal_v[i],per_v[M],z[R]))+0*rect_v(per_v[M])*(Fv/4)*(-Col/2*G_per_p(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                        else:
                                B[i,j] =-0*rect_v(per_v[M])*rect_v(pal_v[i])*(Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==i-1 else 0*rect_v(per_v[M])*(Fv/4)*((U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R])*per_v[M]/2)+0*rect_v(per_v[M])*(Fv/4)*(-Col*H_perp(pal_v[i],per_v[M],z[R]))+0*rect_v(per_v[M])*(Fv/4)*(-Col/2*G_per_e(pal_v[i],per_v[M],z[R]))+0*rect_v(per_v[M])*(Fv/4)*(-Col/2*G_per_p(pal_v[i],per_v[M],z[R])) if j==i else 0*rect_v(per_v[M])*rect_v(pal_v[i])*(Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==i+1 else 0
                elif R==Nr-1:
                        if i==0:
                                B[i,j] =rect_v(per_v[M])*(Fv/4)*((0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R])*per_v[M]/2)+rect_v(per_v[M])*(Fv/4)*(-Col*H_perp(pal_v[i],per_v[M],z[R]))+rect_v(per_v[M])*(Fv/4)*(-Col/2*G_per_e(pal_v[i],per_v[M],z[R]))+rect_v(per_v[M])*(Fv/4)*(-Col/2*G_per_p(pal_v[i],per_v[M],z[R])) if j==0 else rect_v(per_v[M])*rect_v(pal_v[i])*(Fvv/8)*(Col*(-G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==1 else 0
                        elif i==Nv-1:
                                B[i,j] =-rect_v(per_v[M])*rect_v(pal_v[i])*(Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==Nv-2 else rect_v(per_v[M])*(Fv/4)*((0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R])*per_v[M]/2)+rect_v(per_v[M])*(Fv/4)*(-Col*H_perp(pal_v[i],per_v[M],z[R]))+rect_v(per_v[M])*(Fv/4)*(-Col/2*G_per_e(pal_v[i],per_v[M],z[R]))+rect_v(per_v[M])*(Fv/4)*(-Col/2*G_per_p(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                        else:
                                B[i,j] =-rect_v(per_v[M])*rect_v(pal_v[i])*(Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==i-1 else rect_v(per_v[M])*(Fv/4)*((0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R])*per_v[M]/2)+rect_v(per_v[M])*(Fv/4)*(-Col*H_perp(pal_v[i],per_v[M],z[R]))+rect_v(per_v[M])*(Fv/4)*(-Col/2*G_per_e(pal_v[i],per_v[M],z[R]))+rect_v(per_v[M])*(Fv/4)*(-Col/2*G_per_p(pal_v[i],per_v[M],z[R])) if j==i else rect_v(per_v[M])*rect_v(pal_v[i])*(Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==i+1 else 0
                else:
                        if i==0:
                                B[i,j] =rect_v(per_v[M])*(Fv/4)*((0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R])*per_v[M]/2)+rect_v(per_v[M])*(Fv/4)*(-Col*H_perp(pal_v[i],per_v[M],z[R]))+rect_v(per_v[M])*(Fv/4)*(-Col/2*G_per_e(pal_v[i],per_v[M],z[R]))+rect_v(per_v[M])*(Fv/4)*(-Col/2*G_per_p(pal_v[i],per_v[M],z[R])) if j==0 else rect_v(per_v[M])*rect_v(pal_v[i])*(Fvv/8)*(Col*(-G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==1 else 0
                        elif i==Nv-1:
                                B[i,j] =-rect_v(per_v[M])*rect_v(pal_v[i])*(Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==Nv-2 else rect_v(per_v[M])*(Fv/4)*((0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R])*per_v[M]/2)+rect_v(per_v[M])*(Fv/4)*(-Col*H_perp(pal_v[i],per_v[M],z[R]))+rect_v(per_v[M])*(Fv/4)*(-Col/2*G_per_e(pal_v[i],per_v[M],z[R]))+rect_v(per_v[M])*(Fv/4)*(-Col/2*G_per_p(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                        else:
                                B[i,j] =-rect_v(per_v[M])*rect_v(pal_v[i])*(Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==i-1 else rect_v(per_v[M])*(Fv/4)*((0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R])*per_v[M]/2)+rect_v(per_v[M])*(Fv/4)*(-Col*H_perp(pal_v[i],per_v[M],z[R]))+rect_v(per_v[M])*(Fv/4)*(-Col/2*G_per_e(pal_v[i],per_v[M],z[R]))+rect_v(per_v[M])*(Fv/4)*(-Col/2*G_per_p(pal_v[i],per_v[M],z[R])) if j==i else rect_v(per_v[M])*rect_v(pal_v[i])*(Fvv/8)*(-Col*(G_pal_per_e(pal_v[i],per_v[M],z[R])+G_pal_per_p(pal_v[i],per_v[M],z[R]))) if j==i+1 else 0
    return B

def Matrix_C(R,M):
    C=np.zeros(((Nv),(Nv)))
    for i in range(Nv):
        for j in range(Nv):
                if R==0:
                        C[i,j] =0*rect_v(per_v[M])*(Fvv/8)*(-Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))+0*rect_v(per_v[M])*(Fvv/8)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R]))) if j==i else 0
                else:
                        C[i,j] =rect_v(per_v[M])*(Fvv/8)*(-Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))+rect_v(per_v[M])*(Fvv/8)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R]))) if j==i else 0 
    return C

def Matrix_alpha(R,M):
    alpha=np.zeros(((Nv),(Nv)))
    for i in range(Nv):
        for j in range(Nv):
           if R==0:
              alpha[i,j] =-0*(Fz/2)*(U_solar(z[R])+pal_v[i]*cos(z[R])) if j==i else 0
           elif R==Nr-1:
              alpha[i,j] =-(Fz/2)*(0*U_solar(z[R])+pal_v[i]*cos(z[R])) if j==i else 0
           else:
              alpha[i,j] =-(Fz/2)*(0*U_solar(z[R])+pal_v[i]*cos(z[R])) if j==i else 0 #*rect((2.*(t[1]-t[0])/z[R])*f_1[M*Nv+i,R]/Mf[0]+Fz*0.5*(f_1[M*Nv+i,R+1]/Mf[0]-f_1[M*Nv+i,R-1]/Mf[0]))     return alpha
    return alpha

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
                if R==0:
                        if i==0:
                                A[i,j] =1+0*(delt)*(-U_solar(z[R])*dlnB(z[R]))-0*rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))+0*rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+0*rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+0*(delt)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R]))-0*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])-0*(delt)*U_solar(z[R])*dcos(z[R])/cos(z[R]) if j==0 else -0*rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-0*rect_v(pal_v[i])*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-0*rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==1 else -0*rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==2 else 0
                        elif i==1:
                                A[i,j] =0*rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+0*rect_v(pal_v[i])*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+0*rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==0 else 1+0*(delt)*(-U_solar(z[R])*dlnB(z[R]))-0*rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))+0*rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+0*rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+0*(delt)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R]))-0*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])-0*(delt)*U_solar(z[R])*dcos(z[R])/cos(z[R]) if j==1 else -0*rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-0*rect_v(pal_v[i])*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-0*rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==2 else -0*rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==3 else 0
                        elif i==Nv-1:
                                A[i,j] =-0*rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-3 else 0*rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+0*rect_v(pal_v[i])*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+0*rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-2 else 1+0*(delt)*(-U_solar(z[R])*dlnB(z[R]))-0*rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))+0*rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+0*rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+0*(delt)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R]))-0*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])-0*(delt)*U_solar(z[R])*dcos(z[R])/cos(z[R]) if j==Nv-1 else 0
                        elif i==Nv-2:
                                A[i,j] =-0*rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-4 else 0*rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+0*rect_v(pal_v[i])*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+0*rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-3 else 1+0*(delt)*(-U_solar(z[R])*dlnB(z[R]))-0*rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))+0*rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+0*rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+0*(delt)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R]))-0*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])-0*(delt)*U_solar(z[R])*dcos(z[R])/cos(z[R]) if j==Nv-2 else -0*rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-0*rect_v(pal_v[i])*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-0*rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                        else:
                                A[i,j] =-0*rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i-2 else 0*rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+0*rect_v(pal_v[i])*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+0*rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i-1 else 1+0*(delt)*(-U_solar(z[R])*dlnB(z[R]))-0*rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))+0*rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+0*rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+0*(delt)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R]))-0*(U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])-0*(delt)*U_solar(z[R])*dcos(z[R])/cos(z[R]) if j==i else -0*rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-0*rect_v(pal_v[i])*(Fv/4)*(-(U_solar(z[R])+pal_v[i]*cos(z[R]))*(dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-0*rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i+1 else -0*rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i+2 else 0
                elif R==Nr-1:
                        if i==0:
                                A[i,j] =1+Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))-rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))+rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R]))-Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])+Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R]) if j==0 else -rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==1 else -rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==2 else 0
                        elif i==1:
                                A[i,j] =rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==0 else 1+Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))-rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))+rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R]))-Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])+Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R]) if j==1 else -rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==2 else -rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==3 else 0
                        elif i==Nv-1:
                                A[i,j] =-rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-3 else rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-2 else 1+Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))-rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))+rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R]))-Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])+Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R]) if j==Nv-1 else 0
                        elif i==Nv-2:
                                A[i,j] =-rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-4 else rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-3 else 1+Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))-rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))+rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R]))-Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])+Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R]) if j==Nv-2 else -rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                        else:
                                A[i,j] =-rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i-2 else rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i-1 else 1+Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))-rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))+rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R]))-Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])+Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R]) if j==i else -rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i+1 else -rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i+2 else 0
                else:
                        if i==0:
                                A[i,j] =1+Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))-rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))+rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R]))-Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])+Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R]) if j==0 else -rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==1 else -rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==2 else 0
                        elif i==1:
                                A[i,j] =rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==0 else 1+Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))-rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))+rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R]))-Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])+Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R]) if j==1 else -rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==2 else -rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==3 else 0
                        elif i==Nv-1:
                                A[i,j] =-rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-3 else rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-2 else 1+Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))-rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))+rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R]))-Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])+Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R]) if j==Nv-1 else 0
                        elif i==Nv-2:
                                A[i,j] =-rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==Nv-4 else rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-3 else 1+Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))-rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))+rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R]))-Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])+Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R]) if j==Nv-2 else -rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==Nv-1 else 0
                        else:
                                A[i,j] =-rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i-2 else rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])+rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))+rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i-1 else 1+Exp*(delt)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dlnB(z[R]))-rect_v(per_v[M])*(Fvv/4)*(Col/2*(G_per_ee(pal_v[i],per_v[M],z[R])+G_per_pp(pal_v[i],per_v[M],z[R])))+rect_v(per_v[M])*(Fvv/4)*(-Col/2*(G_per_2e(pal_v[i],per_v[M],z[R])+G_per_2p(pal_v[i],per_v[M],z[R])))+rect_v(pal_v[i])*(Fvv/4)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R])))+(delt)*(4*np.pi*Col)*(Collision_Core(pal_v[i],per_v[M],z[R])+(Me/Mp)*Collision_Proton(pal_v[i],per_v[M],z[R]))-Exp*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(2.*delt/z[R])+Exp*(delt)*(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*dcos(z[R])/cos(z[R]) if j==i else -rect_v(pal_v[i])*(Fv/4)*cos(z[R])*electric(z[R])-rect_v(pal_v[i])*(Fv/4)*(-(0*U_solar(z[R])+pal_v[i]*cos(z[R]))*(0*dU_solar(z[R])/cos(z[R])+pal_v[i]*dcos(z[R])/cos(z[R]))-(cos(z[R])*dlnB(z[R])*per_v[M]**2/2))-rect_v(pal_v[i])*(Fv/4)*(-Col*H_palp(pal_v[i],per_v[M],z[R])) if j==i+1 else -rect_v(pal_v[i])*(Fvv/8)*(-Col/2*(G_pal_2e(pal_v[i],per_v[M],z[R])+G_pal_2p(pal_v[i],per_v[M],z[R]))) if j==i+2 else 0
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


#f_1 = np.load('data_next.npy')

Normvalue=np.zeros(shape = (timestep*updatetime))
for p in range(updatetime):
        print(p)


        
        Density_next=np.zeros(shape = (Nr))
        for r in range(Nr):
               tempDensity=0
               for j in range(Nv):
                      for i in range(Nv):
                              if per_v[j]<0:
                                      tempDensity=tempDensity
                              else:
                                      tempDensity=tempDensity+2*np.pi*f_1[j*Nv+i,r]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
               Density_next[r]=tempDensity/(r_s**3)

        Bulk=np.zeros(shape = (Nr))
        for r in range(Nr):
           tempBulk=0
           for j in range(Nv):
              for i in range(Nv):
                      if per_v[j]>=0:
                              tempBulk=tempBulk+2*np.pi*pal_v[i]*f_1[j*Nv+i,r]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
                      else:
                              tempBulk=tempBulk
           Bulk[r]=tempBulk/((r_s**3)*Density_next[r])


        Temperature_pal=np.zeros(shape = (Nr))
        for r in range(Nr):
               temptemp=0
               for j in range(Nv):
                  for i in range(Nv):
                          if per_v[j]<0:
                                  temptemp=temptemp
                          else:
                                  temptemp=temptemp+2*np.pi*(pal_v[i]**2)*f_1[j*Nv+i,r]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
               Temperature_pal[r]=v_Ae_0**2*Me*temptemp/((r_s**3)*Density_next[r]*Bol_k)

        Temperature_per=np.zeros(shape = (Nr))
        for r in range(Nr):
               temptemp=0
               for j in range(Nv):
                  for i in range(Nv):
                          if per_v[j]<0:
                                  temptemp=temptemp
                          else:
                                  temptemp=temptemp+2*np.pi*(per_v[j]**2)*f_1[j*Nv+i,r]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
               Temperature_per[r]=v_Ae_0**2*Me*temptemp/(2*(r_s**3)*Density_next[r]*Bol_k)  

        Temperature_tol=np.zeros(shape = (Nr))
        Temperature_tol=(1/3)*(Temperature_pal+2*Temperature_per)
        


        def Collision_Core(a,b,x):
            for r in range(Nr):
                if abs(x-z[r])<0.5*delz:
                        l=r
            kappa=50.
            d=0
            if (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])==0:
                    d=(r_s**3)*(Density_next[l])/(v_th_function(Temperature_tol[l])**3*np.pi**(3/2))*np.exp(-a**2/v_th_function(Temperature_tol[l])**2-b**2/v_th_function(Temperature_tol[l])**2) #(r_s**3)*(n(r)*10**6)*(np.pi**1.5*v_th_function(temperature(r))**3)**(-1)*(gamma(kappa+1)/(gamma(kappa-0.5)*kappa**1.5))*(1.+((b/v_th_function(temperature(r)))**2)/kappa+((a/v_th_function(temperature(r)))**2)/kappa)**(-kappa-1.)            
            else:
                    d=(r_s**3)*(Density_next[l])/(v_th_function(Temperature_tol[l])**3*np.pi**(3/2))*np.exp(-a**2/v_th_function(Temperature_tol[l])**2-b**2/v_th_function(Temperature_tol[l])**2) #(r_s**3)*(n(r)*10**6)*(np.pi**1.5*v_th_function(temperature(r))**3)**(-1)*(gamma(kappa+1)/(gamma(kappa-0.5)*kappa**1.5))*(1.+((b/v_th_function(temperature(r)))**2)/kappa+((a/v_th_function(temperature(r)))**2)/kappa)**(-kappa-1.)            
            return d

        def Collision_Proton(a,b,x):
            for r in range(Nr):
                if abs(x-z[r])<0.5*delz:
                        l=r
            kappa=50.
            d=0
            if (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])==0:
                    d=(r_s**3)*(Density_next[l])/(v_th_function_p(Temperature_tol[l])**3*np.pi**(3/2))*np.exp(-a**2/v_th_function_p(Temperature_tol[l])**2-b**2/v_th_function_p(Temperature_tol[l])**2) #(r_s**3)*(n(r)*10**6)*(np.pi**1.5*v_th_function_p(temperature(r))**3)**(-1)*(gamma(kappa+1)/(gamma(kappa-0.5)*kappa**1.5))*(1.+((b/v_th_function_p(temperature(r)))**2)/kappa+((a/v_th_function_p(temperature(r)))**2)/kappa)**(-kappa-1.)
            else:
                    d=(r_s**3)*(Density_next[l])/(v_th_function_p(Temperature_tol[l])**3*np.pi**(3/2))*np.exp(-a**2/v_th_function_p(Temperature_tol[l])**2-b**2/v_th_function_p(Temperature_tol[l])**2) #(r_s**3)*(n(r)*10**6)*(np.pi**1.5*v_th_function_p(temperature(r))**3)**(-1)*(gamma(kappa+1)/(gamma(kappa-0.5)*kappa**1.5))*(1.+((b/v_th_function_p(temperature(r)))**2)/kappa+((a/v_th_function_p(temperature(r)))**2)/kappa)**(-kappa-1.)
            return d

        def G_per_2e(a,b,x):
            for r in range(Nr):
                if abs(x-z[r])<0.5*delz:
                        l=r
            d=0
            if (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])<1 and (a**2+b**2)**0.5>0:
                d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*((2/(3*v_th_function(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function(Temperature_tol[l])**3)-(4/15)*(b**2/v_th_function(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function(Temperature_tol[l])**5)+(2/5)*(b**2*(a**2+b**2)/v_th_function(Temperature_tol[l])**5))
            elif (a**2+b**2)==0:
                d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*((2/(3*v_th_function(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function(Temperature_tol[l])**3)-(4/15)*(b**2/v_th_function(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function(Temperature_tol[l])**5)+(2/5)*(b**2*(a**2+b**2)/v_th_function(Temperature_tol[l])**5))
            else:
                d=(r_s**3)*(Density_next[l])*v_th_function(Temperature_tol[l])**2*(0.5/(a**2+b**2)**1.5-1.5*b**2/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function(Temperature_tol[l])))+2*(r_s**3)*(Density_next[l])*b**2/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function(Temperature_tol[l]))
            return d


        def G_per_e(a,b,x):
            for r in range(Nr):
                if abs(x-z[r])<0.5*delz:
                        l=r
            d=0
            if (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])<1 and abs(b)>0:
                d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*(1/b)*((2/(3*v_th_function(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function(Temperature_tol[l])**5))
            elif (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])<1 and abs(b)==0:
                d=0*(2*(2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*(1/(b+delv))*((2/(3*v_th_function(Temperature_tol[l])))-(2/15)*((a**2+(b+delv)**2)/v_th_function(Temperature_tol[l])**3)+(1/10)*((a**2+(b+delv)**2)**2/v_th_function(Temperature_tol[l])**5)))-0*(2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*(1/(b+2*delv))*((2/(3*v_th_function(Temperature_tol[l])))-(2/15)*((a**2+(b+2*delv)**2)/v_th_function(Temperature_tol[l])**3)+(1/10)*((a**2+(b+2*delv)**2)**2/v_th_function(Temperature_tol[l])**5))))
            elif (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])>=1 and abs(b)>0:
                d=(r_s**3)*(Density_next[l])*v_th_function(Temperature_tol[l])**2*(1/b)*(0.5/(a**2+b**2)**1.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function(Temperature_tol[l])))
            elif (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])>=1 and abs(b)==0:
                d=0*(2*((r_s**3)*(Density_next[l])*v_th_function(Temperature_tol[l])**2*(1/(b+delv))*(0.5/(a**2+(b+delv)**2)**1.5)*((2/np.pi**0.5)*((a**2+(b+delv)**2)**0.5/v_th_function(Temperature_tol[l]))*np.exp(-(a**2+(b+delv)**2)/v_th_function(Temperature_tol[l])**2)+(2*(a**2+(b+delv)**2)/v_th_function(Temperature_tol[l])**2-1)*special.erf((a**2+(b+delv)**2)**0.5/v_th_function(Temperature_tol[l]))))-0*((r_s**3)*(Density_next[l])*v_th_function(Temperature_tol[l])**2*(1/(b+2*delv))*(0.5/(a**2+(b+2*delv)**2)**1.5)*((2/np.pi**0.5)*((a**2+(b+2*delv)**2)**0.5/v_th_function(Temperature_tol[l]))*np.exp(-(a**2+(b+2*delv)**2)/v_th_function(Temperature_tol[l])**2)+(2*(a**2+(b+2*delv)**2)/v_th_function(Temperature_tol[l])**2-1)*special.erf((a**2+(b+2*delv)**2)**0.5/v_th_function(Temperature_tol[l])))))
            return d

        def G_per_ee(a,b,x):
            for r in range(Nr):
                if abs(x-z[r])<0.5*delz:
                        l=r
            d=0
            if (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])<1 and abs(b)==0:
                d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*((2/(3*v_th_function(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function(Temperature_tol[l])**3)-(4/15)*(b**2/v_th_function(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function(Temperature_tol[l])**5)+(2/5)*(b**2*(a**2+b**2)/v_th_function(Temperature_tol[l])**5)) #2*(r_s**3)*(n(r)*10**6)/(np.pi**0.5)*((2/(3*v_th_function(temperature(r))))-(2/15)*((a**2+b**2)/v_th_function(temperature(r))**3)+(1/10)*((a**2+b**2)**2/v_th_function(temperature(r))**5))
            elif (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])>=1 and abs(b)==0:
                d=(r_s**3)*(Density_next[l])*v_th_function(Temperature_tol[l])**2*(0.5/(a**2+b**2)**1.5-1.5*b**2/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function(Temperature_tol[l])))+2*(r_s**3)*(Density_next[l])*b**2/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function(Temperature_tol[l])) #(r_s**3)*(n(r)*10**6)*v_th_function(temperature(r))**2*(0.5/(a**2+b**2)**1.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function(temperature(r)))*np.exp(-(a**2+b**2)/v_th_function(temperature(r))**2)+(2*(a**2+b**2)/v_th_function(temperature(r))**2-1)*special.erf((a**2+b**2)**0.5/v_th_function(temperature(r))))
            return d

        def G_pal_2e(a,b,x):
            for r in range(Nr):
                if abs(x-z[r])<0.5*delz:
                        l=r
            d=0
            if (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])<1 and (a**2+b**2)**0.5>0:
                d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*((2/(3*v_th_function(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function(Temperature_tol[l])**3)-(4/15)*(a**2/v_th_function(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function(Temperature_tol[l])**5)+(2/5)*(a**2*(a**2+b**2)/v_th_function(Temperature_tol[l])**5))
            elif (a**2+b**2)==0:
                d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*((2/(3*v_th_function(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function(Temperature_tol[l])**3)-(4/15)*(a**2/v_th_function(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function(Temperature_tol[l])**5)+(2/5)*(a**2*(a**2+b**2)/v_th_function(Temperature_tol[l])**5))
            else:
                d=(r_s**3)*(Density_next[l])*v_th_function(Temperature_tol[l])**2*(0.5/(a**2+b**2)**1.5-1.5*a**2/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function(Temperature_tol[l])))+2*(r_s**3)*(Density_next[l])*a**2/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function(Temperature_tol[l]))
            return d


        def G_pal_per_e(a,b,x):
            for r in range(Nr):
                if abs(x-z[r])<0.5*delz:
                        l=r
            d=0
            if (a**2+b**2)**0.5/v_th_function(Temperature_tol[l])<1 and (a**2+b**2)**0.5>0:
                d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*(-(4/15)*(a*b/v_th_function(Temperature_tol[l])**3)+(2/5)*(a*b*(a**2+b**2)/v_th_function(Temperature_tol[l])**5))
            elif (a**2+b**2)==0:
                d=0
            else:
                d=(-(r_s**3)*(Density_next[l])*v_th_function(Temperature_tol[l])**2*(1.5*a*b/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function(Temperature_tol[l])))+2*(r_s**3)*(Density_next[l])*a*b/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function(Temperature_tol[l])))
            return d




        def H_per(a,b,x):
                return 0

        def H_pal(a,b,x):
                return 0

        def G_per_2p(a,b,x):
            for r in range(Nr):
                if abs(x-z[r])<0.5*delz:
                        l=r
            d=0
            if (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])<1:
                d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*((2/(3*v_th_function_p(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function_p(Temperature_tol[l])**3)-(4/15)*(b**2/v_th_function_p(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function_p(Temperature_tol[l])**5)+(2/5)*(b**2*(a**2+b**2)/v_th_function_p(Temperature_tol[l])**5))
            else:
                d=(r_s**3)*(Density_next[l])*v_th_function_p(Temperature_tol[l])**2*(0.5/(a**2+b**2)**1.5-1.5*b**2/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])))+2*(r_s**3)*(Density_next[l])*b**2/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l]))
            return d

        def G_per_p(a,b,x):
            for r in range(Nr):
                if abs(x-z[r])<0.5*delz:
                        l=r
            d=0
            if (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])<1 and abs(b)>0:
                d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*(1/b)*((2/(3*v_th_function_p(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function_p(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function_p(Temperature_tol[l])**5))
            elif (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])<1 and abs(b)==0:
                d=0*(2*(2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*(1/(b+delv))*((2/(3*v_th_function_p(Temperature_tol[l])))-(2/15)*((a**2+(b+delv)**2)/v_th_function_p(Temperature_tol[l])**3)+(1/10)*((a**2+(b+delv)**2)**2/v_th_function_p(Temperature_tol[l])**5)))-0*(2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*(1/(b+2*delv))*((2/(3*v_th_function_p(Temperature_tol[l])))-(2/15)*((a**2+(b+2*delv)**2)/v_th_function_p(Temperature_tol[l])**3)+(1/10)*((a**2+(b+2*delv)**2)**2/v_th_function_p(Temperature_tol[l])**5))))
            elif (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])>=1 and abs(b)>0:
                d=(r_s**3)*(Density_next[l])*v_th_function_p(Temperature_tol[l])**2*(1/b)*(0.5/(a**2+b**2)**1.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])))
            elif (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])>=1 and abs(b)==0:
                d=0*(2*((r_s**3)*(Density_next[l])*v_th_function_p(Temperature_tol[l])**2*(1/(b+delv))*(0.5/(a**2+(b+delv)**2)**1.5)*((2/np.pi**0.5)*((a**2+(b+delv)**2)**0.5/v_th_function_p(Temperature_tol[l]))*np.exp(-(a**2+(b+delv)**2)/v_th_function_p(Temperature_tol[l])**2)+(2*(a**2+(b+delv)**2)/v_th_function_p(Temperature_tol[l])**2-1)*special.erf((a**2+(b+delv)**2)**0.5/v_th_function_p(Temperature_tol[l]))))-0*((r_s**3)*(Density_next[l])*v_th_function_p(Temperature_tol[l])**2*(1/(b+2*delv))*(0.5/(a**2+(b+2*delv)**2)**1.5)*((2/np.pi**0.5)*((a**2+(b+2*delv)**2)**0.5/v_th_function_p(Temperature_tol[l]))*np.exp(-(a**2+(b+2*delv)**2)/v_th_function_p(Temperature_tol[l])**2)+(2*(a**2+(b+2*delv)**2)/v_th_function_p(Temperature_tol[l])**2-1)*special.erf((a**2+(b+2*delv)**2)**0.5/v_th_function_p(Temperature_tol[l])))))    
            return d

        def G_per_pp(a,b,x):
            for r in range(Nr):
                if abs(x-z[r])<0.5*delz:
                        l=r
            d=0
            if (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])<1 and abs(b)==0:
                d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*((2/(3*v_th_function_p(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function_p(Temperature_tol[l])**3)-(4/15)*(b**2/v_th_function_p(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function_p(Temperature_tol[l])**5)+(2/5)*(b**2*(a**2+b**2)/v_th_function_p(Temperature_tol[l])**5)) #2*(r_s**3)*(n(r)*10**6)/(np.pi**0.5)*((2/(3*v_th_function_p(temperature(r))))-(2/15)*((a**2+b**2)/v_th_function_p(temperature(r))**3)+(1/10)*((a**2+b**2)**2/v_th_function_p(temperature(r))**5))
            elif (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])>=1 and abs(b)==0:
                d=(r_s**3)*(Density_next[l])*v_th_function_p(Temperature_tol[l])**2*(0.5/(a**2+b**2)**1.5-1.5*b**2/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])))+2*(r_s**3)*(Density_next[l])*b**2/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])) #(r_s**3)*(n(r)*10**6)*v_th_function_p(temperature(r))**2*(0.5/(a**2+b**2)**1.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function_p(temperature(r)))*np.exp(-(a**2+b**2)/v_th_function_p(temperature(r))**2)+(2*(a**2+b**2)/v_th_function_p(temperature(r))**2-1)*special.erf((a**2+b**2)**0.5/v_th_function_p(temperature(r))))
            return d


        def G_pal_2p(a,b,x):
            for r in range(Nr):
                if abs(x-z[r])<0.5*delz:
                        l=r
            d=0
            if (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])<1:
                d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*((2/(3*v_th_function_p(Temperature_tol[l])))-(2/15)*((a**2+b**2)/v_th_function_p(Temperature_tol[l])**3)-(4/15)*(a**2/v_th_function_p(Temperature_tol[l])**3)+(1/10)*((a**2+b**2)**2/v_th_function_p(Temperature_tol[l])**5)+(2/5)*(a**2*(a**2+b**2)/v_th_function_p(Temperature_tol[l])**5))
            else:
                d=(r_s**3)*(Density_next[l])*v_th_function_p(Temperature_tol[l])**2*(0.5/(a**2+b**2)**1.5-1.5*a**2/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])))+2*(r_s**3)*(Density_next[l])*a**2/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l]))
            return d

        def G_pal_per_p(a,b,x):
            for r in range(Nr):
                if abs(x-z[r])<0.5*delz:
                        l=r
            d=0
            if (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])<1:
                d=2*(r_s**3)*(Density_next[l])/(np.pi**0.5)*(-(4/15)*(a*b/(v_th_function_p(Temperature_tol[l]))**3)+(2/5)*(a*b*(a**2+b**2)/(v_th_function_p(Temperature_tol[l]))**5))
            else:
                d=(-(r_s**3)*(Density_next[l])*v_th_function_p(Temperature_tol[l])**2*(1.5*a*b/(a**2+b**2)**2.5)*((2/np.pi**0.5)*((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l]))*np.exp(-(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2)+(2*(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2-1)*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])))+2*(r_s**3)*(Density_next[l])*a*b/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])))
            return d

        def H_palp(a,b,x):
            for r in range(Nr):
                if abs(x-z[r])<0.5*delz:
                        l=r
            d=0
            if (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])<1:
                    d=(r_s**3)*(Density_next[l])*(4/np.pi**0.5)*(a/v_th_function_p(Temperature_tol[l]))*(-(1/3)*(1/v_th_function_p(Temperature_tol[l])**2)+(1/5)*((a**2+b**2)/v_th_function_p(Temperature_tol[l])**4))
            else:
                    d=(r_s**3)*(Density_next[l])*(1/v_th_function_p(Temperature_tol[l]))*((2/np.pi**0.5)*(a/(a**2+b**2))*np.exp(-(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2)-(a*v_th_function_p(Temperature_tol[l]))/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])))
            return d

        def H_perp(a,b,x):
            for r in range(Nr):
                if abs(x-z[r])<0.5*delz:
                        l=r
            d=0
            if (a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])<1:
                    d=(r_s**3)*(Density_next[l])*(4/np.pi**0.5)*(b/v_th_function_p(Temperature_tol[l]))*(-(1/3)*(1/v_th_function_p(Temperature_tol[l])**2)+(1/5)*((a**2+b**2)/v_th_function_p(Temperature_tol[l])**4))
            else:
                    d=(r_s**3)*(Density_next[l])*(1/v_th_function_p(Temperature_tol[l]))*((2/np.pi**0.5)*(b/(a**2+b**2))*np.exp(-(a**2+b**2)/v_th_function_p(Temperature_tol[l])**2)-(b*v_th_function_p(Temperature_tol[l]))/(a**2+b**2)**1.5*special.erf((a**2+b**2)**0.5/v_th_function_p(Temperature_tol[l])))
            return d




        e_col=np.zeros(shape = (Nr))
        for r in range(Nr):
               temp=0
               for j in range(Nv):
                      for i in range(Nv):
                              if per_v[j]<0:
                                      temp=temp
                              else:
                                      temp=temp+2*np.pi*pal_v[i]*4*np.pi*Collision_Core(pal_v[i],per_v[j],z[r])*f_1[j*Nv+i,r]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
               e_col[r]=Col*temp



        e_col_G1=np.zeros(shape = (Nr))
        for r in range(Nr):
               temp=0
               for j in range(Nv):
                      for i in range(Nv):
                              if per_v[j]<0:
                                      temp=temp
                              elif per_v[j]>=0 and j!=0 and j!=Nv-1 and j!=1 and j!=Nv-2:
                                      temp=temp+2*np.pi*pal_v[i]*0.5*G_per_2e(pal_v[i],per_v[j],z[r])*((f_1[(j+2)*Nv+i,r]-2*f_1[j*Nv+i,r]+f_1[(j-2)*Nv+i,r])/(4*delv**2))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
               e_col_G1[r]=Col*temp


        e_col_G2=np.zeros(shape = (Nr))
        for r in range(Nr):
               temp=0
               for j in range(Nv):
                      for i in range(Nv):
                              if per_v[j]<0:
                                      temp=temp
                              elif per_v[j]>=0 and i!=0 and i!=Nv-1 and i!=1 and i!=Nv-2:
                                      temp=temp+2*np.pi*pal_v[i]*0.5*G_pal_2e(pal_v[i],per_v[j],z[r])*((f_1[j*Nv+i+2,r]-2*f_1[j*Nv+i,r]+f_1[j*Nv+i-2,r])/(4*delv**2))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
               e_col_G2[r]=Col*temp


        e_col_G3=np.zeros(shape = (Nr))
        for r in range(Nr):
               temp=0
               for j in range(Nv):
                      for i in range(Nv):
                              if per_v[j]<0:
                                      temp=temp
                              elif per_v[j]>=0 and i!=0 and i!=Nv-1 and j!=0 and j!=Nv-1:
                                      temp=temp+2*np.pi*pal_v[i]*G_pal_per_e(pal_v[i],per_v[j],z[r])*((f_1[(j+1)*Nv+i+1,r]-f_1[(j+1)*Nv+i-1,r]-f_1[(j-1)*Nv+i+1,r]+f_1[(j-1)*Nv+i-1,r])/(4*delv**2))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
               e_col_G3[r]=Col*temp

        e_col_G4=np.zeros(shape = (Nr))
        for r in range(Nr):
               temp=0
               for j in range(Nv):
                      for i in range(Nv):
                              if per_v[j]<0:
                                      temp=temp
                              elif per_v[j]>0 and j!=0 and j!=Nv-1:
                                      temp=temp+2*np.pi*pal_v[i]*0.5*G_per_e(pal_v[i],per_v[j],z[r])*((f_1[(j+1)*Nv+i,r]-f_1[(j-1)*Nv+i,r])/(2*delv))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
                              elif per_v[j]==0 and j!=0 and j!=Nv-1 and j!=1 and j!=Nv-2:
                                      temp=temp+2*np.pi*pal_v[i]*0.5*G_per_ee(pal_v[i],per_v[j],z[r])*((f_1[(j+2)*Nv+i,r]-2*f_1[j*Nv+i,r]+f_1[(j-2)*Nv+i,r])/(4*delv**2))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
               e_col_G4[r]=Col*temp




        p_col=np.zeros(shape = (Nr))
        for r in range(Nr):
               temp=0
               for j in range(Nv):
                      for i in range(Nv):
                              if per_v[j]<0:
                                      temp=temp
                              else:
                                      temp=temp+2*np.pi*pal_v[i]*4*np.pi*(Me/Mp)*Collision_Proton(pal_v[i],per_v[j],z[r])*f_1[j*Nv+i,r]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
               p_col[r]=Col*temp



        p_col_G1=np.zeros(shape = (Nr))
        for r in range(Nr):
               temp=0
               for j in range(Nv):
                      for i in range(Nv):
                              if per_v[j]<0:
                                      temp=temp
                              elif per_v[j]>=0 and j!=0 and j!=Nv-1 and j!=1 and j!=Nv-2:
                                      temp=temp+2*np.pi*pal_v[i]*0.5*G_per_2p(pal_v[i],per_v[j],z[r])*((f_1[(j+2)*Nv+i,r]-2*f_1[j*Nv+i,r]+f_1[(j-2)*Nv+i,r])/(4*delv**2))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
               p_col_G1[r]=Col*temp


        p_col_G2=np.zeros(shape = (Nr))
        for r in range(Nr):
               temp=0
               for j in range(Nv):
                      for i in range(Nv):
                              if per_v[j]<0:
                                      temp=temp
                              elif per_v[j]>=0 and i!=0 and i!=Nv-1 and i!=1 and i!=Nv-2:
                                      temp=temp+2*np.pi*pal_v[i]*0.5*G_pal_2p(pal_v[i],per_v[j],z[r])*((f_1[j*Nv+i+2,r]-2*f_1[j*Nv+i,r]+f_1[j*Nv+i-2,r])/(4*delv**2))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
               p_col_G2[r]=Col*temp


        p_col_G3=np.zeros(shape = (Nr))
        for r in range(Nr):
               temp=0
               for j in range(Nv):
                      for i in range(Nv):
                              if per_v[j]<0:
                                      temp=temp
                              elif per_v[j]>=0 and i!=0 and i!=Nv-1 and j!=0 and j!=Nv-1:
                                      temp=temp+2*np.pi*pal_v[i]*G_pal_per_p(pal_v[i],per_v[j],z[r])*((f_1[(j+1)*Nv+i+1,r]-f_1[(j+1)*Nv+i-1,r]-f_1[(j-1)*Nv+i+1,r]+f_1[(j-1)*Nv+i-1,r])/(4*delv**2))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
               p_col_G3[r]=Col*temp

        p_col_G4=np.zeros(shape = (Nr))
        for r in range(Nr):
               temp=0
               for j in range(Nv):
                      for i in range(Nv):
                              if per_v[j]<0:
                                      temp=temp
                              elif per_v[j]>0 and j!=0 and j!=Nv-1:
                                      temp=temp+2*np.pi*pal_v[i]*0.5*G_per_p(pal_v[i],per_v[j],z[r])*((f_1[(j+1)*Nv+i,r]-f_1[(j-1)*Nv+i,r])/(2*delv))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
                              elif per_v[j]==0 and j!=0 and j!=Nv-1 and j!=1 and j!=Nv-2:
                                      temp=temp+2*np.pi*pal_v[i]*0.5*G_per_pp(pal_v[i],per_v[j],z[r])*((f_1[(j+2)*Nv+i,r]-2*f_1[j*Nv+i,r]+f_1[(j-2)*Nv+i,r])/(4*delv**2))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
               p_col_G4[r]=Col*temp



        p_col_H1=np.zeros(shape = (Nr))
        for r in range(Nr):
               temp=0
               for j in range(Nv):
                      for i in range(Nv):
                              if per_v[j]<0:
                                      temp=temp
                              elif per_v[j]>=0 and j!=0 and j!=Nv-1:
                                      temp=temp+2*np.pi*pal_v[i]*H_perp(pal_v[i],per_v[j],z[r])*((f_1[(j+1)*Nv+i,r]-f_1[(j-1)*Nv+i,r])/(2*delv))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
               p_col_H1[r]=Col*temp

        p_col_H2=np.zeros(shape = (Nr))
        for r in range(Nr):
               temp=0
               for j in range(Nv):
                      for i in range(Nv):
                              if per_v[j]<0:
                                      temp=temp
                              elif per_v[j]>=0 and i!=0 and i!=Nv-1:
                                      temp=temp+2*np.pi*pal_v[i]*H_palp(pal_v[i],per_v[j],z[r])*((f_1[j*Nv+i+1,r]-f_1[j*Nv+i-1,r])/(2*delv))*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
               p_col_H2[r]=Col*temp

        def electric(x):
                for r in range(Nr):
                        if abs(x-z[r])<0.5*delz:
                                l=r
                if l!=0:
                        E=-(Bulk[l]/(timestep*delt*cos(x)))-(1/((r_s**3)*Density_next[l]*cos(x)))*(e_col[l]+e_col_G1[l]+e_col_G2[l]+e_col_G3[l]+e_col_G4[l]+p_col[l]+p_col_G1[l]+p_col_G2[l]+p_col_G3[l]+p_col_G4[l]+p_col_H1[l]+p_col_H2[l])+U_solar(x)*dU_solar(x)/(cos(x)**2)+(1/v_Ae_0**2)*(Bol_k)/(Me*Density_next[l])*(Density_next[l]*Temperature_pal[l]-Density_next[l-1]*Temperature_pal[l-1])/delz+2*(1/v_Ae_0**2)*(Bol_k)/(Me)*dcos(x)/cos(x)*Temperature_pal[l]+(1/v_Ae_0**2)*(Bol_k)/(Me)*dlnB(x)*Temperature_per[l]+(1/v_Ae_0**2)*(2*Bol_k)/(Me*x)*Temperature_pal[l]#+(1/Density_next[l])*(Density_next[l]*Bulk_next[l]-Density_pre[l]*Bulk_pre[l])/(10*delt)+(Bulk_next[l]/cos(x))*dU_solar(x)+Bulk_next[l]*(dU_solar(x)/cos(x)+U_solar(x)*dcos_1(x))+(U_solar(x)/cos(x))*Bulk_next[l]/x+(U_solar(x)/(cos(x)*Density_next[l]))*(Density_next[l]*Bulk_next[l]-Density_next[l-1]*Bulk_next[l-1])/delz
                else:
                        E=-(Bulk[l]/(timestep*delt*cos(x)))-(1/((r_s**3)*Density_next[l]*cos(x)))*(e_col[l]+e_col_G1[l]+e_col_G2[l]+e_col_G3[l]+e_col_G4[l]+p_col[l]+p_col_G1[l]+p_col_G2[l]+p_col_G3[l]+p_col_G4[l]+p_col_H1[l]+p_col_H2[l])+U_solar(x)*dU_solar(x)/(cos(x)**2)+(1/v_Ae_0**2)*(Bol_k)/(Me*Density_next[l])*(Density_next[l+1]*Temperature_pal[l+1]-Density_next[l]*Temperature_pal[l])/delz+2*(1/v_Ae_0**2)*(Bol_k)/(Me)*dcos(x)/cos(x)*Temperature_pal[l]+(1/v_Ae_0**2)*(Bol_k)/(Me)*dlnB(x)*Temperature_per[l]+(1/v_Ae_0**2)*(2*Bol_k)/(Me*x)*Temperature_pal[l]#+(1/Density_next[l])*(Density_next[l]*Bulk_next[l]-Density_pre[l]*Bulk_pre[l])/(10*delt)+(Bulk_next[l]/cos(x))*dU_solar(x)+Bulk_next[l]*(dU_solar(x)/cos(x)+U_solar(x)*dcos_1(x))+(U_solar(x)/cos(x))*Bulk_next[l]/x+(U_solar(x)/(cos(x)*Density_next[l]))*(Density_next[l]*Bulk_next[l]-Density_next[l-1]*Bulk_next[l-1])/delz
                return E

        AQ=np.zeros(((Nv)*(Nv),(Nv)*(Nv),Nr))
        AalphaA=np.zeros(((Nv)*(Nv),(Nv)*(Nv),Nr))
        for r in range(Nr):
            AQ[:,:,r]=dot(inv(Matrix_AA(r)),Matrix_QQ(r))
            AalphaA[:,:,r]=dot(inv(Matrix_AA(r)),Matrix_alphaA(r))

        f_initial=np.zeros(shape = (Nv**2, Nr))
        f_fix=np.zeros(shape = (Nv**2, Nr))
        f_initial[:,:]=f_1[:,:]
        kl=50
        l=10
        t=0



        X2,Y2 = np.meshgrid(pal_v,per_v)
        cont_lev = np.linspace(-10,0,25)

        solu1=np.zeros(shape = (Nv, Nv))
        solu2=np.zeros(shape = (Nv))


        
        
        for k in range(timestep):
                print(k)
                f_pre=np.zeros(shape = (Nv**2, Nr))
                f_next=np.zeros(shape = (Nv**2, Nr))
                f_temp1=np.zeros(shape = (Nv**2, Nr))
                f_pre[:,:]=f_1[:,:]

                for r in range(Nr):
                    if r==0:
                        f_1[:,r]=f_initial[:,r]
                    elif r==Nr-1:
                        f_1[:,r]=f_pre[:,r]
                    else:
                        f_1[:,r]=dot(AQ[:,:,r],f_pre[:,r])+dot(AalphaA[:,:,r],f_pre[:,r+1])-dot(AalphaA[:,:,r],f_pre[:,r-1])
            
                maxx=np.amax(f_1)


        
                for r in range(Nr):
                        if r>0:
                                for j in range(Nv):
                                        for i in range(Nv):
                                                if f_1[j*Nv+i,r]<0:
                                                        f_1[j*Nv+i,r]=f_pre[j*Nv+i,r]


                f_1[:,Nr-1]=f_1[:,Nr-2]*ratio_r[:,Nr-2]**(-1)


                f_temp4=np.zeros(shape = (Nv**2, Nr))
                f_temp4[:,:]=f_1[:,:]                                
                for r in range(Nr-2):
                            for j in range(Nv):
                                    for i in range(Nv):
                                                f_temp4[j*Nv+i,r+1]=0.5*(0.5*(f_1[j*Nv+i,r]*ratio_r[j*Nv+i,r]**(-1)+f_1[j*Nv+i,r+1])+0.5*(f_1[j*Nv+i,r+1]+f_1[j*Nv+i,r+2]*ratio_r[j*Nv+i,r+1]))     #0.5*(f_1[(r)*(Nv)*(Nv)+j*Nv+i]*ratio_r[r*(Nv)*(Nv)+j*Nv+i]**(-1)+f_1[(r+2)*(Nv)*(Nv)+j*Nv+i]*ratio_r[(r+1)*(Nv)*(Nv)+j*Nv+i])                                
                f_1[:,:]=f_temp4[:,:]
                f_1[:,0]=f_initial[:,0]

                f_temp1=np.zeros(shape = (Nv**2, Nr))
                f_temp1[:,:]=f_1[:,:]
                for r in range(Nr):                                             #Von neumann boundary condition for v-derivative
                    if r>0:
                            for j in range(Nv):                      
                                    for i in range(Nv):
                                            if i==0 and j!=0 and j!=Nv-1 and j!=1 and j!=Nv-2:
                                                    f_temp1[j*Nv+i,r]=f_1[(j)*Nv+i+1,r]*(f_pre[(j)*Nv+i,r]/f_pre[(j)*Nv+i+1,r])#f_1[(j)*Nv+i+1,r]*d_pal_ne[j,r]#(4*f_1[(j+1)*Nv+i+1,r]+4*f_1[(j-1)*Nv+i+1,r]+4*f_1[(j)*Nv+i+3,r]-4*f_1[(j)*Nv+i+2,r]-4*f_1[(j)*Nv+i+1,r]-f_1[(j+2)*Nv+i+2,r]-f_1[(j-2)*Nv+i+2,r]-f_1[(j)*Nv+i+4,r])#2*f_1[(r)*(Nv)*(Nv)+(j)*Nv+i+1]-f_1[(r)*(Nv)*(Nv)+(j)*Nv+i+2] #np.max(f_1)*10**(2*np.log10(f_1[(r)*(Nv)*(Nv)+(j)*Nv+i+1]/np.max(f_1))-np.log10(f_1[(r)*(Nv)*(Nv)+(j)*Nv+i+2]/np.max(f_1)))    #np.max(f_1)*10**((pal_v[i]-pal_v[i+2])/(pal_v[i+2]-pal_v[i+1]))*(np.log10(f_1[(r)*(Nv)*(Nv)+j*Nv+i+2]/np.max(f_1))-np.log10(f_1[(r)*(Nv)*(Nv)+j*Nv+i+1]/np.max(f_1)))+np.log10(f_1[(r)*(Nv)*(Nv)+j*Nv+i+2]/np.max(f_1))                               #((pal_v[i]-pal_v[i+2])/(pal_v[i+2]-pal_v[i+1]))*(f_1[(r)*(Nv)*(Nv)+j*Nv+i+2]-f_1[(r)*(Nv)*(Nv)+j*Nv+i+1])+f_1[(r)*(Nv)*(Nv)+j*Nv+i+2] 
                                            if i==Nv-1 and j!=0 and j!=Nv-1 and j!=1 and j!=Nv-2:
                                                    f_temp1[j*Nv+i,r]=f_1[(j)*Nv+i-1,r]*(f_pre[(j)*Nv+i,r]/f_pre[(j)*Nv+i-1,r])#f_1[(j)*Nv+i-1,r]*d_pal_po[j,r]#(4*f_1[(j+1)*Nv+i-1,r]+4*f_1[(j-1)*Nv+i-1,r]+4*f_1[(j)*Nv+i-3,r]-4*f_1[(j)*Nv+i-2,r]-4*f_1[(j)*Nv+i-1,r]-f_1[(j+2)*Nv+i-2,r]-f_1[(j-2)*Nv+i-2,r]-f_1[(j)*Nv+i-4,r]) #np.max(f_1)*10**(2*np.log10(f_1[(r)*(Nv)*(Nv)+(j)*Nv+i-1]/np.max(f_1))-np.log10(f_1[(r)*(Nv)*(Nv)+(j)*Nv+i-2]/np.max(f_1)))                                  #((pal_v[i]-pal_v[i-2])/(pal_v[i-2]-pal_v[i-1]))*(f_1[(r)*(Nv)*(Nv)+j*Nv+i-2]-f_1[(r)*(Nv)*(Nv)+j*Nv+i-1])+f_1[(r)*(Nv)*(Nv)+j*Nv+i-2] 
                                            if i==0 and j==1:
                                                    f_temp1[j*Nv+i,r]=f_1[(j)*Nv+i+1,r]*(f_pre[(j)*Nv+i,r]/f_pre[(j)*Nv+i+1,r])#f_1[(j)*Nv+i+1,r]*d_pal_ne[j,r]#2*f_1[(j)*Nv+i+1,r]-f_1[(j)*Nv+i+2,r]
                                            if i==0 and j==Nv-2:
                                                    f_temp1[j*Nv+i,r]=f_1[(j)*Nv+i+1,r]*(f_pre[(j)*Nv+i,r]/f_pre[(j)*Nv+i+1,r])#f_1[(j)*Nv+i+1,r]*d_pal_ne[j,r]#2*f_1[(j)*Nv+i+1,r]-f_1[(j)*Nv+i+2,r]
                                            if i==Nv-1 and j==1:
                                                    f_temp1[j*Nv+i,r]=f_1[(j)*Nv+i-1,r]*(f_pre[(j)*Nv+i,r]/f_pre[(j)*Nv+i-1,r])#f_1[(j)*Nv+i-1,r]*d_pal_po[j,r]#2*f_1[(j)*Nv+i-1,r]-f_1[(j)*Nv+i-2,r]
                                            if i==Nv-1 and j==Nv-2:
                                                    f_temp1[j*Nv+i,r]=f_1[(j)*Nv+i-1,r]*(f_pre[(j)*Nv+i,r]/f_pre[(j)*Nv+i-1,r])#f_1[(j)*Nv+i-1,r]*d_pal_po[j,r]#2*f_1[(j)*Nv+i-1,r]-f_1[(j)*Nv+i-2,r]
                                        
                                            if j==0 and i!=0 and i!=Nv-1 and i!=1 and i!=Nv-2:
                                                    f_temp1[j*Nv+i,r]=f_1[(j+1)*Nv+i,r]*(f_pre[(j)*Nv+i,r]/f_pre[(j+1)*Nv+i,r])#f_1[(j+1)*Nv+i,r]*d_per_ne[i,r]#(4*f_1[(j+1)*Nv+i+1,r]+4*f_1[(j+1)*Nv+i-1,r]+4*f_1[(j+3)*Nv+i,r]-4*f_1[(j+2)*Nv+i,r]-4*f_1[(j+1)*Nv+i,r]-f_1[(j+2)*Nv+i+2,r]-f_1[(j+2)*Nv+i-2,r]-f_1[(j+4)*Nv+i,r])#2*f_1[(r)*(Nv)*(Nv)+(j+1)*Nv+i]-f_1[(r)*(Nv)*(Nv)+(j+2)*Nv+i] #np.max(f_1)*10**(2*np.log10(f_1[(r)*(Nv)*(Nv)+(j+1)*Nv+i]/np.max(f_1))-np.log10(f_1[(r)*(Nv)*(Nv)+(j+2)*Nv+i]/np.max(f_1)))                            #((per_v[j]-per_v[j+2])/(per_v[j+2]-per_v[j+1]))*(f_1[(r)*(Nv)*(Nv)+(j+2)*Nv+i]-f_1[(r)*(Nv)*(Nv)+(j+1)*Nv+i])+f_1[(r)*(Nv)*(Nv)+(j+2)*Nv+i] 
                                            if j==Nv-1 and i!=0 and i!=Nv-1 and i!=1 and i!=Nv-2:
                                                    f_temp1[j*Nv+i,r]=f_1[(j-1)*Nv+i,r]*(f_pre[(j)*Nv+i,r]/f_pre[(j-1)*Nv+i,r])#f_1[(j-1)*Nv+i,r]*d_per_po[i,r]#(4*f_1[(j-1)*Nv+i-1,r]+4*f_1[(j-1)*Nv+i+1,r]+4*f_1[(j-3)*Nv+i,r]-4*f_1[(j-2)*Nv+i,r]-4*f_1[(j-1)*Nv+i,r]-f_1[(j-2)*Nv+i+2,r]-f_1[(j-2)*Nv+i-2,r]-f_1[(j-4)*Nv+i,r])#2*f_1[(r)*(Nv)*(Nv)+(j-1)*Nv+i]-f_1[(r)*(Nv)*(Nv)+(j-2)*Nv+i] #np.max(f_1)*10**(2*np.log10(f_1[(r)*(Nv)*(Nv)+(j-1)*Nv+i]/np.max(f_1))-np.log10(f_1[(r)*(Nv)*(Nv)+(j-2)*Nv+i]/np.max(f_1)))                                #((per_v[j]-per_v[j-2])/(per_v[j-2]-per_v[j-1]))*(f_1[(r)*(Nv)*(Nv)+(j-2)*Nv+i]-f_1[(r)*(Nv)*(Nv)+(j-1)*Nv+i])+f_1[(r)*(Nv)*(Nv)+(j-2)*Nv+i]                                                                                
                                            if j==0 and i==1:
                                                    f_temp1[j*Nv+i,r]=f_1[(j+1)*Nv+i,r]*(f_pre[(j)*Nv+i,r]/f_pre[(j+1)*Nv+i,r])#f_1[(j+1)*Nv+i,r]*d_per_ne[i,r]#2*f_1[(j+1)*Nv+i,r]-f_1[(j+2)*Nv+i,r]
                                            if j==0 and i==Nv-2:
                                                    f_temp1[j*Nv+i,r]=f_1[(j+1)*Nv+i,r]*(f_pre[(j)*Nv+i,r]/f_pre[(j+1)*Nv+i,r])#f_1[(j+1)*Nv+i,r]*d_per_ne[i,r]#2*f_1[(j+1)*Nv+i,r]-f_1[(j+2)*Nv+i,r]
                                            if j==Nv-1 and i==1:
                                                    f_temp1[j*Nv+i,r]=f_1[(j-1)*Nv+i,r]*(f_pre[(j)*Nv+i,r]/f_pre[(j-1)*Nv+i,r])#f_1[(j-1)*Nv+i,r]*d_per_po[i,r]#2*f_1[(j-1)*Nv+i,r]-f_1[(j-2)*Nv+i,r]
                                            if j==Nv-1 and i==Nv-2:
                                                    f_temp1[j*Nv+i,r]=f_1[(j-1)*Nv+i,r]*(f_pre[(j)*Nv+i,r]/f_pre[(j-1)*Nv+i,r])#f_1[(j-1)*Nv+i,r]*d_per_po[i,r]#2*f_1[(j-1)*Nv+i,r]-f_1[(j-2)*Nv+i,r]
                                            if j==0 and i==0:
                                                    f_temp1[j*Nv+i,r]=f_1[(j+1)*Nv+i+1,r]*(f_pre[(j)*Nv+i,r]/f_pre[(j+1)*Nv+i+1,r])#f_1[(j+1)*Nv+i+1,r]*d_pal_ne_per_ne[r]#2*f_1[(j)*Nv+i+1,r]-f_1[(j)*Nv+i+2,r]
                                            if j==0 and i==Nv-1:
                                                    f_temp1[j*Nv+i,r]=f_1[(j+1)*Nv+i-1,r]*(f_pre[(j)*Nv+i,r]/f_pre[(j+1)*Nv+i-1,r])#f_1[(j+1)*Nv+i-1,r]*d_pal_po_per_ne[r]#2*f_1[(j)*Nv+i-1,r]-f_1[(j)*Nv+i-2,r]
                                            if j==Nv-1 and i==0:
                                                    f_temp1[(j)*Nv+i,r]=f_1[(j-1)*Nv+i+1,r]*(f_pre[(j)*Nv+i,r]/f_pre[(j-1)*Nv+i+1,r])#f_1[(j-1)*Nv+i+1,r]*d_pal_ne_per_po[r]#2*f_1[(j)*Nv+i+1,r]-f_1[(j)*Nv+i+2,r]
                                            if j==Nv-1 and i==Nv-1:
                                                    f_temp1[(j)*Nv+i,r]=f_1[(j-1)*Nv+i-1,r]*(f_pre[(j)*Nv+i,r]/f_pre[(j-1)*Nv+i-1,r])#f_1[(j-1)*Nv+i-1,r]*d_pal_po_per_po[r]#2*f_1[(j)*Nv+i-1,r]-f_1[(j)*Nv+i-2,r]
                f_1[:,:]=f_temp1[:,:]
                f_1[:,0]=f_initial[:,0]

                

        
                #f_temp4=np.zeros(shape = (Nv**2, Nr))
                #f_temp4[:,:]=f_1[:,:]                                
                #for r in range(Nr-1):
                #            for j in range(Nv):
                #                    for i in range(Nv):
                #                            if i!=Nv-1 and i!=0 and j!=Nv-1 and j!=0:
                #                                    f_temp4[j*Nv+i,r+1]=(1/4)*(2*f_1[j*Nv+i,r+1]+0.5*f_1[j*Nv+i+1,r+1]*(f_pre[j*Nv+i,r+1]/f_pre[j*Nv+i+1,r+1])+0.5*f_1[j*Nv+i-1,r+1]*(f_pre[j*Nv+i,r+1]/f_pre[j*Nv+i-1,r+1])+0.5*f_1[(j+1)*Nv+i,r+1]*(f_pre[(j)*Nv+i,r+1]/f_pre[(j+1)*Nv+i,r+1])+0.5*f_1[(j-1)*Nv+i,r+1]*(f_pre[(j)*Nv+i,r+1]/f_pre[(j-1)*Nv+i,r+1]))                             
                #                            elif i==Nv-1 and j!=Nv-1 and j!=0:
                #                                    f_temp4[j*Nv+i,r+1]=(1/3)*((3/2)*f_1[j*Nv+i,r+1]+0.5*f_1[j*Nv+i-1,r+1]*(f_pre[j*Nv+i,r+1]/f_pre[j*Nv+i-1,r+1])+0.5*f_1[(j+1)*Nv+i,r+1]*(f_pre[(j)*Nv+i,r+1]/f_pre[(j+1)*Nv+i,r+1])+0.5*f_1[(j-1)*Nv+i,r+1]*(f_pre[(j)*Nv+i,r+1]/f_pre[(j-1)*Nv+i,r+1])) 
                #                            elif i==0 and j!=Nv-1 and j!=0:
                #                                    f_temp4[j*Nv+i,r+1]=(1/3)*((3/2)*f_1[j*Nv+i,r+1]+0.5*f_1[j*Nv+i+1,r+1]*(f_pre[j*Nv+i,r+1]/f_pre[j*Nv+i+1,r+1])+0.5*f_1[(j+1)*Nv+i,r+1]*(f_pre[(j)*Nv+i,r+1]/f_pre[(j+1)*Nv+i,r+1])+0.5*f_1[(j-1)*Nv+i,r+1]*(f_pre[(j)*Nv+i,r+1]/f_pre[(j-1)*Nv+i,r+1])) 
                #                            elif j==Nv-1 and i!=Nv-1 and i!=0:
                #                                    f_temp4[j*Nv+i,r+1]=(1/3)*((3/2)*f_1[j*Nv+i,r+1]+0.5*f_1[(j-1)*Nv+i,r+1]*(f_pre[j*Nv+i,r+1]/f_pre[(j-1)*Nv+i,r+1])+0.5*f_1[(j)*Nv+i+1,r+1]*(f_pre[(j)*Nv+i,r+1]/f_pre[(j)*Nv+i+1,r+1])+0.5*f_1[(j)*Nv+i-1,r+1]*(f_pre[(j)*Nv+i,r+1]/f_pre[(j)*Nv+i-1,r+1])) 
                #                            elif j==0 and i!=Nv-1 and i!=0:
                #                                    f_temp4[j*Nv+i,r+1]=(1/3)*((3/2)*f_1[j*Nv+i,r+1]+0.5*f_1[(j+1)*Nv+i,r+1]*(f_pre[j*Nv+i,r+1]/f_pre[(j+1)*Nv+i,r+1])+0.5*f_1[(j)*Nv+i+1,r+1]*(f_pre[(j)*Nv+i,r+1]/f_pre[(j)*Nv+i+1,r+1])+0.5*f_1[(j)*Nv+i-1,r+1]*(f_pre[(j)*Nv+i,r+1]/f_pre[(j)*Nv+i-1,r+1])) 
                #f_1[:,:]=f_temp4[:,:]
                #f_1[:,0]=f_initial[:,0]    

                
                f_next[:,:]=f_1[:,:]
                norm=0
                for R in range(Nr):
                        for J in range(Nv):
                                for I in range(Nv):
                                        norm=norm+abs((f_next[J*Nv+I,R]/np.max(f_next[:,R])-f_pre[J*Nv+I,R]/np.max(f_pre[:,R])))**2
                        Normvalue[k]=norm**0.5
                #if k>100 and Normvalue[k,R]>=Normvalue[k-1,R]:
                #        Normvalue[:,R]=0
                #        f_1[:,R]=f_pre[:,R]
                Normvalue[k+timestep*p]=norm**0.5
                print(norm**0.5)






np.save('data_next.npy', f_1)          
np.save('data_norm.npy', Normvalue)        
            

X2,Y2 = np.meshgrid(pal_v,per_v)

solu1=np.zeros(shape = (Nv, Nv))
solu2=np.zeros(shape = (Nv))
solu3=np.zeros(shape = (Nv))
solu4=np.zeros(shape = (Nv))
cont_lev = np.linspace(-10,0,25)




for r in range(Nr):
   for j in range(Nv):
       for i in range(Nv):
               if f_1[(j)*Nv+i,r]/np.amax(f_1)>1:
                       solu1[j,i]=0
               elif f_1[(j)*Nv+i,r]/np.amax(f_1)>10**(-10):
                       solu1[j,i]=np.log10(f_1[(j)*Nv+i,r]/np.amax(f_1))
               else:
                       solu1[j,i]=-10
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
   plt.text(pal_v[Nv-13],0.5,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae0}$', fontsize=12)
   plt.text(0.,pal_v[Nv-4],r'$\mathcal{v}_\perp/\mathcal{v}_{Ae0}$', fontsize=12)
   plt.text(pal_v[Nv-23],pal_v[Nv-4], r'$r/r_s=$' "%.2f" % z[r], fontsize=12)
   #plt.text(pal_v[Nv-10],pal_v[Nv-2], r'$T(\mathcal{v}_{Ae0}/r_s):$' "%.2f" % nu, fontsize=8)
   #plt.text(pal_v[Nv-10],pal_v[Nv-4], r'$Nv=$' "%.2f" % Nv, fontsize=8)
   #plt.text(pal_v[Nv-10],pal_v[Nv-5], r'$Nr=$' "%.2f" % Nr, fontsize=8)
   plt.colorbar(label=r'$Log(F/F_{MAX})$')
   plt.savefig(f'{path_current}figure/{r}.png')
   plt.clf()
   plt.close()

for r in range(Nr):
   for i in range(Nv):
        solu2[i]=np.log10(f_1[(40)*Nv+i,r]/np.amax(f_1))
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
   plt.text(-3*delv,-8.7,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae0}$', fontsize=12)
   plt.text(-3*delv,0.5*delv,r'$Log(F/F_{MAX})$', fontsize=12)
   plt.ylim([-8, 0])
   plt.xlim([-Mv, Mv])
   plt.rc('font', size=8)
   plt.tick_params(labelsize=8)
   plt.savefig(f'{path_current}17/1D{r}.png')
   plt.clf()
   plt.close()

for r in range(Nr):
   for j in range(Nv):
        solu4[j]=np.log10(f_1[(j)*Nv+40,r]/np.amax(f_1))
   fig = plt.figure()
   fig.set_dpi(500)
   plt.plot(per_v,solu4,color='k',label=r'$r/r_s=$' "%.2f" % z[r]);
   plt.legend(loc='upper right')
   plt.grid()
   ax = plt.gca()
   ax.spines['left'].set_position('center')
   ax.spines['right'].set_color('none')
   ax.spines['top'].set_color('none')
   ax.xaxis.set_ticks_position('bottom')
   ax.yaxis.set_ticks_position('left')
   ax.set_yticks([-8,-6,-4,-2,-0])
   plt.text(-2*delv,-8.7,r'$\mathcal{v}_\perp/\mathcal{v}_{Ae0}$', fontsize=12)
   plt.text(-2*delv,2*delv,r'$Log(F/F_{MAX})$', fontsize=12)
   plt.ylim([-8, 0])
   plt.xlim([-Mv, Mv])
   plt.rc('font', size=8)
   plt.tick_params(labelsize=8)
   plt.savefig(f'{path_current}per/1D{r}.png')
   plt.clf()
   plt.close()



Density=np.zeros(shape = (Nr))
for r in range(Nr):
   tempDensity=0
   for j in range(Nv):
      for i in range(Nv):
              if per_v[j]<0:
                      tempDensity=tempDensity
              else:
                      tempDensity=tempDensity+2*np.pi*f_1[j*Nv+i,r]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
   Density[r]=tempDensity/(r_s**3)



Bulk=np.zeros(shape = (Nr))
for r in range(Nr):
   tempBulk=0
   for j in range(Nv):
      for i in range(Nv):
              if per_v[j]>=0:
                      tempBulk=tempBulk+2*np.pi*pal_v[i]*f_1[j*Nv+i,r]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
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
plt.savefig(f'{path_current}figure/bulk.png')
plt.clf()
plt.close()


Temperature_pal=np.zeros(shape = (Nr))
for r in range(Nr):
   temptemp=0
   for j in range(Nv):
      for i in range(Nv):
              if per_v[j]<0:
                      temptemp=temptemp
              else:
                      temptemp=temptemp+2*np.pi*(pal_v[i]**2)*f_1[j*Nv+i,r]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
   Temperature_pal[r]=v_Ae_0**2*Me*temptemp/((r_s**3)*Density[r]*Bol_k)

Temperature_per=np.zeros(shape = (Nr))
for r in range(Nr):
   temptemp=0
   for j in range(Nv):
      for i in range(Nv):
              if per_v[j]<0:
                      temptemp=temptemp
              else:
                      temptemp=temptemp+2*np.pi*(per_v[j]**2)*f_1[j*Nv+i,r]*abs(per_v[j])*(pal_v[1]-pal_v[0])**2
   Temperature_per[r]=v_Ae_0**2*Me*temptemp/(2*(r_s**3)*Density[r]*Bol_k)

plt.figure(figsize=(20,15))
plt.grid()
ax = plt.gca()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlim([z[0],z[Nr-1]])
ax.set_ylim([temperature(f_solar_r),temperature(i_solar_r)])
ax.set_xlabel(r'$r/r_s$', fontsize=28)
ax.set_ylabel(r'$T$', fontsize=28)
ax.plot(z,Temperature_pal,linewidth=3.0, color='r',label=r'$Numerical \ Temperature_{pal}$');
ax.plot(z,temperature(z),linewidth=3.0, color='k',linestyle='--',label=r'$Anaytical \ Temperature$');
plt.legend(loc='upper right')
plt.savefig(f'{path_current}figure/temperaturepal.png')
plt.clf()
plt.close()

plt.figure(figsize=(20,15))
plt.grid()
ax = plt.gca()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlim([z[0],z[Nr-1]])
ax.set_ylim([temperature(f_solar_r),temperature(i_solar_r)])
ax.set_xlabel(r'$r/r_s$', fontsize=28)
ax.set_ylabel(r'$T$', fontsize=28)
ax.plot(z,Temperature_per,linewidth=3.0, color='b',label=r'$Numerical \ Temperature_{per}$');
ax.plot(z,temperature(z),linewidth=3.0, color='k',linestyle='--',label=r'$Anaytical \ Temperature$');
plt.legend(loc='upper right')
plt.savefig(f'{path_current}figure/temperatureper.png')
plt.clf()
plt.close()

plt.figure(figsize=(20,15))
plt.grid()
ax = plt.gca()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlim([z[0],z[Nr-1]])
ax.set_ylim([temperature(f_solar_r),temperature(i_solar_r)])
ax.set_xlabel(r'$r/r_s$', fontsize=28)
ax.set_ylabel(r'$Temperature (K)$', fontsize=28)
ax.plot(z,(1/3)*(Temperature_pal+2*Temperature_per),linewidth=3.0, color='k',label=r'$T_{total}$');
ax.plot(z,Temperature_per,linewidth=3.0, color='b',label=r'$T_\perp$');
ax.plot(z,Temperature_pal,linewidth=3.0, color='r',label=r'$T_\parallel$');
ax.plot(z,max(Temperature_pal)*(z[0]/z)**0.8,linewidth=3.0, color='k',linestyle='--',label=r'$1/r^{0.8} \ Profile$');
#ax.plot(z,temperature(z),linewidth=3.0, color='k',linestyle='--',label=r'$Anaytical \ Temperature$');
plt.legend(loc='upper right')
plt.savefig(f'{path_current}figure/temperature.png')
plt.clf()
plt.close()

WS=np.zeros(shape = (Nr))
for r in range(Nr):
        WS[r]=U_solar(z[r])

plt.figure(figsize=(20,15))
plt.grid()
ax = plt.gca()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlim([z[0],z[Nr-1]])
ax.set_ylim([min(WS),max(WS)])
ax.set_xlabel(r'$r/r_s$', fontsize=28)
ax.set_ylabel(r'$Wind \ Speed$', fontsize=28)
ax.plot(z,WS,linewidth=3.0, color='k',label=r'$Solar \ Wind \ Speed$');
plt.savefig(f'{path_current}figure/wind speed.png')
plt.clf()
plt.close()




#for r in range(Nr):
#        print(U_solar(z[r]))
#print("o")
#print(Bulk)

P_flux=np.zeros(shape = (Nr))
for r in range(Nr):
        P_flux[r]=z[r]**2*(U_solar(z[r])+Bulk[r])*Density[r]
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
plt.savefig(f'{path_current}figure/particleflux.png')
plt.clf()
plt.close()


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
ax.plot(z,Density,linewidth=3.0, color='k',label=r'$Calculated \ Density$');
ax.plot(z,max(Density)*(z[0]/z)**2,linewidth=3.0, color='r',linestyle='--',label=r'$1/r^{2} \ Profile$');
ax.plot(z,max(Density)*(z[0]/z)**2*(WS[0]/(WS)),linewidth=3.0, color='b',linestyle='dashdot',label=r'$1/(Ur^{2}) \ Profile$');
ax.plot(z,max(Density)*(z[0]/z)**2*(WS[0]+Bulk[0])/(WS+Bulk),linewidth=3.0, color='r',linestyle='dotted',label=r'$Anaytical \ 1/(U+U_{bulk})r^{2} \ Density$');
plt.legend(loc='upper right')
plt.savefig(f'{path_current}figure/density.png')
plt.clf()
plt.close()


o=np.linspace(1, timestep*updatetime, timestep*updatetime)

plt.figure(figsize=(20,15))
plt.grid()
ax = plt.gca()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlim([o[0],o[timestep*updatetime-1]])
ax.set_ylim([10**(-5),10**(-2)])
ax.set_xlabel(r'$t$', fontsize=28)
ax.set_ylabel(r'$norm$', fontsize=28)
ax.plot(o,Normvalue,linewidth=3.0, color='k');
plt.savefig(f'{path_current}figure/norm.png')
plt.clf()
plt.close()
