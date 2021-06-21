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
#v_Ae=(B_0*10**(-9))/(4*np.pi*10**(-7)*9.1094*10**(-31)*n_0*10**(6))**0.5
q=1.6022*(10**(-19))
Me=9.1094*(10**(-31))
r_s=696340000
U_f=800
U_f2=400000
Omega=2.7*10**(-6)
i_solar_r=5
f_solar_r=20
z=np.linspace(i_solar_r, f_solar_r, 100)
T_e=1.0;
path_home="/Users/user/Desktop/profiles/"
path_current=path_home
def U_solar(r):
        return U_f*(np.exp(r/10.)-np.exp(-r/10.))/(np.exp(r/10.)+np.exp(-r/10.)) 

def U_solar3(r):
        return U_f*(np.exp(r/10.)-np.exp(-r/10.))/(np.exp(r/10.)+np.exp(-r/10.)) 


def U_solar2(r):
        return 800*(1-np.exp(-(r-2.8)/25))**0.5 


def B_0(r):
        return 2*(215/r)**2

def n_0(r):
        return 5*(215/r)**2

def n(r):
        return n_0(i_solar_r)*(i_solar_r/r)**2*(U_solar(215)/U_solar(r))

print(n(20))

def lnn(r):
        return -2/r

def dU_solar(x):
        return U_f*(1/40)*(2/(np.exp(x/40.)+np.exp(-x/40.)))**2

def temperature(r):
        return T_e*(i_solar_r/r)**(0.8)#T_e*np.exp(-(r-i_solar_r)**2/600)#(0.1*T_e-T_e)/(f_solar_r-i_solar_r)*(r-i_solar_r)+T_e #T_e*np.exp(-(r-i_solar_r)**2/600) #-0.75

def lntemperature(r):
        return -(r-i_solar_r)/300#(0.1*T_e-T_e)/(f_solar_r-i_solar_r) #-(r-i_solar_r)/300 #-1.4/(r-2.2)**(1.7)  

def electric(x):
        return 1/(Me*n(x))*(n(x)*temperature(x)*lntemperature(x)+temperature(x)*n(x)*lnn(x))

b=np.linspace(1, 10, 100)
def threshold(x):
        return (2*(1.6**0.5)*(0.912**2))*(1+np.cos(np.pi*60/180))/((x**4)*((1-np.cos(np.pi*60/180))*(np.cos(np.pi*60/180))))

plt.figure(figsize=(20,15))
ax = plt.gca()
plt.plot(b, threshold(b), 'k',linewidth=3.0)
plt.xlabel('U_s/v_Ae',fontsize=28)
plt.ylabel('n_s/n_c',fontsize=28)
plt.grid()
ax.set_ylim([0.005,0.1])
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.savefig(f'{path_current}/threshold.png')
plt.clf()
plt.close()
#for a in range(1):
#	plt.plot(z, electric(z), 'k')
#	plt.xlabel('Solar Radii')
#	plt.ylabel('E-field')
#	plt.show()

plt.figure(figsize=(20,15))
plt.plot(z, temperature(z), 'k',linewidth=3.0)
plt.xlabel(r'$r/r_s$',fontsize=28)
plt.ylabel('Temperature [$10^6$ K]',fontsize=28)
plt.grid()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(f'{path_current}/temperature.png')
plt.clf()
plt.close()

plt.figure(figsize=(20,15))
plt.plot(z, U_solar3(z), 'k', linewidth=3.0)
plt.xlabel(r'$r/r_s$',fontsize=28)
plt.ylabel('Wind Speed U [km/s]',fontsize=28)
plt.grid()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(f'{path_current}/wind speed.png')
plt.clf()
plt.close()

plt.figure(figsize=(20,15))
plt.plot(z, n_0(z), 'k',linewidth=3.0)
plt.xlabel('Solar Radii',fontsize=28)
plt.ylabel('Density [$10^6$ $m^{-3}$]',fontsize=28)
plt.grid()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(f'{path_current}/density.png')
plt.clf()
plt.close()


def B(x):
        return B_0(i_solar_r)*(i_solar_r/x)**2*(1+((x-i_solar_r)*Omega/U_solar(x))**2)**0.5

print(B(20))

plt.figure(figsize=(20,15))
plt.plot(z, B(z), 'k',linewidth=3.0)
plt.xlabel(r'$r/r_s$',fontsize=28)
plt.ylabel('$B_z$ [nT]',fontsize=28)
plt.grid()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig(f'{path_current}/magnetic field.png')
plt.clf()
plt.close()
	


def n(r):
        return n_0(i_solar_r)*(i_solar_r/r)**2


