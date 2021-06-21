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
Me=9.1094*(10**(-31))
Mp=1.6726*(10**(-27))
data = np.loadtxt('/cygwin64/home/user/NHDS_noHDF/output.dat')
ratio=Me/Mp
x=data[:,0]
y=data[:,2]
z=data[:,1]
for k in range(1): #Numer in range indicates the minute.
    plt.figure(figsize=(20,15))
    plt.grid()
    plt.xlabel(r'$\mathit{k_{\parallel} \mathit{v}_{Ae}}/|\Omega_{e}|$',fontsize=20)
    plt.xlim(0, 0.2)
    plt.tick_params(labelsize=20)
    plt.ylabel(r'$\mathit{\gamma_{k}}/|\Omega_{e}|$',fontsize=20)
    plt.ylim(-0,0.01)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.rc('font', size=20)
    plt.tick_params(labelsize=20)
    plt.plot(x*(ratio**0.5),y*(ratio), '--', linewidth=3,label=r'$\theta$'"=59"r'$^\circ$')
    ax2 = plt.twinx()
    plt.ylabel(r'$\mathit{\omega_{k}}/|\Omega_{e}|$',fontsize=20)
    plt.ylim(0,0.3)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tick_params(labelsize=20)
    plt.plot(x*(ratio**0.5),z*(ratio), '-', linewidth=3,label=r'$\theta$'"=59"r'$^\circ$')
    plt.show()

    
num=len(y)
print(num)
p=np.zeros(shape = (num, 1))
k=np.zeros(shape = (num, 1))
p[:,0]=data[:,2]
k[:,0]=data[:,0]
print(p)
for i in range(num):
    if p[i,0]>=0:
        print(p[i,0])
        print("www")
        print(k[i,0]*(ratio**0.5))
