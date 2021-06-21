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

Nv=34  #velocity step number
i_solar_r=5 #10
f_solar_r=20 #30
path_home="/Users/user/Desktop/test/"
path_lab="/disk/plasma4/syj2/Code/JSY2/"
path_current=path_home
#path_current=path_lab
def n_0(r):
        return 1*(215/r)**2

def B_0(r):
        return 1*(215/r)**2

v_Ae_0=(10*B_0(215)*10**(-9))/(4.*np.pi*10**(-7)*9.1094*10**(-31)*10*n_0(215)*10**6)**0.5
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
T_e=15*10**5; #5*(10**(5))
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

#calculate Beta

def n(r):
        return n_0(i_solar_r)*(i_solar_r/r)**2

def lnn(r):
        return -2/r

def U_solar(r):
        return U_f*(np.exp(r/10.)-np.exp(-r/10.))/(np.exp(r/10.)+np.exp(-r/10.)) 

def dU_solar(x):
        return U_f*(1./10.)*(2./(np.exp(x/10.)+np.exp(-x/10.)))**2

def cos(r):
        return (1/(1+(r*Omega/U_solar(r))**2)**0.5)

def dcos_1(r):
        return ((r*(Omega/U_solar(r))**2-(r*Omega/U_solar(r))**2/U_solar(r)*dU_solar(r))/(1+(r*Omega/U_solar(r))**2)**0.5)

def temperature(r):
        return T_e*(i_solar_r/r)**(0.8) #T_e*np.exp(-(r-i_solar_r)**2/600) #T_e*np.exp(2/(r-2.2)**0.7) #(0.1*T_e-T_e)/(f_solar_r-i_solar_r)*(r-i_solar_r)+T_e

def lntemperature(r):
        return -0.8*(1/r)#-(r-i_solar_r)/300 #-1.4/(r-2.2)**(1.7) #(0.1*T_e-T_e)/(f_solar_r-i_solar_r)/((0.1*T_e-T_e)/(f_solar_r-i_solar_r)*(r-i_solar_r)+T_e) 

def temperature_per(r):
        return 0.6*T_e*np.exp(2/(r-2.2)**0.7) #-0.75

def v_th_function(T):
        kappa=20
        return ((2)*Bol_k*T/(Me))**0.5/v_Ae_0

def v_th_function_p(T):
        kappa=20
        return ((2)*Bol_k*T/(Mp))**0.5/v_Ae_0

def kappa_v_th_function(T):
        kappa=2 #3
        return ((2.*kappa-3)*Bol_k*T/(kappa*Me))**0.5/v_Ae_0

def Kappa_Initial_Core(a,b,r):
   kappac=6 #2
   return (U_solar(z[0])/U_solar(r))*(r_s**3)*(n(r)*10**6)*(v_th_function(temperature(r))*v_th_function(temperature(r))**2)**(-1)*(2/(np.pi*(2*kappac-3)))**1.5*(gamma(kappac+1)/gamma(kappac-0.5))*(1.+(2/(2*kappac-3))*((b/v_th_function(temperature(r)))**2)+(2/(2*kappac-3))*((a/v_th_function(temperature(r)))**2))**(-kappac-1.) #(U_f/U_solar(r))*(r_s**3)*(n(r)*10**6)*(2*np.pi*kappa_v_th_function(temperature(r))**3*kappa**1.5)**(-1)*(gamma(kappa+1)/(gamma(kappa-0.5)*gamma(1.5)))*(1.+((b/kappa_v_th_function(temperature(r)))**2)/kappa+((a/kappa_v_th_function(temperature(r)))**2)/kappa)**(-kappa-1.)#+10**(-6)*(r_s**3)*(n(r)*10**6)*(np.pi**1.5*kappa_v_th_function(temperature(r))**3)**(-1)*(gamma(kappa+1)/(gamma(kappa-0.5)*kappa**1.5))*(1.+((b/(kappa_v_th_function(temperature(r))*100000))**2)/kappa+((a/(kappa_v_th_function(temperature(r))*100000))**2)/kappa)**(-kappa-1.) #(((7.5*10**9/r_s)/c)**2+0.05*np.exp(-(c-23)**2))*
#(r_s**3)*(n(r)*10**6)/(v_th_function(temperature(r))**3*np.pi**(3/2))*np.exp(-a**2/v_th_function(temperature(r))**2-b**2/v_th_function(temperature(r))**2)
         
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






#np.set_printoptions(threshold=np.inf)
#print(AQ)

print("here")
print(f_1[(Nr-1)*(Nv)*(Nv)+(15)*Nv+Nv-2])
print("here")
print(f_1[(Nr-1)*(Nv)*(Nv)+(15)*Nv+Nv-3])


print(2*f_1[(29)*(Nv)*(Nv)+(25)*Nv+Nv-2]-f_1[(29)*(Nv)*(Nv)+(25)*Nv+Nv-3])

f_temp1=np.zeros(shape = (Nr*Nv**2, 1))
f_temp1[:,:]=f_1[:,:]
for r in range(Nr):                                             #Von neumann boundary condition for v-derivative
        if r>0:
                for j in range(Nv):                      
                        for i in range(Nv):
                                if i==0 and j!=0 and j!=Nv-1:
                                        f_temp1[(r)*(Nv)*(Nv)+j*Nv+i]=f_1[(r)*(Nv)*(Nv)+(j+1)*Nv+i+1]*(f_1[(r)*(Nv)*(Nv)+(j-1)*Nv+i+1]/f_1[(r)*(Nv)*(Nv)+(j)*Nv+i+2])#f_1[(r)*(Nv)*(Nv)+(j+1)*Nv+i+1]*(f_1[(r)*(Nv)*(Nv)+(j-1)*Nv+i+1]/f_1[(r)*(Nv)*(Nv)+(j)*Nv+i+2]) #4*f_1[(r)*(Nv)*(Nv)+(j+1)*Nv+i+1]+4*f_1[(r)*(Nv)*(Nv)+(j-1)*Nv+i+1]+4*f_1[(r)*(Nv)*(Nv)+(j)*Nv+i+3]-4*f_1[(r)*(Nv)*(Nv)+(j)*Nv+i+2]-4*f_1[(r)*(Nv)*(Nv)+(j)*Nv+i+1]-f_1[(r)*(Nv)*(Nv)+(j+2)*Nv+i+2]-f_1[(r)*(Nv)*(Nv)+(j-2)*Nv+i+2]-f_1[(r)*(Nv)*(Nv)+(j)*Nv+i+4]#2*f_1[(r)*(Nv)*(Nv)+(j)*Nv+i+1]-f_1[(r)*(Nv)*(Nv)+(j)*Nv+i+2] #np.max(f_1)*10**(2*np.log10(f_1[(r)*(Nv)*(Nv)+(j)*Nv+i+1]/np.max(f_1))-np.log10(f_1[(r)*(Nv)*(Nv)+(j)*Nv+i+2]/np.max(f_1)))    #np.max(f_1)*10**((pal_v[i]-pal_v[i+2])/(pal_v[i+2]-pal_v[i+1]))*(np.log10(f_1[(r)*(Nv)*(Nv)+j*Nv+i+2]/np.max(f_1))-np.log10(f_1[(r)*(Nv)*(Nv)+j*Nv+i+1]/np.max(f_1)))+np.log10(f_1[(r)*(Nv)*(Nv)+j*Nv+i+2]/np.max(f_1))                               #((pal_v[i]-pal_v[i+2])/(pal_v[i+2]-pal_v[i+1]))*(f_1[(r)*(Nv)*(Nv)+j*Nv+i+2]-f_1[(r)*(Nv)*(Nv)+j*Nv+i+1])+f_1[(r)*(Nv)*(Nv)+j*Nv+i+2] 
                                if i==Nv-1 and j!=0 and j!=Nv-1:
                                        f_temp1[(r)*(Nv)*(Nv)+j*Nv+i]=f_1[(r)*(Nv)*(Nv)+(j+1)*Nv+i-1]*(f_1[(r)*(Nv)*(Nv)+(j-1)*Nv+i-1]/f_1[(r)*(Nv)*(Nv)+(j)*Nv+i-2])#4*f_1[(r)*(Nv)*(Nv)+(j+1)*Nv+i-1]+4*f_1[(r)*(Nv)*(Nv)+(j-1)*Nv+i-1]+4*f_1[(r)*(Nv)*(Nv)+(j)*Nv+i-3]-4*f_1[(r)*(Nv)*(Nv)+(j)*Nv+i-2]-4*f_1[(r)*(Nv)*(Nv)+(j)*Nv+i-1]-f_1[(r)*(Nv)*(Nv)+(j+2)*Nv+i-2]-f_1[(r)*(Nv)*(Nv)+(j-2)*Nv+i-2]-f_1[(r)*(Nv)*(Nv)+(j)*Nv+i-4] #np.max(f_1)*10**(2*np.log10(f_1[(r)*(Nv)*(Nv)+(j)*Nv+i-1]/np.max(f_1))-np.log10(f_1[(r)*(Nv)*(Nv)+(j)*Nv+i-2]/np.max(f_1)))                                  #((pal_v[i]-pal_v[i-2])/(pal_v[i-2]-pal_v[i-1]))*(f_1[(r)*(Nv)*(Nv)+j*Nv+i-2]-f_1[(r)*(Nv)*(Nv)+j*Nv+i-1])+f_1[(r)*(Nv)*(Nv)+j*Nv+i-2] 
                                        #if i==Nv-1 and j==1:
                                        #        f_temp1[(r)*(Nv)*(Nv)+j*Nv+i]=4*f_1[(r)*(Nv)*(Nv)+(j)*Nv+i-1]+4*f_1[(r)*(Nv)*(Nv)+(j)*Nv+i-3]-6*f_1[(r)*(Nv)*(Nv)+(j)*Nv+i-2]-f_1[(r)*(Nv)*(Nv)+(j)*Nv+i-4]
                                        #if i==Nv-1 and j==Nv-2:
                                        #        f_temp1[(r)*(Nv)*(Nv)+j*Nv+i]=4*f_1[(r)*(Nv)*(Nv)+(j)*Nv+i-1]+4*f_1[(r)*(Nv)*(Nv)+(j)*Nv+i-3]-6*f_1[(r)*(Nv)*(Nv)+(j)*Nv+i-2]-f_1[(r)*(Nv)*(Nv)+(j)*Nv+i-4]
                                
f_1[:,:]=f_temp1[:,:]


f_temp1=np.zeros(shape = (Nr*Nv**2, 1))
f_temp1[:,:]=f_1[:,:]
for r in range(Nr):                                             #Von neumann boundary condition for v-derivative
        if r>0:
                for j in range(Nv):                      
                        for i in range(Nv):
                                if j==0 and i!=0 and i!=Nv-1:
                                        f_temp1[(r)*(Nv)*(Nv)+j*Nv+i]=f_1[(r)*(Nv)*(Nv)+(j+1)*Nv+i+1]*(f_1[(r)*(Nv)*(Nv)+(j+1)*Nv+i-1]/f_1[(r)*(Nv)*(Nv)+(j+2)*Nv+i])#2*f_1[(r)*(Nv)*(Nv)+(j+1)*Nv+i]-f_1[(r)*(Nv)*(Nv)+(j+2)*Nv+i] #np.max(f_1)*10**(2*np.log10(f_1[(r)*(Nv)*(Nv)+(j+1)*Nv+i]/np.max(f_1))-np.log10(f_1[(r)*(Nv)*(Nv)+(j+2)*Nv+i]/np.max(f_1)))                            #((per_v[j]-per_v[j+2])/(per_v[j+2]-per_v[j+1]))*(f_1[(r)*(Nv)*(Nv)+(j+2)*Nv+i]-f_1[(r)*(Nv)*(Nv)+(j+1)*Nv+i])+f_1[(r)*(Nv)*(Nv)+(j+2)*Nv+i] 
                                if j==Nv-1 and i!=0 and i!=Nv-1:
                                        f_temp1[(r)*(Nv)*(Nv)+j*Nv+i]=f_1[(r)*(Nv)*(Nv)+(j-1)*Nv+i+1]*(f_1[(r)*(Nv)*(Nv)+(j-1)*Nv+i-1]/f_1[(r)*(Nv)*(Nv)+(j-2)*Nv+i])#2*f_1[(r)*(Nv)*(Nv)+(j-1)*Nv+i]-f_1[(r)*(Nv)*(Nv)+(j-2)*Nv+i] #np.max(f_1)*10**(2*np.log10(f_1[(r)*(Nv)*(Nv)+(j-1)*Nv+i]/np.max(f_1))-np.log10(f_1[(r)*(Nv)*(Nv)+(j-2)*Nv+i]/np.max(f_1)))                                #((per_v[j]-per_v[j-2])/(per_v[j-2]-per_v[j-1]))*(f_1[(r)*(Nv)*(Nv)+(j-2)*Nv+i]-f_1[(r)*(Nv)*(Nv)+(j-1)*Nv+i])+f_1[(r)*(Nv)*(Nv)+(j-2)*Nv+i]                                                                                
                                if j==0 and i==0:
                                        f_temp1[(r)*(Nv)*(Nv)+j*Nv+i]=f_1[(r)*(Nv)*(Nv)+(j+1)*Nv+i]*(f_1[(r)*(Nv)*(Nv)+(j+1)*Nv+i]/f_1[(r)*(Nv)*(Nv)+(j+2)*Nv+i])
                                if j==Nv-1 and i==Nv-1:
                                        f_temp1[(r)*(Nv)*(Nv)+j*Nv+i]=f_1[(r)*(Nv)*(Nv)+(j-1)*Nv+i]*(f_1[(r)*(Nv)*(Nv)+(j-1)*Nv+i]/f_1[(r)*(Nv)*(Nv)+(j-2)*Nv+i])
f_1[:,:]=f_temp1[:,:]

X2,Y2 = np.meshgrid(pal_v,per_v)
cont_lev = np.linspace(-10,0,25)




number=0
num=np.zeros(shape = (Nv))
for i in range(Nv):
        if abs(pal_v[i])<=3:
                number=number+1
                num[i]=pal_v[i]
                
pal_v_num = np.linspace(-max(num), max(num), number)
per_v_num = np.linspace(-max(num), max(num), number)

X2,Y2 = np.meshgrid(pal_v,per_v)
X1,Y1 = np.meshgrid(pal_v_num,per_v_num)
solu1=np.zeros(shape = (Nv, Nv))
#per_v2 = np.linspace(0, Mv, Nv//2)
#X2,Y2 = np.meshgrid(pal_v,per_v2)

solu1_num=np.zeros(shape = (number, number))
solu2=np.zeros(shape = (Nv))
solu3=np.zeros(shape = (Nv))
solu4=np.zeros(shape = (Nv))
cont_lev = np.linspace(-10,0,25)
difference=np.zeros(shape = ((Nr)*(Nv)*(Nv), 1))



for r in range(Nr):
   for j in range(Nv):
       for i in range(Nv):
               if f_1[(r)*(Nv)*(Nv)+(j)*Nv+i]/np.max(f_1)>1:
                       solu1[j,i]=0
               elif f_1[(r)*(Nv)*(Nv)+(j)*Nv+i]/np.max(f_1)>10**(-10):
                       solu1[j,i]=np.log10(f_1[(r)*(Nv)*(Nv)+(j)*Nv+i]/np.max(f_1))
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
   plt.text(pal_v[Nv-6],0.1,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae0}$', fontsize=12)
   plt.text(0.,pal_v[Nv-2],r'$\mathcal{v}_\perp/\mathcal{v}_{Ae0}$', fontsize=12)
   #plt.text(pal_v[Nv-10],pal_v[Nv-2], r'$T(\mathcal{v}_{Ae0}/r_s):$' "%.2f" % nu, fontsize=8)
   #plt.text(pal_v[Nv-10],pal_v[Nv-4], r'$Nv=$' "%.2f" % Nv, fontsize=8)
   #plt.text(pal_v[Nv-10],pal_v[Nv-5], r'$Nr=$' "%.2f" % Nr, fontsize=8)
   plt.colorbar(label=r'$Log(F/F_{MAX})$')
   plt.savefig(f'{path_current}figure/{r}.png')
   plt.clf()
   plt.close()

for r in range(Nr):
   for i in range(Nv):
        solu2[i]=np.log10(f_1[(r)*(Nv)*(Nv)+(17)*Nv+i]/np.max(f_1))
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
   plt.savefig(f'{path_current}17/1D{r}.png')
   plt.clf()
   plt.close()
