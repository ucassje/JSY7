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
Nv=81  #velocity step number
i_solar_r=5 #10
f_solar_r=20 #30
path_home="/Users/user/Desktop/test/"
path_lab="/disk/plasma4/syj2/Code/JSY7/"
#path_current=path_home
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
print(delv)
Nr=60      #radial step number
r_s=696340000.
z=np.linspace(i_solar_r, f_solar_r, Nr)
delz=z[1]-z[0]
print(delz)
Mt=3600*v_Ae_0/r_s
Nt=3600
t=np.linspace(0, Mt, Nt)
delt=(t[1]-t[0])            #time step
print(delt)
Fv=delt/delv
Fvv=delt/(delv)**2
Fz=delt/delz
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
print((f_solar_r-i_solar_r)/U_f)
print(((f_solar_r-i_solar_r)/U_f)/delt)

#calculate Beta

def n(r):
        return n_0(i_solar_r)*(i_solar_r/r)**2

def lnn(r):
        return -2/r

def U_solar(r):
        return U_f*(np.exp(r/10.)-np.exp(-r/10.))/(np.exp(r/10.)+np.exp(-r/10.)) 

def B(x):
        return B_0(i_solar_r)*(i_solar_r/x)**2*(1+((x-0*i_solar_r)*Omega/U_solar(x))**2)**0.5

def dU_solar(x):
        return U_f*(1./10.)*(2./(np.exp(x/10.)+np.exp(-x/10.)))**2

def cos(r):
        return (1/(1+(r*Omega/U_solar(r))**2)**0.5)

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

X2,Y2 = np.meshgrid(pal_v,per_v)
cont_lev = np.linspace(-10,0,25)

nc=np.zeros(shape = (Nr))
ns=np.zeros(shape = (Nr))
#nh=np.zeros(shape = (Nr))
Tc_pal=np.zeros(shape = (Nr))
Tc_per=np.zeros(shape = (Nr))
Ts_pal=np.zeros(shape = (Nr))
Ts_per=np.zeros(shape = (Nr))
#Th_pal=np.zeros(shape = (Nr))
#Th_per=np.zeros(shape = (Nr))
Uc=np.zeros(shape = (Nr))
Us=np.zeros(shape = (Nr))
#kappah=np.zeros(shape = (Nr))
#kappas=np.zeros(shape = (Nr))
v_Ae=np.zeros(shape = (Nr))
beta_c=np.zeros(shape = (Nr))
beta_s=np.zeros(shape = (Nr))
#beta_h=np.zeros(shape = (Nr))

h=0
f_1 = np.load('data_next.npy')

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

for j in range(Nv):
        if per_v[j]>13.5:
            h=h+1
                
f_11=np.zeros(shape = (Nv*(Nv-2*h), 1))
for j in range(Nv-2*h):
        for i in range(Nv):
            f_11[j*Nv+i]=f_1[(j+h)*Nv+i,Nr-1]


#per_v2 = np.linspace(-13, 13, (Nv-2*h))
#X2,Y2 = np.meshgrid(pal_v,per_v2)
#cont_lev = np.linspace(-10,0,25)

#solu1=np.zeros(shape = ((Nv-2*h), Nv))
#for j in range((Nv-2*h)):
#    for i in range(Nv):
#        solu1[j,i]=np.log10(f_11[j*Nv+i]/np.max(f_11))
#fig = plt.figure()
#fig.set_dpi(500)
#plt.contourf(X2, Y2,solu1, cont_lev,cmap='Blues');
#ax = plt.gca()
#ax.spines['left'].set_position('center')
#ax.spines['left'].set_smart_bounds(True)
#ax.spines['bottom'].set_position('zero')
#ax.spines['bottom'].set_smart_bounds(True)
#ax.spines['right'].set_color('none')
#ax.spines['top'].set_color('none')
#ax.xaxis.set_ticks_position('bottom')
#plt.axis('equal')
#ax.xaxis.set_ticks_position('bottom')
#ax.yaxis.set_ticks_position('left')
#plt.rc('font', size=8)
#plt.tick_params(labelsize=8)
#plt.colorbar(label=r'$Log(F/F_{MAX})$')
#plt.savefig(f'{path_current}/average.png')
#plt.clf()
#plt.close()


for r in range(Nr):
    print(r)
    if r==0:
            p = lmfit.Parameters()
            p.add_many(('nc', 0.7,True,0.7,1), ('Tc_pal', 10*10**5,True,1*10**5,10*10**5), ('Tc_per', 10*10**5,True,1*10**5,10*10**5), ('Ts_pal', 17*10**5,True,1*10**5,20*10**5), ('Ts_per', 17*10**5,True,1*10**5,20*10**5), ('Uc',0,True,-3.,0),('Us',0,True,0,0.5))
    else:                          
            p = lmfit.Parameters()
            p.add_many(('nc', nc[r-1],True,0.7,1), ('Tc_pal', Tc_pal[r-1],True,1*10**5,10*10**5), ('Tc_per', Tc_per[r-1],True,1*10**5,10*10**5), ('Ts_pal', Ts_pal[r-1],True,1*10**5,20*10**5), ('Ts_per', Ts_per[r-1],True,1*10**5,20*10**5), ('Uc',Uc[r-1],True,-3.,-0.),('Us',Us[r-1],True,0,17))

    f_11=np.zeros(shape = (Nv*(Nv-2*h), 1))
    for j in range(Nv-2*h):
        for i in range(Nv):
            f_11[j*Nv+i]=f_1[(j+h)*Nv+i,r]

    maxi=np.max(f_11)
    
    def residual(p):
        v=p.valuesdict()
        fitting=np.zeros(shape = (Nv*(Nv-2*h), 1))
        for j in range((Nv-2*h)):
            for i in range(Nv):
                fitting[j*Nv+i]=(v['nc'])*(r_s**3)*Density[r]*(np.pi**(3/2)*v_th_function(v['Tc_pal'])*v_th_function(v['Tc_per'])**2)**(-1)*np.exp(-(per_v[j+h]/v_th_function(v['Tc_per']))**2)*np.exp(-((pal_v[i]-v['Uc'])/v_th_function(v['Tc_pal']))**2)+(1-v['nc'])*(r_s**3)*Density[r]*(np.pi**(3/2)*v_th_function(v['Ts_pal'])*v_th_function(v['Ts_per'])**2)**(-1)*np.exp(-(per_v[j+h]/v_th_function(v['Ts_per']))**2)*np.exp(-((pal_v[i]-v['Us'])/v_th_function(v['Ts_pal']))**2)#+(v['nh'])*(r_s**3)*Density[r]*(v_th_function(v['Th_pal'])*v_th_function(v['Th_per'])**2)**(-1)*(2/(np.pi*(2*v['kappah']-3)))**1.5*(gamma(v['kappah']+1)/gamma(v['kappah']-0.5))*(1.+(2/(2*v['kappah']-3))*(((per_v[j])/v_th_function(v['Th_per']))**2)+(2/(2*v['kappah']-3))*(((pal_v[i]-v['Uc'])/v_th_function(v['Th_pal']))**2))**(-v['kappah']-1.)      #(v['nc'])*(r_s**3)*Density[r]*(v_th_function(v['Tc_pal'])*v_th_function(v['Tc_per'])**2)**(-1)*(2/(np.pi*(2*30-3)))**1.5*(gamma(30+1)/gamma(30-0.5))*(1.+(2/(2*30-3))*(((per_v[j])/v_th_function(v['Tc_per']))**2)+(2/(2*30-3))*(((pal_v[i]-v['Uc'])/v_th_function(v['Tc_pal']))**2))**(-30-1.)+(v['ns'])*(r_s**3)*Density[r]*(v_th_function(v['Ts_pal'])*v_th_function(v['Ts_per'])**2)**(-1)*(2/(np.pi*(2*30-3)))**1.5*(gamma(30+1)/gamma(30-0.5))*(1.+(2/(2*30-3))*(((per_v[j])/v_th_function(v['Ts_per']))**2)+(2/(2*30-3))*(((pal_v[i]-v['Us'])/v_th_function(v['Ts_pal']))**2))**(-30-1.)
        fit_maxi=np.max(fitting)
        
        DataChosen = np.where((f_11/maxi)> 10**(-8));
        return np.log10(fitting[DataChosen])-np.log10(f_11[DataChosen]) #np.log10(fitting/fit_maxi)-np.log10(f_11/maxi) 

    mi = lmfit.minimize(residual, p, method='nelder', options={'maxiter' : 1900}, nan_policy='omit')
    #lmfit.printfuncs.report_fit(mi.params, min_correl=0.5)
    print(fit_report(mi))
    zx =  mi.params
    nc[r] = zx['nc'].value
    ns[r] = 1-nc[r]#zx['ns'].value
    #nh[r] = zx['nh'].value
    Tc_pal[r] = zx['Tc_pal'].value
    Tc_per[r] = zx['Tc_per'].value
    Ts_pal[r] = zx['Ts_pal'].value
    Ts_per[r] = zx['Ts_per'].value
    #Th_pal[r] = zx['Th_pal'].value
    #Th_per[r] = zx['Th_per'].value
    Uc[r] = zx['Uc'].value
    Us[r] = zx['Us'].value
    #kappah[r] = zx['kappah'].value
    #kappas[r] = zx['kappas'].value

    v_Ae[r]=(B(z[r])*10**(-9))/(4.*np.pi*10**(-7)*9.1094e-31*Density[r])**0.5
    beta_c[r]=8*np.pi*10**(-7)*Bol_k*Density[r]*nc[r]*Tc_pal[r]/(B(z[r])*10**(-9))**2
    beta_s[r]=8*np.pi*10**(-7)*Bol_k*Density[r]*ns[r]*Ts_pal[r]/(B(z[r])*10**(-9))**2
    #beta_h[r]=8*np.pi*10**(-7)*Bol_k*Density[r]*nh[r]*Th_pal[r]/(B(z[r])*10**(-9))**2
    

    fitting=np.zeros(shape = (Nv**2, 1))
    for j in range(Nv):
        for i in range(Nv):
            fitting[j*Nv+i]=(nc[r])*(r_s**3)*Density[r]*(np.pi**(3/2)*v_th_function(Tc_pal[r])*v_th_function(Tc_per[r])**2)**(-1)*np.exp(-(per_v[j]/v_th_function(Tc_per[r]))**2)*np.exp(-((pal_v[i]-Uc[r])/v_th_function(Tc_pal[r]))**2)+(ns[r])*(r_s**3)*Density[r]*(np.pi**(3/2)*v_th_function(Ts_pal[r])*v_th_function(Ts_per[r])**2)**(-1)*np.exp(-(per_v[j]/v_th_function(Ts_per[r]))**2)*np.exp(-((pal_v[i]-Us[r])/v_th_function(Ts_pal[r]))**2)#+nh[r]*(r_s**3)*Density[r]*(v_th_function(Th_pal[r])*v_th_function(Th_per[r])**2)**(-1)*(2/(np.pi*(2*kappah[r]-3)))**1.5*(gamma(kappah[r]+1)/gamma(kappah[r]-0.5))*(1.+(2/(2*kappah[r]-3))*(((per_v[j])/v_th_function(Th_per[r]))**2)+(2/(2*kappah[r]-3))*(((pal_v[i]-Uc[r])/v_th_function(Th_pal[r]))**2))**(-kappah[r]-1.) #nc[r]*(r_s**3)*Density[r]*(v_th_function(Tc_pal[r])*v_th_function(Tc_per[r])**2)**(-1)*(2/(np.pi*(2*30-3)))**1.5*(gamma(30+1)/gamma(30-0.5))*(1.+(2/(2*30-3))*(((per_v[j])/v_th_function(Tc_per[r]))**2)+(2/(2*30-3))*(((pal_v[i]-Uc[r])/v_th_function(Tc_pal[r]))**2))**(-30-1.)+(ns[r])*(r_s**3)*Density[r]*(v_th_function(Ts_pal[r])*v_th_function(Ts_per[r])**2)**(-1)*(2/(np.pi*(2*30-3)))**1.5*(gamma(30+1)/gamma(30-0.5))*(1.+(2/(2*30-3))*(((per_v[j])/v_th_function(Ts_per[r]))**2)+(2/(2*30-3))*(((pal_v[i]-Us[r])/v_th_function(Ts_pal[r]))**2))**(-30-1.)
    
    fitting_c=np.zeros(shape = (Nv**2, 1))
    for j in range(Nv):
        for i in range(Nv):
            fitting_c[j*Nv+i]=(nc[r])*(r_s**3)*Density[r]*(np.pi**(3/2)*v_th_function(Tc_pal[r])*v_th_function(Tc_per[r])**2)**(-1)*np.exp(-(per_v[j]/v_th_function(Tc_per[r]))**2)*np.exp(-((pal_v[i]-Uc[r])/v_th_function(Tc_pal[r]))**2) #nc[r]*(r_s**3)*Density[r]*(v_th_function(Tc_pal[r])*v_th_function(Tc_per[r])**2)**(-1)*(2/(np.pi*(2*30-3)))**1.5*(gamma(30+1)/gamma(30-0.5))*(1.+(2/(2*30-3))*(((per_v[j])/v_th_function(Tc_per[r]))**2)+(2/(2*30-3))*(((pal_v[i]-Uc[r])/v_th_function(Tc_pal[r]))**2))**(-30-1.)

    fitting_s=np.zeros(shape = (Nv**2, 1))
    for j in range(Nv):
        for i in range(Nv):
            fitting_s[j*Nv+i]=(ns[r])*(r_s**3)*Density[r]*(np.pi**(3/2)*v_th_function(Ts_pal[r])*v_th_function(Ts_per[r])**2)**(-1)*np.exp(-(per_v[j]/v_th_function(Ts_per[r]))**2)*np.exp(-((pal_v[i]-Us[r])/v_th_function(Ts_pal[r]))**2) #(ns[r])*(r_s**3)*Density[r]*(v_th_function(Ts_pal[r])*v_th_function(Ts_per[r])**2)**(-1)*(2/(np.pi*(2*30-3)))**1.5*(gamma(30+1)/gamma(30-0.5))*(1.+(2/(2*30-3))*(((per_v[j])/v_th_function(Ts_per[r]))**2)+(2/(2*30-3))*(((pal_v[i]-Us[r])/v_th_function(Ts_pal[r]))**2))**(-30-1.)

    #fitting_h=np.zeros(shape = (Nv**2, 1))
    #for j in range(Nv):
    #    for i in range(Nv):
    #        fitting_h[j*Nv+i]=nh[r]*(r_s**3)*Density[r]*(v_th_function(Th_pal[r])*v_th_function(Th_per[r])**2)**(-1)*(2/(np.pi*(2*kappah[r]-3)))**1.5*(gamma(kappah[r]+1)/gamma(kappah[r]-0.5))*(1.+(2/(2*kappah[r]-3))*(((per_v[j])/v_th_function(Th_per[r]))**2)+(2/(2*kappah[r]-3))*(((pal_v[i]-Uc[r])/v_th_function(Th_pal[r]))**2))**(-kappah[r]-1.) #(ns[r])*(r_s**3)*Density[r]*(v_th_function(Ts_pal[r])*v_th_function(Ts_per[r])**2)**(-1)*(2/(np.pi*(2*30-3)))**1.5*(gamma(30+1)/gamma(30-0.5))*(1.+(2/(2*30-3))*(((per_v[j])/v_th_function(Ts_per[r]))**2)+(2/(2*30-3))*(((pal_v[i]-Us[r])/v_th_function(Ts_pal[r]))**2))**(-30-1.)

    
    if r==0:
        fitting_max=np.max(fitting)
        original_max=np.max(f_11)

    solu1=np.zeros(shape = (Nv, Nv))
    for j in range(Nv):
        for i in range(Nv):
            solu1[j,i]=np.log10(fitting[j*Nv+i]/fitting_max)
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
    plt.text(pal_v[0],pal_v[Nv-4], r'$n_c/n_e=$' "%.3f" % nc[r], fontsize=8)
    plt.text(pal_v[0],pal_v[Nv-6], r'$n_s/n_e=$' "%.3f" % ns[r], fontsize=8)
    plt.text(pal_v[0],pal_v[Nv-8], r'$Tc_{pal}(K)=$' "%.0f" % Tc_pal[r], fontsize=8)
    plt.text(pal_v[0],pal_v[Nv-10], r'$Tc_{per}(K)=$' "%.0f" % Tc_per[r], fontsize=8)
    plt.text(pal_v[0],pal_v[Nv-12], r'$Ts_{pal}(K)=$' "%.0f" % Ts_pal[r], fontsize=8)
    plt.text(pal_v[0],pal_v[Nv-14], r'$Ts_{per}(K)=$' "%.0f" % Ts_per[r], fontsize=8)
    plt.text(pal_v[0],pal_v[Nv-16], r'$U_c/v_{Ae0}=$' "%.3f" % Uc[r], fontsize=8)
    plt.text(pal_v[0],pal_v[Nv-18], r'$U_s/v_{Ae0}=$' "%.3f" % Us[r], fontsize=8)
    #plt.text(pal_v[0],pal_v[Nv-20], r'$kappa_c=$' "%.0f" % kappac[r], fontsize=8)
    #plt.text(pal_v[0],pal_v[Nv-22], r'$kappa_s=$' "%.0f" % kappas[r], fontsize=8)
    plt.text(pal_v[0],pal_v[Nv-24], r'$v_{Ae}(m/s)=$' "%.0f" % v_Ae[r], fontsize=8)
    plt.text(pal_v[0],pal_v[Nv-26], r'$\beta_c=$' "%.4f" % beta_c[r], fontsize=8)
    plt.text(pal_v[0],pal_v[Nv-28], r'$\beta_s=$' "%.4f" % beta_s[r], fontsize=8)
    plt.text(pal_v[0],pal_v[Nv-30], r'$reduced CS=$' "%.3f" % mi.redchi, fontsize=8)
    #plt.text(pal_v[Nv-10],pal_v[Nv-2], r'$T(\mathcal{v}_{Ae0}/r_s):$' "%.2f" % nu, fontsize=8)
    #plt.text(pal_v[Nv-10],pal_v[Nv-4], r'$Nv=$' "%.2f" % Nv, fontsize=8)
    #plt.text(pal_v[Nv-10],pal_v[Nv-5], r'$Nr=$' "%.2f" % Nr, fontsize=8)
    plt.colorbar(label=r'$Log(F/F_{MAX})$')
    plt.savefig(f'{path_current}fitting/{r}.png')
    plt.clf()
    plt.close()




    solu2=np.zeros(shape = (Nv))
    solu2_c=np.zeros(shape = (Nv))
    solu2_s=np.zeros(shape = (Nv))
    solu4=np.zeros(shape = (Nv))
    #solu2_h=np.zeros(shape = (Nv))
    for i in range(Nv):
        solu4[i]=np.log10(f_1[40*Nv+i,r]/fitting_max)
    for i in range(Nv):
        solu2[i]=np.log10(fitting[40*Nv+i]/fitting_max)
    for i in range(Nv):
        solu2_c[i]=np.log10(fitting_c[40*Nv+i]/fitting_max)
    for i in range(Nv):
        solu2_s[i]=np.log10(fitting_s[40*Nv+i]/fitting_max)
    #for i in range(Nv):
    #    solu2_h[i]=np.log10(fitting_h[40*Nv+i]/fitting_max)
    fig = plt.figure()
    fig.set_dpi(500)
    plt.plot(pal_v,solu2_c,color='b');
    plt.plot(pal_v,solu2_s,color='r');
    #plt.plot(pal_v,solu2_h,color='g');
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
    plt.text(pal_v[0],0, r'$n_c/n_e=$' "%.3f" % nc[r], fontsize=8)
    plt.text(pal_v[0],-0.5, r'$n_s/n_e=$' "%.3f" % ns[r], fontsize=8)
    plt.text(pal_v[0],-1, r'$Tc_{pal}(K)=$' "%.0f" % Tc_pal[r], fontsize=8)
    plt.text(pal_v[0],-1.5, r'$Tc_{per}(K)=$' "%.0f" % Tc_per[r], fontsize=8)
    plt.text(pal_v[0],-2, r'$Ts_{pal}(K)=$' "%.0f" % Ts_pal[r], fontsize=8)
    plt.text(pal_v[0],-2.5, r'$Ts_{per}(K)=$' "%.0f" % Ts_per[r], fontsize=8)
    plt.text(pal_v[0],-3, r'$U_c/v_{Ae0}=$' "%.3f" % Uc[r], fontsize=8)
    plt.text(pal_v[0],-3.5, r'$U_s/v_{Ae0}=$' "%.3f" % Us[r], fontsize=8)
    #plt.text(pal_v[0],-4, r'$kappa_c=$' "%.0f" % kappac[r], fontsize=8)
    #plt.text(pal_v[0],-4.5, r'$kappa_s=$' "%.0f" % kappas[r], fontsize=8)
    plt.text(pal_v[0],-5, r'$v_{Ae}(m/s)=$' "%.0f" % v_Ae[r], fontsize=8)
    plt.text(pal_v[0],-5.5, r'$\beta_c=$' "%.4f" % beta_c[r], fontsize=8)
    plt.text(pal_v[0],-6, r'$\beta_s=$' "%.4f" % beta_s[r], fontsize=8)
    plt.text(pal_v[0],-6.5, r'$reduced CS=$' "%.3f" % mi.redchi, fontsize=8)
    plt.text(-3*delv,-8.7,r'$\mathcal{v}_\parallel/\mathcal{v}_{Ae0}$', fontsize=12)
    plt.text(-3*delv,0.5*delv,r'$Log(F/F_{MAX})$', fontsize=12)
    plt.ylim([-8, 0])
    plt.xlim([-Mv, Mv])
    plt.rc('font', size=8)
    plt.tick_params(labelsize=8)
    plt.savefig(f'{path_current}fitting/1D/{r}.png')
    plt.clf()
    plt.close()


np.save('data_nc.npy', nc)         
np.save('data_ns.npy', ns)
#np.save('data_nh.npy', ns)
np.save('data_Tc_pal.npy', Tc_pal)
np.save('data_Tc_per.npy', Tc_per)
np.save('data_Ts_pal.npy', Ts_pal)
np.save('data_Ts_per.npy', Ts_per)
#np.save('data_Th_pal.npy', Ts_pal)
#np.save('data_Th_per.npy', Ts_per)
np.save('data_Uc.npy', Uc)
np.save('data_Us.npy', Us)
#np.save('data_kappac.npy', kappac)
#np.save('data_kappas.npy', kappas)

plt.figure(figsize=(20,15))
plt.grid()
ax = plt.gca()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlim([z[0],z[Nr-1]])
ax.set_ylim([0,1])
ax.set_xlabel(r'$r/r_s$', fontsize=28)
ax.set_ylabel(r'$Relative \ Density$', fontsize=28)
ax.plot(z,nc,linewidth=4.0, color='b',label=r'$n_c/n_e$');
ax.plot(z,ns,linewidth=4.0, color='r',label=r'$n_s/n_e$');
#ax.plot(z,nh,linewidth=4.0, color='g',label=r'$n_h/n_e$');
plt.legend(loc='upper right')
plt.savefig(f'{path_current}fitting/density.png')
plt.clf()
plt.close()

plt.figure(figsize=(20,15))
plt.grid()
ax = plt.gca()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlim([z[0],z[Nr-1]])
ax.set_ylim([10**(5),20*10**(5)])
ax.set_xlabel(r'$r/r_s$', fontsize=28)
ax.set_ylabel(r'$Temperatures (K)$', fontsize=28)
ax.plot(z,Tc_pal,linewidth=4.0, color='b',label=r'$Tc_{pal}$');
ax.plot(z,Tc_per,linewidth=4.0, color='b',linestyle='dotted',label=r'$Tc_{per}$');
ax.plot(z,Ts_pal,linewidth=4.0, color='r',label=r'$Ts_{pal}$');
ax.plot(z,Ts_per,linewidth=4.0, color='r',linestyle='dotted',label=r'$Ts_{per}$');
#ax.plot(z,Th_pal,linewidth=4.0, color='g',label=r'$Th_{pal}$');
#ax.plot(z,Th_per,linewidth=4.0, color='g',linestyle='dotted',label=r'$Th_{per}$');
ax.plot(z,max(Tc_pal)*(z[0]/z)**0.8,linewidth=3.0, color='k',linestyle='--',label=r'$1/r^{0.8} \ Profile$');
plt.legend(loc='upper right')
plt.savefig(f'{path_current}fitting/temperature.png')
plt.clf()
plt.close()

plt.figure(figsize=(20,15))
plt.grid()
ax = plt.gca()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlim([z[0],z[Nr-1]])
ax.set_ylim([-1,17])
ax.set_xlabel(r'$r/r_s$', fontsize=28)
ax.set_ylabel(r'$Bulk \ Velocity$', fontsize=28)
ax.plot(z,Uc,linewidth=4.0, color='b',label=r'$U_c/v_{Ae0}$');
ax.plot(z,Us,linewidth=4.0, color='r',label=r'$U_s/v_{Ae0}$');
plt.legend(loc='upper right')
plt.savefig(f'{path_current}fitting/BulkVelocity.png')
plt.clf()
plt.close()

#plt.figure(figsize=(20,15))
#plt.grid()
#ax = plt.gca()
#plt.rc('font', size=35)
#plt.tick_params(labelsize=40)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax.set_xlim([z[0],z[Nr-1]])
#ax.set_ylim([2,50])
#ax.set_xlabel(r'$r/r_s$', fontsize=28)
#ax.set_ylabel(r'$Kappa \ Value$', fontsize=28)
#ax.plot(z,kappac,linewidth=4.0, color='b',label=r'$kappa_c$');
#ax.plot(z,kappas,linewidth=4.0, color='r',label=r'$kappa_s$');
#plt.legend(loc='upper right')
#plt.savefig(f'{path_current}fitting/Kappa.png')
#plt.clf()
#plt.close()


plt.figure(figsize=(20,15))
plt.grid()
ax = plt.gca()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlim([z[0],z[Nr-1]])
ax.set_ylim([0,0.1])
ax.set_xlabel(r'$r/r_s$', fontsize=28)
ax.set_ylabel(r'$Beta \ Value$', fontsize=28)
ax.plot(z,beta_c,linewidth=4.0, color='b',label=r'$\beta_c$');
ax.plot(z,beta_s,linewidth=4.0, color='r',label=r'$\beta_s$');
#ax.plot(z,beta_h,linewidth=4.0, color='g',label=r'$\beta_h$');
plt.legend(loc='upper right')
plt.savefig(f'{path_current}fitting/beta.png')
plt.clf()
plt.close()

Threshold=np.zeros(shape = (Nr))
for r in range(Nr):
        Threshold[r]=(3*(2*Bol_k*Tc_pal[r]/Me)**0.5)/v_Ae_0
print(Threshold)
plt.figure(figsize=(20,15))
plt.grid()
ax = plt.gca()
plt.rc('font', size=35)
plt.tick_params(labelsize=40)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_xlim([z[0],z[Nr-1]])
ax.set_ylim([0,17])
ax.set_xlabel(r'$r/r_s$', fontsize=28)
ax.set_ylabel(r'$Bulk \ Velocity$', fontsize=28)
ax.plot(z,Threshold,linewidth=4.0, color='k',label=r'$Threshold$');
ax.plot(z,Us,linewidth=4.0, color='r',label=r'$U_s/v_{Ae0}$');
plt.legend(loc='upper right')
plt.savefig(f'{path_current}fitting/threshold.png')
plt.clf()
plt.close()
