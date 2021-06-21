Python 3.8.2 (tags/v3.8.2:7b3ab59, Feb 25 2020, 23:03:10) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> from numpy.linalg import inv
>>> from numpy import dot
>>> import numpy as np
>>> from numpy import pi,exp,sqrt
>>> import matplotlib.pyplot as plt
>>> from mpl_toolkits.mplot3d import axes3d
>>> from mpmath import *
>>> from matplotlib.ticker import MultipleLocator
>>> import matplotlib as mpl
>>> from math import gamma
>>> import math as math
>>> from scipy import integrate
>>> n_c=25.768190*10**(6)
>>> n_s=4.090507*10**6
>>> vth_pal_c=2.817809*10**6
>>> vth_per_c=2.866745*10**6
>>> vth_pal_s=4.725664*10**6
>>> vth_per_s=3.300259*10**6
>>> U_s=3.734004*10**6
>>> n_s/(n_c+n_s)*100
13.699549581818657
>>> n_e=54.336258*10**(6)
>>> n_ee=n_c+n_s
>>> n_c_real=n_e*n_c/n_ee
>>> n_s_real=n_e*n_s/n_ee
>>> n_s_real/(n_c_real+n_s_real)
0.13699549581818657
>>> B_0=38.490482*10**(-9)
>>> v_Ae=(B_0)/(4*np.pi*10**(-7)*9.1094*10**(-31)*n_e)**0.5
>>> beta_pal_c=vth_pal_c**2/v_Ae**2*(n_c_real/n_e)
>>> beta_pal_s=vth_pal_s**2/v_Ae**2*(n_s_real/n_e)
>>> beta_per_c=vth_per_c**2/v_Ae**2*(n_c_real/n_e)
>>> beta_per_s=vth_per_s**2/v_Ae**2*(n_s_real/n_e)
>>> Anisotropy_c=vth_per_c**2/vth_pal_c**2
>>> Anisotropy_s=vth_per_s**2/vth_pal_s**2
>>> v_Ap=(B_0)/(4*np.pi*10**(-7)*1.6726*10**(-27)*n_e)**0.5
>>> vth_pal_p=61680.645
>>> beta_pal_p=vth_pal_p**2/v_Ap**2
>>> U_b=394795.013
>>> U_b=-394795.013
>>> U_s_real=U_s-U_b
>>> U_s_real/v_Ap
36.25068638087187
>>> -U_s_real/v_Ap*(n_s/n_c)
-5.754524721983229
>>> beta_pal_c
0.28768606770423355
>>> beta_pal_s
0.12844423943462926
>>> beta_pal_p
0.29328023569656086
>>> Anisotropy_c
1.0350349708140978
>>> Anisotropy_s
0.4877197797586079
>>> n_c/(n_c+n_s)*100
86.30045041818134
>>> 