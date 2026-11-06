# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 20:55:38 2026

@author: suare
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from matplotlib.patches import Rectangle 
import scipy.io as sio

mat_path = r"./ECG_TP4.mat"
sio.whosmat(mat_path)
mat_struct = sio.loadmat(mat_path)
ecg_one_lead = (mat_struct['ecg_lead']*1.0).flatten()

#%%Mediana

b_median = sig.medfilt(ecg_one_lead[800000+66000:800000+80000], kernel_size=199)
b_median = sig.medfilt(b_median, 599)

ecg_sin_osc_median = ecg_one_lead[800000+66000:800000+80000] - b_median


plt.figure(1)
plt.plot(ecg_one_lead[800000+66000:800000+80000])
#plt.figure(2)
plt.plot(ecg_sin_osc_median)
mat_struct([qrs_detections][800000+66000:800000+80000])
'''
plt.figure(3)
plt.plot(ecg_one_lead)
plt.show()
'''
#%%Interpolacion splines cubicos 

