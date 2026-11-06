# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:33:04 2026

@author: suare
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from matplotlib.patches import Rectangle 
import scipy.io as sio

ruta_npy = './ecg_sin_ruido.npy'
ecg_sin_ruido = np.load(ruta_npy)

print("Tipo:", type(ecg_sin_ruido))
print("Forma (shape):", ecg_sin_ruido.shape)

mat_path = './ECG_TP4.mat'
sio.whosmat(mat_path)
mat_struct = sio.loadmat(mat_path)
ecg_one_lead = mat_struct['ecg_lead']

# Definición de parámetros (Pasabanda)
fs_ecg = 500
wp1 = 1    #[Hz]
ws1 = 0.1   #[Hz]  
wp2 = 35   #[Hz]
ws2 = 50   #[Hz]
Gpass = 3
Gstop = 40

ripple = Gpass
att = Gstop

# Definición del vector de frecuencias personalizado
ff = np.append(np.logspace(np.log10(.001), np.log10(2), 2000), np.logspace(np.log10(2.01), np.log10(50)))
ff = np.append(ff, np.linspace(51, fs_ecg/2, 1000, endpoint=True))

# Funciones auxiliares para el cálculo de frecuencia
def calc_freq_sos(sos):
    resp_freq = sig.freqz_sos(sos, worN=ff, fs=fs_ecg)
    return resp_freq

def calc_freq_fir(bfir):
    resp_freq = sig.freqz(bfir, 1, worN=ff, fs=fs_ecg)
    return resp_freq

# Diseño de los filtros en formato SOS
orden_butter = 4
sos_butt = sig.iirdesign([wp1, wp2+5], [ws1, ws2], Gpass, Gstop, ftype='butter', fs=fs_ecg, output='sos')
sos_cheby2 = sig.iirdesign([wp1, wp2+5], [ws1, ws2], Gpass, Gstop, ftype='cheby2', fs=fs_ecg, output='sos')
#sos_cauer = sig.iirdesign([wp1, wp2], [ws1, ws2], Gpass, Gstop, ftype='cauer', fs=fs_ecg, output='sos')

# FIR design (Ventanas)
taps_win = 2000
b_win_boxcar = sig.firwin2(taps_win+20, [0.0, ws1+0.15, wp1-0.3, wp2, ws2, fs_ecg/2], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0], fs=fs_ecg, window="boxcar")
#b_win_Hamming = sig.firwin2(taps_win, [0.0, ws1+0.3, wp1, wp2, ws2, fs_ecg/2], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0], fs=fs_ecg, window="hamming")

# FIR design (Mínimos Cuadrados - firls requiere número de taps impar)
b_firls = sig.firls(1251, [0.0, ws1+0.25, wp1, wp2+1.5, ws2, fs_ecg/2], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0], fs=fs_ecg)

# FIR design (Remez / Equiripple - orden moderado para asegurar convergencia)
# %%
b_remez = sig.remez(631,[0.0, ws1, wp1, wp2+5, ws2-5, fs_ecg/2], [0.0, 1.0, 0.0], weight=[5, 1, 5], fs=fs_ecg)

y_min, y_max = -100, 5

#%% Graficar la respuesta en frecuencia (Módulo y Plantilla)
plt.close('all')

fig, ax_mod = plt.subplots(figsize=(10, 5))

# Graficamos los módulos de los filtros IIR
for sos, name in zip([sos_butt, sos_cheby2], ['butter', 'cheby2']): #acá se borró Cauer
    f, h = calc_freq_sos(sos)
    modulo_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    ax_mod.plot(f, modulo_db, linewidth=1.5, label=name)

# Graficamos los filtros FIR (Ventanas, Mínimos Cuadrados y Remez)
fir_filters = [
    (b_win_boxcar, 'win_boxcar'), 
    #(b_win_Hamming, 'win_Hamming'), 
    (b_firls, 'FIR_LeastSquares'),
    (b_remez, 'FIR_Remez')
]

for b_fir, name in fir_filters:
    f, h = calc_freq_fir(b_fir)
    modulo_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    ax_mod.plot(f, modulo_db, linewidth=1.5, linestyle='--', label=name)

ax_mod.set_ylim(y_min, y_max)
ax_mod.set_ylabel("Módulo (dB)")
ax_mod.set_xlabel("Frecuencia (Hz)")
ax_mod.set_title("Respuesta en frecuencia (Módulo) y plantilla pasabanda")
ax_mod.grid(True)

# Plantilla pasabanda
rect_rechazo_inf = Rectangle(
    (0, -att), ws1, y_max + att,
    facecolor="red", alpha=0.12, edgecolor="red", hatch="xxx", linewidth=1.2, label='Plantilla'
)
ax_mod.add_patch(rect_rechazo_inf)

rect_rechazo_sup = Rectangle(
    (ws2, -att), fs_ecg / 2 - ws2, y_max + att,
    facecolor="red", alpha=0.12, edgecolor="red", hatch="xxx", linewidth=1.2
)
ax_mod.add_patch(rect_rechazo_sup)

rect_paso_inf = Rectangle(
    (wp1, y_min), wp2 - wp1, -Gpass - y_min,
    facecolor="red", alpha=0.12, edgecolor="red", hatch="xxx", linewidth=1.2
)
ax_mod.add_patch(rect_paso_inf)

rect_paso_sup = Rectangle(
    (wp1, 0), wp2 - wp1, y_max - 0,
    facecolor="red", alpha=0.12, edgecolor="red", hatch="xxx", linewidth=1.2
)
ax_mod.add_patch(rect_paso_sup)

ax_mod.axvline(wp1, color="gray", linestyle=":")
ax_mod.axvline(wp2, color="gray", linestyle=":")
ax_mod.axvline(ws1, color="gray", linestyle="--")
ax_mod.axvline(ws2, color="gray", linestyle="--")
ax_mod.legend(loc="lower right", fontsize=7)
plt.show()

#%% FASE DE TODOS LOS FILTROS EN UN MISMO GRÁFICO (IIR y FIR)
plt.figure(figsize=(10, 5))

# Fase filtros IIR
for sos, name in zip([sos_butt, sos_cheby2], ['butter', 'cheby2']):
    f, h = calc_freq_sos(sos)
    fase_rad = np.unwrap(np.angle(h))
    plt.plot(f, fase_rad, linewidth=1.2, label=name)

# Fase todos los filtros FIR
for b_fir, name in fir_filters:
    f, h = calc_freq_fir(b_fir)
    fase_rad = np.unwrap(np.angle(h))
    plt.plot(f, fase_rad, linewidth=1.2, linestyle='--', label=name)

plt.axvline(wp1, color="gray", linestyle=":")
plt.axvline(wp2, color="gray", linestyle=":")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Fase (rad)")
plt.title("Fase de todos los filtros (IIR y FIR)")
plt.grid(True)
plt.legend(loc='lower left', fontsize=7)
plt.show()

#%% RETARDO DE GRUPO DE LOS FILTROS IIR EN UN MISMO GRÁFICO
plt.figure(figsize=(10, 5))
for sos, name in zip([sos_butt, sos_cheby2], ['butter', 'cheby2']):
    b_temp, a_temp = sig.sos2tf(sos)
    w_gd, gd_muestras = sig.group_delay((b_temp, a_temp), w=ff, fs=fs_ecg)
    gd_segundos = gd_muestras / fs_ecg
     
    plt.plot(w_gd, gd_segundos, label=name)

plt.axvline(wp1, color="orange", linestyle="--", label=f"wp1 = {wp1} Hz")
plt.axvline(wp2, color="orange", linestyle="--", label=f"wp2 = {wp2} Hz")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Retardo de grupo (s)")
plt.title("Retardo de grupo de los filtros IIR")
plt.grid(True)
plt.legend()
plt.show()

#%% RESPUESTA AL IMPULSO DE LOS FILTROS IIR EN UN MISMO GRÁFICO
'''
delta_k = sig.unit_impulse(1000)
t_ms = (np.arange(len(delta_k)) / fs_ecg) * 1000   

resp_imp_butt = sig.sosfiltfilt(sos_butt, delta_k)
resp_imp_cheby2 = sig.sosfiltfilt(sos_cheby2, delta_k)
#resp_imp_cauer = sig.sosfiltfilt(sos_cauer, delta_k)

plt.figure(figsize=(10, 4))
plt.plot(t_ms, resp_imp_butt, linewidth=1.25, label='butter')
plt.plot(t_ms, resp_imp_cheby2, linewidth=1.25, label='cheby2')
#plt.plot(t_ms, resp_imp_cauer, linewidth=1.25, label='cauer')
plt.legend()

plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud")
plt.title("Respuesta al impulso de los filtros IIR (Fase Cero - sosfiltfilt)")
plt.grid(True)
plt.show()
'''

#%% APLICACIÓN Y FILTRADO DEL ECG
ecg_flat = ecg_one_lead.flatten()

ecg_filt_butt = sig.sosfilt(sos_butt, ecg_flat)
ecg_filt_cheby2 = sig.sosfilt(sos_cheby2, ecg_flat)
#ecg_filt_cauer = sig.sosfilt(sos_cauer, ecg_flat)

N_plot = min(5000, len(ecg_flat))
muestras_plot = np.arange(N_plot)

plt.figure(figsize=(10, 5))
plt.plot(muestras_plot, ecg_flat[:N_plot], color='purple', alpha=0.3, linewidth=2, label='sin filtrado (Original)')
plt.plot(muestras_plot, ecg_filt_butt[:N_plot], color='dodgerblue', linewidth=1.5, label='filtrado_butter')
plt.plot(muestras_plot, ecg_filt_cheby2[:N_plot], color='orange', linewidth=1.5, label='filtrado_cheby2')
#plt.plot(muestras_plot, ecg_filt_cauer[:N_plot], color='crimson', linewidth=1.5, label='filtrado_cauer')

plt.legend(loc='upper right')
plt.ylabel('Adimensional')
plt.xlabel('Muestras (#)')
plt.title("ECG filtrado con IIR Pasabanda")
plt.grid(True)
plt.show()