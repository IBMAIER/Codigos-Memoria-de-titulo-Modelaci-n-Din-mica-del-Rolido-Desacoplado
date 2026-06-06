"""
VERIFICACION DE lote_rolido.py vs Excel de referencia
=======================================================
"""

import sys, os, time as _time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from solver_rolido_RK4 import simular_rolido, RHO, NABLA
from lote_rolido import phi_significativa, periodo_dominante, damping_desde_decay
from funciones_dinamicas import CASOS_ESTUDIO

# Parámetros exactos de los Excel de referencia (no los del lote nuevo)
K44  = 5.5      # m
KG   = 5.0      # m
SEED = 42
MU   = 90.0
DT   = 0.05
T_TOTAL_REF = 600.0
T_DECAY_REF = 120.0
PHI0_DECAY = 15.0
UMBRAL = 3.0    # % tolerancia

def stats_no_discard(phi_rad, phi_d_rad, t_vec):
    """Calcula stats SIN descartar los 100s, para comparar igual a los Excel"""
    ph  = np.degrees(phi_rad)
    phd = np.degrees(phi_d_rad)
    T_pk, _ = periodo_dominante(t_vec, phi_rad)
    return {
        'phi_rms':    round(float(np.sqrt(np.mean(ph**2))), 4),
        'phi_13':     round(float(np.degrees(phi_significativa(phi_rad))), 4),
        'phi_max':    round(float(np.max(ph)), 4),
        'phi_min':    round(float(np.min(ph)), 4),
        'phi_std':    round(float(np.std(ph)), 4),
        'phd_rms':    round(float(np.sqrt(np.mean(phd**2))), 4),
        'phd_max':    round(float(np.max(np.abs(phd))), 4),
        'T_peak':     round(T_pk, 3)
    }

def diff_pct(ref, new):
    return 100*(new-ref)/abs(ref) if abs(ref)>1e-9 else float('nan')

print('='*60)
print('INFORME DE VERIFICACION (Comparando con Excel)')
print('='*60)

resultados = []

# ----------------------------------------------------------------
# CASO 0: Roll Decay
# ----------------------------------------------------------------
print(f'\\nCASO 0 - Roll Decay (V=0, phi0={PHI0_DECAY}deg, T={T_DECAY_REF}s, dt={DT}s)')
ref0 = pd.read_excel('rolido_RK4_C0_KG5.0_V0.0_dt0.05.xlsx')
phi_r0 = ref0['phi_rad'].values
z_r, d_r, Tn_r = damping_desde_decay(phi_r0)

td,phd_arr,_,_ = simular_rolido(k44=K44,kg_val=KG,caso_id=0,V_knots=0.0,
    mu_deg=MU,dt=DT,T_total=T_DECAY_REF,phi0_deg=PHI0_DECAY,
    seed=None,exportar=False,verbose=False)
z_l, d_l, Tn_l = damping_desde_decay(phd_arr)

dz=diff_pct(z_r,z_l); dT=diff_pct(Tn_r,Tn_l)
ok0=(abs(dz)<10 and abs(dT)<5)
print(f'  Ref:  zeta={z_r:.4f}  delta={d_r:.4f}  Tn={Tn_r:.2f}s')
print(f'  Lote: zeta={z_l:.4f}  delta={d_l:.4f}  Tn={Tn_l:.2f}s')
print(f'  Diff: zeta={dz:.1f}%   Tn={dT:.1f}%')
print(f'  => {"OK" if ok0 else "FALLA"}')

# ----------------------------------------------------------------
# CASO 2: SS3, V=20
# ----------------------------------------------------------------
print(f'\\nCASO 2 - SS3 (V=20kn, seed={SEED}, T={T_TOTAL_REF}s, dt={DT}s)')
ref2 = pd.read_excel('rolido_RK4_C2_KG5.0_V20.0_dt0.05.xlsx')
sr2  = stats_no_discard(ref2['phi_rad'].values, ref2['phi_d_rad_s'].values, ref2['t_s'].values)

t2,phi2,phd2,_ = simular_rolido(k44=K44,kg_val=KG,caso_id=2,V_knots=20.0,
    mu_deg=MU,dt=DT,T_total=T_TOTAL_REF,phi0_deg=0.0,
    seed=SEED,exportar=False,verbose=False)
sl2 = stats_no_discard(phi2, phd2, t2)

crit2=[]
for k in ['phi_rms','phi_13','phi_max']:
    r,l = sr2[k], sl2[k]
    dp  = diff_pct(r,l)
    crit2.append(abs(dp))
    print(f'  {k:<18} {r:>10.4f} {l:>10.4f} {dp:>8.2f}%')
ok2 = all(v<UMBRAL for v in crit2)
print(f'  => {"OK" if ok2 else "REVISAR"}')

# ----------------------------------------------------------------
# CASO 6: SS4, V=20
# ----------------------------------------------------------------
print(f'\\nCASO 6 - SS4 (V=20kn, seed={SEED}, T={T_TOTAL_REF}s, dt={DT}s)')
ref6 = pd.read_excel('rolido_RK4_C6_KG5.0_V20.0_dt0.05.xlsx')
sr6  = stats_no_discard(ref6['phi_rad'].values, ref6['phi_d_rad_s'].values, ref6['t_s'].values)

t6,phi6,phd6,_ = simular_rolido(k44=K44,kg_val=KG,caso_id=6,V_knots=20.0,
    mu_deg=MU,dt=DT,T_total=T_TOTAL_REF,phi0_deg=0.0,
    seed=SEED,exportar=False,verbose=False)
sl6 = stats_no_discard(phi6, phd6, t6)

crit6=[]
for k in ['phi_rms','phi_13','phi_max']:
    r,l = sr6[k], sl6[k]
    dp  = diff_pct(r,l)
    crit6.append(abs(dp))
    print(f'  {k:<18} {r:>10.4f} {l:>10.4f} {dp:>8.2f}%')
ok6 = all(v<UMBRAL for v in crit6)
print(f'  => {"OK" if ok6 else "REVISAR"}')

# ----------------------------------------------------------------
# RESUMEN
# ----------------------------------------------------------------
all_ok = ok0 and ok2 and ok6
print('\\n' + '='*60)
print(f'  VEREDICTO: {"=== VERIFICACION EXITOSA ===" if all_ok else "*** REVISAR ***"}')
print('='*60)

if all_ok:
    print("\\nVerificacion OK -> lanzando lote_rolido.py (T_total=700s, descartando 100s transiente)")
    import lote_rolido as L
    dm = L.fase_decay()
    df_res = L.fase_irregular(dm)
    L.guardar_excel(df_res, dm)
    print("\\nLote finalizado.")
else:
    print("\\nVerificacion no paso -> lote NO lanzado.")
