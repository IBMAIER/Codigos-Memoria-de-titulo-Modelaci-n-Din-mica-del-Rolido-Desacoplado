import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from solver_rolido_RK4 import simular_rolido
from lote_rolido import phi_significativa

OUT_DIR = os.path.join(_SCRIPT_DIR, "estudio_convergencia")
os.makedirs(OUT_DIR, exist_ok=True)

# Parámetros Fijos
KG = 5.0
K44 = 5.5
V_KNOTS = 10.0
MU = 90.0
SEED = 42

CASOS = list(range(1, 13))

def simular_y_extraer(dt, T_tot, caso_id):
    try:
        t_vec, phi_vec, phi_d_vec, _ = simular_rolido(
            k44=K44, kg_val=KG, caso_id=caso_id, V_knots=V_KNOTS,
            mu_deg=MU, dt=dt, T_total=T_tot, phi0_deg=0.0, seed=SEED,
            exportar=False, verbose=False, K_aleta=0.0
        )
        
        idx = t_vec >= 100.0
        if not np.any(idx):
            return np.nan, np.nan, np.nan
            
        phi_est = np.degrees(phi_vec[idx])
        
        phi_max = np.max(np.abs(phi_est))
        phi_rms = np.sqrt(np.mean(phi_est**2))
        phi_13 = np.degrees(phi_significativa(phi_vec[idx]))
        
        return phi_max, phi_rms, phi_13
    except Exception as e:
        return np.nan, np.nan, np.nan

def worker_dt(args):
    dt, caso_id = args
    T_tot = 3600.0
    p_max, p_rms, p_13 = simular_y_extraer(dt, T_tot, caso_id)
    return {'caso_id': caso_id, 'dt': dt, 'phi_max': p_max, 'phi_rms': p_rms, 'phi_13': p_13}

def worker_T(args):
    T_tot, caso_id = args
    dt = 0.1  # dt fijo a 0.1 (0.5 puede inestabilizar el RK4 por completo)
    p_max, p_rms, p_13 = simular_y_extraer(dt, T_tot, caso_id)
    return {'caso_id': caso_id, 'T_total': T_tot, 'phi_max': p_max, 'phi_rms': p_rms, 'phi_13': p_13}

# Controla si se re-ejecuta el análisis de dt (ya validado) o solo T_total
RUN_DT = False

def ejecutar_analisis():
    # =========================================================================
    # 1. Variación de DT (Para T_total = 3600s fijo) — saltar si RUN_DT=False
    # =========================================================================
    dts = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    
    if RUN_DT:
        print(f"Iniciando Análisis de Convergencia 1: Variación de dt (T_total=3600s)...")
        args_dt = [(dt, cid) for dt in dts for cid in CASOS]
        res_dt = []
        with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 1)) as executor:
            futuros = [executor.submit(worker_dt, arg) for arg in args_dt]
            for idx, f in enumerate(as_completed(futuros), 1):
                res_dt.append(f.result())
                print(f"  Progreso DT: {idx}/{len(args_dt)} simulaciones...", end='\r')
        df_dt = pd.DataFrame(res_dt)
    else:
        print("[SKIP] Análisis de dt omitido (RUN_DT=False). Cargando resultados previos...")
        df_dt = pd.read_excel(os.path.join(OUT_DIR, "datos_convergencia_dt.xlsx"))
    
    # =========================================================================
    # 2. Variación de T_total (Para dt = 0.1s fijo)
    # =========================================================================
    T_tots = [600, 1200, 1800, 2400, 3000, 3600, 4200, 5400, 7200, 8600, 10000]
    print("\n\nIniciando Análisis de Convergencia 2: Variación de T_total (dt=0.1s)...")
    args_T = [(T, cid) for T in T_tots for cid in CASOS]
    res_T = []
    
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 1)) as executor:
        futuros = [executor.submit(worker_T, arg) for arg in args_T]
        for idx, f in enumerate(as_completed(futuros), 1):
            res_T.append(f.result())
            print(f"  Progreso T_total: {idx}/{len(args_T)} simulaciones...", end='\r')
            
    df_T = pd.DataFrame(res_T)
    
    # Exportar datos a Excel
    df_dt.to_excel(os.path.join(OUT_DIR, "datos_convergencia_dt.xlsx"), index=False)
    df_T.to_excel(os.path.join(OUT_DIR, "datos_convergencia_Ttotal.xlsx"), index=False)
    
    sns.set_theme(style="whitegrid", context="paper")
    
    print("\n\nGenerando Gráficos...")
    # Gráficos de DT
    for cid in CASOS:
        data = df_dt[df_dt['caso_id'] == cid].sort_values('dt')
        plt.figure(figsize=(10, 6))
        plt.plot(data['dt'], data['phi_max'], marker='o', label='$\\phi_{max}$', color='red')
        plt.plot(data['dt'], data['phi_13'], marker='s', label='$\\phi_{1/3}$', color='orange')
        plt.plot(data['dt'], data['phi_rms'], marker='^', label='$\\phi_{rms}$', color='blue')
        plt.title(f"Convergencia Numérica vs Paso de Integración (dt)\nEstado de Mar {cid} | T_total = 3600 s")
        plt.xlabel("Paso de integración temporal dt [s]")
        plt.ylabel("Amplitud de Rolido [°]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"01_Conv_DT_CasoMar_{cid:02d}.png"), dpi=200)
        plt.close()
        
    # Gráficos de T_total
    for cid in CASOS:
        data = df_T[df_T['caso_id'] == cid].sort_values('T_total')
        plt.figure(figsize=(10, 6))
        plt.plot(data['T_total'], data['phi_max'], marker='o', label='$\\phi_{max}$', color='red')
        plt.plot(data['T_total'], data['phi_13'], marker='s', label='$\\phi_{1/3}$', color='orange')
        plt.plot(data['T_total'], data['phi_rms'], marker='^', label='$\\phi_{rms}$', color='blue')
        plt.title(f"Convergencia Estadística vs Tiempo de Simulación ($T_{{total}}$)\nEstado de Mar {cid} | dt = 0.1 s")
        plt.xlabel("Tiempo de Simulación $T_{total}$ [s]")
        plt.ylabel("Amplitud de Rolido [°]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"02_Conv_Ttotal_CasoMar_{cid:02d}.png"), dpi=200)
        plt.close()

    print(f"\n¡Análisis completado! Revisa la carpeta: {OUT_DIR}")

if __name__ == "__main__":
    ejecutar_analisis()
