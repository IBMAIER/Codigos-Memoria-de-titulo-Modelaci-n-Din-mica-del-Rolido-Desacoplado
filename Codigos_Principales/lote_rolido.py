"""
LOTE DE SIMULACIONES DE ROLIDO — ESTUDIO PARAMÉTRICO
=====================================================
Combinaciones:
  KG:         5.0, 5.15, 5.30 m
  I'_xx:      53.9, 75.3, 100.0 × 10⁶ kg·m²  → k44 [m]
  Espectros:  caso_id 1–12
  Semillas:   N_SEEDS por espectro
  Velocidades: 10, 15, 20 nudos
  μ:          90° (beam seas)

Total: 3 × 3 × 12 × N_SEEDS × 3  simulaciones de mar irregular
       + 9 roll decays (uno por cada combinación KG × k44)

Salida: estudio_parametrico_rolido.xlsx  (sin gráficos)
"""

import numpy as np
import pandas as pd
import os, sys, traceback, time as _time
from scipy import signal
from scipy.signal import find_peaks
from concurrent.futures import ProcessPoolExecutor, as_completed

# Forzar UTF-8 en stdout (necesario en Windows con cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from solver_rolido_RK4 import simular_rolido, RHO, NABLA, G
from funciones_dinamicas import CASOS_ESTUDIO

# ══════════════════════════════════════════════════════════════════════════════
# Configuración de parámetros
# ══════════════════════════════════════════════════════════════════════════════
m_ship   = RHO * NABLA                        # masa del buque [kg]

I_XX_LIST = [53.9e6, 75.3e6, 100.0e6]        # I'_xx [kg·m²]
K44_LIST  = [float(np.sqrt(I / m_ship)) for I in I_XX_LIST]   # k44 [m]

KG_LIST   = [5.0, 5.15, 5.30]                # KG [m]
V_LIST    = [5.0, 10.0, 15.0, 20.0, 25.0]    # velocidades [kn]
CASO_IDS  = list(range(1, 13))               # espectros 1–12
N_SEEDS   = 10                               # semillas por caso
K_ALETA_LIST = [0.0, 2e6, 4e6, 6e6, 8e6, 10e6]    # Valores de K_aleta = Alfa * 2e6

MU_DEG    = 90.0    # angulo de encuentro [deg]: 90 = beam seas
DT        = 0.8     # paso de integracion [s]  
T_TOTAL   = 5400.0  # duracion por simulacion [s]
T_DECAY   = 120.0   # duracion del roll decay [s] -- igual que referencia
PHI0_DECAY = 15.0   # escora inicial decay [deg]

# Guardado incremental: guarda el DataFrame cada SAVE_EVERY simulaciones
SAVE_EVERY = 100
OUTPUT_FILE = os.path.join(_SCRIPT_DIR, os.path.join(os.path.dirname(__file__), "..", "Resultados", "estudio_parametrico_rolido_ALETAS.xlsx"))
CHECKPOINT  = os.path.join(_SCRIPT_DIR, "_checkpoint_lote_aletas.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE ANÁLISIS
# ══════════════════════════════════════════════════════════════════════════════

def phi_significativa(phi_vec: np.ndarray) -> float:
    """φ₁/₃: promedio del tercio superior de amplitudes de picos."""
    peaks, _ = find_peaks(np.abs(phi_vec))
    if len(peaks) < 3:
        return float(np.max(np.abs(phi_vec)))
    amps = np.sort(np.abs(phi_vec[peaks]))[::-1]
    return float(np.mean(amps[: max(1, len(amps) // 3)]))


def periodo_dominante(t_vec: np.ndarray, phi_vec: np.ndarray):
    """Período e frecuencia dominante via Welch PSD. Retorna (T_peak, omega_peak)."""
    dt_v = t_vec[1] - t_vec[0]
    f, Pxx = signal.welch(phi_vec, fs=1.0 / dt_v, nperseg=min(len(phi_vec) // 4, 512))
    if len(f) < 2:
        return np.nan, np.nan
    idx = int(np.argmax(Pxx[1:])) + 1
    fp  = float(f[idx])
    return (1.0 / fp, 2 * np.pi * fp) if fp > 1e-6 else (np.nan, np.nan)


def damping_desde_decay(phi_vec: np.ndarray):
    """Extrae ζ y δ (decremento logarítmico) de una señal de decay.
    Retorna (zeta, delta, Tn_s)."""
    phi_deg = np.degrees(phi_vec)
    peaks, _ = find_peaks(phi_deg, height=0.5, prominence=0.3)
    if len(peaks) < 3:
        return np.nan, np.nan, np.nan
    amps   = phi_deg[peaks]
    deltas = [np.log(amps[i] / amps[i + 1])
              for i in range(len(amps) - 1)
              if amps[i] > 0 and amps[i + 1] > 0 and amps[i] > amps[i + 1]]
    if not deltas:
        return np.nan, np.nan, np.nan
    delta  = float(np.mean(deltas))
    zeta   = delta / np.sqrt(4 * np.pi**2 + delta**2)
    # Tn desde el espaciado entre picos
    t_peaks = np.linspace(0, len(phi_vec) * DT, len(phi_vec))[peaks]   # approx
    Tn = float(np.mean(np.diff(t_peaks))) if len(t_peaks) > 1 else np.nan
    return zeta, delta, Tn


def estadisticas(t_vec, phi_vec, phi_d_vec,
                 Hs, Tp, Tn,
                 kg, k44, I_xx, V, caso_id, seed, kaleta,
                 zeta_d, delta_d) -> dict:
    """Compila todas las métricas para una simulación."""
    # Descartar los primeros 100 segundos (transientes)
    idx = t_vec >= 100.0
    t_vec_est = t_vec[idx]
    phi_vec_est = phi_vec[idx]
    phi_d_vec_est = phi_d_vec[idx]

    ph  = np.degrees(phi_vec_est)
    phd = np.degrees(phi_d_vec_est)

    phi_rms = float(np.sqrt(np.mean(ph**2)))
    phi_std = float(np.std(ph))
    phi_max = float(np.max(ph))
    phi_min = float(np.min(ph))
    phi_13  = float(np.degrees(phi_significativa(phi_vec_est)))
    
    phi_sorted = np.sort(np.abs(ph))[::-1]
    phi_99 = float(phi_sorted[int(0.01 * len(phi_sorted))]) if len(phi_sorted) > 0 else 0.0
    
    phd_sorted = np.sort(np.abs(phd))[::-1]
    phi_dot_rms = float(np.sqrt(np.mean(phd**2)))
    phi_dot_13  = float(phi_significativa(phd))
    phi_dot_99  = float(phd_sorted[int(0.01 * len(phd_sorted))]) if len(phd_sorted) > 0 else 0.0
    phi_dot_max = float(np.max(np.abs(phd)))

    phdd = np.gradient(phd, t_vec_est)
    phdd_sorted = np.sort(np.abs(phdd))[::-1]
    phi_ddot_rms = float(np.sqrt(np.mean(phdd**2)))
    phi_ddot_13  = float(phi_significativa(phdd))
    phi_ddot_99  = float(phdd_sorted[int(0.01 * len(phdd_sorted))]) if len(phdd_sorted) > 0 else 0.0
    phi_ddot_max = float(np.max(np.abs(phdd)))

    T_pk, w_pk = periodo_dominante(t_vec_est, phi_vec_est)

    # Cálculo Real vs Estadístico del tiempo excedido (>30°)
    if len(t_vec_est) > 1:
        dt_est = t_vec_est[1] - t_vec_est[0]
        tiempo_total_est = t_vec_est[-1] - t_vec_est[0]
        pasos_excedidos = np.sum(np.abs(ph) > 30.0)
        Pt_real = (pasos_excedidos * dt_est / tiempo_total_est) * 100.0 if tiempo_total_est > 0 else 0.0
    else:
        Pt_real = 0.0
        
    from scipy.stats import norm
    Pt_stat = 2 * (1 - norm.cdf(30.0, loc=0, scale=phi_rms)) * 100.0 if phi_rms > 0 else 0.0

    caso = CASOS_ESTUDIO[caso_id]
    return {
        # ── Identificadores ──────────────────────────────────────────────────
        "caso_id":           caso_id,
        "ss":                caso["ss"],
        "desc":              caso["desc"],
        "Hs_m":              Hs,
        "Tp_s":              Tp,
        "KG_m":              kg,
        "I_xx_e6_kgm2":      round(I_xx / 1e6, 1),
        "k44_m":             round(k44, 4),
        "V_knots":           V,
        "K_aleta":           kaleta,
        "seed":              seed,
        "Tn_s":              round(Tn, 3) if not np.isnan(Tn) else np.nan,
        # ── Respuesta en rolido ───────────────────────────────────────────────
        "phi_rms_deg":       round(phi_rms, 4),
        "phi_13_deg":        round(phi_13, 4),
        "phi_99_deg":        round(phi_99, 4),
        "phi_max_deg":       round(phi_max, 4),
        "phi_min_deg":       round(phi_min, 4),
        "phi_std_deg":       round(phi_std, 4),
        "phi_p2p_deg":       round(phi_max - phi_min, 4),
        # ── Velocidad angular ─────────────────────────────────────────────────
        "phi_dot_rms_deg_s":  round(phi_dot_rms, 4),
        "phi_dot_13_deg_s":   round(phi_dot_13, 4),
        "phi_dot_99_deg_s":   round(phi_dot_99, 4),
        "phi_dot_max_deg_s":  round(phi_dot_max, 4),
        # ── Aceleración angular ───────────────────────────────────────────────
        "phi_ddot_rms_deg_s2": round(phi_ddot_rms, 4),
        "phi_ddot_13_deg_s2":  round(phi_ddot_13, 4),
        "phi_ddot_99_deg_s2":  round(phi_ddot_99, 4),
        "phi_ddot_max_deg_s2": round(phi_ddot_max, 4),
        # ── Frecuencia dominante ──────────────────────────────────────────────
        "T_peak_s":          round(T_pk, 3) if not np.isnan(T_pk) else np.nan,
        "omega_peak_rad_s":  round(w_pk, 4) if not np.isnan(w_pk) else np.nan,
        # ── Relaciones adimensionales ─────────────────────────────────────────
        "phi_rms_over_Hs":   round(phi_rms / Hs, 4) if Hs > 0 else np.nan,
        "Tp_over_Tn":        round(Tp / Tn, 4) if (not np.isnan(Tn) and Tn > 0) else np.nan,
        # ── Damping (desde roll decay) ────────────────────────────────────────
        "zeta_decay":        round(zeta_d, 5) if not np.isnan(zeta_d) else np.nan,
        "delta_decay":       round(delta_d, 5) if not np.isnan(delta_d) else np.nan,
        # ── Distribución ──────────────────────────────────────────────────────
        "kurtosis":          round(float(pd.Series(ph).kurt()), 4),
        "skewness":          round(float(pd.Series(ph).skew()), 4),
        # ── Tiempos Excedidos (P_t > 30) ──────────────────────────────────────
        "Pt_real_pct":       round(Pt_real, 4),
        "Pt_stat_pct":       round(Pt_stat, 4),
        "Pt_error_abs":      round(abs(Pt_real - Pt_stat), 4)
    }

def procesar_un_caso(args):
    """Función top-level para poder ser serializada por multiprocessing."""
    n, kg, k44, I_xx, V, cid, seed, kaleta, decay_map_data, dt_sim, t_tot_sim, mu_sim = args
    key_d = (kg, round(k44, 4))
    zeta_d, delta_d, Tn = decay_map_data.get(key_d, (np.nan, np.nan, np.nan))
    caso  = CASOS_ESTUDIO[cid]
    Hs, Tp = caso["Hs"], caso["Tp"]

    try:
        t_vec, phi_vec, phi_d_vec, _ = simular_rolido(
            k44=k44, kg_val=kg,
            caso_id=cid, V_knots=V, mu_deg=mu_sim,
            dt=dt_sim, T_total=t_tot_sim,
            phi0_deg=0.0, seed=seed,
            exportar=False, verbose=False, K_aleta=kaleta
        )
        row = estadisticas(t_vec, phi_vec, phi_d_vec,
                           Hs, Tp, Tn,
                           kg, k44, I_xx, V, cid, seed, kaleta,
                           zeta_d, delta_d)
        return (n, row, None)
    except Exception as e:
        err_msg = f"KG={kg} k44={k44:.2f} V={V} C{cid} s={seed} Kaleta={kaleta}: {e}"
        return (n, None, err_msg)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — ROLL DECAY  (una vez por KG × k44)
# ══════════════════════════════════════════════════════════════════════════════

def fase_decay() -> dict:
    """Retorna dict {(kg, k44_round): (zeta, delta, Tn)}."""
    decay_map = {}
    combos = [(kg, k44) for kg in KG_LIST for k44 in K44_LIST]
    print(f"\n{'='*60}")
    print(f"FASE 1: Roll Decay -- {len(combos)} combinaciones")
    print(f"{'='*60}")
    for kg, k44 in combos:
        key = (kg, round(k44, 4))
        lbl = f"  KG={kg}m  k44={k44:.3f}m"
        print(f"{lbl}...", end="", flush=True)
        try:
            t, phi, _, _ = simular_rolido(
                k44=k44, kg_val=kg,
                caso_id=0, V_knots=0.0, mu_deg=90.0,
                dt=DT, T_total=T_DECAY, phi0_deg=PHI0_DECAY,
                seed=None, exportar=False, verbose=False,
            )
            zeta, delta, Tn = damping_desde_decay(phi)
            decay_map[key] = (zeta, delta, Tn)
            print(f"  zeta={zeta:.4f}  delta={delta:.4f}  Tn={Tn:.2f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            decay_map[key] = (np.nan, np.nan, np.nan)
    return decay_map


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — BUCLE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def fase_irregular(decay_map: dict) -> pd.DataFrame:
    # Construir lista completa de casos
    casos = [
        (kg, k44, I_xx, V, cid, s, kaleta)
        for kg in KG_LIST
        for k44, I_xx in zip(K44_LIST, I_XX_LIST)
        for V in V_LIST
        for cid in CASO_IDS
        for s in range(N_SEEDS)
        for kaleta in K_ALETA_LIST
    ]
    n_total = len(casos)
    print(f"\n{'='*60}")
    print(f"FASE 2: Mar irregular -- {n_total} simulaciones (MULTINÚCLEO)")
    print(f"{'='*60}")

    resultados = []
    n_err = 0
    t0 = _time.time()

    # Preparamos los argumentos para cada proceso
    args_list = []
    for n, (kg, k44, I_xx, V, cid, seed, kaleta) in enumerate(casos, 1):
        args_list.append((n, kg, k44, I_xx, V, cid, seed, kaleta, decay_map, DT, T_TOTAL, MU_DEG))

    max_workers = max(1, os.cpu_count() - 2) # Deja dos núcleos libres para el OS
    print(f"  Usando {max_workers} núcleos...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futuros = {executor.submit(procesar_un_caso, arg): arg for arg in args_list}
        
        completados = 0
        for futuro in as_completed(futuros):
            completados += 1
            n_orig, row, err_msg = futuro.result()
            
            elapsed = _time.time() - t0
            eta_m   = (elapsed / completados) * (n_total - completados) / 60 if completados > 1 else 0
            
            if row is not None:
                resultados.append(row)
            else:
                n_err += 1
                print(f"\n  [ERR] {err_msg}")

            # Imprime el avance
            if completados % SAVE_EVERY == 0 or completados == n_total or completados == 1:
                print(f"  [{completados:4d}/{n_total}] {100*completados/n_total:5.1f}%  "
                      f"ETA~{eta_m:.0f}min", flush=True)

            # Checkpoint incremental
            if completados % SAVE_EVERY == 0 and resultados:
                pd.DataFrame(resultados).to_pickle(CHECKPOINT)

    print(f"\n\n  Completadas: {len(resultados)}/{n_total}  ({n_err} errores)")
    print(f"  Tiempo total: {(_time.time()-t0)/60:.1f} min")
    return pd.DataFrame(resultados)


# ══════════════════════════════════════════════════════════════════════════════
# GUARDADO EN EXCEL
# ══════════════════════════════════════════════════════════════════════════════

def guardar_excel(df: pd.DataFrame, decay_map: dict):
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        # Hoja completa
        df.to_excel(writer, sheet_name="Resultados", index=False)

        # Resumen promediado sobre semillas
        grp_cols = ["KG_m", "I_xx_e6_kgm2", "k44_m", "V_knots", "K_aleta", "caso_id", "ss", "Hs_m", "Tp_s"]
        agg = {
            "phi_rms_deg":       ["mean", "std"],
            "phi_13_deg":        ["mean", "std"],
            "phi_99_deg":        ["mean", "std"],
            "phi_max_deg":       ["mean", "max"],
            "phi_std_deg":       ["mean"],
            "phi_dot_rms_deg_s": ["mean"],
            "phi_dot_13_deg_s":  ["mean"],
            "phi_dot_99_deg_s":  ["mean"],
            "phi_dot_max_deg_s": ["mean", "max"],
            "phi_ddot_rms_deg_s2": ["mean"],
            "phi_ddot_13_deg_s2":  ["mean"],
            "phi_ddot_99_deg_s2":  ["mean"],
            "phi_ddot_max_deg_s2": ["mean", "max"],
            "T_peak_s":          ["mean"],
            "phi_rms_over_Hs":   ["mean"],
            "Tp_over_Tn":        ["mean"],
            "kurtosis":          ["mean"],
            "skewness":          ["mean"],
        }
        summary = df.groupby(grp_cols, sort=False).agg(agg)
        summary.columns = ["_".join(c).strip("_") for c in summary.columns]
        summary = summary.reset_index()
        summary.to_excel(writer, sheet_name="Resumen_semillas", index=False)

        # Hoja roll decay
        decay_rows = [
            {"KG_m": k[0], "k44_m": k[1],
             "zeta": v[0], "delta": v[1], "Tn_s": v[2]}
            for k, v in decay_map.items()
        ]
        pd.DataFrame(decay_rows).to_excel(writer, sheet_name="Roll_Decay", index=False)

    print(f"\n  Excel guardado -> {OUTPUT_FILE}")
    # Limpiar checkpoint
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nRadios de giro (k44):")
    for I, k in zip(I_XX_LIST, K44_LIST):
        print(f"  I_xx = {I/1e6:.1f}e6 kg*m2  ->  k44 = {k:.4f} m")

    n_irr = len(KG_LIST) * len(K44_LIST) * len(V_LIST) * len(CASO_IDS) * N_SEEDS
    n_dec = len(KG_LIST) * len(K44_LIST)
    print(f"\nPlan:")
    print(f"  Roll decay:    {n_dec} runs  (phi0={PHI0_DECAY} deg, T={T_DECAY}s)")
    print(f"  Mar irregular: {n_irr} runs  (dt={DT}s, T={T_TOTAL}s, mu={MU_DEG} deg)")
    print(f"  Estimado ~ {n_irr * 8 / 3600:.1f} - {n_irr * 20 / 3600:.1f} horas (depende del hardware)")

    decay_map = fase_decay()
    df_result = fase_irregular(decay_map)
    guardar_excel(df_result, decay_map)

    print("\n¡Estudio completado!")
    print(f"  Filas en Excel:  {len(df_result)}")
    print(f"  Columnas:        {len(df_result.columns)}")
