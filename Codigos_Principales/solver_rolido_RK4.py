"""
SOLVER RK4 — ECUACIÓN DE ROLIDO ACOPLADO EN MAR IRREGULAR
===========================================================
Ecuación gobernante:
    (I44 + A44) * phi_dd + B44(phi, phi_d, omega_E) * phi_d + Delta * GZ(phi) = Mw(t)

Estado:  x = [phi, phi_d]
         x_d = [phi_d, phi_dd]

Donde:
  phi_dd = [ Mw(t) - B44*phi_d - Delta*GZ(phi) ] / (I44 + A44)

Momento de ola (Forma B — precomputado):
  Mw(t) = sum_i  rho*g*zeta_i * [ Ci(phi)*cos(theta_i) + Si(phi)*sin(theta_i) ]
  theta_i = omega_i * t + epsilon_i   (fase aleatoria)

Interpolación de GZ y Ci/Si:  scipy.interpolate.interp1d sobre los arrays precomputados.
"""

import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time as _time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── rutas ──────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from funciones_dinamicas import discretizar_espectro_no_uniforme, CASOS_ESTUDIO
from calcular_damping     import calcular_damping_B44

# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS GLOBALES DEL BUQUE  (constantes físicas del casco)
# ══════════════════════════════════════════════════════════════════════════════
RHO   = 1025.0        # densidad agua [kg/m³]
G     = 9.81          # gravedad [m/s²]
L_PP  = 71.75         # eslora [m]
B_ship = 14.402       # manga [m]
D_ship = 4.7          # calado diseño [m]
C_B   = 0.4698        # coeficiente de bloque
C_M   = 0.8681        # coeficiente sección maestra (calculo_hidrostatico.py: A_M/B*T)
NABLA = 2281.744      # volumen de carena [m³]
NU    = 1.19e-6       # viscosidad cinemática [m²/s]
B_BK  = 0.0           # ancho quilla de balance [m] — sin quillas instaladas
L_BK  = 0.0           # longitud quilla de balance [m] — sin quillas instaladas
DELTA = RHO * G * NABLA   # desplazamiento [N]

# NOTA: el k44 que se pasa a simular_rolido es el radio de giro efectivo/ajustado
# que ya incluye la contribución de la masa añadida A44.
# Por lo tanto:  I_tot = m * k44_eff^2 = (rho*nabla) * k44^2
# NO se aplica ningún factor adicional sobre A44.

# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS PRECOMPUTADOS
# ══════════════════════════════════════════════════════════════════════════════

def _cargar_gz(kg_val: float) -> interp1d:
    """
    Lee resultados_curva_GZ.csv y devuelve un interpolador GZ(phi_rad).
    Convención: GZ positivo → momento restaurador (empuja al buque a 0).
    """
    ruta = os.path.join(_SCRIPT_DIR, os.path.join(os.path.dirname(__file__), "..", "Resultados", "resultados_curva_GZ.csv"))
    df   = pd.read_csv(ruta, sep=";", decimal=",")

    col_gz = f"GZ_{kg_val}"
    if col_gz not in df.columns:
        # busca la más cercana
        gz_cols = [c for c in df.columns if c.startswith("GZ_") and not c.startswith("GZ_lin")]
        nums    = [float(c.split("_")[1]) for c in gz_cols]
        closest = gz_cols[int(np.argmin(np.abs(np.array(nums) - kg_val)))]
        print(f"[Aviso] Columna '{col_gz}' no encontrada; usando '{closest}'.")
        col_gz = closest

    phi_pos = np.radians(df["phi_deg"].values)          # [0 … π/2]
    gz_pos  = df[col_gz].values

    # Extender simétricamente: GZ(-phi) = -GZ(phi)
    phi_all = np.concatenate([-phi_pos[::-1][:-1], phi_pos])
    gz_all  = np.concatenate([-gz_pos[::-1][:-1],  gz_pos])

    return interp1d(phi_all, gz_all, kind="cubic",
                    bounds_error=False, fill_value=(gz_all[0], gz_all[-1]))


def _cargar_matrices_excitacion(kg_val: float):
    """
    Lee matrices_excitacion_KG{kg}.xlsx.
    Retorna:
        phis_rad : (N_phi,)  ángulos de escora [rad]
        Ci_mat   : (N_omega, N_phi)
        Si_mat   : (N_omega, N_phi)
        omegas_exc : (N_omega,)  frecuencias del Excel [rad/s]
    """
    fname = os.path.join(_SCRIPT_DIR, os.path.join(os.path.dirname(__file__), "..", "Resultados", f"matrices_excitacion_KG{kg_val}.xlsx"))
    if not os.path.exists(fname):
        # busca el más cercano
        for f in os.listdir(_SCRIPT_DIR):
            if f.startswith("matrices_excitacion_KG") and f.endswith(".xlsx"):
                fname = os.path.join(_SCRIPT_DIR, f)
                print(f"[Aviso] Usando archivo de excitación alternativo: {f}")
                break
        else:
            raise FileNotFoundError(f"No se encontró {fname}. Ejecuta lote_matrices.py primero.")

    df_Ci = pd.read_excel(fname, sheet_name="Ci(phi)", index_col=0)
    df_Si = pd.read_excel(fname, sheet_name="Si(phi)", index_col=0)

    omegas_exc = df_Ci.index.to_numpy(dtype=float)
    phis_deg   = df_Ci.columns.to_numpy(dtype=float)
    phis_rad   = np.radians(phis_deg)

    Ci_mat = df_Ci.to_numpy(dtype=float)   # (N_omega, N_phi)
    Si_mat = df_Si.to_numpy(dtype=float)

    return phis_rad, Ci_mat, Si_mat, omegas_exc


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL DEL SOLVER
# ══════════════════════════════════════════════════════════════════════════════

def simular_rolido(
    k44: float       = 5.5,    # radio de giro [m]
    kg_val: float    = 5.0,    # centro de gravedad vertical [m]
    caso_id: int     = 6,      # caso de espectro (1-12 de CASOS_ESTUDIO)
    V_knots: float   = 20.0,    # velocidad de avance [nudos]
    mu_deg: float    = 90.0,  # ángulo de encuentro [°]: 0=proa, 90=través, 180=popa
    dt: float        = 0.05,   # paso de tiempo [s]
    T_total: float   = 600.0,  # tiempo total de simulación [s]
    phi0_deg: float  = 2.0,    # condición inicial de escora [grados]
    phi_d0_deg_s: float = 0.0, # condición inicial de velocidad angular [grados/s]
    seed: int        = None,   # semilla aleatoria para fases (None = aleatorio)
    exportar: bool   = True,   # exportar resultados a Excel y PNG
    verbose: bool    = True,
    K_aleta: float   = 0.0,    # amortiguamiento de aleta activo/pasivo [N·m·s/rad]
):
    """
    Resuelve la ecuación de rolido acoplado usando RK4 en dominio temporal.

    Parámetros
    ----------
    k44          : Radio de giro efectivo en rolido [m], ya incluye la masa añadida
                   (radio ajustado). I_tot = (Delta/g) * k44^2 = m * k44^2.
    kg_val       : KG para seleccionar GZ y matrices de excitación.
    caso_id      : ID del estado de mar (1-12) definido en CASOS_ESTUDIO.
    V_knots      : Velocidad de avance del buque [nudos].
    mu_deg       : Ángulo de encuentro [°]. Convención: 0° = proa (seas a proa),
                   90° = través, 180° = popa (seas a popa). Afecta la corrección
                   Doppler en la frecuencia de encuentro.
    dt           : Paso de integración RK4 [s].
    T_total      : Duración total de la simulación [s].
    phi0_deg     : Ángulo inicial de rolido [grados].
    phi_d0_deg_s : Velocidad angular inicial [grados/s].
    seed         : Semilla para fases aleatorias del oleaje.
    exportar     : Si True, guarda Excel y PNGs en plots/.
    verbose      : Si True, imprime progreso.

    Retorna
    -------
    t_vec     : ndarray (N,)   — vector de tiempo [s]
    phi_vec   : ndarray (N,)   — ángulo de rolido [rad]
    phi_d_vec : ndarray (N,)   — velocidad angular [rad/s]
    df_out    : DataFrame      — tabla completa de resultados
    """
    t0_wall = _time.time()

    # ── 1. Momento de inercia total (k44 ya incluye masa añadida) ───────────
    # I_tot = (Delta/g) * k44^2  =  rho * nabla * k44^2
    m_ship = RHO * NABLA            # masa del buque [kg]  (= Delta/g)
    I_tot  = m_ship * k44**2        # inercia total ajustada [kg·m²]

    if verbose:
        print("=" * 60)
        caso = CASOS_ESTUDIO[caso_id]
        if caso_id == 0:
            print("SOLVER RK4 — ROLL DECAY (SIN EXCITACIÓN DE OLAS)")
            print("=" * 60)
            print(f"  Modo: {caso['desc']}")
        else:
            print("SOLVER RK4 — ROLIDO EN MAR IRREGULAR")
            print("=" * 60)
            print(f"  Caso espectro: {caso_id} ({caso['ss']}) Hs={caso['Hs']}m Tp={caso['Tp']}s")
        print(f"  KG={kg_val}m  k44_eff={k44}m  V={V_knots}kn  dt={dt}s  T={T_total}s")
        print(f"  m={m_ship:.0f}kg  I_tot=m·k44²={I_tot:.3e} kg·m²  [k44 incluye A44]")

    # ── 2. Espectro y amplitudes de ola ──────────────────────────────────────
    V_ms = V_knots * 0.5144   # velocidad en m/s

    if caso_id == 0:
        # ── Roll decay: sin excitación — arrays vacíos ──────────────────────
        omegas_w  = np.array([])
        zeta_a    = np.array([])
        S_vals    = np.array([])
        d_omegas  = np.array([])
        omegas_E  = np.array([])
        N_freq    = 0
        epsilon   = np.array([])
        omega_E_dam = None          # se calculará desde ω_n tras cargar GZ
        if verbose:
            print(f"  Mw(t) ≡ 0  (modo roll decay)")
            print(f"  V = {V_ms:.2f} m/s  |  μ = {mu_deg:.1f}°")
    else:
        df_esp  = discretizar_espectro_no_uniforme(caso_id=caso_id, exportar_excel=False, verbose=verbose)
        omegas_w = df_esp["omega_i"].values    # frecuencias de ola en agua quieta [rad/s]
        zeta_a   = df_esp["zeta_a"].values     # amplitudes [m]
        S_vals   = df_esp["S_omega"].values    # densidad espectral S(ω) [m²·s/rad]
        d_omegas = df_esp["d_omega"].values    # anchos de banda Δω [rad/s]
        N_freq   = len(omegas_w)

        # Frecuencia de encuentro por componente:
        #   ω_E(ω) = ω − (ω²/g) · V · cos(μ)
        mu_rad   = np.radians(mu_deg)
        omegas_E = omegas_w - (omegas_w**2 / G) * V_ms * np.cos(mu_rad)
        omegas_E = np.maximum(omegas_E, 1e-4)

        # Promedio energético de la frecuencia de encuentro:
        #   ω̄_E = Σ[ ω_E(ωᵢ) · S(ωᵢ)·Δωᵢ ] / Σ[ S(ωᵢ)·Δωᵢ ]
        energia_total = np.sum(S_vals * d_omegas)
        if energia_total > 1e-10:
            omega_E_dam = float(np.sum(omegas_E * S_vals * d_omegas) / energia_total)
        else:
            omega_E_dam = float(np.mean(omegas_E))

        # Fases aleatorias uniformes en [0, 2π]
        rng     = np.random.default_rng(seed)
        epsilon = rng.uniform(0, 2 * np.pi, N_freq)

        if verbose:
            print(f"  Frecuencias espectro: {N_freq}  rango [{omegas_w.min():.3f}, {omegas_w.max():.3f}] rad/s")
            print(f"  μ (ángulo encuentro) = {mu_deg:.1f}°   V = {V_ms:.2f} m/s")
            print(f"  ω_E rango: [{omegas_E.min():.3f}, {omegas_E.max():.3f}] rad/s")
            print(f"  ω̄_E (promedio energético) = {omega_E_dam:.4f} rad/s  "
                  f"→ T_E = {2*np.pi/omega_E_dam:.2f} s  [usado en B44]")

    # ── 3. Cargar GZ interpolado ──────────────────────────────────────────────
    gz_interp = _cargar_gz(kg_val)

    # En modo roll decay, omega_E_dam = frecuencia natural ω_n estimada desde
    # la pendiente de GZ en phi=0 (≈ g·GM / k44²)^0.5
    if caso_id == 0:
        d_phi    = 1e-4          # incremento infinitesimal [rad]
        gz_slope = (float(gz_interp(d_phi)) - float(gz_interp(-d_phi))) / (2.0 * d_phi)
        # gz_slope ≈ GM_L  (lever arm gradient at upright) [m/rad]
        omega_n  = float(np.sqrt(max(DELTA * gz_slope / I_tot, 1e-6)))
        omega_E_dam = omega_n
        if verbose:
            print(f"  GM (pendiente GZ@0) ≈ {gz_slope:.4f} m")
            print(f"  ω_n = {omega_n:.4f} rad/s  →  T_n = {2*np.pi/omega_n:.2f} s")
            print(f"  ω̄_E = ω_n = {omega_E_dam:.4f} rad/s  [usado en B44 del decay]")

    # ── 4. Cargar matrices Ci, Si ─────────────────────────────────────────────
    if caso_id == 0:
        # Roll decay: sin excitacion — matrices vacias
        _Ci   = None
        _Si   = None
        _phis = None
        pref  = np.array([])
    else:
        phis_exc, Ci_mat, Si_mat, omegas_exc = _cargar_matrices_excitacion(kg_val)

        # Verificar alineacion de frecuencias
        if len(omegas_exc) != N_freq or not np.allclose(omegas_exc, omegas_w, rtol=1e-3):
            Ci_aligned = np.zeros((N_freq, len(phis_exc)))
            Si_aligned = np.zeros((N_freq, len(phis_exc)))
            for j in range(len(phis_exc)):
                fi_c = interp1d(omegas_exc, Ci_mat[:, j], kind="linear",
                                bounds_error=False, fill_value=0.0)
                fi_s = interp1d(omegas_exc, Si_mat[:, j], kind="linear",
                                bounds_error=False, fill_value=0.0)
                Ci_aligned[:, j] = fi_c(omegas_w)
                Si_aligned[:, j] = fi_s(omegas_w)
            Ci_mat = Ci_aligned
            Si_mat = Si_aligned
            if verbose:
                print("  [Info] Matrices Ci/Si interpoladas al vector de frecuencias del espectro.")

        # Guardar referencias directas — la interpolacion en phi se hace
        # de forma vectorizada en momento_ola (sin listas de closures).
        _Ci   = Ci_mat     # (N_freq, N_phi)
        _Si   = Si_mat     # (N_freq, N_phi)
        _phis = phis_exc   # (N_phi,)  array ordenado en rad

        # Prefactor de excitacion: RHO * G (zeta_a ya incorpora delta_omega)
        pref = RHO * G * zeta_a    # (N_freq,)

    # ── 5. Parámetros del amortiguamiento ─────────────────────────────────────
    ship_params = {
        "L_PP":  L_PP,
        "B":     B_ship,
        "d":     D_ship,
        "C_B":   C_B,
        "nabla": NABLA,
        "C_M":   C_M,
        "OG":    D_ship - kg_val,   # convención Ikeda
        "b_BK":  B_BK,
        "l_BK":  L_BK,
        "rho":   RHO,
        "nu":    NU,
    }

    # === PRE-INTERPOLACIÓN DE AMORTIGUAMIENTO (Speed-Up) ===
    # Precalculamos el B44 para amplitudes de 0.1 a 60 grados.
    # Así evitamos llamar a las fórmulas complejas de Ikeda 48,000 veces por simulación.
    phi_a_grid = np.linspace(0.1, 60.0, 60)
    B44_grid = np.zeros_like(phi_a_grid)
    for i, pa in enumerate(phi_a_grid):
        B44_grid[i] = calcular_damping_B44(pa, V_knots, omega_E_dam, ship_params)["B44_total"]
    
    # Creamos un interpolador súper rápido
    interp_B44 = interp1d(phi_a_grid, B44_grid, kind='linear', bounds_error=False, 
                          fill_value=(B44_grid[0], B44_grid[-1]))
    # =======================================================

    # ── 6. Vectores de tiempo ─────────────────────────────────────────────────
    N_steps = int(T_total / dt) + 1
    t_vec     = np.linspace(0.0, T_total, N_steps)
    phi_vec   = np.zeros(N_steps)
    phi_d_vec = np.zeros(N_steps)
    Mw_vec    = np.zeros(N_steps)
    B44_vec   = np.zeros(N_steps)

    # Condiciones iniciales
    phi_vec[0]   = np.radians(phi0_deg)
    phi_d_vec[0] = np.radians(phi_d0_deg_s)

    # Amplitud efectiva para el amortiguamiento (actualizada cada paso)
    phi_a_eff = max(abs(phi0_deg), 5.0)  # [grados]; mínimo 5° para evitar B44→0

    # ── 7. Funciones auxiliares del RHS ───────────────────────────────────────

    def momento_ola(t_val: float, phi_val: float) -> float:
        """Mw(t, phi) vectorizado — interpolacion numpy pura, sin loop Python.
        En modo roll decay (N_freq==0) retorna 0."""
        if N_freq == 0:
            return 0.0
        theta = omegas_w * t_val + epsilon          # (N_freq,)

        # Interpolacion lineal vectorizada de Ci y Si en phi_val
        # _phis es (N_phi,) ordenado; _Ci/_Si son (N_freq, N_phi)
        phi_c = float(np.clip(phi_val, _phis[0], _phis[-1]))
        idx   = int(np.searchsorted(_phis, phi_c))  # indice derecho
        idx   = max(1, min(idx, len(_phis) - 1))
        w     = (phi_c - _phis[idx - 1]) / (_phis[idx] - _phis[idx - 1] + 1e-15)
        ci_phi = _Ci[:, idx - 1] * (1.0 - w) + _Ci[:, idx] * w   # (N_freq,)
        si_phi = _Si[:, idx - 1] * (1.0 - w) + _Si[:, idx] * w   # (N_freq,)

        return float(np.dot(pref, ci_phi * np.cos(theta) + si_phi * np.sin(theta)))

    def amortiguamiento(phi_val: float, phi_d_val: float, phi_a_deg_val: float) -> float:
        """B44 en [kg·m²/s] × phi_d_val → momento de amortiguamiento [N·m].
        Usa la tabla pre-interpolada en vez de calcular Ikeda repetitivamente."""
        B44_val = float(interp_B44(phi_a_deg_val))
        return B44_val * phi_d_val   # B44 * phi_dot  [N·m]

    def momento_restaurador(phi_val: float) -> float:
        """C(phi) = Delta * GZ(phi) [N·m]"""
        return DELTA * float(gz_interp(phi_val))

    def derivadas(t_val: float, phi_val: float, phi_d_val: float,
                  phi_a_deg_val: float):
        """Retorna (phi_d, phi_dd)"""
        Mw   = momento_ola(t_val, phi_val)
        M_aleta = -K_aleta * phi_d_val
        B_mo = amortiguamiento(phi_val, phi_d_val, phi_a_deg_val)
        C_mo = momento_restaurador(phi_val)
        phi_dd = (Mw + M_aleta - B_mo - C_mo) / I_tot
        return phi_d_val, phi_dd, Mw, B_mo

    # ── 8. Bucle RK4 ──────────────────────────────────────────────────────────
    if verbose:
        print(f"\n  Iniciando integración RK4 ({N_steps} pasos)...")

    # Ventana para calcular phi_a efectivo (últimos N_win pasos)
    N_win = max(1, int(20.0 / dt))   # ~20 s de historial

    for n in range(N_steps - 1):
        t_n   = t_vec[n]
        ph_n  = phi_vec[n]
        phd_n = phi_d_vec[n]

        # Amplitud efectiva del último ciclo (en grados)
        win_start = max(0, n - N_win)
        phi_a_eff = max(
            np.degrees(np.max(np.abs(phi_vec[win_start:n+1]))),
            3.0  # mínimo 3° para evitar B44 degenerado
        )

        # Coeficientes RK4 ────────────────────────────────────────────────────
        k1_ph, k1_phd, Mw_n, _ = derivadas(t_n,            ph_n,            phd_n,            phi_a_eff)
        k2_ph, k2_phd, _,    _ = derivadas(t_n + dt/2,     ph_n + dt/2*k1_ph,  phd_n + dt/2*k1_phd, phi_a_eff)
        k3_ph, k3_phd, _,    _ = derivadas(t_n + dt/2,     ph_n + dt/2*k2_ph,  phd_n + dt/2*k2_phd, phi_a_eff)
        k4_ph, k4_phd, _,    _ = derivadas(t_n + dt,       ph_n + dt*k3_ph,    phd_n + dt*k3_phd,    phi_a_eff)

        phi_vec[n+1]   = ph_n  + dt/6*(k1_ph  + 2*k2_ph  + 2*k3_ph  + k4_ph)
        phi_d_vec[n+1] = phd_n + dt/6*(k1_phd + 2*k2_phd + 2*k3_phd + k4_phd)
        Mw_vec[n]      = Mw_n

        # Limitar ángulo a ±90° (fuera de validez del modelo)
        phi_vec[n+1] = np.clip(phi_vec[n+1], -np.pi/2, np.pi/2)

        # Progreso
        if verbose and (n % (N_steps // 10) == 0):
            print(f"    t={t_n:7.1f}s  phi={np.degrees(ph_n):+6.2f}°  "
                  f"phi_a_eff={phi_a_eff:.1f}°  Mw={Mw_n:.2e} N·m")

    # ── 9. Post-proceso ───────────────────────────────────────────────────────
    phi_deg_vec = np.degrees(phi_vec)
    phi_a_final = np.max(np.abs(phi_deg_vec))

    elapsed = _time.time() - t0_wall
    if verbose:
        print(f"\n  Simulación completada en {elapsed:.1f} s")
        print(f"  Amplitud máxima de rolido: {phi_a_final:.2f}°")

    # ── 10. Exportar ──────────────────────────────────────────────────────────
    df_out = pd.DataFrame({
        "t_s":         t_vec,
        "phi_deg":     phi_deg_vec,
        "phi_rad":     phi_vec,
        "phi_d_rad_s": phi_d_vec,
        "Mw_Nm":       Mw_vec,
    })

    tag = f"C{caso_id}_KG{kg_val}_V{V_knots}_Kaleta{K_aleta:.0f}_dt{dt}"

    if exportar:
        os.makedirs(os.path.join(_SCRIPT_DIR, "plots"), exist_ok=True)

        # Excel
        xlsx_path = os.path.join(_SCRIPT_DIR, f"rolido_RK4_{tag}.xlsx")
        try:
            df_out.to_excel(xlsx_path, index=False)
            print(f"  Excel → {xlsx_path}")
        except PermissionError:
            xlsx_path2 = xlsx_path.replace(".xlsx", "_v2.xlsx")
            df_out.to_excel(xlsx_path2, index=False)
            print(f"  Excel → {xlsx_path2}")

        # ── Gráfico 1: phi(t) ─────────────────────────────────────────────
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"Simulación RK4 — Rolido en Mar Irregular\n"
                     f"Caso {caso_id} | KG={kg_val}m | k44={k44}m | V={V_knots}kn | Kaleta={K_aleta:.1e}",
                     fontsize=13, fontweight="bold")

        caso = CASOS_ESTUDIO[caso_id]
        axes[0].plot(t_vec, phi_deg_vec, color="#1f77b4", lw=0.8)
        axes[0].axhline(0, color="k", lw=0.5)
        axes[0].set_ylabel("$\\phi$ [°]", fontsize=11)
        axes[0].set_title(f"Ángulo de rolido — Hs={caso['Hs']}m, Tp={caso['Tp']}s", fontsize=10)
        axes[0].grid(True, alpha=0.35)

        axes[1].plot(t_vec, np.degrees(phi_d_vec), color="#d62728", lw=0.8)
        axes[1].set_ylabel("$\\dot{\\phi}$ [°/s]", fontsize=11)
        axes[1].set_title("Velocidad angular de rolido", fontsize=10)
        axes[1].grid(True, alpha=0.35)

        axes[2].plot(t_vec, Mw_vec / 1e6, color="#2ca02c", lw=0.8)
        axes[2].set_ylabel("$M_w$ [MN·m]", fontsize=11)
        axes[2].set_xlabel("Tiempo [s]", fontsize=11)
        axes[2].set_title("Momento de excitación de ola", fontsize=10)
        axes[2].grid(True, alpha=0.35)

        fig.tight_layout()
        png_t = os.path.join(_SCRIPT_DIR, "plots", f"rolido_RK4_{tag}_series.png")
        fig.savefig(png_t, dpi=200)
        plt.close(fig)
        print(f"  PNG  → {png_t}")

        # ── Gráfico 2: plano de fase phi vs phi_d ─────────────────────────
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        sc = ax2.scatter(phi_deg_vec, np.degrees(phi_d_vec),
                         c=t_vec, cmap="viridis", s=1.5, alpha=0.7)
        plt.colorbar(sc, ax=ax2, label="t [s]")
        ax2.set_xlabel("$\\phi$ [°]", fontsize=12)
        ax2.set_ylabel("$\\dot{\\phi}$ [°/s]", fontsize=12)
        ax2.set_title(f"Plano de fase — Caso {caso_id} | KG={kg_val}m | Kaleta={K_aleta:.1e}", fontsize=12)
        ax2.axhline(0, color="k", lw=0.5)
        ax2.axvline(0, color="k", lw=0.5)
        ax2.grid(True, alpha=0.35)
        fig2.tight_layout()
        png_f = os.path.join(_SCRIPT_DIR, "plots", f"rolido_RK4_{tag}_fase.png")
        fig2.savefig(png_f, dpi=200)
        plt.close(fig2)
        print(f"  PNG  → {png_f}")

    return t_vec, phi_vec, phi_d_vec, df_out


# ══════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN DIRECTA
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Configuración para Roll Decay
    # Configuración de aletas
    # K_aleta = 0 deshabilita el estabilizador
    
    t, phi, phi_d, df = simular_rolido(
        k44      = 5.5,      # radio de giro [m]
        kg_val   = 5.0,      # centro de gravedad vertical [m]
        caso_id  = 10,        # ID de espectro (0 = Roll Decay sin olas)
        V_knots  = 10.0,     # velocidad de avance [nudos]
        mu_deg   = 90.0,     # ángulo de encuentro [°]
        dt       = 1,     # paso de integración [s]
        T_total  = 3600.0,    # tiempo total [s] (para roll decay basta con 120.0)
        phi0_deg = 0.0,      # escora inicial [°] (usar ej. 15.0 si es roll decay)
        seed     = 42,       # semilla para olas
        exportar = True,
        verbose  = True,
        K_aleta  = 0,      # amortiguamiento de aleta [N·m·s/rad]
    )
    
    print(f"\n  Escora máxima alcanzada: {np.degrees(np.max(np.abs(phi))):.2f}°")
