import os

filepath = r'c:\Users\ibm\Documents\Seminario\waves\SCRIPS\solver_rolido_RK4.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Add numba import
content = content.replace('import time as _time', 'import time as _time\nfrom numba import njit')

njit_block = """
# --------------------------------------------------
# Numba JIT compiled routines
# --------------------------------------------------
@njit
def momento_ola_numba(t_val, N_freq, omegas_w, epsilon, _phis, _Ci, _Si, pref, phi_val):
    if N_freq == 0:
        return 0.0
    theta = omegas_w * t_val + epsilon
    
    phi_c = phi_val
    if phi_c < _phis[0]: phi_c = _phis[0]
    if phi_c > _phis[-1]: phi_c = _phis[-1]
    
    idx = np.searchsorted(_phis, phi_c)
    if idx < 1: idx = 1
    if idx > len(_phis) - 1: idx = len(_phis) - 1
    
    w = (phi_c - _phis[idx - 1]) / (_phis[idx] - _phis[idx - 1] + 1e-15)
    ci_phi = np.empty(N_freq)
    si_phi = np.empty(N_freq)
    for i in range(N_freq):
        ci_phi[i] = _Ci[i, idx - 1] * (1.0 - w) + _Ci[i, idx] * w
        si_phi[i] = _Si[i, idx - 1] * (1.0 - w) + _Si[i, idx] * w
        
    ans = 0.0
    for i in range(N_freq):
        ans += pref[i] * (ci_phi[i] * np.cos(theta[i]) + si_phi[i] * np.sin(theta[i]))
    return ans

@njit
def derivadas_numba(t_val, phi_val, phi_d_val, phi_a_deg_val, N_freq, omegas_w, epsilon, _phis, _Ci, _Si, pref, phi_a_grid, B44_grid, DELTA, phi_all, gz_all, K_aleta, I_tot):
    Mw = momento_ola_numba(t_val, N_freq, omegas_w, epsilon, _phis, _Ci, _Si, pref, phi_val)
    M_aleta = -K_aleta * phi_d_val
    B44_val = np.interp(phi_a_deg_val, phi_a_grid, B44_grid)
    B_mo = B44_val * phi_d_val
    gz_val = np.interp(phi_val, phi_all, gz_all)
    C_mo = DELTA * gz_val
    phi_dd = (Mw + M_aleta - B_mo - C_mo) / I_tot
    return phi_d_val, phi_dd, Mw, B_mo

@njit
def rk4_loop_numba(N_steps, t_vec, phi_vec, phi_d_vec, Mw_vec, dt, N_win, 
                   N_freq, omegas_w, epsilon, _phis, _Ci, _Si, pref, 
                   phi_a_grid, B44_grid, DELTA, phi_all, gz_all, K_aleta, I_tot):
    for n in range(N_steps - 1):
        t_n   = t_vec[n]
        ph_n  = phi_vec[n]
        phd_n = phi_d_vec[n]

        win_start = max(0, n - N_win)
        max_val = 0.0
        for i in range(win_start, n+1):
            v = abs(phi_vec[i])
            if v > max_val: max_val = v
        phi_a_eff = max(max_val * 180.0 / 3.141592653589793, 3.0)

        k1_ph, k1_phd, Mw_n, _ = derivadas_numba(t_n, ph_n, phd_n, phi_a_eff, N_freq, omegas_w, epsilon, _phis, _Ci, _Si, pref, phi_a_grid, B44_grid, DELTA, phi_all, gz_all, K_aleta, I_tot)
        k2_ph, k2_phd, _, _    = derivadas_numba(t_n + dt/2.0, ph_n + dt/2.0*k1_ph, phd_n + dt/2.0*k1_phd, phi_a_eff, N_freq, omegas_w, epsilon, _phis, _Ci, _Si, pref, phi_a_grid, B44_grid, DELTA, phi_all, gz_all, K_aleta, I_tot)
        k3_ph, k3_phd, _, _    = derivadas_numba(t_n + dt/2.0, ph_n + dt/2.0*k2_ph, phd_n + dt/2.0*k2_phd, phi_a_eff, N_freq, omegas_w, epsilon, _phis, _Ci, _Si, pref, phi_a_grid, B44_grid, DELTA, phi_all, gz_all, K_aleta, I_tot)
        k4_ph, k4_phd, _, _    = derivadas_numba(t_n + dt, ph_n + dt*k3_ph, phd_n + dt*k3_phd, phi_a_eff, N_freq, omegas_w, epsilon, _phis, _Ci, _Si, pref, phi_a_grid, B44_grid, DELTA, phi_all, gz_all, K_aleta, I_tot)

        phi_vec[n+1]   = ph_n  + (dt/6.0)*(k1_ph  + 2.0*k2_ph  + 2.0*k3_ph  + k4_ph)
        phi_d_vec[n+1] = phd_n + (dt/6.0)*(k1_phd + 2.0*k2_phd + 2.0*k3_phd + k4_phd)
        Mw_vec[n]      = Mw_n

        if phi_vec[n+1] < -1.5707963267948966: phi_vec[n+1] = -1.5707963267948966
        if phi_vec[n+1] > 1.5707963267948966: phi_vec[n+1] = 1.5707963267948966
        
    return phi_vec, phi_d_vec, Mw_vec

def simular_rolido(
"""
content = content.replace('def simular_rolido(', njit_block)

content = content.replace('_Ci   = None', '_Ci   = np.zeros((1, 1))')
content = content.replace('_Si   = None', '_Si   = np.zeros((1, 1))')
content = content.replace('_phis = None', '_phis = np.zeros(1)')

loop_old = '''    # ── 7. Funciones auxiliares del RHS ───────────────────────────────────────

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
        print(f"\\n  Iniciando integración RK4 ({N_steps} pasos)...")

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
                  f"phi_a_eff={phi_a_eff:.1f}°  Mw={Mw_n:.2e} N·m")'''

loop_new = '''    # ── 7. Extraer arrays para Numba ─────────────────────────────────────────
    phi_all = gz_interp.x
    gz_all = gz_interp.y

    # ── 8. RK4 Loop (Numba) ──────────────────────────────────────────────
    if verbose:
        print(f"\\n  Starting RK4 integration ({N_steps} pasos)...")

    N_win = max(1, int(20.0 / dt))
    
    phi_vec, phi_d_vec, Mw_vec = rk4_loop_numba(
        N_steps, t_vec, phi_vec, phi_d_vec, Mw_vec, dt, N_win, 
        N_freq, omegas_w, epsilon, _phis, _Ci, _Si, pref, 
        phi_a_grid, B44_grid, DELTA, phi_all, gz_all, K_aleta, I_tot
    )'''

content = content.replace(loop_old, loop_new)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print('Listo, solver_rolido_RK4 modificado con Numba.')
