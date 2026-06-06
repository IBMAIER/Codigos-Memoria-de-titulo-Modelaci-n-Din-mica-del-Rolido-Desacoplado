import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random

from funciones_dinamicas import discretizar_espectro_no_uniforme, CASOS_ESTUDIO

def order_points_by_angle(y, z):
    """
    Ordena los puntos de una sección transversal barriendo el ángulo
    desde babor hacia estribor alrededor de un centro elevado.
    """
    zc = np.max(z) + 10.0 # Punto central por encima de la cubierta
    yc = 0.0
    angles = np.arctan2(z - zc, y - yc)
    idx = np.argsort(angles)
    return y[idx], z[idx]

def generar_verificaciones(x_vals, secciones_data, phis_deg, line_c, kg_val, df_gz, n_cases=4):
    print(f"\n--- Generando {n_cases} casos de verificación aleatorios ---")
    random.seed(42) # Para reproducibilidad
    os.makedirs("plots", exist_ok=True)
    
    dfs_verificacion = []
    
    # Asegurar al menos un ángulo negativo, uno positivo y uno exactamente cero
    idx_neg = [j for j, p in enumerate(phis_deg) if p <= -5.0]
    idx_pos = [j for j, p in enumerate(phis_deg) if p >= 5.0]
    idx_zero = [j for j, p in enumerate(phis_deg) if np.isclose(p, 0.0, atol=1e-3)]
    
    j_phis_elegidos = []
    if idx_neg: j_phis_elegidos.append(random.choice(idx_neg))
    if idx_pos: j_phis_elegidos.append(random.choice(idx_pos))
    if idx_zero: j_phis_elegidos.append(idx_zero[0])
    
    while len(j_phis_elegidos) < n_cases:
        j_phis_elegidos.append(random.randint(0, len(phis_deg) - 1))
        
    random.shuffle(j_phis_elegidos)
    
    for i, j_phi in enumerate(j_phis_elegidos):
        idx_x = random.randint(0, len(x_vals) - 1)
        x_val = x_vals[idx_x]
        sec_dict = secciones_data[idx_x]
        phi_deg = phis_deg[j_phi]
        phi_rad = np.radians(phi_deg)
        C_val = line_c[j_phi]
        
        # Extraer Line_A_y y Line_B_z desde df_gz
        idx_gz = np.where(np.isclose(df_gz['phi_deg'].values, abs(phi_deg), atol=1e-3))[0][0]
        line_A = df_gz['Line_A_y'].values[idx_gz]
        line_B = df_gz['Line_B_z'].values[idx_gz]
        if phi_deg < 0:
            line_A = -line_A # A is sin(phi), so it flips sign for negative phi
            
        eta_G = -(C_val + kg_val * np.cos(phi_rad))
        
        y1 = sec_dict['y1']
        y2 = sec_dict['y2']
        z1 = sec_dict['z1']
        z2 = sec_dict['z2']
        y_mid = sec_dict['y_mid']
        z_mid = sec_dict['z_mid']
        dS = sec_dict['dS']
        B_palanca = sec_dict['B_palanca']
        ny = sec_dict['ny']
        nz = sec_dict['nz']
        
        yf = y_mid * np.cos(phi_rad) - z_mid * np.sin(phi_rad)
        zf = y_mid * np.sin(phi_rad) + z_mid * np.cos(phi_rad)
        
        H_filter = np.where(zf <= eta_G, 1, 0)
        
        df_case = pd.DataFrame({
            'Caso': i+1,
            'X_section_m': x_val,
            'Phi_deg': phi_deg,
            'Line_A_y_K': line_A,
            'Line_B_z_K': line_B,
            'Line_C_K': C_val,
            'y1_Body': y1,
            'z1_Body': z1,
            'y2_Body': y2,
            'z2_Body': z2,
            'y_mid_Body': y_mid,
            'z_mid_Body': z_mid,
            'ny': ny,
            'nz': nz,
            'dS': dS,
            'B_palanca': B_palanca,
            'Yf_Fluido': yf,
            'Zf_Fluido': zf,
            'eta_G': eta_G,
            'Sumergido_H': H_filter
        })
        dfs_verificacion.append(df_case)
        
        # Gráfica en el sistema Body
        plt.figure(figsize=(10, 8))
        plt.plot(0, 0, 'ro', markersize=8, label='Centro de Gravedad (G)')
        
        submerged = H_filter == 1
        plt.scatter(y_mid[submerged], z_mid[submerged], c='blue', s=12, label='Sumergido (Paneles)')
        plt.scatter(y_mid[~submerged], z_mid[~submerged], c='gray', s=12, label='Emergido (Paneles)')
        
        # Línea de agua
        y_wl_f = np.linspace(yf.min() - 2, yf.max() + 2, 100)
        z_wl_f = np.full_like(y_wl_f, eta_G)
        
        # Transformar a Body para plotear
        y_wl_b = y_wl_f * np.cos(phi_rad) + z_wl_f * np.sin(phi_rad)
        z_wl_b = -y_wl_f * np.sin(phi_rad) + z_wl_f * np.cos(phi_rad)
        
        plt.plot(y_wl_b, z_wl_b, 'c--', linewidth=2, label='Línea de Agua')

        # DIBUJAR VECTORES NORMALES (Distancias Consideradas / Palancas)
        # Mostrar las normales solo de algunos puntos para no saturar el gráfico
        step_pts = max(1, len(y_mid[submerged]) // 15)
        for idx in range(0, len(y_mid[submerged]), step_pts):
            y_pt = y_mid[submerged][idx]
            z_pt = z_mid[submerged][idx]
            ny_pt = ny[submerged][idx]
            nz_pt = nz[submerged][idx]
            # Escalar el vector normal visualmente (por ejemplo, longitud 1m)
            plt.arrow(y_pt, z_pt, ny_pt*1.0, nz_pt*1.0, color='orange', head_width=0.2, alpha=0.8)

        # Solo agregar a la leyenda el vector normal si se dibujaron
        if len(y_mid[submerged]) > 0:
            plt.plot([], [], color='orange', label='Normales (Integrales)')
        plt.title(f'Caso {i+1}: Sec X={x_val:.2f}m, Escora $\phi$={phi_deg}$^\circ$\n(Sistema Buque)')
        plt.xlabel('Yb [m]')
        plt.ylabel('Zb [m]')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"plots/verificacion_excitacion_caso_{i+1}_X{x_val:.1f}_phi{phi_deg}.png")
        plt.close()
        
    df_all = pd.concat(dfs_verificacion, ignore_index=True)
    try:
        df_all.to_excel("verificacion_geometria_paneles.xlsx", index=False)
        print("   -> 3 Casos de verificación generados en 'plots/' y Excel en 'verificacion_geometria_paneles.xlsx'.")
    except PermissionError:
        df_all.to_excel("verificacion_geometria_paneles_v2.xlsx", index=False)
        print("   -> (Aviso: El excel original estaba abierto). Datos de verificación guardados en 'verificacion_geometria_paneles_v2.xlsx'.")

def generar_informe_malla(ds_promedio, paneles_promedio, step, phis_deg, omegas, x_vals, filename="informe_calidad_malla.txt"):
    # Cada panel es un segmento de línea definido por 2 puntos consecutivos
    pts_por_panel = 2
    
    informe = (
        "--- INFORME DE CALIDAD DE MALLA ---\n"
        f"ds promedio (paneles sumergidos): {ds_promedio:.6f} m\n"
        f"Número de puntos promedio por panel: {pts_por_panel}\n"
        f"Número de paneles sumergidos promedio por sección: {paneles_promedio:.2f}\n"
        f"Diferencia de ángulo para cálculo: {step:.2f} grados\n"
        f"Rango de ángulo: [{phis_deg.min():.2f}, {phis_deg.max():.2f}] grados\n"
        f"Cantidad de frecuencias analizadas: {len(omegas)}\n"
        f"Rango de frecuencias: [{omegas.min():.4f}, {omegas.max():.4f}] rad/s\n"
        f"Secciones longitudinales analizadas: {len(x_vals)}\n"
        "-----------------------------------\n"
    )
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(informe)
    print(f"   -> Informe de calidad de malla generado en '{filename}'.")

def calcular_matrices(kg_val=5.0, exportar_verificacion=True):
    print("="*60)
    print("CÁLCULO DE MATRICES DE EXCITACIÓN Ci(phi) y Si(phi)")
    print("="*60)
    
    print("\n1. Obteniendo frecuencias de discretización...")
    # Utilizamos la función del script funciones_dinamicas.py
    df_freq = discretizar_espectro_no_uniforme(caso_id=6, exportar_excel=False)
    omegas = df_freq['omega_i'].values
    print(f"[Trazabilidad] Frecuencias (omega_i):")
    print(f" - Cantidad total de frecuencias: {len(omegas)}")
    print(f" - Rango considerado: [{omegas.min():.4f}, {omegas.max():.4f}] rad/s")
    
    print("\n2. Cargando geometría del casco (secciones_contorno_exterior.csv)...")
    df_hull = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "Resultados", "secciones_contorno_exterior.csv"), sep=";", decimal=",")
    # Convertir de mm a metros
    df_hull['X'] = df_hull['X'] / 1000.0
    df_hull['Y'] = df_hull['Y'] / 1000.0
    df_hull['Z'] = df_hull['Z'] / 1000.0
    
    x_vals = np.sort(df_hull['X'].unique())
    print(f" - Encontradas {len(x_vals)} secciones transversales (eslora de {x_vals.min():.2f}m a {x_vals.max():.2f}m).")
    
    print(f"\n3. Cargando curva GZ y extendiendo para rolido simétrico (-30 a +30 grados)...")
    df_gz = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "Resultados", "resultados_curva_GZ.csv"), sep=";", decimal=",")
    
    # El intervalo debe ser el mismo que resultados_curva_GZ.csv
    step = df_gz['phi_deg'].iloc[1] - df_gz['phi_deg'].iloc[0]
    phis_deg = np.arange(-30.0, 30.0 + step/2, step)
    
    line_c = np.zeros_like(phis_deg)
    for idx, p in enumerate(phis_deg):
        # Asumiendo simetría transversal para la línea de agua referenciada a K
        idx_gz = np.where(np.isclose(df_gz['phi_deg'].values, abs(p), atol=1e-3))[0][0]
        line_c[idx] = df_gz['Line_C'].values[idx_gz]
    
    # Matrices finales de almacenamiento
    Ci = np.zeros((len(omegas), len(phis_deg)))
    Si = np.zeros((len(omegas), len(phis_deg)))
    
    g = 9.81
    K_vals = (omegas**2) / g
    
    print(f"\n4. Iniciando doble integración (Secciones transversales y longitudinal)...")
    print(f"   Parámetro de CG: KG = {kg_val} m")
    
    # Determinar la línea base K desde el CAD (el punto más bajo)
    z_k_cad = df_hull['Z'].min()
    print(f"   -> Línea base del CAD (punto K) detectada en Z = {z_k_cad:.3f} m")
    
    # Pre-procesar geometría para evitar filtrar el DataFrame en cada iteración
    print("   Pre-procesando perfiles de las secciones...")
    secciones_data = []
    for x_val in x_vals:
        sec_data = df_hull[df_hull['X'] == x_val]
        y_sec = sec_data['Y'].values
        z_sec = sec_data['Z'].values
        # Ordenar puntos para garantizar un contorno continuo de babor a estribor
        y_sec, z_sec = order_points_by_angle(y_sec, z_sec)
        
        # Discretización del contorno en el sistema b (Origen en G)
        # Z respecto a K es (z_sec - z_k_cad). Luego restamos KG.
        y_b = y_sec
        z_b = z_sec - z_k_cad - kg_val
        
        y_mid = 0.5 * (y_b[:-1] + y_b[1:])
        z_mid = 0.5 * (z_b[:-1] + z_b[1:])
        dy = y_b[1:] - y_b[:-1]
        dz = z_b[1:] - z_b[:-1]
        dS = np.sqrt(dy**2 + dz**2)
        
        valid = dS > 1e-6
        y1, y2 = y_b[:-1][valid], y_b[1:][valid]
        z1, z2 = z_b[:-1][valid], z_b[1:][valid]
        y_mid, z_mid, dy, dz, dS = y_mid[valid], z_mid[valid], dy[valid], dz[valid], dS[valid]
        ny = dz / dS
        nz = -dy / dS
        B_palanca = y_mid * nz - z_mid * ny
        
        secciones_data.append({
            'y1': y1,
            'y2': y2,
            'z1': z1,
            'z2': z2,
            'y_mid': y_mid,
            'z_mid': z_mid,
            'dS': dS,
            'B_palanca': B_palanca,
            'ny': ny,
            'nz': nz
        })

    # Llamar a la función de verificación
    if exportar_verificacion:
        generar_verificaciones(x_vals, secciones_data, phis_deg, line_c, kg_val, df_gz, n_cases=4)

    total_ds_sumergido = 0.0
    total_paneles_sumergidos = 0

    for j_phi, phi_deg in enumerate(phis_deg):
        phi_rad = np.radians(phi_deg)
        C_val = line_c[j_phi]
        
        # Nivel de la superficie libre referenciado al CG (sistema rotado fluido F)
        eta_G = -(C_val + kg_val * np.cos(phi_rad))
        
        # Almacenamiento intermedio para integración de Simpson/Trapecio longitudinal
        C_long = np.zeros((len(omegas), len(x_vals)))
        S_long = np.zeros((len(omegas), len(x_vals)))
        
        for idx_x, sec_dict in enumerate(secciones_data):
            y_mid = sec_dict['y_mid']
            z_mid = sec_dict['z_mid']
            dS = sec_dict['dS']
            B_palanca = sec_dict['B_palanca']
            
            # Transformación de coordenadas del sistema Body (b) al sistema Fluido Fijo (f)
            yf = y_mid * np.cos(phi_rad) - z_mid * np.sin(phi_rad)
            zf = y_mid * np.sin(phi_rad) + z_mid * np.cos(phi_rad)
            
            # Filtro del dominio mojado (Función de Heaviside)
            H_filter = np.where(zf <= eta_G, 1.0, 0.0)
            
            total_ds_sumergido += np.sum(dS * H_filter)
            total_paneles_sumergidos += np.sum(H_filter)
            
            # Profundidad relativa
            z_rel = zf - eta_G
            
            # z_rel shape: (N_p,) -> K_vals shape: (N_w,)
            # z_rel_matrix shape: (N_w, N_p)
            z_rel_matrix = K_vals[:, np.newaxis] * z_rel[np.newaxis, :]
            presion_amp = np.exp(z_rel_matrix)
            
            # yf_matrix shape: (N_w, N_p)
            yf_matrix = K_vals[:, np.newaxis] * yf[np.newaxis, :]
            
            # termino comun shape: (N_p,)
            term_geom = H_filter * B_palanca * dS
            
            # C_long y S_long actualización
            C_long[:, idx_x] = np.sum(presion_amp * np.cos(yf_matrix) * term_geom, axis=1)
            S_long[:, idx_x] = np.sum(presion_amp * np.sin(yf_matrix) * term_geom, axis=1)
                
        # Integración longitudinal usando regla del trapecio
        for i_w in range(len(omegas)):
            Ci[i_w, j_phi] = np.trapezoid(C_long[i_w, :], x_vals)
            Si[i_w, j_phi] = np.trapezoid(S_long[i_w, :], x_vals)
            
        if (j_phi % 10 == 0) or (j_phi == len(phis_deg)-1):
            print(f"   -> Progreso: Phi = {phi_deg:5.1f} grados completado.")
            
    ds_promedio_calc = total_ds_sumergido / total_paneles_sumergidos if total_paneles_sumergidos > 0 else 0.0
    paneles_promedio_calc = total_paneles_sumergidos / (len(phis_deg) * len(x_vals))
            
    print("\n5. Generando informe de malla y exportando resultados...")
    filename_txt = f"informe_calidad_malla_KG{kg_val}.txt"
    generar_informe_malla(ds_promedio_calc, paneles_promedio_calc, step, phis_deg, omegas, x_vals, filename=filename_txt)
    
    # Formateo de DataFrames
    cols = [f"{phi:.1f}" for phi in phis_deg]
    df_Ci = pd.DataFrame(Ci, index=omegas, columns=cols)
    df_Si = pd.DataFrame(Si, index=omegas, columns=cols)
    
    filename_excel = os.path.join(os.path.dirname(__file__), "..", "Resultados", f"matrices_excitacion_KG{kg_val}.xlsx")
    filename_excel_v2 = f"matrices_excitacion_KG{kg_val}_v2.xlsx"
    
    # Exportar a hojas separadas
    try:
        with pd.ExcelWriter(filename_excel) as writer:
            df_Ci.to_excel(writer, sheet_name='Ci(phi)')
            df_Si.to_excel(writer, sheet_name='Si(phi)')
        print(f"[Exito] Matrices de excitación guardadas en: {filename_excel}")
    except PermissionError:
        with pd.ExcelWriter(filename_excel_v2) as writer:
            df_Ci.to_excel(writer, sheet_name='Ci(phi)')
            df_Si.to_excel(writer, sheet_name='Si(phi)')
        print(f"[Exito] (Aviso: El excel original estaba abierto). Matrices guardadas en: {filename_excel_v2}")

if __name__ == "__main__":
    calcular_matrices(kg_val=5.0, exportar_verificacion=True)
