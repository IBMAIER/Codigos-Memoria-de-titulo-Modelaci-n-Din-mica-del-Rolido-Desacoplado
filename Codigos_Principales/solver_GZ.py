"""
SOLVER ITERATIVO DE CURVA DE ESTABILIDAD ESTÁTICA GZ (solver_GZ.py)

Construye la curva de brazos adrizantes (Curva GZ) de la carena.
El algoritmo toma las secciones del buque y, paramétricamente:
1. Para cada ángulo de escora, busca iterativamente el calado de equilibrio que iguale
   el volumen sumergido al volumen de diseño del buque (Método raiz Brent).
2. Con la flotación convergida, calcula los brazos estabilizadores (KN, SZ, GZ) y S.
3. Evalúa la estabilidad inicial o GZ bajo distintos centros de gravedad KG.
4. Exporta gráficos visuales con matplotlib de secciones y trayectoria de carena.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import simpson
import sys
import os

from funciones_naval import cargar_y_ordenar_secciones, calcular_parametros_hidrostaticos, recortar_poligono

def graficar_seccion_sumergida(poligono, plano_z, phi_rad, phi_deg, x_val, res_punto=None, kg_valores=None, ks=None, output_dir="plots_secciones_GZ"):
    """
    Genera un gráfico de una sección específica, mostrando el contorno original
    y el área sumergida según el plano de flotación inclinado.
    """
    if len(poligono) < 3:
        return
        
    poligono_sumergido = recortar_poligono(poligono, plano_z, phi_rad)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Hacer figura más grande (ancho, alto) para que quepa la leyenda lateral
    plt.figure(figsize=(10, 8))
    
    # Original
    plt.plot(poligono[:, 0], poligono[:, 1], 'k-', label='Casco Original', linewidth=1.5)
        
    # Sumergido
    if len(poligono_sumergido) > 2:
        plt.fill(poligono_sumergido[:, 0], poligono_sumergido[:, 1], 'blue', alpha=0.3, label='Área Sumergida')
        plt.plot(poligono_sumergido[:, 0], poligono_sumergido[:, 1], 'b-', linewidth=1.5)
        
    # Línea de flotación (Waterline)
    y_min, y_max = np.min(poligono[:, 0]) - 2, np.max(poligono[:, 0]) + 2
    if abs(np.cos(phi_rad)) > 1e-6:
        y_line = np.array([y_min, y_max])
        z_line = (plano_z - y_line * np.sin(phi_rad)) / np.cos(phi_rad)
        plt.plot(y_line, z_line, 'c--', label='Plano de flotación', linewidth=2)
    else:
        # Caso de 90 grados exactos (poco común, pero por si acaso)
        plt.axvline(x=plano_z, color='c', linestyle='--', label='Plano de flotación', linewidth=2)
        
    # Limites del plot para mantener proporciones fijas en todas las imágenes
    z_min, z_max = np.min(poligono[:, 1]) - 1, np.max(poligono[:, 1]) + 2
    y_min, y_max = np.min(poligono[:, 0]) - 2, np.max(poligono[:, 0]) + 2
    plt.xlim(y_min, y_max)
    plt.ylim(min(-1, z_min), z_max + 2) # extend explicitly a bit for the S point
        
    # Línea de Crujía
    plt.axvline(x=0, color='gray', linestyle='-.', alpha=0.5, label='Línea de Crujía')
    
    # Marcar puntos e información calculada
    if res_punto is not None:
        YB = res_punto["YB"]
        ZB = res_punto["ZB"]
        SZ = res_punto["SZ"]
        
        # Plot K
        plt.plot(0, 0, 'ks', markersize=8, label='K (Quilla)')
        plt.text(0.2, -0.4, 'K', fontsize=12, weight='bold')

        # Plot B
        plt.plot(YB, ZB, 'bo', markersize=8, label='B (Centro Carena)')
        plt.text(YB+0.2, ZB+0.2, 'B', fontsize=12, weight='bold', color='blue')

        # Plot S
        if ks is not None:
            plt.plot(0, ks, 'g^', markersize=8, label=f'S (KS={ks}m)')
            plt.text(0.2, ks+0.2, 'S', fontsize=12, weight='bold', color='green')

        # Plot G points
        if kg_valores is not None and len(kg_valores) > 0:
            plt.plot(0, kg_valores[0], 'ro', markersize=5, label='Centros de masa')
            for kg in kg_valores[1:]:
                plt.plot(0, kg, 'ro', markersize=5)
                
        # Línea de acción del empuje (Vertical por B)
        # Vertical inclinada según phi
        t_vals = np.linspace(-10, 20, 2)
        y_vert = YB + t_vals * np.sin(phi_rad)
        z_vert = ZB + t_vals * np.cos(phi_rad)
        plt.plot(y_vert, z_vert, 'b:', alpha=0.6, label='Línea Empuje')
        
        # Brazo SZ (Proyección ortogonal desde S hacia la vertical por B)
        if ks is not None:
            dot_v_SB = YB * np.sin(phi_rad) + (ZB - ks) * np.cos(phi_rad)
            Z_proj_y = YB - dot_v_SB * np.sin(phi_rad)
            Z_proj_z = ZB - dot_v_SB * np.cos(phi_rad)
            
            plt.plot([0, Z_proj_y], [ks, Z_proj_z], 'g-', linewidth=2, label=f'Brazo SZ')
            plt.plot(Z_proj_y, Z_proj_z, 'gx', markersize=8)
        
        # Cuadro de texto con los valores
        textstr = f'plano_z = {plano_z:.3f} m\nYB = {YB:.3f} m\nZB = {ZB:.3f} m\nKN = {res_punto["KN"]:.3f} m\nSZ = {SZ:.3f} m\n'
        if kg_valores is not None:
            for kg in kg_valores:
                textstr += f'GZ(KG={kg}) = {res_punto[f"GZ_{kg}"]:.3f} m\n'
            
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        plt.gca().text(1.05, 1.0, textstr.strip(), transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Ajustamos layout para hacer espacio al texto fuera
        plt.subplots_adjust(right=0.65)
        
    plt.title(f'Sección X = {x_val:.3f} m | $\phi$ = {phi_deg}° | plano_z = {plano_z:.3f} m')
    plt.xlabel('Y [m]')
    plt.ylabel('Z (desde quilla) [m]')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.0), fontsize=9)
    
    filename = f'seccion_X_{x_val:.3f}_phi_{int(phi_deg):02d}.png'.replace('.', '_').replace('_png', '.png')
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def calcular_curva_gz(filepath_secciones, volumen_diseno, kg_valores, ks=6.0, angulos_grad=None, x_plot_list=None):
    """
    Función principal que determina el calado de equilibrio (plano_z) para iterativos ángulos de escora
    de modo que el volumen sumergido iguale al de diseño. Al hallar el equilibrio, 
    calcula la posición del centro de carena y los brazos adrizantes KZ, SZ, GZ para diferentes KGs,
    incluye el cálculo del Superficie Mojada Total (S).
    """
    if angulos_grad is None:
        angulos_grad = np.linspace(0, 90, 181) # 181 points, every 0.5 degrees
        
    secciones_raw = cargar_y_ordenar_secciones(filepath_secciones)
    
    # Encontrar Z de la quilla en mm
    z_mins = [np.min(pts[:, 1]) for pts in secciones_raw.values() if len(pts) > 0]
    z_fondo = np.min(z_mins)
    
    # Convertir a metros y trasladar Z para que la quilla sea Z=0                                      
    secciones = {}
    for x, pts in secciones_raw.items():
        if len(pts) == 0: continue
        pts_m = pts.copy()
        pts_m[:, 0] = pts_m[:, 0] / 1000.0                 # Y en metros
        pts_m[:, 1] = (pts_m[:, 1] - z_fondo) / 1000.0     # Z en metros desde la quilla
        secciones[x / 1000.0] = pts_m                      # X en metros
        
    # Encontrar las secciones de X más cercanas si se pide graficar
    x_plot_keys = []
    if x_plot_list is not None and len(secciones) > 0:
        for x_target in x_plot_list:
            x_closest = min(secciones.keys(), key=lambda k: abs(k - x_target))
            x_plot_keys.append(x_closest)
        x_plot_keys = sorted(list(set(x_plot_keys)))
        print(f"Graficando cortes para las secciones X = {[round(x, 3) for x in x_plot_keys]} m")
    
    resultados = []
    
    # Calcular KM a 0 grados para aproximacion lineal (GZ = GM * phi)
    def error_volumen_0(plano_z):
        params = calcular_parametros_hidrostaticos(secciones, plano_z, escora_rad=0.0, solo_volumen=True)
        return params["Volumen"] - volumen_diseno
    plano_z_eq_0 = brentq(error_volumen_0, -20.0, 20.0)
    params_0 = calcular_parametros_hidrostaticos(secciones, plano_z_eq_0, escora_rad=0.0)
    KM_0 = params_0["ZB"] + params_0["BM_T"]
    GM_vals = {kg: KM_0 - kg for kg in kg_valores}
    
    for phi_deg in angulos_grad:
        phi_rad = np.radians(phi_deg)
        
        def error_volumen(plano_z):
            params = calcular_parametros_hidrostaticos(secciones, plano_z, escora_rad=phi_rad, solo_volumen=True)
            return params["Volumen"] - volumen_diseno
            
        # Rango de busqueda razonable para el calado en metros: -20.0 a 20.0
        print(f"  Calculando para phi={phi_deg} grados...")
        try:
            plano_z_eq = brentq(error_volumen, -20.0, 20.0)
        except ValueError as e:
            print(f"No se pudo encontrar raiz para phi={phi_deg}. Error: {e}")
            continue
            
        params = calcular_parametros_hidrostaticos(secciones, plano_z_eq, escora_rad=phi_rad)
        YB = params["TCB"]
        ZB = params["ZB"]
        
        # Brazo desde la Quilla (KN)
        KN = np.abs(YB) * np.cos(phi_rad) + ZB * np.sin(phi_rad)
        
        # Brazo desde el polo auxiliar S (KS = 6m en este caso)
        SZ = KN - ks * np.sin(phi_rad)
        
        res_punto = {
            "phi_deg": phi_deg,
            "plano_z": plano_z_eq,
            "Line_A_y": np.sin(phi_rad),   # Ecuación A*y + B*z + C = 0
            "Line_B_z": np.cos(phi_rad),
            "Line_C": -plano_z_eq,
            "YB": YB,
            "ZB": ZB,
            "KN": KN,
            "SZ": SZ,
            "S": params.get("S", 0.0),
            "Volumen_Calculado": params["Volumen"],
            "Volumen_Error_%": abs((params["Volumen"] - volumen_diseno) / volumen_diseno) * 100.0
        }
        
        for kg in kg_valores:
            # Distancia SG (asumiendo que S está por sobre G)
            SG = ks - kg
            # Según la nomenclatura clásica (ej: GZ = SZ + SG * sin(phi))
            gz = SZ + SG * np.sin(phi_rad)
            res_punto[f"GZ_{kg}"] = gz
            
            # Aproximación Lineal: GZ ≈ GM * phi_rad
            gz_lin = GM_vals[kg] * phi_rad
            res_punto[f"GZ_lin_{kg}"] = gz_lin
            
            # Error de la Aproximación Lineal
            if abs(gz) > 1e-6:
                res_punto[f"Error_Lin_{kg}_%"] = abs((gz_lin - gz) / gz) * 100.0
            else:
                res_punto[f"Error_Lin_{kg}_%"] = 0.0
            
        resultados.append(res_punto)
        
        # Generar gráfico visual para las secciones en este ángulo si procede
        for x_plot_key in x_plot_keys:
            graficar_seccion_sumergida(
                poligono=secciones[x_plot_key],
                plano_z=plano_z_eq,
                phi_rad=phi_rad,
                phi_deg=phi_deg,
                x_val=x_plot_key,
                res_punto=res_punto,
                kg_valores=kg_valores,
                ks=ks
            )
        
    df_res = pd.DataFrame(resultados)
    return df_res, GM_vals

def graficar_gz(df_res, kg_valores):
    """
    Exporta una imagen de la Curva de Estabilidad GZ versus el ángulo de escora
    para todas las condiciones de carga (KGs) especificadas, inclyendo la aprox. lineal inicial.
    """
    plt.figure(figsize=(10, 6))
    
    for kg in kg_valores:
        col_name = f"GZ_{kg}"
        col_name_lin = f"GZ_lin_{kg}"
        
        p = plt.plot(df_res['phi_deg'], df_res[col_name], label=f'KG = {kg} m', linewidth=2)
        color = p[0].get_color()
        plt.plot(df_res['phi_deg'], df_res[col_name_lin], color=color, linestyle='--', alpha=0.7, label=f'Lineal (KG={kg}m)')
        
    plt.axhline(0, color='black', linewidth=1)
    
    # Límite del eje Y para evitar que la aproximación lineal dañe la escala
    max_gz = df_res[[f"GZ_{kg}" for kg in kg_valores]].max().max()
    min_gz = df_res[[f"GZ_{kg}" for kg in kg_valores]].min().min()
    plt.ylim(min(0, min_gz - 0.2), max_gz + 0.5)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Curva de Estabilidad Estática (Curva GZ) y Aproximación Lineal')
    plt.xlabel('Ángulo de Escora $\\phi$ [grados]')
    plt.ylabel('Brazo Adrizante GZ [m]')
    plt.legend()
    plt.tight_layout()
    plt.savefig('curva_GZ.png')
    plt.close()

def graficar_movimiento_B(df_res):
    """
    Exporta una imagen con la trayectoria bidimensional del Centro de Carena (B)
    durante la escora progresiva (YB vs ZB) con respecto a la línea de crujía.
    """
    plt.figure(figsize=(8, 8))
    
    yb = df_res['YB']
    zb = df_res['ZB']
    phi = df_res['phi_deg']
    
    plt.plot(yb, zb, 'b.-', label='Trayectoria de B')
    
    # Añadir marcadores para el primer y último punto calculados
    plt.plot(yb.iloc[0], zb.iloc[0], 'go', markersize=8, label=f'Inicio ($\phi$={phi.iloc[0]}°)')
    plt.plot(yb.iloc[-1], zb.iloc[-1], 'ro', markersize=8, label=f'Fin ($\phi$={phi.iloc[-1]}°)')
    
    plt.axvline(0, color='gray', linestyle='-.', alpha=0.5, label='Línea de Crujía')
    
    # Cuadro indicando el origen en la quilla K
    plt.plot(0, 0, 'ks', markersize=8, label='K (Quilla)')
    plt.text(0.1, -0.2, 'K', fontsize=12, weight='bold')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Movimiento del Centro de Carena (B)')
    plt.xlabel('Desplazamiento Transversal YB [m]')
    plt.ylabel('Desplazamiento Vertical ZB [m]')
    
    # Ajustar a la misma escala para X e Y para no deformar la trayectoria visualmente
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('movimiento_B.png')
    plt.close()

def generar_informe_estabilidad(df_res, gm_vals, kg_valores, filepath="informe_estabilidad.txt"):
    """
    Calcula dinámicamente el área bajo la curva GZ entre diferentes rangos angulares
    (ej: 0-30, 0-40) y verifica de manera automatizada si la embarcación cumple 
    con los criterios estatutarios iniciales de estabilidad intacta.
    Imprime un informe detallado texto evaluando 'OK' o 'FALLA' en cada condición.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=========================================================\n")
        f.write("      INFORME DE CRITERIOS DE ESTABILIDAD INTACTA\n")
        f.write("=========================================================\n\n")
        
        for kg in kg_valores:
            f.write(f"--- Condicion de Carga: KG = {kg:.2f} m ---\n")
            
            # Extraer phi_rad para integracion
            phi_rad = np.radians(df_res['phi_deg'])
            col_gz = f"GZ_{kg}"
            
            # Filtros booleanos para los angulos
            mask_30 = df_res['phi_deg'] <= 30
            mask_40 = df_res['phi_deg'] <= 40
            mask_30_40 = (df_res['phi_deg'] >= 30) & (df_res['phi_deg'] <= 40)
            
            # Criterios (Simpson Rule en numpy para scipy)
            area_0_30 = simpson(df_res.loc[mask_30, col_gz], x=phi_rad[mask_30])
            area_0_40 = simpson(df_res.loc[mask_40, col_gz], x=phi_rad[mask_40])
            area_30_40 = simpson(df_res.loc[mask_30_40, col_gz], x=phi_rad[mask_30_40])
            
            # GZ a 30 grados
            gz_30 = df_res.loc[df_res['phi_deg'] == 30, col_gz].iloc[0]
            
            # GM inicial
            gm_0 = gm_vals[kg]
            
            # Evaluacion frente al criterio
            check_0_30 = "OK" if area_0_30 >= 0.055 else "FALLA"
            check_0_40 = "OK" if area_0_40 >= 0.090 else "FALLA"
            check_30_40 = "OK" if area_30_40 >= 0.030 else "FALLA"
            check_gz_30 = "OK" if gz_30 >= 0.200 else "FALLA"
            check_gm_0 = "OK" if gm_0 >= 0.150 else "FALLA"
            
            f.write(f"  1. Area bajo GZ (0 a 30 grados):     {area_0_30:.4f} m.rad   (Min: 0.055)  -> {check_0_30}\n")
            f.write(f"  2. Area bajo GZ (0 a 40 grados):     {area_0_40:.4f} m.rad   (Min: 0.090)  -> {check_0_40}\n")
            f.write(f"  3. Area bajo GZ (30 a 40 grados):    {area_30_40:.4f} m.rad   (Min: 0.030)  -> {check_30_40}\n")
            f.write(f"  4. Brazo GZ max a 30 grados:         {gz_30:.4f} m       (Min: 0.200)  -> {check_gz_30}\n")
            f.write(f"  5. Altura metacentrica (GM0):        {gm_0:.4f} m       (Min: 0.150)  -> {check_gm_0}\n\n")

    print(f"\nInforme de estabilidad guardado en '{filepath}'")

if __name__ == "__main__":
    filepath = os.path.join(os.path.dirname(__file__), "..", "Resultados", "secciones_contorno_exterior.csv")
    if not os.path.exists(filepath):
        print(f"Error: No se encontro el archivo: {filepath}")
        sys.exit(1)
        
    VOLUMEN_DISENO = 2281.744
    KS = 6.0
    KG_LIST = [5.00, 5.15, 5.30]
    
    # Escogemos 3 secciones a lo largo de la eslora para tener diferentes visualizaciones.
    # Sabemos que la eslora es ~71.75 m. Podemos usar 15m, 35.875m y 55m.
    X_PLOTS_LIST = [15.0, 35.875, 55.0]
    
    print(f"Iniciando cálculo iterativo de la Curva GZ...")
    print(f"Volumen objetivo = {VOLUMEN_DISENO} m3, Polo KS = {KS} m")
    
    df_resultados, gm_valores = calcular_curva_gz(filepath, VOLUMEN_DISENO, KG_LIST, KS, x_plot_list=X_PLOTS_LIST)
    
    if df_resultados.empty:
        print("Error: No se calcularon resultados.")
        sys.exit(1)
        
    print("\nResultados calculados:")
    print(df_resultados.to_string(index=False))
    
    df_resultados.to_csv(os.path.join(os.path.dirname(__file__), "..", "Resultados", "resultados_curva_GZ.csv"), index=False, sep=";", decimal=",")
    print("\nResultados guardados en 'resultados_curva_GZ.csv'")
    
    graficar_gz(df_resultados, KG_LIST)
    print("Gráfico guardado en 'curva_GZ.png'")
    
    graficar_movimiento_B(df_resultados)
    print("Gráfico guardado en 'movimiento_B.png'")
    
    generar_informe_estabilidad(df_resultados, gm_valores, KG_LIST)
