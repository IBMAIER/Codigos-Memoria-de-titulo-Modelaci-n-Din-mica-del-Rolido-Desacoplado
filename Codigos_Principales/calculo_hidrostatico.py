"""
CALCULADORA DE HIDROSTÁTICA ESTÁTICA SIMPLE (calculo_hidrostatico.py)

Este programa es una herramienta de consulta instantánea que ejecuta de manera estática el 
motor `funciones_naval.py`. Toma un archivo de cuadernas y un calado específico fijo, 
sin considerar escoras complejas. Imprime en consola y en un reporte txt los resultados 
finales de: Volumen, Centros de Carena (LCB, TCB, VCB), Inercias, y Superficie mojada.
Sirve para hacer chequeos rápidos de las propiedades del buque en condición "Upright".
"""
import os
import numpy as np
from funciones_naval import cargar_y_ordenar_secciones, calcular_parametros_hidrostaticos

# ==========================================================
# RUTAS Y ARCHIVOS
# ==========================================================

archivo_entrada = os.path.join(os.path.dirname(__file__), "..", "Resultados", "secciones_contorno_exterior.csv")

# ==========================================================
# RUTINA DE PRUEBA (ESCORA 0)
# ==========================================================
if __name__ == "__main__":
    
    print(f"Leyendo secciones desde: {archivo_entrada}")
    if os.path.exists(archivo_entrada):
        secciones = cargar_y_ordenar_secciones(archivo_entrada)
        print(f"Cargadas {len(secciones)} secciones.")
        
        # Obtener el Z mínimo del fondo de la quilla para usarlo como referencia 0
        z_mins = [np.min(pts[:, 1]) for pts in secciones.values()]
        z_fondo = np.min(z_mins)
        
        # ======== NUEVOS PARÁMETROS DEL USUARIO ========
        calado_T = 4700.0   # T = 4.7 m (4700 mm)
        
        # El plano de corte real en coordenadas del CAD es el fondo + el calado
        plano_corte_z = z_fondo + calado_T
        
        resultados = calcular_parametros_hidrostaticos(secciones, plano_corte_z)
        
        # Cálculo de dimensiones principales útiles
        # Eslora en la flotación (L_{wl})
        x_vals = resultados["x_vals"]
        L_wl = np.max(x_vals) - np.min(x_vals) if len(x_vals) > 0 else 0.0
        
        # Manga máxima en la flotación (B_{wl})
        b_wps = resultados["b_wps"]
        B_wl = np.max(b_wps) if len(b_wps) > 0 else 0.0
        
        # Conversiones de mm a m
        L_wl_m = L_wl / 1000.0
        B_wl_m = B_wl / 1000.0
        T_m = calado_T / 1000.0
        
        Vol_m3 = resultados["Volumen"] / 1e9
        A_wp_m2 = resultados["A_wp"] / 1e6
        I_T_m4 = resultados["I_T"] / 1e12
        BM_T_m = resultados["BM_T"] / 1000.0
        
        # Coeficientes de Forma
        # Coeficiente de Bloque (Cb)
        vol_caja = L_wl_m * B_wl_m * T_m
        Cb = Vol_m3 / vol_caja if vol_caja > 0 else 0.0
        
        # Coeficiente de Plano de flotación (Cwp)
        area_caja_wp = L_wl_m * B_wl_m
        Cwp = A_wp_m2 / area_caja_wp if area_caja_wp > 0 else 0.0

        # Sección Maestra (C_M)
        A_master_raw = resultados["A_master"]  # en mm² (coordenadas CAD en mm)
        x_master_raw = resultados["x_master"]  # en mm (coordenadas CAD)
        A_master_m2  = A_master_raw / 1e6      # convertir a m²
        x_master_m   = x_master_raw / 1000.0  # convertir a m
        caja_maestra = B_wl_m * T_m            # B × T [m²]
        C_M = A_master_m2 / caja_maestra if caja_maestra > 0 else 0.0

        # Desplazamiento (Peso) en toneladas métricas (Asumiendo agua de mar rho = 1.025 t/m3)
        Desplazamiento = Vol_m3 * 1.025
        
        # Generar Reporte de Salida
        # Se guarda directamente en el directorio actual: nuevos(febrero2026)
        reporte_path = "reporte_hidrostatico_completo.txt"
        
        with open(reporte_path, "w", encoding="utf-8") as f:
            f.write("=================================================================\n")
            f.write("           REPORTE DE ARQUITECTURA NAVAL (HIDROSTÁTICA)          \n")
            f.write("=================================================================\n\n")
            f.write("1. CONDICIONES DE DISEÑO\n")
            f.write(f"   Calado de Diseño (T): {T_m:.3f} m\n")
            f.write(f"   Eslora en la flotación (Lwl): {L_wl_m:.3f} m\n")
            f.write(f"   Manga máxima en la flotación (Bwl): {B_wl_m:.3f} m\n\n")
            
            f.write("2. PARÁMETROS GLOBALES DE VOLUMEN\n")
            f.write(f"   Volumen Desplazado (∇): {Vol_m3:.3f} m^3\n")
            f.write(f"   Desplazamiento (\u0394): {Desplazamiento:.3f} t (Densidad = 1.025 t/m^3)\n")
            f.write(f"   Coeficiente de Bloque (Cb): {Cb:.4f}\n\n")
            
            f.write("3. CENTRO DE CARENA (B)\n")
            f.write(f"   LCB (Longitudinal Center of Buoyancy): {resultados['LCB']:.2f} mm\n")
            f.write(f"   TCB (Transverse Center of Buoyancy): {resultados['TCB']:.2f} mm\n")
            f.write(f"   KB  (Vertical Center of Buoyancy desde quilla): {resultados['ZB'] - z_fondo:.2f} mm\n\n")
            
            f.write("4. PROPIEDADES DEL PLANO DE FLOTACIÓN\n")
            f.write(f"   Área del plano de agua (Awp): {A_wp_m2:.3f} m^2\n")
            f.write(f"   Coeficiente del plano de flotación (Cwp): {Cwp:.4f}\n")
            f.write(f"   Inercia Transversal Geométrica (IT): {I_T_m4:.3f} m^4\n")
            f.write("   * NOTA IMPORTANTE: This is not the mass moment of inertia of the ship, \n")
            f.write("     but the geometric second moment of area of the waterplane.\n\n")
            
            f.write("5. ESTABILIDAD TRANSVERSAL INICIAL\n")
            f.write(f"   Radio Metacéntrico Transversal (BMT): {BM_T_m:.3f} m\n")
            f.write("=================================================================\n\n")

            f.write("6. SECCIÓN MAESTRA Y COEFICIENTE C_M\n")
            f.write(f"   Posición longitudinal Sección Maestra (X_M): {x_master_m:.3f} m\n")
            f.write(f"   Área sumergida Sección Maestra (A_M): {A_master_m2:.4f} m^2\n")
            f.write(f"   Caja de referencia B x T: {B_wl_m:.3f} m x {T_m:.3f} m = {caja_maestra:.4f} m^2\n")
            f.write(f"   Coeficiente de Sección Maestra (C_M = A_M / (B x T)): {C_M:.4f}\n")
            f.write("   NOTA: Este C_M debe usarse en calcular_damping.py y solver_rolido_RK4.py\n")
            f.write("=================================================================\n")
            
        print(f"Reporte exportado exitosamente en: {reporte_path}")
        print("Muestra en consola de datos principales:")
        print(f" -> Volumen: {Vol_m3:.2f} m3")
        print(f" -> Desplazamiento: {Desplazamiento:.2f} t")
        print(f" -> Coeficiente de Bloque (Cb): {Cb:.4f}")
        print(f" -> IT: {I_T_m4:.2f} m4")
        print(f" -> BMT: {BM_T_m:.2f} m")
        print(f" -> Sección Maestra en X = {x_master_m:.3f} m  |  A_M = {A_master_m2:.4f} m2")
        print(f" -> C_M (Coef. Sección Maestra) = {C_M:.4f}")
    else:
        print("El archivo base de contornos no existe. Es necesario generarlo primero.")
