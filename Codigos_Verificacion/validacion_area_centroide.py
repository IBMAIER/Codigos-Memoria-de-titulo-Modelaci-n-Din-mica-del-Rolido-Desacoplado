"""
VALIDADOR 2D DE ÁREAS Y CENTROIDES SECCIONALES (validacion_area_centroide.py)

Este script complementario audita a nivel 2D (sección por sección) la precisión 
matemática del método del polígono cerrado (Teorema de Green / Shoelace) y del 
recorte poligonal (Sutherland-Hodgman). Genera evidencia visual comprobando que 
el área local y el centroide 2D (y_c, z_c) sigan la geometría analítica correcta.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

# Importamos las funciones desarrolladas en nuestro módulo cálculo_hidrostatico
from funciones_naval import (
    cargar_y_ordenar_secciones,
    recortar_poligono,
    calcular_area_y_centroide
)

# ==========================================================
# RUTAS
# ==========================================================
carpeta_base = r"C:\Users\ibm\Documents\Seminario\plots"
archivo_entrada = os.path.join(carpeta_base, os.path.join(os.path.dirname(__file__), "..", "Resultados", "secciones_contorno_exterior.csv"))

def run_auditoria_visual(calado_T=4500.0):
    print(f"\n[INICIANDO AUDITORÍA VISUAL GEOMÉTRICA - CALADO = {calado_T} mm]")
    
    if not os.path.exists(archivo_entrada):
        print(f"Error: No se encuentra {archivo_entrada}")
        return
        
    secciones = cargar_y_ordenar_secciones(archivo_entrada)
    if not secciones:
        print("Error: No se cargaron secciones.")
        return
        
    # Calcular Z fondo
    z_mins = [np.min(pts[:, 1]) for pts in secciones.values()]
    z_fondo = np.min(z_mins)
    plano_corte_z = z_fondo + calado_T
    
    # Seleccionaremos 3 secciones representativas: Proa, Medio, Popa
    x_keys = sorted(secciones.keys())
    
    # Índices aproximados al 15%, 50% y 85% de la eslora
    idx_popa = int(len(x_keys) * 0.15)
    idx_centro = int(len(x_keys) * 0.50)
    idx_proa = int(len(x_keys) * 0.85)
    
    secciones_a_probar = [
        ("Sección en Popa", x_keys[idx_popa]),
        ("Sección Maestra (Centro)", x_keys[idx_centro]),
        ("Sección en Proa", x_keys[idx_proa])
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.suptitle(f"AUDITORÍA DE RECORTE POLIGONAL Y CENTROIDES (Calado Diseño T={calado_T}mm)", fontsize=16)
    
    for ax, (nombre, x_val) in zip(axes, secciones_a_probar):
        poligono = secciones[x_val]
        
        # 1. EJECUTAR SUTHERLAND-HODGMAN
        poligono_sumergido = recortar_poligono(poligono, plano_corte_z)
        
        # 2. EJECUTAR SHOELACE
        area, y_c, z_c = calcular_area_y_centroide(poligono_sumergido)
        
        # 3. PLOTEAR CONTORNO ORIGINAL
        ax.plot(poligono[:, 0], poligono[:, 1], 'bo-', markersize=2, alpha=0.5, label='Polígono Original Completo')
        
        # 4. PLOTEAR LÍNEA DE FLOTACIÓN
        y_min_g, y_max_g = ax.get_xlim()
        # Ampliamos ficticiamente la línea de agua para que cruce todo el plot
        ancho_plot = max(np.max(poligono[:, 0]), abs(np.min(poligono[:, 0]))) + 1000
        ax.axhline(y=plano_corte_z, color='c', linestyle='--', linewidth=2, label=f'Plano de Flotación Z={plano_corte_z:.0f}')
        
        # 5. PLOTEAR PARTE SUMERGIDA (Sutherland-Hodgman)
        if len(poligono_sumergido) > 0:
            # Rellenar área sumergida
            ax.fill(poligono_sumergido[:, 0], poligono_sumergido[:, 1], 'skyblue', alpha=0.4, label='Área Sumergida Intersectada')
            # Marcar el contorno del polígono nuevo
            ax.plot(poligono_sumergido[:, 0], poligono_sumergido[:, 1], 'r.-', linewidth=2, label='Contorno Recortado')
            
            # 6. PLOTEAR CENTRO DE CARENA LOCAL (Shoelace)
            ax.plot(y_c, z_c, 'k*', markersize=15, label=f'Centroide (A={area/1e6:.1f} m²)')
        
        ax.set_title(f"{nombre} (X = {x_val})")
        ax.set_xlabel("Y [mm]")
        ax.set_ylabel("Z [mm]")
        ax.axis('equal')  # Mantener proporciones geométricas reales
        
        # Mejorar grilla y límites
        ax.grid(True, linestyle=':', alpha=0.7)
        # Asegurar que el calado siempre se vea si el barco es muy alto
        ax.set_ylim(z_fondo - 500, np.max(poligono[:, 1]) + 500)
        
        # Leyenda solo si hay datos sumergidos
        ax.legend(loc='lower left', fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Guardar la evidencia empírica
    ruta_guardado = os.path.join(carpeta_base, "Auditoria_Recortes_Centroides.png")
    plt.savefig(ruta_guardado, dpi=300)
    print(f"✅ Evidencia visual guardada exitosamente en:\n -> {ruta_guardado}")
    
    # Muestra en pantalla (puede ser ruidoso en un servidor sin display, lo evitamos o comentamos)
    # plt.show() 

if __name__ == "__main__":
    run_auditoria_visual()
