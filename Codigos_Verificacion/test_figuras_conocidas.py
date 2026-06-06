"""
VALIDACIÓN MATEMÁTICA Y PRUEBAS UNITARIAS (test_figuras_conocidas.py)

Somete el motor de cálculo hidrostático (`funciones_naval.py`) a pruebas con formas 
geométricas donde los parámetros (Volumen, LCB, KB, Área de flotación, Superficie Mojada) 
son teóricamente exactos y conocidos (cajas, prismas, semiesferas).
Confirma que los algoritmos de integración funcionen correctamente y exporta
los resultados a un informe de texto delimitado.
"""

import numpy as np
import os
import pandas as pd
from funciones_naval import (
    cargar_y_ordenar_secciones,
    recortar_poligono,
    calcular_area_y_centroide,
    calcular_parametros_hidrostaticos
)

def log_and_print(msg, file_handle):
    print(msg)
    file_handle.write(msg + '\n')

def test_caja_rectangular(f):
    log_and_print("========================================", f)
    log_and_print(" TEST: CAJA RECTANGULAR", f)
    log_and_print("========================================", f)
    secciones_originales = {}
    x_vals = [0.0, 5.0, 10.0]
    pts_rect = np.array([[-2.0, 0.0], [2.0, 0.0], [2.0, 6.0], [-2.0, 6.0], [-2.0, 0.0]])
    for x in x_vals: secciones_originales[x] = pts_rect.copy()
    plano_z = 3.0
    
    log_and_print(f"Dimensiones -> L: 10, B: 4, D: 6", f)
    log_and_print(f"Calado -> Z: {plano_z}", f)
    log_and_print(f"Volumen esperado = 120.0", f)
    log_and_print(f"ZB (KB) esperado = 1.5", f)
    log_and_print(f"Superficie Mojada S esperada = 100.0 (perímetro 10 * eslora 10)", f)
    log_and_print("----------------------------------------", f)
    
    res = calcular_parametros_hidrostaticos(secciones_originales, plano_z)
    
    log_and_print("Resultados:", f)
    log_and_print(f"Volumen: {res['Volumen']:.2f}", f)
    log_and_print(f"LCB: {res['LCB']:.2f}", f)
    log_and_print(f"TCB: {res['TCB']:.2f}", f)
    log_and_print(f"ZB (KB): {res['ZB']:.2f}", f)
    log_and_print(f"A_wp: {res['A_wp']:.2f}", f)
    log_and_print(f"S (Área Mojada): {res['S']:.2f}", f)
    log_and_print("========================================\n", f)

def test_prisma_triangular(f):
    log_and_print("========================================", f)
    log_and_print(" TEST: PRISMA TRIANGULAR (FORMA DE 'V')", f)
    log_and_print("========================================", f)
    secciones_originales = {}
    x_vals = [0.0, 5.0, 10.0]
    pts_tri = np.array([[0.0, 0.0], [3.0, 6.0], [-3.0, 6.0], [0.0, 0.0]])
    for x in x_vals: secciones_originales[x] = pts_tri.copy()
    plano_z = 3.0
    
    esperado_s = 6.7082039 * 10
    
    log_and_print(f"Dimensiones -> L: 10, B_max: 6, D: 6 | Calado Z: 3.0", f)
    log_and_print(f"Volumen esperado = 45.0", f)
    log_and_print(f"ZB (KB) esperado = 2.0", f)
    log_and_print(f"Superficie Mojada S esperada = {esperado_s:.2f}", f)
    log_and_print("----------------------------------------", f)
    
    res = calcular_parametros_hidrostaticos(secciones_originales, plano_z)
    
    log_and_print("Resultados:", f)
    log_and_print(f"Volumen: {res['Volumen']:.2f}", f)
    log_and_print(f"LCB: {res['LCB']:.2f}", f)
    log_and_print(f"TCB: {res['TCB']:.2f}", f)
    log_and_print(f"ZB (KB): {res['ZB']:.2f}", f)
    log_and_print(f"A_wp: {res['A_wp']:.2f}", f)
    log_and_print(f"S (Área Mojada): {res['S']:.2f}", f)
    log_and_print("========================================\n", f)

def test_semiesfera(f):
    log_and_print("========================================", f)
    log_and_print(" TEST: SEMIESFERA", f)
    log_and_print("========================================", f)
    # R = 5.0, Centro en (X=0, Y=0, Z=R)
    R = 5.0
    secciones_originales = {}
    x_vals = np.linspace(-R, R, 21) # Resolución reducida para velocidad
    
    for x in x_vals:
        if abs(x) == R:
            pts = np.array([[0.0, R], [0.0, R]])
        else:
            r_x = np.sqrt(R**2 - x**2)
            theta = np.linspace(np.pi, 2*np.pi, 51) 
            y = r_x * np.cos(theta)
            z = R + r_x * np.sin(theta)
            pts = np.vstack((y, z)).T
            pts = np.vstack((pts, pts[0]))
        secciones_originales[x] = pts

    plano_z = R
    
    vol_esp = (2/3) * np.pi * R**3
    kb_esp = (5/8) * R # Porque la base está arriba
    awp_esp = np.pi * R**2
    it_esp = (np.pi * R**4) / 4
    
    # IMPORTANTE: La superficie mojada S calculada integrando mediante franjas 2D (simpson a lo largo de X)
    # asume que no hay curvatura longitudinal (o la ignora). La integral analítica de los perímetros 
    # transversales de una esfera es int_(-R)^R pi * sqrt(R^2 - x^2) dx = (pi^2 / 2) * R^2.
    # El área 3D real de la semiesfera es 2*pi*R^2 (aprox 157.08), pero el algoritmo matemático bidimensional 
    # de perimetros seccionales SIEMPRE arrojará teóricamente la integral plana:
    s_esp_strip = (np.pi**2 / 2) * R**2

    log_and_print(f"Radio R = {R:.1f} m | Calado Z = {plano_z:.1f} m", f)
    log_and_print(f"Volumen esperado = {vol_esp:.3f}", f)
    log_and_print(f"ZB (KB) esperado = {kb_esp:.3f}", f)
    log_and_print(f"Área Superficie Flotación = {awp_esp:.3f}", f)
    log_and_print(f"Inercia Transversal I_T = {it_esp:.3f}", f)
    log_and_print(f"Superficie Mojada S (Teórica 2D Strips) = {s_esp_strip:.3f} m2 (Superficie 3D Real = {2 * np.pi * R**2:.3f} m2)", f)
    log_and_print("----------------------------------------", f)
    
    res = calcular_parametros_hidrostaticos(secciones_originales, plano_z)
    
    log_and_print("Resultados del Algoritmo:", f)
    log_and_print(f"Volumen: {res['Volumen']:.3f} (Error: {abs(res['Volumen']-vol_esp)/vol_esp*100:.2f}%)", f)
    log_and_print(f"LCB: {res['LCB']:.3f}", f)
    log_and_print(f"TCB: {res['TCB']:.3f}", f)
    log_and_print(f"ZB (KB): {res['ZB']:.3f} (Error: {abs(res['ZB']-kb_esp)/kb_esp*100:.2f}%)", f)
    log_and_print(f"A_wp: {res['A_wp']:.3f} (Error: {abs(res['A_wp']-awp_esp)/awp_esp*100:.2f}%)", f)
    log_and_print(f"I_T: {res['I_T']:.3f} (Error: {abs(res['I_T']-it_esp)/it_esp*100:.2f}%)", f)
    log_and_print(f"S (Área Mojada): {res['S']:.3f} (Error vs 2D Strips: {abs(res['S']-s_esp_strip)/s_esp_strip*100:.2f}%)", f)
    log_and_print("========================================\n", f)

if __name__ == '__main__':
    with open("informe_resultados_pruebas.txt", "w", encoding='utf-8') as f:
        log_and_print("=== INFORME DE VALIDACIÓN MATEMÁTICA Y PRUEBAS ===\n", f)
        test_caja_rectangular(f)
        test_prisma_triangular(f)
        test_semiesfera(f)
        log_and_print("Pruebas completadas y volcadas en el informe.", f)
