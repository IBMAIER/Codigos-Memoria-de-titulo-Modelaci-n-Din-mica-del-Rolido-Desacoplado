import sys
import os
import numpy as np
import pandas as pd

# Importar el motor de cálculo
sys.path.append(r'C:\Users\ibm\Documents\Seminario\Scrips_seminario\nuevos(febrero2026)')
from funciones_naval import calcular_parametros_hidrostaticos

def crear_seccion_rectangular(manga, z_min=0.0, z_max=10.0):
    if manga <= 1e-5:
        return np.array([[0, z_min], [0, z_max], [0, z_min]])
    y = manga / 2.0
    return np.array([
        [-y, z_min],
        [ y, z_min],
        [ y, z_max],
        [-y, z_max],
        [-y, z_min]
    ])

def test_caja_rectangular():
    print("="*60)
    print(" TEST 1: CAJA RECTANGULAR (BARCAZA)")
    print("="*60)
    L = 100.0
    B = 20.0
    
    # Crear geometría
    secciones = {}
    x_vals = np.linspace(0, L, 51)
    for x in x_vals:
        secciones[x] = crear_seccion_rectangular(B)
        
    plano_z = 5.0
    res = calcular_parametros_hidrostaticos(secciones, plano_z)
    
    it_calc = res["I_T"]
    it_teorico = (L * B**3) / 12.0
    error = abs(it_calc - it_teorico) / it_teorico * 100
    
    print(f"I_T Calculado : {it_calc:12.4f} m^4")
    print(f"I_T Teórico   : {it_teorico:12.4f} m^4")
    print(f"Error         : {error:12.6f} %")
    return error

def test_forma_barco_simplificada():
    print("\n" + "="*60)
    print(" TEST 2: FORMA DE BARCO SIMPLIFICADA (VISTA SUPERIOR)")
    print(" Mezcla de proa triangular, cuerpo central rectangular y popa trapezoidal")
    print("="*60)
    
    # Dimensiones
    L_popa = 20.0
    L_centro = 50.0
    L_proa = 30.0
    B_max = 16.0
    B_espejo = 8.0 # Popa truncada
    
    # 1. Popa (Trapecio desde x=0 hasta x=20)
    # y(x) va desde B_espejo/2 hasta B_max/2
    # 2. Centro (Rectángulo desde x=20 hasta x=70)
    # y(x) = B_max/2
    # 3. Proa (Triángulo desde x=70 hasta x=100)
    # y(x) va desde B_max/2 hasta 0
    
    secciones = {}
    x_vals = np.linspace(0, 100, 201)
    for x in x_vals:
        if x <= L_popa:
            b = B_espejo + (B_max - B_espejo) * (x / L_popa)
        elif x <= L_popa + L_centro:
            b = B_max
        else:
            b = B_max * (1.0 - (x - L_popa - L_centro) / L_proa)
            
        secciones[x] = crear_seccion_rectangular(b)
        
    res = calcular_parametros_hidrostaticos(secciones, plano_z=5.0)
    it_calc = res["I_T"]
    
    # Cálculo Teórico Exacto: Integral de (b(x)^3)/12 dx
    # 1. Popa: integral( (8 + 8*(x/20))^3 / 12 ) de 0 a 20
    # b(x) = 8 + 0.4*x
    # I = (1/12) * [ (8 + 0.4x)^4 / (4 * 0.4) ]_0^20
    #   = (1/12) * (1/1.6) * [ 16^4 - 8^4 ]
    #   = (1/19.2) * [ 65536 - 4096 ] = 61440 / 19.2 = 3200.0
    it_popa = 3200.0
    
    # 2. Centro: L * B^3 / 12 = 50 * 16^3 / 12 = 50 * 4096 / 12 = 17066.6667
    it_centro = 50.0 * (B_max**3) / 12.0
    
    # 3. Proa: L * B^3 / 48 (inercia de romboide truncado o triángulo)
    # b(x) = 16 - (16/30)*x'  (x' de 0 a 30)
    # integral( (16 - 16/30 * x')^3 / 12 ) = L * B^3 / 48 = 30 * 16^3 / 48 = 2560.0
    it_proa = (L_proa * B_max**3) / 48.0
    
    it_teorico = it_popa + it_centro + it_proa
    
    error = abs(it_calc - it_teorico) / it_teorico * 100
    
    print(f"Inercia Popa   (Teórica): {it_popa:12.4f} m^4")
    print(f"Inercia Centro (Teórica): {it_centro:12.4f} m^4")
    print(f"Inercia Proa   (Teórica): {it_proa:12.4f} m^4")
    print("-" * 40)
    print(f"I_T Calculado  : {it_calc:12.4f} m^4")
    print(f"I_T Teórico    : {it_teorico:12.4f} m^4")
    print(f"Error          : {error:12.6f} %")
    return error

def test_rombo():
    print("\n" + "="*60)
    print(" TEST 3: ROMBO DIAMANTE (VISTA SUPERIOR)")
    print("="*60)
    L = 100.0
    B = 20.0
    secciones = {}
    x_vals = np.linspace(0, L, 201)
    for x in x_vals:
        if x <= L/2:
            b = B * (x / (L/2))
        else:
            b = B * (1.0 - (x - L/2) / (L/2))
        secciones[x] = crear_seccion_rectangular(b)
        
    res = calcular_parametros_hidrostaticos(secciones, plano_z=5.0)
    it_calc = res["I_T"]
    
    # Inercia de un rombo (dos triángulos unidos por su base):
    # = 2 * ( (L/2) * B^3 / 48 ) = L * B^3 / 48
    it_teorico = (L * B**3) / 48.0
    error = abs(it_calc - it_teorico) / it_teorico * 100
    
    print(f"I_T Calculado : {it_calc:12.4f} m^4")
    print(f"I_T Teórico   : {it_teorico:12.4f} m^4")
    print(f"Error         : {error:12.6f} %")
    return error

def test_cilindro():
    print("\n" + "="*60)
    print(" TEST 4: CILINDRO HORIZONTAL SUMERGIDO HASTA SU EJE")
    print(" (Plano de agua es un rectángulo)")
    print("="*60)
    L = 100.0
    R = 10.0
    
    secciones = {}
    x_vals = np.linspace(0, L, 51)
    # Cada sección es un círculo.
    # Lo discretizamos con N puntos.
    theta = np.linspace(0, 2*np.pi, 360)
    for x in x_vals:
        # El círculo centrado en Z=5.0
        y = R * np.cos(theta)
        z = 5.0 + R * np.sin(theta)
        secciones[x] = np.column_stack((y, z))
        
    res = calcular_parametros_hidrostaticos(secciones, plano_z=5.0)
    it_calc = res["I_T"]
    
    # A Z=5.0, el ancho es exactamente 2*R. El plano de agua es rectángulo de L x 2R
    it_teorico = (L * (2*R)**3) / 12.0
    error = abs(it_calc - it_teorico) / it_teorico * 100
    
    print(f"I_T Calculado : {it_calc:12.4f} m^4")
    print(f"I_T Teórico   : {it_teorico:12.4f} m^4")
    print(f"Error         : {error:12.6f} %")
    return error

if __name__ == "__main__":
    print("************************************************************")
    print(" REPORTE DE VALIDACIÓN: MOTOR DE INERCIA TRANSVERSAL (I_T)")
    print("************************************************************")
    e1 = test_caja_rectangular()
    e2 = test_forma_barco_simplificada()
    e3 = test_rombo()
    e4 = test_cilindro()
    
    print("\n************************************************************")
    print(" RESUMEN DE VALIDACIÓN")
    print("************************************************************")
    print(f"Test 1 (Caja Rectangular) : {'EXITO' if e1 < 1.0 else 'FALLO'} (Err: {e1:.4f}%)")
    print(f"Test 2 (Forma Barco)      : {'EXITO' if e2 < 1.0 else 'FALLO'} (Err: {e2:.4f}%)")
    print(f"Test 3 (Rombo)            : {'EXITO' if e3 < 1.0 else 'FALLO'} (Err: {e3:.4f}%)")
    print(f"Test 4 (Cilindro a la Mitad): {'EXITO' if e4 < 1.0 else 'FALLO'} (Err: {e4:.4f}%)")
    print("************************************************************")

def plot_geometria_barco():
    import matplotlib.pyplot as plt
    L_popa = 20.0
    L_centro = 50.0
    L_proa = 30.0
    B_max = 16.0
    B_espejo = 8.0 
    
    x_vals = np.linspace(0, 100, 201)
    y_pos = []
    y_neg = []
    
    for x in x_vals:
        if x <= L_popa:
            b = B_espejo + (B_max - B_espejo) * (x / L_popa)
        elif x <= L_popa + L_centro:
            b = B_max
        else:
            b = B_max * (1.0 - (x - L_popa - L_centro) / L_proa)
        y_pos.append(b / 2.0)
        y_neg.append(-b / 2.0)
        
    plt.figure(figsize=(10, 4))
    plt.plot(x_vals, y_pos, 'b-', linewidth=2, label="Babor/Estribor")
    plt.plot(x_vals, y_neg, 'b-', linewidth=2)
    plt.fill_between(x_vals, y_neg, y_pos, color='lightblue', alpha=0.5)
    
    plt.title("Plano de Flotación: Forma de Buque Simplificada (Test 2)")
    plt.xlabel("Eslora (X) [m]")
    plt.ylabel("Manga (Y) [m]")
    plt.axvline(20, color='r', linestyle='--', alpha=0.5, label='Fin Popa')
    plt.axvline(70, color='g', linestyle='--', alpha=0.5, label='Inicio Proa')
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.axis("equal") # Mantener proporciones geométricas reales
    plt.tight_layout()
    plt.savefig("21_Plano_Flotacion_Test2.png", dpi=300)
    plt.close()
    print("Gráfico generado: 21_Plano_Flotacion_Test2.png")

if __name__ == "__main__":
    plot_geometria_barco()
