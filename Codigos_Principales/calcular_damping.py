import numpy as np
import pandas as pd
import math
import sys
import os

# Agregamos la ruta del directorio que contiene funciones_naval.py para no duplicar código
sys.path.append(r"C:\Users\ibm\Documents\Seminario\Scrips_seminario\nuevos(febrero2026)")

from funciones_dinamicas import discretizar_espectro_no_uniforme, CASOS_ESTUDIO
from calcular_coeficientes_excitacion import order_points_by_angle
from funciones_naval import recortar_poligono, calcular_area_y_centroide

def precomputar_geometria_adrizada(hull_sections_df, d):
    """
    Precomputa el área sumergida (A_x), manga (B_x) y calado (d_x) para cada sección
    en condición adrizada (phi=0), usando Sutherland-Hodgman.
    Trabaja en un sistema referenciado a la quilla (K) global.
    """
    precomputed = {}
    if hull_sections_df is not None and d > 0:
        # 1. Encontrar el punto más bajo global (Quilla o K) para unificar la referencia de Z
        z_min_global = np.min(hull_sections_df['Z'].values)
        
        x_vals = np.sort(hull_sections_df['X'].unique())
        for x_val in x_vals:
            sec_data = hull_sections_df[hull_sections_df['X'] == x_val]
            y_sec = sec_data['Y'].values
            
            # 2. Desplazar las coordenadas Z para que la quilla K esté en Z=0
            z_sec = sec_data['Z'].values - z_min_global
            
            if not np.allclose([y_sec[0], z_sec[0]], [y_sec[-1], z_sec[-1]], atol=1e-5):
                y_sec_close = np.append(y_sec, y_sec[0])
                z_sec_close = np.append(z_sec, z_sec[0])
            else:
                y_sec_close = y_sec
                z_sec_close = z_sec
                
            poligono = np.column_stack((y_sec_close, z_sec_close))
            
            # 3. La línea de agua global está a altura d exacta desde la quilla K
            plano_z = d
            poligono_sumergido = recortar_poligono(poligono, plano_z, escora_rad=0.0)
            
            A_x, B_x, d_x = 0.0, 0.0, 0.0
            if len(poligono_sumergido) > 2:
                y_sub = poligono_sumergido[:, 0]
                z_sub = poligono_sumergido[:, 1]
                
                B_x = np.max(y_sub) - np.min(y_sub)
                
                # El calado local d_x es la diferencia de alturas dentro de la parte sumergida
                d_x = np.max(z_sub) - np.min(z_sub)
                
                if B_x > 0 and d_x > 0:
                    A_x, _, _ = calcular_area_y_centroide(poligono_sumergido)
                    if A_x <= 1e-4:
                        A_x = B_x * d_x * 0.9
                        
            precomputed[x_val] = {'A_x': A_x, 'B_x': B_x, 'd_x': d_x}
            
    return precomputed

def calcular_damping_B44(phi_a_deg, V_knots, omega_E, ship_params, exportar_eddy_txt=False):
    """
    Calcula el amortiguamiento al rolido B44 en unidades dimensionales [kg*m^2/s]
    utilizando la metodología híbrida Ikeda-Himeno + Kawahara SIM (2009).
    
    Parámetros:
    -----------
    phi_a_deg: Amplitud de rolido característica [grados]
    V_knots: Velocidad de avance [nudos]
    omega_E: Frecuencia de encuentro [rad/s]
    ship_params: Diccionario con los datos principales del buque
        (L_PP, B, d, C_B, C_M, OG, b_BK, l_BK, rho, nu)
    hull_sections_df: DataFrame con geometría (X, Y, Z) para integrar Eddy. 
                      Opcional; si es None, B44E será 0.0.
                      
    Retorna:
    --------
    dict: Desglose de componentes B44 y el total.
    """
    
    # Conversiones y datos base
    phi_a = np.radians(phi_a_deg)
    V = V_knots * 0.514444  # De nudos a m/s
    
    L = ship_params.get('L_PP', 0.0)
    B = ship_params.get('B', 0.0)
    d = ship_params.get('d', 0.0)
    CB = ship_params.get('C_B', 0.0)
    CM = ship_params.get('C_M', 0.0)
    OG = ship_params.get('OG', 0.0) # Convención: z_flotación - z_G
    
    rho = ship_params.get('rho', 1025.0)
    nu = ship_params.get('nu', 1.19e-6)
    g = 9.81
    
    b_BK = ship_params.get('b_BK', 0.0)
    l_BK = ship_params.get('l_BK', 0.0)
    
    # Factor de estandarización ITTC (Para pasar de Adimensional a Dimensional)
    nabla = ship_params.get('nabla', L * B * d * CB)
    if B > 0:
        factor_ittc = rho * nabla * (B**2) * np.sqrt(2.0 * g / B)
    else:
        factor_ittc = 0.0
        
    B44_total = 0.0
    
    # ==========================================
    # 1. FRICCIÓN (Kato) - B44F
    # ==========================================
    if L > 0 and d > 0 and phi_a > 1e-4 and omega_E > 1e-4:
        Sf = L * (1.7 * d + CB * B)
        rf = (1.0 / np.pi) * ((0.887 + 0.145 * CB) * (1.7 * d + CB * B) - 2.0 * OG)
        
        Re = 0.512 * (rf**2) * (phi_a**2) * omega_E / nu
        Cf = 1.328 * Re**(-0.5)
        
        B44F0 = (4.0 / (3.0 * np.pi)) * rho * Sf * (rf**3) * phi_a * omega_E * Cf
        
        if V > 0:
            B44F = B44F0 * (1.0 + 4.1 * V / (L * omega_E))
        else:
            B44F = B44F0
    else:
        B44F = 0.0
        
    B44_total += B44F
    
    # ==========================================
    # 2. LIFT (Himeno) - B44L
    # ==========================================
    if V > 0 and L > 0 and d > 0:
        if CM <= 0.92:
            kappa = 0.0
        elif CM <= 0.97:
            kappa = 0.1
        elif CM < 0.99:
            kappa = 0.3
        else:
            kappa = 0.3 # Tope
            
        kN = 2.0 * np.pi * (d / L) + kappa * (4.1 * (B / L) - 0.045)
        l0 = 0.3 * d
        lR = 0.5 * d
        
        B44L = 0.5 * rho * V * L * d * kN * l0 * lR * (1.0 - 1.4 * OG / lR + 0.7 * (OG**2) / (l0 * lR))
    else:
        B44L = 0.0
        
    B44_total += B44L
    
    # ==========================================
    # 3. RADIACIÓN DE OLAS (Kawahara SIM) - B44W
    # ==========================================
    if d > 0 and B > 0 and omega_E > 1e-4:
        x1 = B / d
        x2 = CB
        x3 = CM
        x4 = 1.0 - OG / d
        x5 = omega_E * np.sqrt(B / (2.0 * g))
        
        # Coeficientes A1
        AA111 = 17.945*x1**3 - 166.294*x1**2 + 489.799*x1 - 493.142
        AA112 = -25.507*x1**3 + 236.275*x1**2 - 698.683*x1 + 701.494
        AA113 = 9.077*x1**3 - 84.332*x1**2 + 249.983*x1 - 250.787
        AA121 = -16.872*x1**3 + 156.399*x1**2 - 460.689*x1 + 463.848
        AA122 = 24.015*x1**3 - 222.507*x1**2 + 658.027*x1 - 660.665
        AA123 = -8.56*x1**3 + 79.549*x1**2 - 235.827*x1 + 236.579
        
        AA11 = AA111*x2**2 + AA112*x2 + AA113
        AA12 = AA121*x2**2 + AA122*x2 + AA123
        AA1 = (AA11 * x3 + AA12) * (1.0 - x4) + 1.0
        
        A111 = -0.002222*x1**3 + 0.040871*x1**2 - 0.286866*x1 + 0.599424
        A112 = 0.010185*x1**3 - 0.161176*x1**2 + 0.904989*x1 - 1.641389
        A113 = -0.015422*x1**3 + 0.220371*x1**2 - 1.084987*x1 + 1.834167
        A121 = -0.0628667*x1**4 + 0.4989259*x1**3 + 0.52735*x1**2 - 10.7918672*x1 + 16.616327
        A122 = 0.1140667*x1**4 - 0.8108963*x1**3 - 2.2186833*x1**2 + 25.1269741*x1 - 37.7729778
        A123 = -0.0589333*x1**4 + 0.2639704*x1**3 + 3.1949667*x1**2 - 21.8126569*x1 + 31.4113508
        A124 = 0.0107667*x1**4 + 0.0018704*x1**3 - 1.2494083*x1**2 + 6.9427931*x1 - 10.2018992
        A131 = 0.192207*x1**3 - 2.787462*x1**2 + 12.507855*x1 - 14.764856
        A132 = -0.350563*x1**3 + 5.222348*x1**2 - 23.974852*x1 + 29.007851
        A133 = 0.237096*x1**3 - 3.535062*x1**2 + 16.368376*x1 - 20.539908
        A134 = -0.067119*x1**3 + 0.966362*x1**2 - 4.407535*x1 + 5.894703
        
        A11 = A111*x2**2 + A112*x2 + A113
        A12 = A121*x2**3 + A122*x2**2 + A123*x2 + A124
        A13 = A131*x2**3 + A132*x2**2 + A133*x2 + A134
        
        A1 = (A11 * x4**2 + A12 * x4 + A13) * AA1
        
        # Coeficiente A2
        A2 = -1.402 * x4**3 + 7.189 * x4**2 - 10.993 * x4 + 9.45
        
        # Coeficiente A3
        A31 = -7686.0287*x2**6 + 30131.5678*x2**5 - 49048.9664*x2**4 + 42480.7709*x2**3 - 20665.147*x2**2 + 5355.2035*x2 - 577.8827
        A32 = 61639.9103*x2**6 - 241201.0598*x2**5 + 392579.5937*x2**4 - 340629.4699*x2**3 + 166348.6917*x2**2 - 43358.7938*x2 + 4714.7918
        A33 = -130677.4903*x2**6 + 507996.2604*x2**5 - 826728.7127*x2**4 + 722677.104*x2**3 - 358360.7392*x2**2 + 95501.4948*x2 - 10682.8619
        A34 = -110034.6584*x2**6 + 446051.22*x2**5 - 724186.4643*x2**4 + 599411.9264*x2**3 - 264294.7189*x2**2 + 58039.7328*x2 - 4774.6414
        A35 = 709672.0656*x2**6 - 2803850.2395*x2**5 + 4553780.5017*x2**4 - 3888378.9905*x2**3 + 1839829.259*x2**2 - 457313.6939*x2 + 46600.823
        A36 = -822735.9289*x2**6 + 3238899.7308*x2**5 - 5256636.5472*x2**4 + 4500543.147*x2**3 - 2143487.3508*x2**2 + 538548.1194*x2 - 55751.1528
        A37 = 299122.8727*x2**6 - 1175773.1606*x2**5 + 1907356.1357*x2**4 - 1634256.8172*x2**3 + 780020.9393*x2**2 - 196679.7143*x2 + 20467.0904
        
        AA311 = (-17.102*x2**3 + 41.495*x2**2 - 33.234*x2 + 8.8007)*x4 + 36.566*x2**3 - 89.203*x2**2 + 71.8*x2 - 18.108
        AA32 = -0.0727*x1**2 + 0.7*x1 - 1.2818
        AA31 = (-0.3767*x1**3 + 3.39*x1**2 - 10.356*x1 + 11.588) * AA311
        
        x6_val = x4 - AA32
        AA3 = AA31 * (-1.05584*x6_val**9 + 12.688*x6_val**8 - 63.70534*x6_val**7 + 172.84571*x6_val**6 \
                      - 274.05701*x6_val**5 + 257.68705*x6_val**4 - 141.40915*x6_val**3 + 44.13177*x6_val**2 \
                      - 7.1654*x6_val - 0.0495*x1**2 + 0.4518*x1 - 0.61655)
        
        A3 = A31 * x4**6 + A32 * x4**5 + A33 * x4**4 + A34 * x4**3 + A35 * x4**2 + A36 * x4 + A37 + AA3
        
        if x5 > 1e-6:
            arg_exp = -A2 * ((np.log(x5) - A3)**2) / 1.44
            B44W0_adim = (A1 / x5) * np.exp(arg_exp)
        else:
            B44W0_adim = 0.0
            
        # Corrección por velocidad V > 0
        if V > 0:
            xi_d = max((omega_E**2) * d / g, 1e-6)
            Omega = V * omega_E / g
            
            Aw1 = 1.0 + (xi_d**(-1.2)) * np.exp(-2.0 * xi_d)
            Aw2 = 0.5 + (xi_d**(-1.0)) * np.exp(-2.0 * xi_d)
            
            ratio_W = 0.5 * ((Aw2 + 1.0) + (Aw2 - 1.0) * np.tanh(20.0 * Omega - 0.3) + \
                             (2.0 * Aw1 - Aw2 - 1.0) * np.exp(-150.0 * (Omega - 0.25)**2))
                             
            B44W_adim = B44W0_adim * ratio_W
        else:
            B44W_adim = B44W0_adim
            
        B44W = B44W_adim * factor_ittc
    else:
        B44W = 0.0
        
    B44_total += B44W
    
    # ==========================================
    # 4. BILGE KEEL (Kawahara SIM) - B44BK
    # ==========================================
    if b_BK > 0 and l_BK > 0 and d > 0 and omega_E > 1e-4:
        x1_bk = B / d
        x2_bk = CB
        x3_bk = CM
        x4_bk = OG / d   # Note que aquí Kawahara define x4 distinto que en Wave
        x5_bk = omega_E * np.sqrt(B / (2.0 * g))
        x6_bk = phi_a_deg # En GRADOS
        x7_bk = b_BK / B
        x8_bk = l_BK / L
        
        f1 = (-0.3651 * x2_bk + 0.3907) * (x1_bk - 2.83)**2 - 2.21 * x2_bk + 2.632
        f2 = 0.00255 * x6_bk**2 + 0.122 * x6_bk + 0.4794
        f3 = (-0.8913 * x7_bk**2 - 0.0733 * x7_bk) * x8_bk**2 + (5.2857 * x7_bk**2 - 0.01185 * x7_bk + 0.00189) * x8_bk
        ABK = f1 * f2 * f3
        
        BBK1 = (5.0 * x7_bk + 0.3 * x1_bk - 0.2 * x8_bk + 0.00125 * x6_bk**2 - 0.0425 * x6_bk - 1.86) * x4_bk
        BBK2 = -15.0 * x7_bk + 1.2 * x2_bk - 0.1 * x1_bk - 0.0657 * x4_bk**2 + 0.0586 * x4_bk + 1.6164
        BBK3 = 2.5 * x4_bk + 15.75
        
        if x3_bk > 0:
            B44BK_adim = ABK * np.exp(BBK1 + BBK2 * (x3_bk**BBK3)) * x5_bk
        else:
            B44BK_adim = 0.0
            
        B44BK = B44BK_adim * factor_ittc
    else:
        B44BK = 0.0
        
    B44_total += B44BK
    
    # ==========================================
    # 5. EDDY (Kawahara SIM 2009) - B44E
    # ==========================================
    B44E = 0.0
    if phi_a > 1e-4 and omega_E > 1e-4 and d > 0:
        # Parámetros SIM
        x1_e = B / d
        x2_e = CB
        x3_e = CM
        x4_e = OG / d
        
        omega_hat = omega_E * np.sqrt(B / (2.0 * g))
        
        # Coeficientes
        AE = (-0.0182 * x2_e + 0.0155) * (x1_e - 1.8)**3 - 79.414 * x2_e**4 + 215.695 * x2_e**3 - 215.883 * x2_e**2 + 93.894 * x2_e - 14.848
        
        BE1 = (-0.2 * x1_e + 1.6) * (3.98 * x2_e - 5.1525) * x4_e - ((0.9717 * x2_e**2 - 1.55 * x2_e + 0.723) * x4_e + (0.04567 * x2_e + 0.9408))
        BE2 = (0.25 * x4_e + 0.95) * x4_e - 219.2 * x2_e**3 + 443.7 * x2_e**2 - 283.3 * x2_e + 59.6
        BE3 = (46.5 - 15.0 * x1_e) * x2_e + 11.2 * x1_e - 28.6
        
        CR = AE * np.exp(BE1 + BE2 * (x3_e**BE3))
        
        B44E_adim = (4.0 * omega_hat * phi_a) / (3.0 * np.pi * x2_e * (x1_e**3)) * CR
        
        # Convertir a dimensional
        B44E0 = B44E_adim * factor_ittc
        
        if exportar_eddy_txt:
            import os
            debug_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eddy_debug_SIM.txt")
            with open(debug_path, "w") as f:
                f.write("DEBUG EDDY (Kawahara SIM 2009)\n")
                f.write("="*40 + "\n")
                f.write(f"x1 (B/d):  {x1_e:.4f}\n")
                f.write(f"x2 (CB):   {x2_e:.4f}\n")
                f.write(f"x3 (CM):   {x3_e:.4f}\n")
                f.write(f"x4 (OG/d): {x4_e:.4f}\n")
                f.write(f"AE:        {AE:.4f}\n")
                f.write(f"BE1:       {BE1:.4f}\n")
                f.write(f"BE2:       {BE2:.4f}\n")
                f.write(f"BE3:       {BE3:.4f}\n")
                f.write(f"CR:        {CR:.6f}\n")
                f.write(f"B44E_adim: {B44E_adim:.6f}\n")
            print(f"  [Debug] Exportados valores SIM Eddy a: {debug_path}")
            
        # Corrección V > 0
        if V > 0:
            K_val = omega_E * L / V
            B44E = B44E0 * (0.04 * K_val**2) / (1.0 + 0.04 * K_val**2)
        else:
            B44E = B44E0
            
    B44_total += B44E
    
    return {
        'B44_F': B44F,
        'B44_E': B44E,
        'B44_L': B44L,
        'B44_W': B44W,
        'B44_BK': B44BK,
        'B44_total': B44_total
    }

def evaluar_damping_condicion_unica():
    print("="*60)
    print("CÁLCULO DE AMORTIGUAMIENTO (Condición Única)")
    print("="*60)
    
    kg_val = 5.0
    V_knots = 1.0
    phi_a_deg = 10.0
    omega_E = 0.5
    d_val = 4.7
        
    ship_params = {
        'L_PP': 71.75,
        'B': 14.402,
        'd': d_val,
        'C_B': 0.4698,
        'nabla': 2281.744,
        'C_M': 0.8681,        # calculado desde geometría CAD (calculo_hidrostatico.py)
        'OG': d_val - kg_val, # Convención Ikeda: z_flotación - z_G
        'b_BK': 0.0,          # sin quillas de balance instaladas
        'l_BK': 0.0,
        'rho': 1025.0,
        'nu': 1.19e-6
    }
    

    
    print(f"\nEvaluando amortiguamiento para:")
    print(f" - KG: {kg_val} m")
    print(f" - Amplitud de Rolido (phi_a): {phi_a_deg} grados")
    print(f" - Frecuencia de encuentro (omega_E): {omega_E} rad/s")
    print(f" - Velocidad (V): {V_knots} nudos")
    
    res = calcular_damping_B44(
        phi_a_deg=phi_a_deg, 
        V_knots=V_knots, 
        omega_E=omega_E, 
        ship_params=ship_params,
        exportar_eddy_txt=False
    )
    
    print("\n" + "="*40)
    print("RESULTADOS B44 [kg*m^2/s]:")
    print("="*40)
    print(f"B44_F  (Fricción):   {res['B44_F']:.2f}")
    print(f"B44_E  (Eddy):       {res['B44_E']:.2f}")
    print(f"B44_L  (Lift):       {res['B44_L']:.2f}")
    print(f"B44_W  (Wave):       {res['B44_W']:.2f}")
    print(f"B44_BK (Bilge Keel): {res['B44_BK']:.2f}")
    print("-" * 40)
    print(f"B44_TOTAL:           {res['B44_total']:.2f}")
    print("="*40)

if __name__ == "__main__":
    evaluar_damping_condicion_unica()
