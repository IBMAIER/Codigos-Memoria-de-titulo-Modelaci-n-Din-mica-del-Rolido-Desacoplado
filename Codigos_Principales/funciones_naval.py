"""
MÓDULO DE CÁLCULO HIDROSTÁTICO NAVAL (funciones_naval.py)

Contiene las funciones matemáticas y geométricas principales para procesar las cuadernas
de un buque y extraer sus propiedades físicas. Este módulo actúa como el motor,
provee algoritmos como:
- Rotación de planos de flotura (escora).
- Recorte de polígonos 2D (algoritmos matemáticos y Sutherland-Hodgman).
- Cálculo de áreas y perímetros mojados (Teorema de Green / Shoelace).
- Integración longitudinal a lo largo de la eslora (Regla de Simpson).

Es importado como librería principal por `solver_GZ.py` y calculadoras estáticas.
"""
import pandas as pd
import numpy as np
from scipy.integrate import simpson

# ==========================================================
# 1. LECTURA Y PREPROCESAMIENTO DE SECCIONES (ORDENAMIENTO)
# ==========================================================
def cargar_y_ordenar_secciones(filepath):
    """
    Carga el CSV validado y se asegura de que los puntos de cada sección
    estén ordenados consistentemente (cerrando el polígono).
    Se asume que el archivo 'secciones_contorno_exterior.csv' ya fue
    procesado por 'plotea_exterior.py' y contiene un único contorno simple.
    """
    df = pd.read_csv(filepath, sep=';', decimal=',')
    valores_x = sorted(df["X"].unique())
    
    secciones = {}
    for x_val in valores_x:
        df_sec = df[df["X"] == x_val].copy()
        # Asegurarse de ordenar por PointIndex si existe
        if "PointIndex" in df_sec.columns:
            df_sec = df_sec.sort_values("PointIndex")
            
        pts = df_sec[["Y", "Z"]].to_numpy()
        
        if len(pts) < 3:
            continue
            
        # Cerrar el polígono si no lo está (último punto igual al primero)
        if not np.allclose(pts[0], pts[-1], atol=1e-5):
            pts = np.vstack((pts, pts[0]))
            
        secciones[x_val] = pts
        
    return secciones

# ==========================================================
# 2. DEFINICIÓN DEL PLANO DE CORTE (LÍNEA DE FLOTACIÓN)
# ==========================================================
def evaluar_plano(punto, plano_z, escora_rad=0.0):
    """
    Evalúa si un punto (y, z) se encuentra sumergido.
    Para escora=0, el plano es Z = plano_z.
    f(p) = Z(p)*cos(ang) + Y(p)*sin(ang) - plano_z.
    Para escora != 0, el plano rota en torno a Y=0 (u otro eje transversal).
    """
    y, z = punto
    # Ecuación de plano rotado: Z * cos(ang) + Y * sin(ang) - Z0 = 0
    f_p = z * np.cos(escora_rad) + y * np.sin(escora_rad) - plano_z
    return f_p <= 1e-9  # Tolerancia numérica

def interpolar_interseccion(p1, p2, plano_z, escora_rad=0.0):
    """
    Calcula el punto de intersección de un segmento con el plano rotado.
    """
    y1, z1 = p1
    y2, z2 = p2
    
    f1 = z1 * np.cos(escora_rad) + y1 * np.sin(escora_rad) - plano_z
    f2 = z2 * np.cos(escora_rad) + y2 * np.sin(escora_rad) - plano_z
    
    # Si f1 y f2 son del mismo signo no hay intersección,
    # y si f1 == f2 el segmento es paraleo al plano
    denominador = f2 - f1
    if abs(denominador) < 1e-9:
        return None 
        
    t = -f1 / denominador
    
    # Validar que t esté entre 0 y 1
    if not (0.0 <= t <= 1.0):
        return None
        
    y_int = y1 + t * (y2 - y1)
    z_int = z1 + t * (z2 - z1)
    
    return np.array([y_int, z_int])

# ==========================================================
# 3. RECORTE POLIGONAL (Sutherland-Hodgman)
# ==========================================================
def recortar_poligono(poligono, plano_z, escora_rad=0.0):
    """
    Aplica el algoritmo de Sutherland-Hodgman para recortar el polígono
    dejando solo la parte sumergida (Z_rotado <= plano_z).
    """
    sumergido = []
    
    for i in range(len(poligono) - 1):
        p_actual = poligono[i]
        p_siguiente = poligono[i+1]
        
        actual_sumergido = evaluar_plano(p_actual, plano_z, escora_rad)
        siguiente_sumergido = evaluar_plano(p_siguiente, plano_z, escora_rad)
        
        if actual_sumergido and siguiente_sumergido:
            # Caso 2: Ambos sumergidos -> agregar el siguiente
            sumergido.append(p_siguiente)
        elif actual_sumergido and not siguiente_sumergido:
            # Caso 3: Sale del agua -> calcular intersección
            p_int = interpolar_interseccion(p_actual, p_siguiente, plano_z, escora_rad)
            if p_int is not None:
                sumergido.append(p_int)
        elif not actual_sumergido and siguiente_sumergido:
            # Caso 4: Entra al agua -> calcular intersección y luego el punto
            p_int = interpolar_interseccion(p_actual, p_siguiente, plano_z, escora_rad)
            if p_int is not None:
                sumergido.append(p_int)
            sumergido.append(p_siguiente)
        # Caso 1: Ambos fuera del agua -> No hacer nada
            
    # Si resultó un polígono válido, nos aseguramos que esté cerrado cerrándolo con el primer punto si hace falta
    if len(sumergido) > 2:
        sum_arr = np.array(sumergido)
        if not np.allclose(sum_arr[0], sum_arr[-1], atol=1e-5):
            sum_arr = np.vstack((sum_arr, sum_arr[0]))
        return sum_arr
    return np.array([])

# ==========================================================
# 4. CÁLCULO DE ÁREA Y CENTROIDE (SHOELACE / GREEN)
# ==========================================================
def calcular_area_y_centroide(poligono):
    """
    Calcula el área y el centroide (Y_c, Z_c) de un polígono cerrado (sentido consistente).
    Basado en el Teorema de Green (Shoelace).
    """
    if len(poligono) < 4:  # Menos de 3 puntos únicos (+ cierre)
        return 0.0, 0.0, 0.0
        
    y = poligono[:, 0]
    z = poligono[:, 1]
    
    # Productos cruzados para el área
    cross = y[:-1] * z[1:] - y[1:] * z[:-1]
    
    area = 0.5 * np.sum(cross)
    
    if abs(area) < 1e-9:
        return 0.0, 0.0, 0.0
        
    # Ecuaciones para el centroide
    y_c = (1.0 / (6.0 * area)) * np.sum((y[:-1] + y[1:]) * cross)
    z_c = (1.0 / (6.0 * area)) * np.sum((z[:-1] + z[1:]) * cross)
    
    # El área podría ser negativa dependiendo del sentido de ordenamiento, devolvemos valor absoluto
    return abs(area), y_c, z_c

# ==========================================================
# 5. INTEGRACIÓN LONGITUDINAL Y PARÁMETROS HIDROSTÁTICOS
# ==========================================================
def calcular_parametros_hidrostaticos(secciones, plano_z, escora_rad=0.0, solo_volumen=False):
    """
    Procesa todas las secciones, recorta, calcula áreas locales e integra longitudinalmente.
    Añade el cálculo de la inercia transversal (I_T) del plano de flotación y superficie mojada (S).
    Si solo_volumen=True, omite cálculos detallados para optimizar iteraciones.
    """
    x_vals = []
    areas = []

    # Sección Maestra: sección transversal de mayor área sumergida (phi=0)
    x_master  = None
    A_master  = 0.0
    
    if not solo_volumen:
        y_cs = []
        z_cs = []
        b_wps = []
        i_ts = []
        p_ss = []
    
    for x_val in sorted(secciones.keys()):
        poligono = secciones[x_val]
        # Aplica la escora al crear el plano de flotación inclinado localmente
        poligono_sumergido = recortar_poligono(poligono, plano_z, escora_rad)
        
        area, y_c, z_c = calcular_area_y_centroide(poligono_sumergido)
        
        x_vals.append(x_val)
        areas.append(area)

        # Registrar sección de mayor área (candidata a Sección Maestra)
        if area > A_master:
            A_master = area
            x_master = x_val
        
        if solo_volumen:
            continue
            
        y_cs.append(y_c if area > 0 else 0.0)
        z_cs.append(z_c if area > 0 else 0.0)
        
        # Cálculo del perímetro mojado local (p_s) excluyendo la "tapa" de agua
        if len(poligono_sumergido) > 2:
            p_s_local = 0.0
            for i in range(len(poligono_sumergido) - 1):
                p1 = poligono_sumergido[i]
                p2 = poligono_sumergido[i+1]
                
                f_p1 = p1[1] * np.cos(escora_rad) + p1[0] * np.sin(escora_rad) - plano_z
                f_p2 = p2[1] * np.cos(escora_rad) + p2[0] * np.sin(escora_rad) - plano_z
                
                if abs(f_p1) < 1e-4 and abs(f_p2) < 1e-4:
                    continue
                    
                dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                p_s_local += dist
            p_ss.append(p_s_local)
        else:
            p_ss.append(0.0)
        
        # Extraer puntos que se encuentran sobre el plano de flotación
        y_sup = []
        z_sup = []
        for p in poligono_sumergido:
            f_p = p[1] * np.cos(escora_rad) + p[0] * np.sin(escora_rad) - plano_z
            if abs(f_p) < 1e-4:  # Usar una ventana ligeramente más amplia para absorber redondeo
                y_sup.append(p[0])
                z_sup.append(p[1])
                
        if len(y_sup) >= 2:
            y_max, y_min = np.max(y_sup), np.min(y_sup)
            z_max, z_min = np.max(z_sup), np.min(z_sup)
            
            # Distancia sobre el plano sumergido transversal (Manga local)
            ds = np.sqrt((y_max - y_min)**2 + (z_max - z_min)**2)
            b_wps.append(ds)
            
            # Geometría: Integral paramétrica de y^2 sobre el segmento recto ds
            # Permite calcular I_T_local independientemente del ángulo de rotación
            if abs(y_max - y_min) > 1e-6:
                i_t_local = ds * (y_max**2 + y_max*y_min + y_min**2) / 3.0
            else:
                i_t_local = ds * (y_max**2)
                
            i_ts.append(i_t_local)
        else:
            b_wps.append(0.0)
            i_ts.append(0.0)
        
    x_vals = np.array(x_vals)
    areas = np.array(areas)
    
    # Volumen desplazado (Integración del área local A(x) respecto de x)
    Volumen = simpson(areas, x=x_vals) if len(x_vals) > 2 else 0.0
    
    if solo_volumen:
        return {"Volumen": Volumen}
        
    y_cs = np.array(y_cs)
    z_cs = np.array(z_cs)
    b_wps = np.array(b_wps)
    i_ts = np.array(i_ts)
    p_ss = np.array(p_ss)
    
    # Integración de momentos para Centro de Carena Global (B)
    if Volumen > 0:
        L_CB = simpson(areas * x_vals, x=x_vals) / Volumen   # X_B
        T_CB = simpson(areas * y_cs, x=x_vals) / Volumen     # Y_B (debería ser cercano a 0 para escora 0)
        V_CB = simpson(areas * z_cs, x=x_vals) / Volumen     # Z_B (KB = Height of Buoyancy)
    else:
        L_CB, T_CB, V_CB = 0.0, 0.0, 0.0
        
    # Integración del Área del Plano de Agua (A_wp), Inercia Transversal (I_T) y Superficie Mojada (S)
    if len(x_vals) > 2:
        A_wp = simpson(b_wps, x=x_vals)
        I_T = simpson(i_ts, x=x_vals)
        S = simpson(p_ss, x=x_vals)
    else:
        A_wp, I_T, S = 0.0, 0.0, 0.0
        
    BM_T = I_T / Volumen if Volumen > 0 else 0.0
        
    return {
        "Volumen": Volumen, # Volumen desplazado sumergido total (m³)
        "LCB": L_CB,        # Centro de Carena Longitudinal (Longitudinal Center of Buoyancy) (m)
        "TCB": T_CB,        # Centro de Carena Transversal (Transverse Center of Buoyancy) (m)
        "ZB": V_CB,         # Centro de Carena Vertical (Vertical Center of Buoyancy o KB) (m)
        "A_wp": A_wp,       # Área del plano de agua / flotación (Waterplane Area) (m²)
        "I_T": I_T,         # Momento de Inercia Transversal del plano de agua (m⁴)
        "BM_T": BM_T,       # Radio metacéntrico transversal (Distancia de B a M) (m)
        "S": S,             # Superficie Mojada total (m²)
        "x_vals": x_vals,   # (Array) Posiciones longitudinales X de cada sección procesada
        "areas": areas,     # (Array) Áreas sumergidas de cada sección transversal (m²)
        "b_wps": b_wps,     # (Array) Mangas locales en el plano de agua por sección (m)
        "i_ts": i_ts,       # (Array) Inercias transversales locales del plano de agua por sección (m⁴)
        "p_ss": p_ss,       # (Array) Perímetros mojados locales de cada sección (m)
        "A_master": A_master, # Área de la Sección Maestra (m²) — máxima área transversal sumergida
        "x_master": x_master, # Posición longitudinal X de la Sección Maestra (mm en coordenadas CAD)
    }
