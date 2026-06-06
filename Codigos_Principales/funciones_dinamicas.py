import numpy as np
import pandas as pd

# Casos de estudio estándar para la tesis / simulación de rolido.
# Cruzan la zona de resonancia natural (T_roll ~ 5.6 - 8.1s) en los id 3 al 8.
CASOS_ESTUDIO = {
    0:  {"ss": "Decay", "desc": "Roll decay libre (sin olas)",   "Hs": 0.0, "Tp": 0.0},
    1:  {"ss": "SS2", "desc": "Marejadilla",       "Hs": 0.8, "Tp": 3.8},
    2:  {"ss": "SS3", "desc": "Marejada ligera",   "Hs": 1.2, "Tp": 4.8},
    3:  {"ss": "SS3", "desc": "Marejada ligera",   "Hs": 1.4, "Tp": 5.5},
    4:  {"ss": "SS3", "desc": "Marejada ligera",   "Hs": 1.6, "Tp": 6.0},
    5:  {"ss": "SS4", "desc": "Marejada moderada", "Hs": 2.2, "Tp": 6.5},
    6:  {"ss": "SS4", "desc": "Marejada moderada", "Hs": 2.5, "Tp": 7.0},
    7:  {"ss": "SS4", "desc": "Marejada moderada", "Hs": 2.8, "Tp": 7.5},
    8:  {"ss": "SS4", "desc": "Marejada moderada", "Hs": 3.0, "Tp": 8.0},
    9:  {"ss": "SS5", "desc": "Mar gruesa",        "Hs": 3.8, "Tp": 9.5},
    10: {"ss": "SS5", "desc": "Mar gruesa",        "Hs": 4.5, "Tp": 10.5},
    11: {"ss": "SS6", "desc": "Mar muy gruesa",    "Hs": 5.5, "Tp": 11.5},
    12: {"ss": "SS6", "desc": "Mar muy gruesa",    "Hs": 6.5, "Tp": 13.0},
}

def S_ITTC(omega, Hs, Tp):
    """
    Calcula la densidad espectral S(omega) basándose en el espectro ITTC 
    (Bretschneider de dos parámetros).
    
    Parámetros:
    -----------
    omega : Array o float. Frecuencia(s) angular(es) en [rad/s]
    Hs    : Float. Altura significativa de ola en [m]
    Tp    : Float. Período pico de ola en [s]
    
    Retorna:
    --------
    S     : Array o float con el valor del espectro en [m^2*s/rad].
    """
    if Hs <= 0 or Tp <= 0:
        return np.zeros_like(omega)
    
    omega_p = 2.0 * np.pi / Tp
    ratio = omega_p / omega
    
    S = (5.0 / 16.0) * (Hs**2) * (omega_p**4) / (omega**5) * np.exp(-1.25 * ratio**4)
    return S

def discretizar_espectro_no_uniforme(
    caso_id=None,
    Hs=None, 
    Tp=None, 
    w_min=0.25, 
    w_max=1.75, 
    w_res_min=0.7, 
    w_res_max=1.2, 
    N_low=50, 
    N_res=100, 
    N_high=50,
    exportar_excel=False,
    ruta_exportacion="espectro_discretizado.xlsx",
    verbose=True,
):
    """
    Discretiza el espectro espacialmente en 3 zonas (baja frecuencia, resonancia, 
    alta frecuencia) basándose en N subdivisiones (como particiones de Riemann).
    
    En lugar de np.linspace (que da problemas en los bordes), calcula el centro 
    exacto de cada partición (omega_i) y le asigna su delta_omega correspondiente.
    Esto permite que la integración numérica y la reconstrucción de la ola sean 
    matemáticamente exactas.
    
    Parámetros:
    -----------
    caso_id          : Entero (1 al 12). Si se provee, ignora Hs y Tp y usa los de CASOS_ESTUDIO.
    Hs, Tp           : Altura significativa [m] y Período pico [s]. (Usados solo si caso_id es None).
    w_min, w_max     : Límites absolutos del dominio [rad/s].
    w_res_min, max   : Límites de la zona de resonancia (donde se necesita máxima resolución).
    N_low            : Número de intervalos para [w_min, w_res_min].
    N_res            : Número de intervalos para [w_res_min, w_res_max].
    N_high           : Número de intervalos para [w_res_max, w_max].
    
    Retorna:
    --------
    df : pandas.DataFrame con las columnas:
         [id_global, zona, id_zona, omega_i, d_omega, S_omega, zeta_a]
    """
    if caso_id is not None:
        if caso_id not in CASOS_ESTUDIO:
            raise ValueError(f"El caso_id {caso_id} no existe en CASOS_ESTUDIO (1 al 12).")
        Hs = CASOS_ESTUDIO[caso_id]["Hs"]
        Tp = CASOS_ESTUDIO[caso_id]["Tp"]
        if verbose:
            print(f"[Info] Utilizando Caso ID {caso_id}: {CASOS_ESTUDIO[caso_id]['ss']} - {CASOS_ESTUDIO[caso_id]['desc']} (Hs={Hs}m, Tp={Tp}s)")
    elif Hs is None or Tp is None:
        raise ValueError("Se debe proveer explícitamente caso_id, o bien Hs y Tp.")
    
    # 1. Zona Baja Frecuencia
    L_low = w_res_min - w_min
    dw_low = L_low / N_low if N_low > 0 else 0
    w_low = [w_min + dw_low/2.0 + i*dw_low for i in range(N_low)]
    
    # 2. Zona de Resonancia
    L_res = w_res_max - w_res_min
    dw_res = L_res / N_res if N_res > 0 else 0
    w_res = [w_res_min + dw_res/2.0 + i*dw_res for i in range(N_res)]
    
    # 3. Zona Alta Frecuencia
    L_high = w_max - w_res_max
    dw_high = L_high / N_high if N_high > 0 else 0
    w_high = [w_res_max + dw_high/2.0 + i*dw_high for i in range(N_high)]
    
    # Ensamblar vectores globales
    omegas = np.array(w_low + w_res + w_high)
    d_omegas = np.array([dw_low]*N_low + [dw_res]*N_res + [dw_high]*N_high)
    zonas = ['Baja_Freq']*N_low + ['Resonancia']*N_res + ['Alta_Freq']*N_high
    IDs_zona = list(range(N_low)) + list(range(N_res)) + list(range(N_high))
    
    # Calcular espectro y amplitudes
    S_vals = S_ITTC(omegas, Hs, Tp)
    
    # Amplitud de ola unitaria para cada componente i: zeta_ai = sqrt( 2 * S(w) * dw )
    zeta_vals = np.sqrt(2.0 * S_vals * d_omegas)
    
    # Armar un DataFrame ordenado para devolver/exportar
    df = pd.DataFrame({
        'id_global': np.arange(len(omegas)), # i global de 0 a (N_tot - 1)
        'zona'     : zonas,                  # Etiqueta en texto
        'id_zona'  : IDs_zona,               # El entero (id) dentro de su propia zona
        'omega_i'  : omegas,                 # La frecuencia representativa [rad/s]
        'd_omega'  : d_omegas,               # El ancho del paso para ese intervalo
        'S_omega'  : S_vals,                 # Valor de densidad espectral
        'zeta_a'   : zeta_vals               # Amplitud de la ola para construir eta(t) [m]
    })
    
    if exportar_excel:
        df.to_excel(ruta_exportacion, index=False)
        print(f"[OK] Espectro discretizado guardado en: {ruta_exportacion}")
        
    return df

if __name__ == "__main__":
    # Prueba del script directamente
    # Configuración de prueba llamando directly por id de caso (ej: Caso 6)
    print("Probando función de discretización ITTC no uniforme con caso parametrizado...")
    caso_prueba = 6
    
    df_prueba = discretizar_espectro_no_uniforme(
        caso_id=caso_prueba,
        w_min=0.1, w_max=2.5,
        w_res_min=0.7, w_res_max=1.15,
        N_low=30, N_res=150, N_high=40,
        exportar_excel=True
    )
    
    print("\nMuestra de la zona de Baja Frecuencia (primeros 3 del id_zona):")
    print(df_prueba[df_prueba['zona'] == 'Baja_Freq'].head(3)[['id_zona', 'omega_i', 'd_omega', 'S_omega']])
    
    print("\nMuestra de la zona de Resonancia (primeros 3 del id_zona):")
    print(df_prueba[df_prueba['zona'] == 'Resonancia'].head(3)[['id_zona', 'omega_i', 'd_omega', 'S_omega']])
    
    # Integración para comprobar la energía:
    m0_numerico = np.sum(df_prueba['S_omega'] * df_prueba['d_omega'])
    Hs_chequeo = 4 * np.sqrt(m0_numerico)
    
    Hs_original = CASOS_ESTUDIO[caso_prueba]['Hs']
    print(f"\nChequeo de Energía Conservada para el Caso {caso_prueba}:")
    print(f"  Hs original  : {Hs_original:.4f} m")
    print(f"  Hs integrado : {Hs_chequeo:.4f} m (Debido a ser sumatoria Riemann exacta)")
