import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(_SCRIPT_DIR, "estudio_parametrico_rolido_ALETAS.xlsx")

k_aleta_target = None
if len(sys.argv) > 1:
    k_aleta_target = float(sys.argv[1])
    if len(sys.argv) > 2:
        OUT_DIR = os.path.join(_SCRIPT_DIR, sys.argv[2])
    else:
        OUT_DIR = os.path.join(_SCRIPT_DIR, f"reporte_parametrico_aletas_K_{int(k_aleta_target)}")
else:
    OUT_DIR = os.path.join(_SCRIPT_DIR, "reporte_parametrico_aletas")
os.makedirs(OUT_DIR, exist_ok=True)

def generar_informe_y_graficos():
    print(f"Cargando datos desde {FILE_PATH}...")
    df_raw = pd.read_excel(FILE_PATH, sheet_name="Resultados")
    df_res = pd.read_excel(FILE_PATH, sheet_name="Resumen_semillas")

    if k_aleta_target is not None:
        df_raw = df_raw[df_raw['K_aleta'] == k_aleta_target].copy()
        df_res = df_res[df_res['K_aleta'] == k_aleta_target].copy()

    # Clasificación de severidad basada en promedio de semillas
    df_res['severidad'] = pd.cut(
        df_res['phi_max_deg_max'], 
        bins=[-np.inf, 15, 30, np.inf], 
        labels=['Operacional (<15°)', 'Severa (15°-30°)', 'Extrema (>=30°)']
    )

    df_res['Tp_over_Tn_bin'] = pd.cut(
        df_res['Tp_over_Tn_mean'], 
        bins=np.arange(0.4, 2.2, 0.2), 
        labels=[f"{x:.1f}" for x in np.arange(0.5, 2.1, 0.2)]
    )

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # === REPORTE DE TEXTO ===
    informe_path = os.path.join(OUT_DIR, "informe_ejecutivo.txt")
    with open(informe_path, "w", encoding="utf-8") as f:
        f.write("=========================================================\n")
        f.write("INFORME DE COMPORTAMIENTO DINÁMICO AL ROLIDO (LÍNEA BASE)\n")
        f.write("=========================================================\n\n")

        f.write("1. DOMINIO DE VALIDEZ DEL MODELO\n")
        f.write("--------------------------------\n")
        f.write("El modelo dinámico no lineal desarrollado se considera confiable y representativo\n")
        f.write("para amplitudes de rolido de hasta aproximadamente 30° a 35°. En este régimen,\n")
        f.write("la formulación hidrodinámica y el modelo de amortiguamiento mantienen validez.\n")
        f.write("Para amplitudes superiores (Región Extrema), aparecen fenómenos físicos no\n")
        f.write("modelados tales como efectos de supervivencia, embarque de agua en cubierta,\n")
        f.write("impactos severos no lineales y pérdida extrema de estabilidad.\n\n")
        f.write("La aparición de grandes amplitudes en el estudio NO invalida el modelo;\n")
        f.write("por el contrario, demuestra que el modelo captura exitosamente la pérdida\n")
        f.write("progresiva de capacidad restauradora (no linealidad de GZ) y la amplificación\n")
        f.write("dinámica en mares severos.\n\n")

        f.write("2. ESTADÍSTICAS GLOBALES ROBUSTAS (Promediadas sobre semillas)\n")
        f.write("------------------------------------------------------------\n")
        phi_rms_global = df_res['phi_rms_deg_mean'].mean()
        phi_13_global = df_res['phi_13_deg_mean'].mean()
        phi_13_p95 = np.percentile(df_res['phi_13_deg_mean'].dropna(), 95)
        phi_dot_rms_global = df_res['phi_dot_rms_deg_s_mean'].mean()
        
        f.write(f"- Promedio Global phi_rms:       {phi_rms_global:.2f}°\n")
        f.write(f"- Promedio Global phi_1/3:       {phi_13_global:.2f}°\n")
        f.write(f"- Percentil 95 de phi_1/3:       {phi_13_p95:.2f}° (Eventos Típicos Severos)\n")
        f.write(f"- Promedio Global phi_dot_rms:   {phi_dot_rms_global:.2f}°/s\n\n")

        f.write("3. CLASIFICACIÓN DE SEVERIDAD (Según Escora Máxima)\n")
        f.write("------------------------------------------------------------\n")
        conteo = df_res['severidad'].value_counts()
        total = len(df_res)
        f.write(f"De los {total} casos consolidados:\n")
        f.write(f"- Región Operacional (phi < 15°):    {conteo.get('Operacional (<15°)', 0)} casos ({(conteo.get('Operacional (<15°)', 0)/total)*100:.1f}%)\n")
        f.write(f"- Región Severa (15° <= phi < 30°):  {conteo.get('Severa (15°-30°)', 0)} casos ({(conteo.get('Severa (15°-30°)', 0)/total)*100:.1f}%)\n")
        f.write(f"- Región Extrema (phi >= 30°):       {conteo.get('Extrema (>=30°)', 0)} casos ({(conteo.get('Extrema (>=30°)', 0)/total)*100:.1f}%)\n\n")
        
        f.write("INTERPRETACIÓN DE EVENTOS EXTREMOS:\n")
        f.write("En los casos clasificados como 'Extrema (>=30°)', el modelo predice amplitudes de\n")
        f.write("rolido extremadamente elevadas, indicando posible pérdida de operabilidad y\n")
        f.write("entrada en una región dinámica donde la formulación utilizada deja de ser\n")
        f.write("representativa. Estos puntos sirven como detectores de condiciones críticas\n")
        f.write("potenciales y no deben ser interpretados literalmente como volcamiento.\n")

    # Función auxiliar para marcar zona de validez
    def marcar_limite_validez(ax, max_val):
        limite = 30.0
        if max_val > limite:
            ax.axhspan(limite, max_val * 1.05, color='red', alpha=0.1, label='Región Extrema (Fuera de Validez)')
            ax.axhline(limite, color='red', linestyle='--', alpha=0.5)

    print(f"Generando los gráficos actualizados en '{OUT_DIR}'...")

    # 1. phi_rms vs Tp_over_Tn
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(data=df_res, x="Tp_over_Tn_mean", y="phi_rms_deg_mean", hue="V_knots", 
                    palette="coolwarm", alpha=0.8, edgecolor=None)
    sns.lineplot(data=df_res, x="Tp_over_Tn_mean", y="phi_rms_deg_mean", color="k", alpha=0.3, errorbar=None)
    plt.axvline(1.0, color='red', linestyle='--', alpha=0.6, label='Resonancia (Tp/Tn = 1)')
    marcar_limite_validez(ax, df_res["phi_rms_deg_mean"].max())
    plt.title("Identificación de Resonancia: Rolido RMS vs Relación de Períodos (Tp/Tn)")
    plt.xlabel("Relación de Períodos Tp / Tn [-]")
    plt.ylabel("Rolido RMS $\\phi_{rms}$ [°]")
    plt.legend(title="Velocidad [kn]")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_phi_rms_vs_Tp_Tn.png"), dpi=300)
    plt.close()

    # 2. phi_rms vs velocidad
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df_res, x="V_knots", y="phi_rms_deg_mean", hue="Hs_m", palette="viridis", marker="o", errorbar=None)
    marcar_limite_validez(ax, df_res["phi_rms_deg_mean"].max())
    plt.title("Sensibilidad Operacional: Rolido RMS vs Velocidad")
    plt.xlabel("Velocidad de Avance [nudos]")
    plt.ylabel("Rolido RMS $\\phi_{rms}$ [°]")
    plt.legend(title="Altura Sig. Hs [m]", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_phi_rms_vs_velocidad.png"), dpi=300)
    plt.close()

    # 3. phi_rms_over_Hs vs velocidad
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_res, x="V_knots", y="phi_rms_over_Hs_mean", hue="Tp_s", palette="magma", marker="s", errorbar=None)
    plt.title("Eficiencia Dinámica: $\\phi_{rms}$ / Hs vs Velocidad")
    plt.xlabel("Velocidad de Avance [nudos]")
    plt.ylabel("Respuesta Normalizada $\\phi_{rms}$ / Hs [°/m]")
    plt.legend(title="Período Pico Tp [s]", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_phi_rms_over_Hs_vs_velocidad.png"), dpi=300)
    plt.close()

    # 4. phi_max vs Hs
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df_res, x="Hs_m", y="phi_max_deg_max", hue="V_knots", palette="coolwarm", marker="D", errorbar=None)
    marcar_limite_validez(ax, df_res["phi_max_deg_max"].max())
    plt.title("Detección de Condiciones Críticas: Rolido Máximo Absoluto vs Altura de Ola")
    plt.xlabel("Altura Significativa Hs [m]")
    plt.ylabel("Rolido Máximo Absoluto $\\phi_{max}$ [°]")
    plt.legend(title="Velocidad [kn]")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_phi_max_vs_Hs.png"), dpi=300)
    plt.close()

    # 5. phi_dot_rms vs velocidad
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_res, x="V_knots", y="phi_dot_rms_deg_s_mean", hue="Hs_m", palette="plasma", marker="^", errorbar=None)
    plt.title("Agresividad Dinámica: Velocidad Angular RMS vs Velocidad de Avance")
    plt.xlabel("Velocidad de Avance [nudos]")
    plt.ylabel("Velocidad Angular RMS $\\dot{\\phi}_{rms}$ [°/s]")
    plt.legend(title="Altura Sig. Hs [m]", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_phi_dot_rms_vs_velocidad.png"), dpi=300)
    plt.close()

    # 6. phi_rms vs KG
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df_res, x="KG_m", y="phi_rms_deg_mean", hue="Hs_m", palette="Set1", marker="o", errorbar=None)
    marcar_limite_validez(ax, df_res["phi_rms_deg_mean"].max())
    plt.title("Influencia de la Estabilidad: Rolido RMS vs Altura de Centro de Gravedad")
    plt.xlabel("Altura del Centro de Gravedad KG [m]")
    plt.ylabel("Rolido RMS $\\phi_{rms}$ [°]")
    plt.legend(title="Altura Sig. Hs [m]", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_phi_rms_vs_KG.png"), dpi=300)
    plt.close()

    # 7. phi_rms vs k44
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df_res, x="k44_m", y="phi_rms_deg_mean", hue="Hs_m", palette="Set2", marker="o", errorbar=None)
    marcar_limite_validez(ax, df_res["phi_rms_deg_mean"].max())
    plt.title("Influencia de la Inercia: Rolido RMS vs Radio de Giro en Rolido")
    plt.xlabel("Radio de Giro k44 [m]")
    plt.ylabel("Rolido RMS $\\phi_{rms}$ [°]")
    plt.legend(title="Altura Sig. Hs [m]", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "07_phi_rms_vs_k44.png"), dpi=300)
    plt.close()

    # 8. Heatmap: Velocidad vs Tp/Tn
    pivot_v_tp = df_res.pivot_table(index="V_knots", columns="Tp_over_Tn_bin", 
                                    values="phi_rms_deg_mean", aggfunc="mean", observed=False)
    plt.figure(figsize=(10, 5))
    # Saturamos el color a 30 grados para indicar el límite de validez
    sns.heatmap(pivot_v_tp, cmap="YlOrRd", annot=True, fmt=".1f", cbar_kws={'label': '$\\phi_{rms}$ [°]'}, vmax=30.0)
    plt.title("Zonas Críticas de Operación: Velocidad vs Tp/Tn\n(Región en rojo oscuro indica límite operacional o extremo)")
    plt.xlabel("Relación de Períodos Tp / Tn [-]")
    plt.ylabel("Velocidad [nudos]")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "08_heatmap_V_vs_TpTn.png"), dpi=300)
    plt.close()

    # 9. Heatmap: KG vs Tp/Tn
    pivot_kg_tp = df_res.pivot_table(index="KG_m", columns="Tp_over_Tn_bin", 
                                     values="phi_rms_deg_mean", aggfunc="mean", observed=False)
    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot_kg_tp, cmap="YlOrRd", annot=True, fmt=".1f", cbar_kws={'label': '$\\phi_{rms}$ [°]'}, vmax=30.0)
    plt.title("Sensibilidad a la Estabilidad: KG vs Tp/Tn\n(Región en rojo oscuro indica límite operacional o extremo)")
    plt.xlabel("Relación de Períodos Tp / Tn [-]")
    plt.ylabel("KG [m]")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "09_heatmap_KG_vs_TpTn.png"), dpi=300)
    plt.close()

    # =========================================================================
    # 10. phi_1/3 vs Hs (Comportamiento severo típico / Operacional)
    # =========================================================================
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df_res, x="Hs_m", y="phi_13_deg_mean", hue="V_knots", palette="coolwarm", marker="o", errorbar=None)
    marcar_limite_validez(ax, df_res["phi_13_deg_mean"].max())
    plt.title("Comportamiento Severo Típico: Escora Significativa $\\phi_{1/3}$ vs Altura de Ola")
    plt.xlabel("Altura Significativa Hs [m]")
    plt.ylabel("Escora Significativa $\\phi_{1/3}$ [°]")
    plt.legend(title="Velocidad [kn]")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "10_phi_13_vs_Hs.png"), dpi=300)
    plt.close()

    print(f"\n¡Informe semántico y gráficos exportados en: {OUT_DIR}")
    
    # Exportar copia de los datos consolidados (promedio de semillas) a la carpeta de reporte final
    resumen_path = os.path.join(OUT_DIR, "datos_resumen_semillas.xlsx")
    df_res.to_excel(resumen_path, index=False)
    print(f"Copia de los datos promediados guardada en: {resumen_path}")

if __name__ == "__main__":
    generar_informe_y_graficos()
