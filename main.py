... import math
... import numpy as np
... import matplotlib.pyplot as plt
... import streamlit as st
... 
... st.set_page_config(page_title="Percentil del Aumento de Peso (OMS)", page_icon="👶", layout="centered")
... 
... # ---------- Estilos suaves ----------
... st.markdown("""
...     <style>
...       .main {background-color: #fafafa;}
...       .metric-card {padding: 18px 20px; border-radius: 16px; background: white; 
...                     box-shadow: 0 2px 14px rgba(0,0,0,.06); border: 1px solid #eee;}
...       .muted {color:#666; font-size:0.9rem;}
...       .title {font-weight: 800; letter-spacing:.2px;}
...       .percentil {font-size: 2.2rem; font-weight: 800;}
...     </style>
... """, unsafe_allow_html=True)
... 
... # ---------- Datos OMS (2006) ----------
... DELTA_G = 400.0  # desplazamiento δ (g) definido por OMS para velocity
... # Niñas: L constante por intervalo
... GIRLS_L = 0.7781
... GIRLS = {
...     1:  (1279.4834, 0.21479),
...     2:  (1411.1075, 0.19384),
...     3:  (1118.0098, 0.19766),
    4:  (984.8825,  0.20995),
    5:  (888.9803,  0.22671),
    6:  (801.3910,  0.24596),
    7:  (744.3023,  0.26515),
    8:  (710.6923,  0.28409),
    9:  (672.6072,  0.30106),
    10: (644.6032,  0.31676),
    11: (633.2166,  0.33208),
    12: (631.7383,  0.34627),
}
# Niños: L varía por intervalo
BOYS = {
    1:  (1.3828, 1423.0783, 0.22048),
    2:  (0.7241, 1596.3470, 0.19296),
    3:  (0.6590, 1215.3989, 0.19591),
    4:  (0.7003, 1017.0488, 0.20965),
    5:  (0.7419,  921.6249, 0.22790),
    6:  (0.7668,  822.1842, 0.24854),
    7:  (0.7688,  756.5306, 0.26783),
    8:  (0.7624,  715.6257, 0.28677),
    9:  (0.7620,  684.7459, 0.30439),
    10: (0.7659,  658.5809, 0.32154),
    11: (0.7713,  643.4374, 0.33882),
    12: (0.7761,  639.4743, 0.35502),
}

# ---------- Utilidades LMS ----------
def normal_cdf(z):
    z_arr = np.asarray(z, dtype=float)
    vec = np.vectorize(lambda t: 0.5 * (1.0 + math.erf(t / math.sqrt(2.0))))
    return vec(z_arr)

def lms_z_from_inc(inc_g: float, L: float, M_shift: float, S: float) -> float:
    x_shift = inc_g + DELTA_G
    return ((x_shift / M_shift) ** L - 1.0) / (L * S)

def lms_inc_from_z(z, L: float, M_shift: float, S: float):
    x_shift = M_shift * (1.0 + L * S * z) ** (1.0 / L)
    return x_shift - DELTA_G

# ---------- UI ----------
st.markdown("<h1 class='title'>👶 Percentil del aumento de peso del último mes (OMS)</h1>", unsafe_allow_html=True)
st.markdown("<p class='muted'>Basado en los Estándares de Crecimiento Infantil de la OMS (2006), <i>1-month weight increments</i>.</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Datos del bebé")
    sex = st.radio("Sexo", ["Niña", "Niño"], horizontal=True)
    end_month = st.slider("Mes final del intervalo", 1, 12, 7,
                          help="Ej.: para evaluar 6→7 meses, selecciona 7.")
    colw1, colw2 = st.columns(2)
    with colw1:
        w_prev = st.number_input("Peso anterior (kg)", min_value=0.0, step=0.01, value=7.00)
    with colw2:
        w_curr = st.number_input("Peso reciente (kg)", min_value=0.0, step=0.01, value=7.30)

    show_details = st.checkbox("Mostrar detalles avanzados (Z-score, mediana)", value=True)
    st.markdown("---")
    st.caption("Nota: para prematuros/as, use **edad corregida** para elegir el intervalo.")

calc = st.button("Calcular percentil")

# ---------- Cálculo y resultados ----------
if calc:
    if w_curr < w_prev:
        st.error("El peso reciente es menor que el anterior. Verifica las entradas.")
        st.stop()

    inc_g = (w_curr - w_prev) * 1000.0

    if sex == "Niña":
        L, (M_shift, S) = GIRLS_L, GIRLS[end_month]
        grupo = "Niña"
    else:
        L, M_shift, S = BOYS[end_month]
        grupo = "Niño"

    z = lms_z_from_inc(inc_g, L, M_shift, S)
    pct = float(normal_cdf(z) * 100.0)
    median_g = M_shift - DELTA_G

    # Tarjeta principal
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='percentil'>Percentil {pct:.1f}</div>", unsafe_allow_html=True)
    st.write(f"**Este mes, el aumento de peso de tu bebé se sitúa en el percentil {pct:.1f}.**")
    st.write(f"Incremento observado: **{inc_g:.0f} g**  ·  Intervalo: **{end_month-1}→{end_month} meses**  ·  {grupo}")
    if show_details:
        st.write(f"Mediana OMS del intervalo: **{median_g:.0f} g**  ·  Z-score (LMS): **{z:.2f}**")
    st.markdown("</div>", unsafe_allow_html=True)

    # Curva CDF
    z_grid = np.linspace(-3.5, 3.5, 400)
    inc_grid = lms_inc_from_z(z_grid, L, M_shift, S)
    cdf_grid = normal_cdf(z_grid) * 100.0

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(inc_grid, cdf_grid, linewidth=2)
    ax.scatter([inc_g], [pct], s=60)
    ax.axvline(inc_g, linestyle="--", linewidth=1)
    ax.axhline(pct, linestyle="--", linewidth=1)
    ax.set_xlabel("Incremento mensual (g)")
    ax.set_ylabel("Frecuencia acumulada (%)")
    ax.set_title(f"{grupo} {end_month-1}→{end_month} meses • Percentil {pct:.1f}")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)

    with st.expander("¿Qué estoy viendo?"):
        st.markdown("""
        - La **línea** muestra la *curva de frecuencia acumulada* (CDF) de la ganancia mensual esperada según la OMS para ese mes.
        - El **punto** marca el incremento de tu bebé y su **percentil exacto**.
        - **Mediana** ≈ percentil 50: la mitad de los bebés gana más que ese valor y la otra mitad, menos.
        - Recomendación clínica: interpretar junto a la **trayectoria** de los últimos meses y consultas pediátricas.
        """)

else:
    st.info("Completa los datos en la barra lateral y pulsa **Calcular percentil**.")

# ---------- Pie ----------
st.markdown("---")
st.caption("Fuentes: WHO Child Growth Standards (2006), Weight velocity standards (1-month increments). "
           "Este material es informativo y no reemplaza la evaluación pediátrica profesional.")

