import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="OMS: Crecimiento (unificado)", page_icon="👶", layout="centered")

# ---------- Estilos ----------
st.markdown(
    """
    <style>
      .main {background-color: #fafafa;}
      .metric-card {padding: 18px 20px; border-radius: 16px; background: white;
                    box-shadow: 0 2px 14px rgba(0,0,0,.06); border: 1px solid #eee;}
      .muted {color:#666; font-size:0.9rem;}
      .title {font-weight: 800; letter-spacing:.2px;}
      .percentil {font-size: 2.2rem; font-weight: 800;}
      .small {font-size:.9rem;color:#666;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Utilidades ----------
def normal_cdf(z):
    z_arr = np.asarray(z, dtype=float)
    vec = np.vectorize(lambda t: 0.5 * (1.0 + math.erf(t / math.sqrt(2.0))))
    return vec(z_arr)

# ---------- (1) AUMENTO MENSUAL: parámetros OMS (1-month) ----------
DELTA_G = 400.0  # desplazamiento δ (g)
# Niñas: L constante
GIRLS_L_V = 0.7781
GIRLS_V = {
    1:  (1279.4834, 0.21479), 2:  (1411.1075, 0.19384), 3:  (1118.0098, 0.19766),
    4:  (984.8825,  0.20995), 5:  (888.9803,  0.22671), 6:  (801.3910,  0.24596),
    7:  (744.3023,  0.26515), 8:  (710.6923,  0.28409), 9:  (672.6072,  0.30106),
    10: (644.6032,  0.31676), 11: (633.2166,  0.33208), 12: (631.7383,  0.34627),
}
# Niños: L varía
BOYS_V = {
    1:(1.3828,1423.0783,0.22048), 2:(0.7241,1596.3470,0.19296), 3:(0.6590,1215.3989,0.19591),
    4:(0.7003,1017.0488,0.20965), 5:(0.7419, 921.6249,0.22790), 6:(0.7668, 822.1842,0.24854),
    7:(0.7688, 756.5306,0.26783), 8:(0.7624, 715.6257,0.28677), 9:(0.7620, 684.7459,0.30439),
    10:(0.7659, 658.5809,0.32154),11:(0.7713, 643.4374,0.33882),12:(0.7761, 639.4743,0.35502),
}

def lms_z_from_inc(inc_g: float, L: float, M_shift: float, S: float) -> float:
    x_shift = inc_g + DELTA_G
    return ((x_shift / M_shift) ** L - 1.0) / (L * S)

def lms_inc_from_z(z, L: float, M_shift: float, S: float):
    x_shift = M_shift * (1.0 + L * S * z) ** (1.0 / L)
    return x_shift - DELTA_G

# ---------- (2) Peso-para-edad (0–24 m) ----------
# Estructura: month -> (L, M, S)
GIRLS_WFA = {
     0:(0.3809,3.2322,0.14171), 1:(0.1714,4.1873,0.13724), 2:(0.0962,5.1282,0.13000),
     3:(0.0402,5.8458,0.12619), 4:(-0.0050,6.4237,0.12402), 5:(-0.0430,6.8985,0.12274),
     6:(-0.0756,7.2970,0.12204), 7:(-0.1039,7.6422,0.12178), 8:(-0.1288,7.9487,0.12181),
     9:(-0.1507,8.2254,0.12199),10:(-0.1700,8.4800,0.12223),11:(-0.1872,8.7192,0.12247),
    12:(-0.2024,8.9481,0.12268),13:(-0.2158,9.1699,0.12283),14:(-0.2278,9.3870,0.12294),
    15:(-0.2384,9.6008,0.12299),16:(-0.2478,9.8124,0.12303),17:(-0.2562,10.0226,0.12306),
    18:(-0.2637,10.2315,0.12309),19:(-0.2703,10.4393,0.12315),20:(-0.2762,10.6464,0.12323),
    21:(-0.2815,10.8534,0.12335),22:(-0.2862,11.0608,0.12350),23:(-0.2903,11.2688,0.12369),
    24:(-0.2941,11.4775,0.12390),
}
BOYS_WFA = {
     0:(0.3487,3.3464,0.14602), 1:(0.2297,4.4709,0.13395), 2:(0.1970,5.5675,0.12385),
     3:(0.1738,6.3762,0.11727), 4:(0.1553,7.0023,0.11316), 5:(0.1395,7.5105,0.11080),
     6:(0.1257,7.9340,0.10958), 7:(0.1134,8.2970,0.10902), 8:(0.1021,8.6151,0.10882),
     9:(0.0917,8.9014,0.10881),10:(0.0820,9.1649,0.10891),11:(0.0730,9.4122,0.10906),
    12:(0.0644,9.6479,0.10925),13:(0.0563,9.8749,0.10949),14:(0.0487,10.0953,0.10976),
    15:(0.0413,10.3108,0.11007),16:(0.0343,10.5228,0.11041),17:(0.0275,10.7319,0.11079),
    18:(0.0211,10.9385,0.11119),19:(0.0148,11.1430,0.11164),20:(0.0087,11.3462,0.11211),
    21:(0.0029,11.5486,0.11261),22:(-0.0028,11.7504,0.11314),23:(-0.0083,11.9514,0.11369),
    24:(-0.0137,12.1515,0.11426),
}

def lms_z_from_wfa(weight_kg: float, L: float, M: float, S: float) -> float:
    return ((weight_kg / M) ** L - 1.0) / (L * S)

def lms_wfa_from_z(z, L: float, M: float, S: float):
    return M * (1.0 + L * S * z) ** (1.0 / L)

# ---------- (3) Longitud-para-edad (subset 0–6, 12, 18, 24) ----------
GIRLS_LFA = {
     0:(1.0,49.1477,0.03790), 1:(1.0,53.6872,0.03640), 2:(1.0,57.0673,0.03568),
     3:(1.0,59.8029,0.03520), 4:(1.0,62.0899,0.03486), 5:(1.0,64.0301,0.03463),
     6:(1.0,65.7311,0.03448), 12:(1.0,74.0150,0.03479), 18:(1.0,80.7079,0.03598),
    24:(1.0,86.4153,0.03734),
}
BOYS_LFA = {
     0:(1.0,49.8842,0.03795), 1:(1.0,54.7244,0.03557), 2:(1.0,58.4249,0.03424),
     3:(1.0,61.4292,0.03328), 4:(1.0,63.8860,0.03257), 5:(1.0,65.9026,0.03204),
     6:(1.0,67.6236,0.03165), 12:(1.0,75.7488,0.03137), 18:(1.0,82.2587,0.03279),
    24:(1.0,87.8161,0.03479),
}

def lms_z_from_lfa(length_cm: float, L: float, M: float, S: float) -> float:
    return ((length_cm / M) ** L - 1.0) / (L * S)

def lms_lfa_from_z(z, L: float, M: float, S: float):
    return M * (1.0 + L * S * z) ** (1.0 / L)

# ---------- Controles globales (SEXO + EDAD) ----------
st.markdown("<h1 class='title'>👶 OMS – Percentiles unificados</h1>", unsafe_allow_html=True)
st.markdown("<p class='muted'>Selecciona sexo y edad una sola vez. Reusa el mismo peso/talla en todos los módulos.</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Datos del bebé (global)")
    sex = st.radio("Sexo", ["Niña", "Niño"], horizontal=True, key="sex_global")
    age_m = st.slider("Edad (meses)", 0, 24, 5, key="age_global")
    st.markdown("---")
    st.subheader("Entradas")
    w_curr = st.number_input("Peso actual (kg)", min_value=0.0, step=0.01, value=5.72, key="w_curr")
    w_prev = st.number_input("Peso anterior (kg) – hace ~1 mes", min_value=0.0, step=0.01, value=5.20, key="w_prev")
    length_cm = st.number_input("Talla / Longitud (cm)", min_value=20.0, step=0.1, value=64.0, key="len_curr")
    st.caption("Tip: el 'peso actual' también se usa como 'peso reciente' en el cálculo de aumento mensual.")
    st.markdown("---")
    st.subheader("Opciones avanzadas")
    # Mes final del intervalo para velocity; por defecto usamos la edad si 1–12.
    default_end_m = int(age_m) if 1 <= int(age_m) <= 12 else 12
    end_month = st.slider("Mes final del intervalo (para aumento mensual)", 1, 12, default_end_m,
                          help="OMS define velocity mensual para 1–12 meses.", key="endm_global")
    show_details = st.checkbox("Mostrar detalles avanzados (Z-score, mediana)", value=True, key="details")

tab_vel, tab_wfa, tab_lfa = st.tabs(["Aumento mensual", "Peso acumulado", "Talla (longitud)"])

# =========================
# TAB – AUMENTO MENSUAL
# =========================
with tab_vel:
    st.markdown("<h3>Percentil del aumento de peso del último mes</h3>", unsafe_allow_html=True)
    if w_curr < w_prev:
        st.error("El peso actual es menor que el anterior. Verifica las entradas.")
    else:
        inc_g = (w_curr - w_prev) * 1000.0
        if sex == "Niña":
            L = GIRLS_L_V
            M_shift, S = GIRLS_V[end_month]
            grupo = "Niña"
        else:
            L, M_shift, S = BOYS_V[end_month]
            grupo = "Niño"
        z = lms_z_from_inc(inc_g, L, M_shift, S)
        pct = float(normal_cdf(z) * 100.0)
        median_g = M_shift - DELTA_G

        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='percentil'>Percentil {pct:.1f}</div>", unsafe_allow_html=True)
        st.write(f"**Aumento mensual** ({end_month-1}→{end_month} m, {grupo}): **{inc_g:.0f} g**.")
        if show_details:
            st.write(f"Mediana OMS: **{median_g:.0f} g** · Z-score: **{z:.2f}**")
        st.markdown("</div>", unsafe_allow_html=True)

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
    st.caption("Nota: el módulo de 'aumento' está definido por OMS sólo para 1–12 meses.")

# =========================
# TAB – PESO ACUMULADO
# =========================
with tab_wfa:
    st.markdown("<h3>Percentil de peso acumulado (peso-para-edad)</h3>", unsafe_allow_html=True)
    if sex == "Niña":
        L, M, S = GIRLS_WFA[age_m]
        grupo = "Niña"
    else:
        L, M, S = BOYS_WFA[age_m]
        grupo = "Niño"
    z = lms_z_from_wfa(w_curr, L, M, S)
    pct = float(normal_cdf(z) * 100.0)

    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='percentil'>Percentil {pct:.1f}</div>", unsafe_allow_html=True)
    st.write(f"**Peso actual:** {w_curr:.2f} kg · **Edad:** {age_m} m · {grupo}")
    if show_details:
        st.write(f"Mediana OMS: **{M:.2f} kg** · Z: **{z:.2f}** · L={L:.4f}, S={S:.5f}")
    st.markdown("</div>", unsafe_allow_html=True)

    z_grid = np.linspace(-3.5, 3.5, 400)
    w_grid = lms_wfa_from_z(z_grid, L, M, S)
    cdf_grid = normal_cdf(z_grid) * 100.0
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(w_grid, cdf_grid, linewidth=2)
    ax.scatter([w_curr], [pct], s=60)
    ax.axvline(w_curr, linestyle="--", linewidth=1)
    ax.axhline(pct, linestyle="--", linewidth=1)
    ax.set_xlabel("Peso (kg)")
    ax.set_ylabel("Frecuencia acumulada (%)")
    ax.set_title(f"{grupo} • {age_m} meses • Percentil {pct:.1f}")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)

# =========================
# TAB – TALLA ACUMULADA
# =========================
with tab_lfa:
    st.markdown("<h3>Percentil de talla (longitud-para-edad)</h3>", unsafe_allow_html=True)
    months_avail = sorted(GIRLS_LFA.keys())
    if age_m not in months_avail:
        st.info(f"Para talla, esta versión incluye meses {months_avail}. Seleccionaste {age_m} m. "
                f"Usando el mes más cercano disponible.")
        # Elegimos el mes disponible más cercano
        age_use = min(months_avail, key=lambda m: abs(m - age_m))
    else:
        age_use = age_m

    if sex == "Niña":
        L, M, S = GIRLS_LFA[age_use]
        grupo = "Niña"
    else:
        L, M, S = BOYS_LFA[age_use]
        grupo = "Niño"
    z = lms_z_from_lfa(length_cm, L, M, S)
    pct = float(normal_cdf(z) * 100.0)

    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='percentil'>Percentil {pct:.1f}</div>", unsafe_allow_html=True)
    st.write(f"**Talla:** {length_cm:.1f} cm · **Edad de referencia:** {age_use} m · {grupo}")
    if show_details:
        st.write(f"Mediana OMS: **{M:.1f} cm** · Z: **{z:.2f}** · L={L:.1f}, S={S:.5f}")
    st.markdown("</div>", unsafe_allow_html=True)

    z_grid = np.linspace(-3.5, 3.5, 400)
    l_grid = lms_lfa_from_z(z_grid, L, M, S)
    cdf_grid = normal_cdf(z_grid) * 100.0
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(l_grid, cdf_grid, linewidth=2)
    ax.scatter([length_cm], [pct], s=60)
    ax.axvline(length_cm, linestyle="--", linewidth=1)
    ax.axhline(pct, linestyle="--", linewidth=1)
    ax.set_xlabel("Talla / Longitud (cm)")
    ax.set_ylabel("Frecuencia acumulada (%)")
    ax.set_title(f"{grupo} • {age_use} meses • Percentil {pct:.1f}")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)

# ---------- Pie ----------
st.markdown("---")
st.caption("Fuentes: WHO Child Growth Standards (2006). Velocity 1–12 m; Weight-for-age 0–24 m; Length-for-age (subset). "
           "Esta app no sustituye la valoración pediátrica profesional.")
