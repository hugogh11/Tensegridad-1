from collections import deque
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
import numpy as np


class MaquetasRealesNormalizadas:

    def __init__(self):
        self.L_palo = 0.30
        self.R_reposo = 0.11
        self.modo = "1 Piso (Vídeo 2)"
        self.actualizar_fisica(0.0, 0.0)

    def calcular_geometria_piso(self, z_base, aplastamiento_0_a_1, tiron_dedo_lat, factor_escala=1.0):
        h_reposo = 0.24 * factor_escala
        h_actual = h_reposo * (1.0 - 0.78 * aplastamiento_0_a_1)

        apertura_radial = math.sqrt(max(0.01, (self.L_palo * factor_escala) ** 2 - h_actual ** 2)) / math.sqrt(3.0)
        r_actual = max(self.R_reposo * factor_escala, apertura_radial * 0.92)

        giro_twist = (5.0 * np.pi / 6.0) + 0.42 * aplastamiento_0_a_1
        ang_inf = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
        ang_sup = ang_inf + giro_twist

        desp_dedo_x = (tiron_dedo_lat / 30.0) * 0.035

        n_inf = np.column_stack([
            r_actual * np.cos(ang_inf),
            r_actual * np.sin(ang_inf),
            np.full(3, z_base),
        ])
        n_sup = np.column_stack([
            r_actual * np.cos(ang_sup) + desp_dedo_x,
            r_actual * np.sin(ang_sup),
            np.full(3, z_base + h_actual),
        ])

        comba_cable = max(0.0, (aplastamiento_0_a_1 - 0.35) * 1.4)
        return n_inf, n_sup, comba_cable

    def actualizar_fisica(self, fuerza_mano_arriba, tiron_dedo_lateral):
        self.f_mano = fuerza_mano_arriba
        self.f_dedo = tiron_dedo_lateral
        self.fuerza_total_ejercida = fuerza_mano_arriba + tiron_dedo_lateral

        aplastamiento = fuerza_mano_arriba / 50.0

        if self.modo == "1 Piso (Vídeo 2)":
            ni1, ns1, comba1 = self.calcular_geometria_piso(0.0, aplastamiento, tiron_dedo_lateral, 1.0)
            self.pisos_inf = [ni1]
            self.pisos_sup = [ns1]
            self.combas = [comba1]

            # Pesos de cómo se reparte TU fuerza exacta entre los 9 cables:
            # Al aplastar mucho (comba > 0), los cables verticales dejan de trabajar (0 N)
            # y toda tu fuerza se la reparten los triángulos superior e inferior.
            peso_vert = max(0.0, 1.0 - aplastamiento * 2.2)
            pesos_brutos = np.array([
                1.1, 1.1, 1.1 + (tiron_dedo_lateral / 10.0),  # 3 Gomas Triángulo Superior
                1.0, 1.0, 1.0 + (tiron_dedo_lateral / 12.0),  # 3 Gomas Triángulo Mesa
                peso_vert, peso_vert, peso_vert + (tiron_dedo_lateral / 6.0),  # 3 Gomas Verticales
            ])
            # Normalizamos para que la SUMA DE LOS 9 CABLES sea EXACTAMENTE igual a la Fuerza que tú ejerces
            if self.fuerza_total_ejercida > 0.01:
                self.reparto_newtons = self.fuerza_total_ejercida * (pesos_brutos / np.sum(pesos_brutos))
            else:
                self.reparto_newtons = np.zeros(9)
        else:
            ni1, ns1, comba1 = self.calcular_geometria_piso(0.0, aplastamiento * 0.75, tiron_dedo_lateral, 0.95)
            z_inicio_p2 = np.mean(ns1[:, 2]) - 0.045 * (1.0 + 0.5 * aplastamiento)
            ni2, ns2, comba2 = self.calcular_geometria_piso(z_inicio_p2, aplastamiento * 0.90, tiron_dedo_lateral * 0.7, 0.95)
            self.pisos_inf = [ni1, ni2]
            self.pisos_sup = [ns1, ns2]
            self.combas = [comba1, comba2]

            peso_v = max(0.0, 0.8 - aplastamiento * 1.6)
            pesos_2p = np.array([
                1.1,       # P1: Base Mesa
                1.2,       # P1: Anillo Medio
                peso_v + (tiron_dedo_lateral / 8.0),  # P1: Laterales Negros
                1.4,       # Silla Central Inter-pisos
                1.2,       # P2: Base Flotante
                1.4,       # P2: Tapa Blanca Superior
                peso_v,    # P2: Laterales Blancos
            ])
            if self.fuerza_total_ejercida > 0.01:
                self.reparto_newtons = self.fuerza_total_ejercida * (pesos_2p / np.sum(pesos_2p))
            else:
                self.reparto_newtons = np.zeros(7)


sim = MaquetasRealesNormalizadas()

fig = plt.figure(figsize=(15.8, 8.2), facecolor="#f4f5f7")
fig.canvas.manager.set_window_title("Reparto Exacto de la Fuerza Ejercida (Suma de Cables = Fuerza Aplicada)")

ax_3d = fig.add_axes([0.02, 0.24, 0.38, 0.72], projection="3d")
ax_barras = fig.add_axes([0.45, 0.55, 0.52, 0.38])
ax_curvas = fig.add_axes([0.45, 0.09, 0.52, 0.36])

ax_radio = fig.add_axes([0.03, 0.04, 0.13, 0.14], facecolor="#e9ecef")
ax_radio.set_title("Selecciona Maqueta:", fontsize=9, fontweight="bold")
radio_modo = RadioButtons(ax_radio, ("1 Piso (Vídeo 2)", "2 Pisos (Vídeo 1)"), active=0)

ax_s_mano = fig.add_axes([0.23, 0.13, 0.16, 0.035])
s_mano = Slider(ax_s_mano, "APLASTAR CON\nLA MANO (0-50 N) ", 0.0, 50.0, valinit=0.0, color="#dc3545")

ax_s_dedo = fig.add_axes([0.23, 0.075, 0.16, 0.035])
s_dedo = Slider(ax_s_dedo, "TIRAR DE GOMA\nCON DEDOS (N) ", 0.0, 30.0, valinit=0.0, color="#28a745")

ax_btn = fig.add_axes([0.23, 0.02, 0.12, 0.04])
btn_reset = Button(ax_btn, "Soltar (0 N)", color="#dee2e6", hovercolor="#adb5bd")

max_pts = 60
hist_mano = deque([0.0] * max_pts, maxlen=max_pts)
hist_sup = deque([0.0] * max_pts, maxlen=max_pts)
hist_inf = deque([0.0] * max_pts, maxlen=max_pts)
hist_vert = deque([0.0] * max_pts, maxlen=max_pts)

x_eje = np.arange(max_pts)
(l_f,) = ax_curvas.plot(x_eje, hist_mano, "k--", linewidth=2.0, label="Fuerza Total que Tú Ejerces (N)")
(l_sup,) = ax_curvas.plot(x_eje, hist_sup, color="#dc3545", linewidth=2.5, label="Suma Absorbida por Parte Superior (N)")
(l_inf,) = ax_curvas.plot(x_eje, hist_inf, color="#007bff", linewidth=2.5, label="Suma Absorbida por Parte Inferior (N)")
(l_vert,) = ax_curvas.plot(x_eje, hist_vert, color="#28a745", linewidth=2.5, label="Suma Absorbida por Cables Laterales (N)")

ax_curvas.set_ylim(-1, 55)
ax_curvas.set_title("Reparto Directo: La suma de las 3 curvas de colores es IGUAL a la Fuerza que ejerces", fontweight="bold", fontsize=9.5)
ax_curvas.set_xlabel("Tiempo real (Al mover los deslizadores)")
ax_curvas.set_ylabel("Fuerza (Newtons)")
ax_curvas.legend(loc="upper left", fontsize=8.5)
ax_curvas.grid(True, alpha=0.3)


def dibujar_cable_con_comba(ax, p1, p2, comba, color, grosor):
    if comba <= 0.001:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=grosor)
    else:
        t = np.linspace(0, 1, 18)
        pts = np.outer(1 - t, p1) + np.outer(t, p2)
        panza = 4 * t * (1 - t)
        pts[:, 0] += panza * comba * 0.04 * np.sign(p1[0] + 0.01)
        pts[:, 1] += panza * comba * 0.04 * np.sign(p1[1] + 0.01)
        pts[:, 2] = np.maximum(0.0, pts[:, 2] - panza * comba * 0.05)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=1.5, linestyle="--")


def actualizar_todo(val=None):
    sim.modo = radio_modo.value_selected
    f_m = s_mano.val
    f_d = s_dedo.val

    sim.actualizar_fisica(f_m, f_d)

    elev, azim = ax_3d.elev, ax_3d.azim
    ax_3d.cla()
    ax_3d.set_xlim(-0.22, 0.22)
    ax_3d.set_ylim(-0.22, 0.22)
    ax_3d.set_zlim(0.0, 0.30 if sim.modo == "1 Piso (Vídeo 2)" else 0.48)

    num_p = len(sim.pisos_inf)
    for p in range(num_p):
        ni = sim.pisos_inf[p]
        ns = sim.pisos_sup[p]
        comba = sim.combas[p]

        for i in range(3):
            ax_3d.plot([ni[i, 0], ns[i, 0]], [ni[i, 1], ns[i, 1]], [ni[i, 2], ns[i, 2]], color="#deb887", linewidth=6.5, solid_capstyle="round")
            ax_3d.scatter(*ni[i], color="#111111", s=40, zorder=5)
            ax_3d.scatter(*ns[i], color="#111111", s=40, zorder=5)

        for i in range(3):
            j = (i + 1) % 3
            ax_3d.plot([ni[i, 0], ni[j, 0]], [ni[i, 1], ni[j, 1]], [ni[i, 2], ni[j, 2]], color="#007bff" if p == 0 else "#8e44ad", linewidth=2.4)
            ax_3d.plot([ns[i, 0], ns[j, 0]], [ns[i, 1], ns[j, 1]], [ns[i, 2], ns[j, 2]], color="#dc3545" if p == num_p - 1 else "#222222", linewidth=2.5)
            dibujar_cable_con_comba(ax_3d, ns[i], ni[j], comba, "#28a745", 2.2)

        if p == 1:
            ns_abajo = sim.pisos_sup[0]
            for i in range(3):
                ax_3d.plot([ni[i, 0], ns_abajo[i, 0]], [ni[i, 1], ns_abajo[i, 1]], [ni[i, 2], ns_abajo[i, 2]], color="#8e44ad", linewidth=2.5)

    top_z = np.mean(sim.pisos_sup[-1][:, 2])
    ax_3d.set_title(f"Maqueta: {sim.modo} | Fuerza Ejercida: {sim.fuerza_total_ejercida:.1f} N", fontweight="bold", fontsize=10)

    if f_m > 0.5:
        ax_3d.quiver(0, 0, top_z + 0.08, 0, 0, -0.07, color="#dc3545", linewidth=4, arrow_length_ratio=0.35)

    ax_3d.view_init(elev=elev if elev is not None else 18, azim=azim if azim is not None else 35)

    ax_barras.cla()
    vals = sim.reparto_newtons
    if sim.modo == "1 Piso (Vídeo 2)":
        nombres = ["Sup 1", "Sup 2", "Sup 3", "Mesa 1", "Mesa 2", "Mesa 3", "Vert 1", "Vert 2", "Vert 3"]
        cols = ["#dc3545", "#dc3545", "#dc3545", "#007bff", "#007bff", "#007bff", "#28a745", "#28a745", "#28a745"]
    else:
        nombres = ["P1: Base\n(Mesa)", "P1: Anillo\nMedio", "P1: Lat\nNegros", "UNIÓN:\nSilla Central", "P2: Base\nFlotante", "P2: Tapa\nBlanca", "P2: Lat\nBlancos"]
        cols = ["#007bff", "#007bff", "#28a745", "#8e44ad", "#f39c12", "#dc3545", "#28a745"]

    ax_barras.set_title(
        f"Reparto de tus {sim.fuerza_total_ejercida:.1f} N en cada parte (Suma de barras = {np.sum(vals):.1f} N)",
        fontweight="bold",
        fontsize=10,
    )

    suma_t = np.sum(vals)
    rects = ax_barras.bar(nombres, vals, color=cols, width=0.58)
    for idx, r in enumerate(rects):
        v = vals[idx]
        pct = (v / suma_t * 100.0) if suma_t > 0.01 else 0.0
        ax_barras.text(idx, v + 0.6, f"{v:.1f} N\n({pct:.1f}%)", ha="center", fontsize=8, fontweight="bold")

    ax_barras.set_ylim(0, max(15.0, np.max(vals) * 1.28))
    ax_barras.set_ylabel("Fuerza Absorbida (N)")
    ax_barras.grid(True, axis="y", alpha=0.3)

    hist_mano.append(sim.fuerza_total_ejercida)
    if sim.modo == "1 Piso (Vídeo 2)":
        hist_sup.append(np.sum(vals[0:3]))
        hist_inf.append(np.sum(vals[3:6]))
        hist_vert.append(np.sum(vals[6:9]))
    else:
        hist_sup.append(np.sum(vals[4:7]))
        hist_inf.append(np.sum(vals[0:3]))
        hist_vert.append(vals[3])

    l_f.set_ydata(hist_mano)
    l_sup.set_ydata(hist_sup)
    l_inf.set_ydata(hist_inf)
    l_vert.set_ydata(hist_vert)

    fig.canvas.draw_idle()


def soltar_mano(event):
    s_mano.reset()
    s_dedo.reset()


s_mano.on_changed(actualizar_todo)
s_dedo.on_changed(actualizar_todo)
radio_modo.on_clicked(actualizar_todo)
btn_reset.on_clicked(soltar_mano)

actualizar_todo()
plt.show()
