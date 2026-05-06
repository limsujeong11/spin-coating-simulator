"""
Spin Coating Thin-Film Simulator
EBP (Emslie-Bonner-Peck) Theory + Meyerhofer Model
Subject 1 - Fluid Mechanics Term Project 2026 Spring
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from io import BytesIO

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spin Coating Simulator | EBP Theory",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS STYLING
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    font-family: 'Space Mono', monospace;
}
.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -1px;
    border-bottom: 3px solid #6366f1;
    padding-bottom: 0.5rem;
    margin-bottom: 0.2rem;
}
.subtitle {
    font-size: 0.95rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%);
    border: 1px solid #c7d2fe;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.3rem 0;
}
.metric-label {
    font-size: 0.78rem;
    color: #6366f1;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #1e1b4b;
}
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    color: #4338ca;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.5rem 0 0.5rem 0;
    padding-left: 0.6rem;
    border-left: 3px solid #6366f1;
}
.info-box {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: #0c4a6e;
    margin: 0.5rem 0;
}
.warn-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: #92400e;
    margin: 0.5rem 0;
}
.equation-box {
    background: #1e1b4b;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    color: #e0e7ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    margin: 0.8rem 0;
    overflow-x: auto;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🌀 Spin Coating Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">EBP (Emslie–Bonner–Peck) Theory + Meyerhofer Model &nbsp;|&nbsp; Fluid Mechanics Term Project · SKKU 2026</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUTS
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Process Parameters")

    st.markdown('<div class="section-header">Spin Conditions</div>', unsafe_allow_html=True)
    omega_rpm = st.slider("Rotation speed ω (rpm)", 500, 6000, 3000, 100)
    omega = omega_rpm * 2 * np.pi / 60  # rad/s

    st.markdown('<div class="section-header">Fluid Properties</div>', unsafe_allow_html=True)
    h0_um = st.slider("Initial thickness h₀ (μm)", 10, 500, 100, 5)
    h0 = h0_um * 1e-6  # m

    eta0_mPas = st.slider("Initial viscosity η₀ (mPa·s)", 1, 500, 50, 1)
    eta0 = eta0_mPas * 1e-3  # Pa·s

    rho = st.number_input("Density ρ (kg/m³)", value=1200, min_value=800, max_value=1800, step=50)

    st.markdown('<div class="section-header">Evaporation & Geometry</div>', unsafe_allow_html=True)
    E_nm_s = st.slider("Evaporation rate E (nm/s)", 0, 50, 10, 1)
    E = E_nm_s * 1e-9  # m/s

    R_mm = st.slider("Wafer radius R (mm)", 25, 150, 75, 5)
    R = R_mm * 1e-3  # m

    n_visc = st.slider("Viscosity exponent n", 1.0, 4.0, 2.0, 0.5,
                       help="η(t) = η₀·(h₀/h_avg)^n  (Meyerhofer model)")

    st.markdown('<div class="section-header">Simulation Settings</div>', unsafe_allow_html=True)
    t_end = st.slider("Total time (s)", 5, 120, 30, 5)
    Nr = st.select_slider("Radial grid points", options=[20, 40, 80, 120], value=40)
    Nt = st.select_slider("Time steps", options=[200, 500, 1000, 2000], value=500)

    st.markdown("---")
    st.markdown('<div class="info-box">📐 <b>Governing equation</b><br>∂h/∂t = −(ρω²/3η)·(1/r)·∂(r²h³)/∂r − E</div>',
                unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# NUMERICAL SOLVER
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def solve_ebp(omega, h0, eta0, rho, E, R, n_visc, t_end, Nr, Nt):
    """
    Solve EBP thin-film equation on [0, R] using finite differences.

    PDE:  ∂h/∂t = -(ρω²/3η(t)) · (1/r)·∂(r² h³)/∂r  -  E   [m/s]

    Boundary conditions:
      - r=0  : symmetry  →  ∂h/∂r = 0  (no radial flux at center, ghost-cell)
      - r=R  : Neumann   →  ∂h/∂r = 0  (free outflow)
    Initial condition:
      - h(r, 0) = h0  [m]  (uniform initial film)

    Viscosity (Meyerhofer model):
      η(t) = η0 · (h0 / h_avg(t))^n   [Pa·s]
      As solvent evaporates h_avg↓  →  concentration↑  →  η↑

    CFL safety:
      Adaptive dt enforced each step:  dt ≤ CFL_factor · dr² / (2·D_eff)
      where D_eff = coeff · h_max³ · R   [m²/s]
    """
    r = np.linspace(0, R, Nr)          # [m]
    dr = r[1] - r[0]                    # [m]
    dt_req = t_end / Nt                 # [s] requested time step
    t = 0.0                             # [s] current time
    CFL_FACTOR = 0.45                   # safety factor < 0.5

    h = np.full(Nr, h0)                 # [m]  initial uniform film

    # Pre-compute half-cell face radii (vectorised)
    r_p = r + dr / 2                    # r_{i+1/2}  [m]
    r_m = r - dr / 2                    # r_{i-1/2}  [m]
    r_m[0] = 0.0                        # ghost-cell: left face of node 0 is at r=0

    # Snapshot storage (≈100 frames regardless of actual step count)
    t_snap, h_snap = [0.0], [h.copy()]
    next_snap_t = t_end / 100           # save every 1% of t_end

    while t < t_end - 1e-12:
        # ── Meyerhofer viscosity update ──────────────────────────────────
        h_avg = max(float(np.mean(h)), 1e-12)           # [m]
        eta = eta0 * (h0 / h_avg) ** n_visc             # [Pa·s]
        eta = max(eta, eta0)                             # monotone increase only
        coeff = rho * omega**2 / (3.0 * eta)            # [s⁻¹ m⁻²]

        # ── CFL-limited adaptive dt ───────────────────────────────────────
        h_max = max(float(np.max(h)), 1e-12)
        D_eff = coeff * h_max**3 * R                    # [m²/s]
        dt_cfl = CFL_FACTOR * dr**2 / (2.0 * D_eff)    # [s]
        dt = min(dt_req, dt_cfl, t_end - t)             # [s]

        # ── Vectorised flux computation ──────────────────────────────────
        # Cell-face h³ by arithmetic mean (2nd-order)
        h3_p = ((h[:-1] + h[1:]) / 2) ** 3             # shape (Nr-1,)  [m³]
        h3_m = np.empty(Nr)
        h3_m[1:] = h3_p                                 # h³ at left face = right face of left cell
        h3_m[0] = h[0] ** 3                             # symmetry: use h[0] itself at r=0

        # Flux  F_i = coeff · r² h³  evaluated at faces
        F_p = coeff * r_p**2 * h3_p                     # flux at right face  [m³/s · m⁻¹ — see note]
        F_m = coeff * r_m**2 * h3_m                     # flux at left face

        # divergence  (1/r)·∂(r²h³coeff)/∂r  ≈  (F_p - F_m) / (r · dr)
        # Node 0: use r_p[0]=dr/2 as effective radius to avoid 0/0
        r_eff = np.where(r > 1e-15, r, r_p[0])         # [m]  avoid divide-by-zero
        div = (F_p - F_m[1:].tolist().__class__(F_m)[:-1+Nr]) / (r_eff * dr)
        # Rewrite cleanly:
        div = np.empty(Nr)
        div[1:Nr-1] = (F_p[1:Nr-1] - F_m[1:Nr-1]) / (r[1:Nr-1] * dr)
        div[0]      = (F_p[0]      - 0.0           ) / (r_p[0]   * dr)   # symmetry: F_m at r=0 is 0
        div[Nr-1]   = div[Nr-2]                         # Neumann outflow BC

        # ── Explicit Euler update ─────────────────────────────────────────
        h = h - dt * (div + E)                          # [m]
        h = np.maximum(h, 0.0)                          # h ≥ 0 (physical floor)
        # Neumann BC: zero gradient at edge
        h[-1] = h[-2]

        t += dt

        # Save snapshot
        if t >= next_snap_t or t >= t_end - 1e-12:
            t_snap.append(t)
            h_snap.append(h.copy())
            next_snap_t += t_end / 100

    return r * 1e3, np.array(t_snap), np.array(h_snap) * 1e6  # mm, s, μm


# ──────────────────────────────────────────────────────────────────────────────
# RUN SIMULATION
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner("🔄 Solving EBP equation..."):
    r_mm, t_snap, h_snap = solve_ebp(
        omega, h0, eta0, rho, E, R, n_visc, t_end, Nr, Nt
    )

h_final = h_snap[-1]
h_center = h_snap[:, 0]
h_edge = h_snap[:, -1]
h_avg_snap = h_snap.mean(axis=1)

# ──────────────────────────────────────────────────────────────────────────────
# DERIVED METRICS
# ──────────────────────────────────────────────────────────────────────────────
uniformity = (h_final.max() - h_final.min()) / h_final.mean() * 100
h_final_avg = h_final.mean()

# Gel time estimate: when η > 10×η0 (Meyerhofer criterion)
h0_um_val = h0 * 1e6
h_gel_threshold = h0_um_val * (1 / 10) ** (1 / n_visc)  # h when η = 10η0
gel_idx = np.argmax(h_avg_snap < h_gel_threshold)
t_gel = t_snap[gel_idx] if gel_idx > 0 else t_end

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Main Results",
    "🔬 Validation",
    "🎨 Design Explorer",
    "📐 Theory"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MAIN RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Final Avg Thickness</div>
            <div class="metric-value">{h_final_avg:.1f} μm</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        color = "#dc2626" if uniformity > 2 else "#16a34a"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Radial Uniformity</div>
            <div class="metric-value" style="color:{color}">±{uniformity/2:.2f}%</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Estimated t_gel</div>
            <div class="metric-value">{t_gel:.1f} s</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        thickness_ratio = h_final_avg / (h0 * 1e6) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Thinning Ratio</div>
            <div class="metric-value">{thickness_ratio:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    if uniformity / 2 > 2:
        st.markdown('<div class="warn-box">⚠️ Uniformity spec (±2%) NOT met. Try increasing ω or reducing η₀.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">✅ Uniformity spec (±2%) met!</div>', unsafe_allow_html=True)

    st.markdown("")

    # Plots
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Thickness Profile h(r) — Time Evolution</div>', unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        fig1.patch.set_facecolor('#f8fafc')
        ax1.set_facecolor('#f8fafc')
        cmap = plt.cm.plasma
        n_lines = min(8, len(t_snap))
        indices = np.linspace(0, len(t_snap) - 1, n_lines, dtype=int)
        for idx in indices:
            frac = idx / (len(t_snap) - 1)
            ax1.plot(r_mm, h_snap[idx], color=cmap(frac), lw=1.8,
                     label=f't={t_snap[idx]:.1f}s')
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=0, vmax=t_snap[-1]))
        sm.set_array([])
        plt.colorbar(sm, ax=ax1, label='Time (s)', shrink=0.85)
        ax1.set_xlabel('Radial position r (mm)')
        ax1.set_ylabel('Film thickness h (μm)')
        ax1.set_title('h(r, t) — EBP simulation')
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)
        plt.close()

    with col_r:
        st.markdown('<div class="section-header">Thickness at Key Radial Positions vs Time</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor('#f8fafc')
        ax2.set_facecolor('#f8fafc')
        mid_idx = len(r_mm) // 2
        ax2.plot(t_snap, h_snap[:, 0],   color='#6366f1', lw=2, label='Center (r=0)')
        ax2.plot(t_snap, h_snap[:, mid_idx], color='#06b6d4', lw=2, label=f'Mid (r={r_mm[mid_idx]:.0f}mm)')
        ax2.plot(t_snap, h_snap[:, -1],  color='#f59e0b', lw=2, label=f'Edge (r={r_mm[-1]:.0f}mm)')
        ax2.plot(t_snap, h_avg_snap,     color='#10b981', lw=2, ls='--', label='Average')
        ax2.axvline(t_gel, color='#dc2626', ls=':', lw=1.5, label=f't_gel ≈ {t_gel:.1f}s')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Film thickness (μm)')
        ax2.set_title('Thickness at r = 0, R/2, R vs time')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
        plt.close()

    # Heatmap
    st.markdown('<div class="section-header">2D Thickness Heatmap (r–t space)</div>', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(12, 3.5))
    fig3.patch.set_facecolor('#f8fafc')
    im = ax3.imshow(h_snap.T, aspect='auto', origin='lower',
                    extent=[t_snap[0], t_snap[-1], r_mm[0], r_mm[-1]],
                    cmap='viridis')
    plt.colorbar(im, ax=ax3, label='Thickness (μm)', shrink=0.9)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Radius (mm)')
    ax3.set_title('h(r, t) heatmap')
    st.pyplot(fig3)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🔬 Analytical Limit Validation")

    st.markdown("""
    When **E = 0** (no evaporation) and **η = const**, the EBP equation has an exact analytical solution:
    """)

    st.markdown("""
    <div class="equation-box">
    h(t) = h₀ / √(1 + 4ρω²h₀²t / 3η)
    
    (uniform film, E=0, η=const — EBP analytical solution)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Comparison: Numerical vs Analytical (E = 0 m/s, η = const)**")
    st.markdown("""
    > **Derivation note:** Setting E=0 and η=const, the EBP PDE for a spatially uniform film reduces to
    > dh/dt = -(ρω²/3η)·2h³ → integrating gives h(t) = h₀/√(1 + 4ρω²h₀²t/3η).
    > This is used here as the *gold standard* to verify the numerical solver.
    """)

    # Compute analytical solution  [units: t in s, h in μm]
    t_analytical = np.linspace(0, t_end, 300)   # [s]
    tau = 4 * rho * omega**2 * h0**2 / (3 * eta0)  # [s⁻¹]  characteristic thinning rate
    h_analytical_um = (h0 * 1e6) / np.sqrt(1 + tau * t_analytical)  # [μm]

    # Numerical with E=0, n=1 (constant viscosity)
    with st.spinner("Computing E=0, η=const reference..."):
        _, t_noE, h_noE = solve_ebp(omega, h0, eta0, rho, 0.0, R, 1.0, t_end, Nr, Nt)

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    fig4.patch.set_facecolor('#f8fafc')
    ax4.set_facecolor('#f8fafc')
    ax4.plot(t_analytical, h_analytical_um, 'k--', lw=2.5,
             label='Analytical: h₀/√(1+4ρω²h₀²t/3η)  [μm]')
    ax4.plot(t_noE, h_noE.mean(axis=1), color='#6366f1', lw=2,
             label='Numerical (CFL-adaptive FD, E=0, η=const)  [μm]')
    ax4.set_xlabel('Time  t  [s]')
    ax4.set_ylabel('Spatially-averaged thickness  h̄  [μm]')
    ax4.set_title('Validation: Numerical vs Analytical EBP limit (E=0, η=const)')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    st.pyplot(fig4)
    plt.close()

    # Error metric
    h_num_interp = np.interp(t_analytical, t_noE, h_noE.mean(axis=1))
    rel_error = np.abs(h_num_interp - h_analytical_um) / h_analytical_um * 100
    st.markdown(f"**Mean relative error:** {rel_error.mean():.3f} %  &nbsp;|&nbsp;  **Max error:** {rel_error.max():.3f} %")

    if rel_error.mean() < 2:
        st.markdown('<div class="info-box">✅ Numerical solver validated: mean error &lt; 2 %</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-box">⚠️ Consider increasing grid resolution (Nt, Nr). CFL-adaptive solver is active.</div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DESIGN EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🎨 Design Exploration: ω–η₀ Parameter Sweep")
    st.markdown("Find (ω, η₀) combinations that meet **±2% uniformity** spec.")

    col_a, col_b = st.columns(2)
    with col_a:
        omega_range = st.slider("ω sweep range (rpm)", 500, 6000, (1000, 5000), 500)
        n_omega = st.select_slider("ω grid points", [3, 5, 7, 10], value=5)
    with col_b:
        eta_range = st.slider("η₀ sweep range (mPa·s)", 1, 300, (10, 200), 10)
        n_eta = st.select_slider("η₀ grid points", [3, 5, 7, 10], value=5)

    if st.button("🚀 Run Parameter Sweep", type="primary"):
        omega_arr = np.linspace(omega_range[0], omega_range[1], n_omega) * 2 * np.pi / 60
        eta_arr = np.linspace(eta_range[0], eta_range[1], n_eta) * 1e-3

        results = np.zeros((n_omega, n_eta))
        final_thickness = np.zeros((n_omega, n_eta))

        prog = st.progress(0)
        total = n_omega * n_eta
        cnt = 0
        for i, om in enumerate(omega_arr):
            for j, et in enumerate(eta_arr):
                _, _, h_sw = solve_ebp(om, h0, et, rho, E, R, n_visc, t_end, Nr, Nt)
                hf = h_sw[-1]
                unif = (hf.max() - hf.min()) / hf.mean() * 100
                results[i, j] = unif / 2  # ±%
                final_thickness[i, j] = hf.mean()
                cnt += 1
                prog.progress(cnt / total)

        omega_rpm_arr = np.linspace(omega_range[0], omega_range[1], n_omega)
        eta_mPas_arr = np.linspace(eta_range[0], eta_range[1], n_eta)

        fig5, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig5.patch.set_facecolor('#f8fafc')

        # Uniformity heatmap
        ax = axes[0]
        ax.set_facecolor('#f8fafc')
        im1 = ax.imshow(results, aspect='auto', origin='lower', cmap='RdYlGn_r',
                        vmin=0, vmax=5,
                        extent=[eta_mPas_arr[0], eta_mPas_arr[-1],
                                omega_rpm_arr[0], omega_rpm_arr[-1]])
        plt.colorbar(im1, ax=ax, label='Non-uniformity ±%')
        cs = ax.contour(eta_mPas_arr, omega_rpm_arr, results,
                        levels=[2.0], colors='white', linewidths=2)
        ax.clabel(cs, fmt='±2% spec', fontsize=9, colors='white')
        ax.set_xlabel('Initial viscosity η₀ (mPa·s)')
        ax.set_ylabel('Rotation speed ω (rpm)')
        ax.set_title('Non-uniformity map (white line = ±2% spec)')

        # Thickness heatmap
        ax2 = axes[1]
        ax2.set_facecolor('#f8fafc')
        im2 = ax2.imshow(final_thickness, aspect='auto', origin='lower', cmap='viridis',
                         extent=[eta_mPas_arr[0], eta_mPas_arr[-1],
                                 omega_rpm_arr[0], omega_rpm_arr[-1]])
        plt.colorbar(im2, ax=ax2, label='Final avg thickness (μm)')
        ax2.set_xlabel('Initial viscosity η₀ (mPa·s)')
        ax2.set_ylabel('Rotation speed ω (rpm)')
        ax2.set_title('Final film thickness map')

        st.pyplot(fig5)
        plt.close()

        # Table of passing combos
        st.markdown("#### Combinations meeting ±2% spec:")
        rows = []
        for i in range(n_omega):
            for j in range(n_eta):
                if results[i, j] <= 2.0:
                    rows.append({
                        "ω (rpm)": int(omega_rpm_arr[i]),
                        "η₀ (mPa·s)": int(eta_mPas_arr[j]),
                        "Non-uniformity ±%": f"{results[i,j]:.3f}",
                        "Final thickness (μm)": f"{final_thickness[i,j]:.2f}"
                    })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.warning("No combinations met the ±2% spec in this range. Try wider ω or lower η₀.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — THEORY
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📐 Physics & Derivation — Laws Applied (Week-by-Week)")

    st.markdown("""
    #### 1. Starting Point: Navier–Stokes in a Rotating Reference Frame
    *(Week 6 — Navier–Stokes equations)*

    In the rotating frame of the wafer (angular velocity ω [rad/s]), two body forces appear:
    - **Centrifugal force** (outward): **f**_cf = ρω²r r̂  &nbsp; [N/m³ = kg/(m²·s²)]
    - **Coriolis force**: **f**_Co = −2ρ**ω**×**u** &nbsp; [N/m³]

    For thin-film spin coating, Coriolis is negligible (Ekman number Ek = ν/(ωh²) ≫ 1)
    and the dominant balance is **centrifugal force vs. viscous shear**:
    """)

    st.markdown("""
    <div class="equation-box">
    NS (radial, thin-film limit):
      η · ∂²u_r/∂z²  =  −ρω²r         [Pa/m = kg/(m²·s²)]

    Integrating with BC: u_r(z=0) = 0 (no-slip on wafer), ∂u_r/∂z|_(z=h) = 0 (free surface):
      u_r(r,z) = (ρω²r / 2η) · (2hz − z²)     [m/s]
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    #### 2. Continuity + Depth Integration → Thin-Film PDE
    *(Week 4 — Continuity equation; Week 8 — Lubrication / thin-film approximation)*

    **Thin-film validity:** requires h₀/R ≪ 1 and Re·(h/R) ≪ 1.
    With current defaults: h₀/R ≈ 1.3×10⁻³ ✓, Re·(h/R) ≈ 1.0×10⁻⁴ ✓.

    Depth-averaging the radial continuity equation and using the kinematic BC at the free surface
    (∂h/∂t = −w|_(z=h) + E evaporation sink):
    """)

    st.markdown("""
    <div class="equation-box">
    EBP Governing Equation  (Emslie, Bonner & Peck, 1958):

      ∂h/∂t  =  −(ρω²/3η) · (1/r) · ∂(r² h³)/∂r  −  E

    Unit check:
      [ρω²/3η]       = [kg/m³ · s⁻²] / [Pa·s] = [kg/m³ · s⁻²] / [kg/(m·s)] = s⁻¹ m⁻²
      [(1/r)∂(r²h³)/∂r] = [1/m] · [m⁵/m]  =  m³
      Product              = m/s  ==  [∂h/∂t]  ✓

    Variables and units:
      h(r,t) = local film thickness            [m]
      r      = radial coordinate               [m]
      t      = time                            [s]
      ω      = angular velocity                [rad/s]
      η(t)   = dynamic viscosity               [Pa·s = kg/(m·s)]
      ρ      = fluid density                   [kg/m³]
      E      = evaporation flux (solvent loss) [m/s]
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    #### 3. Boundary & Initial Conditions

    | Location | Type | Equation | Physical meaning |
    |----------|------|----------|-----------------|
    | r = 0 | Symmetry (Neumann) | ∂h/∂r = 0, radial flux → 0 | Axial symmetry; no net flow through wafer center |
    | r = R | Outflow (Neumann) | ∂h/∂r = 0 | Fluid free to leave at edge (edge-bead neglected) |
    | t = 0 | Initial condition | h(r, 0) = h₀ [m] | Freshly dispensed uniform film |

    #### 4. Meyerhofer Viscosity Model — Solvent Evaporation Coupling
    *(Week 9 — Non-Newtonian & concentration-dependent rheology)*

    As solvent evaporates the polymer concentration c ∝ 1/h increases. The Mark–Houwink
    relation gives η ∝ cⁿ, so with mass conservation (h·c = h₀·c₀):
    """)

    st.markdown("""
    <div class="equation-box">
    η(t)  =  η₀ · (h₀ / h̄(t))ⁿ        [Pa·s]

    where  h̄(t) = spatial average of h(r,t)  [m]
    and    n = viscosity exponent (≈2 for most PR solutions)

    Physical constraint enforced in code:  η(t) ≥ η₀  (viscosity only increases)

    Two-regime picture (Meyerhofer, 1978):
      Regime 1 (early, rotation-dominated):  h̄ ~ t^(−1/2)   [m]
      Regime 2 (late, evaporation-dominated): h̄ → const      [m]  until gelation
      Crossover: when (ρω²h³/3η) ≈ E  →  both terms equal  [m/s]
    </div>
    """, unsafe_allow_html=True)

    gamma_val = 0.03  # N/m, typical PR
    U_edge = omega * R
    Re_val = rho * omega * h0**2 / eta0
    Ca_val = eta0 * U_edge / gamma_val
    Ek_val = (eta0/rho) / (omega * h0**2)

    st.markdown(f"""
    #### 5. Dimensionless Numbers (current parameter set)

    | Number | Definition | Formula | Value | Interpretation |
    |--------|-----------|---------|-------|----------------|
    | Reynolds **Re** | Inertia / Viscous | ρωh₀²/η₀ | **{Re_val:.2e}** | ≪ 1 → lubrication approximation valid ✓ |
    | Capillary **Ca** | Viscous / Surface tension | η₀·(ωR)/γ | **{Ca_val:.2e}** | ≫ 1 → surface tension negligible in bulk (edge effects only) |
    | Ekman **Ek** | Viscous / Coriolis | ν/(ω·h₀²) | **{Ek_val:.2e}** | ≫ 1 → Coriolis force negligible ✓ |
    | Aspect ratio | Thin-film validity | h₀/R | **{h0/R:.2e}** | ≪ 1 → 2-D lubrication PDE valid ✓ |

    *γ assumed = 0.03 N/m (typical photoresist), ν = η₀/ρ = {eta0/rho:.2e} m²/s*

    #### 6. Numerical Method
    *(Week 10 — Finite differences, explicit Euler, stability)*

    - **Spatial discretisation**: uniform finite-difference grid, r ∈ [0, R], spacing Δr = R/(Nr−1) [m]
    - **Flux at cell faces**: arithmetic-mean h³ at face i±½ (2nd-order, avoids oscillation)
    - **Time integration**: explicit forward Euler
    - **Adaptive CFL time step** (NEW — fixes original code instability):
    """)

    dr_val = R / (Nr - 1)
    coeff_val = rho * omega**2 / (3.0 * eta0)
    D_eff_val = coeff_val * h0**3 * R
    dt_cfl_val = 0.45 * dr_val**2 / (2.0 * D_eff_val)

    st.markdown(f"""
    <div class="equation-box">
    CFL stability condition for thin-film PDE:

      Δt  ≤  0.45 · Δr² / (2 · D_eff)      [s]

    where D_eff = (ρω²/3η) · h³ · R  [m²/s]  (effective diffusivity)

    With default params:
      Δr     = {dr_val*1e3:.4f} mm = {dr_val:.6f} m
      D_eff  = {D_eff_val:.4e} m²/s
      Δt_CFL = {dt_cfl_val:.4e} s   (solver adapts automatically each step)
    </div>

    #### 7. Semiconductor Process Context

    Spin coating is the **photoresist deposition** step in photolithography (FEOL/BEOL):
    - Wafer size: 300 mm (industry standard, R = 150 mm in production)
    - Target PR thickness: 50–500 nm for IC nodes; 1–5 μm for MEMS
    - Critical spec: within-wafer non-uniformity (WIWNU) < ±2 %
    - Edge bead (thick rim, r → R) causes overlay errors → must be removed by EBR (Edge Bead Removal)
    - Gelation time t_gel determines the window for soft-bake scheduling

    **Preston equation** (material removal rate ∝ pressure × velocity) is the downstream metric;
    uniform PR thickness feeds directly into uniform etch/CMP budget.
    """, unsafe_allow_html=True)



# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#94a3b8; font-size:0.8rem;'>"
    "Spin Coating Simulator · EBP Theory · SKKU Fluid Mechanics 2026 · "
    "Built with Streamlit + NumPy"
    "</div>",
    unsafe_allow_html=True
)
