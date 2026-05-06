# 🌀 Spin Coating Thin-Film Simulator
### EBP (Emslie–Bonner–Peck) Theory + Meyerhofer Model
**SKKU Fluid Mechanics Term Project · 2026 Spring**

---

## 📐 Physics Background

This simulator solves the **Emslie-Bonner-Peck (EBP)** thin-film equation for spin coating:

```
∂h/∂t = -(ρω²/3η) · (1/r) · ∂(r² h³)/∂r  -  E
```

With **Meyerhofer viscosity model**:
```
η(t) = η₀ · (h₀ / h_avg(t))^n
```

### Boundary Conditions
| Location | Condition | Meaning |
|----------|-----------|---------|
| r = 0 | ∂h/∂r = 0 (Neumann) | Symmetry — no flux at center |
| r = R | ∂h/∂r = 0 (Neumann) | Free outflow at wafer edge |
| t = 0 | h(r,0) = h₀ | Uniform initial film |

---

## 🚀 How to Run Locally

### Step 1: Clone this repo
```bash
git clone https://github.com/YOUR_USERNAME/spin-coating-simulator.git
cd spin-coating-simulator
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the app
```bash
streamlit run spin_coating_app.py
```

Your browser will open at `http://localhost:8501`

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository and set:
   - **Main file**: `spin_coating_app.py`
5. Click **Deploy** → Done! You get a public URL like:
   `https://your-app.streamlit.app`

---

## 📁 File Structure
```
spin-coating-simulator/
├── spin_coating_app.py    ← Main Streamlit app
├── requirements.txt       ← Python dependencies
└── README.md              ← This file
```

---

## 🔢 Simulator Features

| Tab | Content |
|-----|---------|
| 📊 Main Results | h(r,t) profiles, time evolution, 2D heatmap, metrics |
| 🔬 Validation | Numerical vs analytical (E=0 limit) comparison |
| 🎨 Design Explorer | ω–η₀ parameter sweep, ±2% spec map |
| 📐 Theory | Full derivation, BCs, dimensionless numbers |

---

## 📚 References
- Emslie, Bonner, Peck (1958) *J. Appl. Phys.* 29, 858
- Meyerhofer (1978) *J. Appl. Phys.* 49, 3993
- Scriven (1988) — Lubrication theory review
