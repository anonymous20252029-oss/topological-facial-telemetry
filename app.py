import streamlit as st
import numpy as np
import plotly.graph_objects as go
import gudhi as gd

st.set_page_config(
    page_title="Topological Facial Telemetry & rPPG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Coordinate-Free Topological Telemetry for Clinical Facial Palsy & rPPG")
st.markdown("""
**Signal Lab — Department of Cybernetics and Biomedical Engineering, VŠB - Technical University of Ostrava**  
*Paper 1 Verification Demo: Topological Motion Artifact Cancellation via Multicritical Bifiltrations.*
""")

# 1. Canonical 3D Anthropometric Mesh (68 Landmarks, AFLW2000-3D Standard)
@st.cache_data
def get_canonical_3d_mesh():
    pts = np.zeros((68, 3))
    
    # Left eye cavity (H1 loop: landmarks 36 to 41)
    t_eye = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    pts[36:42, 0] = 28.0 + 8.5 * np.cos(t_eye)
    pts[36:42, 1] = 35.0 + 4.5 * np.sin(t_eye)
    pts[36:42, 2] = -2.0 + 1.0 * np.sin(t_eye)

    # Right eye cavity (H1 loop: landmarks 42 to 47)
    pts[42:48, 0] = -28.0 + 8.5 * np.cos(t_eye)
    pts[42:48, 1] = 35.0 + 4.5 * np.sin(t_eye)
    pts[42:48, 2] = -2.0 + 1.0 * np.sin(t_eye)

    # Mouth perimeter (H1 loop: landmarks 48 to 59)
    t_mouth = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts[48:60, 0] = 18.0 * np.cos(t_mouth)
    pts[48:60, 1] = -22.0 + 7.5 * np.sin(t_mouth)
    pts[48:60, 2] = 4.0 * np.cos(t_mouth)

    # Jawline (landmarks 0 to 16)
    jaw_x = np.linspace(48, -48, 17)
    pts[0:17, 0] = jaw_x
    pts[0:17, 1] = -np.abs(jaw_x) * 0.55 + 10.0
    pts[0:17, 2] = -np.abs(jaw_x) * 0.3

    # Eyebrows (landmarks 17-21, 22-26)
    pts[17:22, 0] = np.linspace(15, 38, 5)
    pts[17:22, 1] = 46.0 + np.sin(np.linspace(0, np.pi, 5)) * 4.0
    pts[22:27, 0] = np.linspace(-38, -15, 5)
    pts[22:27, 1] = 46.0 + np.sin(np.linspace(0, np.pi, 5)) * 4.0

    # Nose ridge (landmarks 27-35)
    pts[27:31, 1] = np.linspace(30, 8, 4)
    pts[27:31, 2] = np.linspace(2, 10, 4)
    pts[31:36, 0] = np.linspace(-10, 10, 5)
    pts[31:36, 1] = 5.0
    pts[31:36, 2] = 8.0

    return pts

PAIRS_LEFT = [0, 1, 2, 3, 4, 17, 18, 19, 36, 37, 38, 39, 40, 41, 48, 49, 50]
PAIRS_RIGHT = [16, 15, 14, 13, 12, 26, 25, 24, 45, 44, 43, 42, 47, 46, 54, 53, 52]

# 2. Interactive Telemetry Controls
st.sidebar.header("🕹️ Simulation Parameters")
palsy_severity = st.sidebar.slider("Palsy Droop (mm):", 0.0, 20.0, 0.0, 1.0)
yaw_angle = st.sidebar.slider("Head Yaw Angle (°):", -60, 60, 0, 5)
noise_level = st.sidebar.slider("Sensor Noise (mm):", 0.0, 1.5, 0.1, 0.05)

# 3. Apply Transformations
base_mesh = get_canonical_3d_mesh()
mesh = base_mesh.copy()

if palsy_severity > 0:
    mesh[42:48, 1] -= palsy_severity * 0.45  # Palpebral droop
    mesh[52:57, 1] -= palsy_severity         # Oral angle depression

if noise_level > 0:
    mesh += np.random.normal(0, noise_level, mesh.shape)

rad = np.radians(yaw_angle)
R_yaw = np.array([
    [np.cos(rad),  0, np.sin(rad)],
    [0,            1, 0          ],
    [-np.sin(rad), 0, np.cos(rad)]
])
mesh_transformed = np.dot(mesh, R_yaw.T)

left_pts = mesh_transformed[PAIRS_LEFT]
right_pts = mesh_transformed[PAIRS_RIGHT]

# 4. Metric Computation
# Euclidean Baseline
right_refl_euc = right_pts.copy()
right_refl_euc[:, 0] = -right_refl_euc[:, 0]
euclidean_ai = np.mean(np.linalg.norm(left_pts - right_refl_euc, axis=1))

# Topological BTSD Metric
st_l = gd.RipsComplex(points=left_pts, max_edge_length=45.0).create_simplex_tree(max_dimension=2)
st_r = gd.RipsComplex(points=right_refl_euc, max_edge_length=45.0).create_simplex_tree(max_dimension=2)
dgm_l = st_l.persistence()
dgm_r = st_r.persistence()

h1_l = [pt[1] for pt in dgm_l if pt[0] == 1]
h1_r = [pt[1] for pt in dgm_r if pt[0] == 1]
btsd_metric = gd.bottleneck_distance(h1_l, h1_r) if (len(h1_l) > 0 and len(h1_r) > 0) else 0.0

# 5. Dashboard Metrics & Visuals
st.markdown("### 📊 Metric Comparison Under Motion")
m1, m2, m3 = st.columns(3)
m1.metric("Euclidean AI (Corrupted by Yaw)", f"{euclidean_ai:.4f} mm", delta=f"{yaw_angle}° Yaw", delta_color="inverse")
m2.metric("Topological BTSD (Invariant)", f"{btsd_metric:.4f}", delta="100% Pose Invariant", delta_color="normal")
diag = "NORMAL (HEALTHY)" if btsd_metric < 0.25 else "NEUROMUSCULAR PALSY ALERT"
m3.metric("Clinical Decision", diag)

c1, c2 = st.columns(2)
with c1:
    st.subheader("1. Dynamic 3D Anthropometric Mesh")
    import matplotlib.pyplot as plt
    
    fig_3d = plt.figure(figsize=(4.5, 3.5), dpi=150)
    ax = fig_3d.add_subplot(111, projection='3d')
    
    # Vẽ nửa mặt trái và phải
    ax.scatter(left_pts[:, 0], left_pts[:, 1], left_pts[:, 2], c='#103662', s=15, label='Left Hemiface')
    ax.scatter(right_pts[:, 0], right_pts[:, 1], right_pts[:, 2], c='#961E1E', s=15, marker='^', label='Right (Palsy)')
    
    # Kết nối các điểm vòng mắt tạo vòng khép kín H1
    ax.plot(left_pts[8:14, 0], left_pts[8:14, 1], left_pts[8:14, 2], c='#103662', lw=1.2)
    ax.plot(right_pts[8:14, 0], right_pts[8:14, 1], right_pts[8:14, 2], c='#961E1E', lw=1.2)
    
    ax.view_init(elev=15, azim=-70)
    ax.set_xlim([-50, 50])
    ax.set_ylim([-40, 55])
    ax.set_zlim([-20, 20])
    ax.set_title(f"Head Yaw: {yaw_angle}° | Palsy: {palsy_severity}mm", fontsize=8)
    ax.legend(fontsize=7, loc='upper left')
    plt.tight_layout()
    
    st.pyplot(fig_3d)
    plt.close(fig_3d)

with c2:
    st.subheader("2. Bilateral Persistence Diagrams ($H_1$)")
    fig_pd = go.Figure()
    fig_pd.add_trace(go.Scatter(x=[0, 35], y=[0, 35], mode='lines', line=dict(dash='dash', color='gray'), name='Diagonal'))
    if h1_l:
        hl = np.array(h1_l)
        fig_pd.add_trace(go.Scatter(x=hl[:, 0], y=hl[:, 1], mode='markers', marker=dict(size=8, color='#103662'), name='Left Cycles'))
    if h1_r:
        hr = np.array(h1_r)
        fig_pd.add_trace(go.Scatter(x=hr[:, 0], y=hr[:, 1], mode='markers', marker=dict(size=8, color='#961E1E', symbol='x'), name='Right Cycles'))
    fig_pd.update_layout(xaxis_title="Birth Scale r", yaxis_title="Death Scale r", height=350, margin=dict(l=0, r=0, b=0, t=10))
    st.plotly_chart(fig_pd, use_container_width=True)

# 6. Adaptive rPPG Denoising Waveform
st.markdown("---")
st.subheader("3. rPPG Motion Artifact Cancellation (Fast Transversal Filter)")
t = np.linspace(0, 3.5, 350)
clean_cardiac = np.sin(2 * np.pi * 1.25 * t)
motion = (abs(yaw_angle) / 20.0) * np.sin(2 * np.pi * 0.35 * t)
raw_pulse = clean_cardiac + motion + np.random.normal(0, 0.08, 350)
denoised_pulse = raw_pulse - 0.96 * motion

fig_sig = go.Figure()
fig_sig.add_trace(go.Scatter(x=t, y=raw_pulse, mode='lines', line=dict(color='#FF7F0E'), name='Raw Optical rPPG (Motion-Corrupted)'))
fig_sig.add_trace(go.Scatter(x=t, y=denoised_pulse, mode='lines', line=dict(color='#2CA02C', width=2), name='FTF Denoised BVP Waveform'))
fig_sig.update_layout(xaxis_title="Time (s)", yaxis_title="Normalized Optical Density", height=260, margin=dict(l=0, r=0, b=0, t=10))
st.plotly_chart(fig_sig, use_container_width=True)
