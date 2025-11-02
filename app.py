import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import imageio
from dataclasses import dataclass
from typing import List

# =============================
# --- Model Core
# =============================

NEIGH_OFFSETS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]

def init_board(n: int = 30, m: int = 30, vacancy: float = 0.1, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = n * m
    n_empty = int(round(vacancy * total))
    n_occ = total - n_empty
    n_a = n_occ // 2
    n_b = n_occ - n_a
    cells = np.array([0] * n_empty + [1] * n_a + [-1] * n_b)
    rng.shuffle(cells)
    return cells.reshape((n, m))

def neighbors(grid, i, j):
    n, m = grid.shape
    vals = []
    for dx, dy in NEIGH_OFFSETS:
        x, y = i + dx, j + dy
        if 0 <= x < n and 0 <= y < m and grid[x, y] != 0:
            vals.append(grid[x, y])
    return np.array(vals)

def satisfaction(grid, i, j, bias):
    me = grid[i, j]
    nbrs = neighbors(grid, i, j)
    if len(nbrs) == 0:
        return True, 1.0
    same = np.sum(nbrs == me)
    share = same / len(nbrs)
    return (share >= bias), share

def segregation_degree(grid, bias=None):
    shares = []
    dissatisfied = 0
    occ = 0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 0:
                continue
            occ += 1
            ok, share = satisfaction(grid, i, j, bias if bias else 0)
            shares.append(share)
            if bias and not ok:
                dissatisfied += 1
    mean_same = np.mean(shares) if shares else 1.0
    diss_share = dissatisfied / occ if bias and occ > 0 else np.nan
    return mean_same, diss_share

def step_once_random(grid, bias, rng):
    diss = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 0:
                continue
            ok, _ = satisfaction(grid, i, j, bias)
            if not ok:
                diss.append((i, j))
    if not diss:
        return grid, 0
    empty = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i, j] == 0]
    rng.shuffle(diss)
    rng.shuffle(empty)
    moved = 0
    for (i, j), (ei, ej) in zip(diss, empty):
        grid[ei, ej] = grid[i, j]
        grid[i, j] = 0
        moved += 1
    return grid, moved

@dataclass
class RunResult:
    frames: List[np.ndarray]
    tipping_idx: int
    final_grid: np.ndarray
    iters: int
    time_same: List[float]
    time_diss: List[float]

def run_model(n=30, m=30, vacancy=0.1, bias=0.6, seed=123, max_iters=200):
    rng = np.random.default_rng(seed)
    grid = init_board(n, m, vacancy, seed)
    frames = [grid.copy()]
    time_same, time_diss = [], []
    tipping_idx = 0
    for t in range(max_iters):
        mean_same, diss_share = segregation_degree(grid, bias)
        time_same.append(mean_same)
        time_diss.append(diss_share)
        if tipping_idx == 0 and mean_same >= 0.95:
            tipping_idx = t
        grid, moved = step_once_random(grid, bias, rng)
        frames.append(grid.copy())
        if moved == 0:
            break
    return RunResult(frames, tipping_idx, grid, t + 1, time_same, time_diss)

def draw_grid(grid, title=""):
    vis = grid.copy().astype(float)
    vis[vis == -1] = 0.1
    vis[vis == 0] = 0.5
    vis[vis == 1] = 0.9
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(vis, cmap='bwr', vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    return fig

# =============================
# --- Streamlit UI
# =============================

st.title("🏙️ Schelling Segregation Model — Interactive Simulation")

st.sidebar.header("Simulation Parameters")
n = st.sidebar.slider("Grid rows (n)", 10, 80, 30)
m = st.sidebar.slider("Grid cols (m)", 10, 80, 30)
vacancy = st.sidebar.slider("Vacancy rate", 0.0, 0.5, 0.1, 0.01)
bias = st.sidebar.slider("Bias (tolerance threshold)", 0.0, 1.0, 0.6, 0.01)
seed = st.sidebar.number_input("Random seed", 0, 9999, 123)
max_iters = st.sidebar.slider("Max iterations", 10, 500, 200)
speed = st.sidebar.slider("🎞️ Animation speed (s per frame)", 0.01, 1.0, 0.1, 0.01)

if st.button("🚀 Run Simulation"):
    rr = run_model(n=n, m=m, vacancy=vacancy, bias=bias, seed=seed, max_iters=max_iters)
    st.success(f"Simulation completed in {rr.iters} iterations.")

    # === GIF 动画生成（显示所有帧） ===
    frames_uint8 = []
    for i, grid in enumerate(rr.frames):
        fig = draw_grid(grid, f"Iteration {i+1}")
        fig.canvas.draw()  # 渲染
        # 获取 RGBA 缓冲区并转为 uint8
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]  # 去掉 alpha
        frames_uint8.append(img)
        plt.close(fig)
    
    st.subheader("📽️ Animation")
    gif_path = "schelling.gif"
    imageio.mimsave(gif_path, frames_uint8, duration=speed)
    st.image(gif_path)

    # === 三列对比图 ===
    st.subheader("🏁 Initial / Final States")
    col1, col2 = st.columns(2)
    with col1:
        fig = draw_grid(rr.frames[0], "Initial State")
        st.pyplot(fig)
    with col2:
        fig = draw_grid(rr.final_grid, "Final State")
        st.pyplot(fig)

    # === 折线图 ===
    st.subheader("📊 Time Series")
    fig, ax = plt.subplots()
    ax.plot(range(1, len(rr.time_same)+1), rr.time_same, label="Mean same-neighbor share")
    ax.plot(range(1, len(rr.time_diss)+1), rr.time_diss, label="Dissatisfied share")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

    # === 输出指标 ===
    st.subheader("📈 Simulation Summary")
    st.write(f"**Final mean same-neighbor share:** {round(rr.time_same[-1], 3)}")
    st.write(f"**Final dissatisfied share:** {round(rr.time_diss[-1], 3)}")
