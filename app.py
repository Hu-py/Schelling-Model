import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio
import os
from io import BytesIO
from dataclasses import dataclass
from typing import List, Tuple
from scipy.optimize import curve_fit

# =============================
# --- Model Core (从原代码拷贝关键部分)
# =============================

NEIGH_OFFSETS = [(dx,dy) for dx in (-1,0,1) for dy in (-1,0,1) if not (dx==0 and dy==0)]

def init_board(n: int=30, m: int=30, vacancy: float=0.1, seed: int=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = n*m
    n_empty = int(round(vacancy * total))
    n_occ = total - n_empty
    n_a = n_occ//2
    n_b = n_occ - n_a
    cells = np.array([0]*n_empty + [1]*n_a + [-1]*n_b)
    rng.shuffle(cells)
    return cells.reshape((n,m))

def neighbors(grid, i, j):
    n,m = grid.shape
    vals = []
    for dx,dy in NEIGH_OFFSETS:
        x=i+dx; y=j+dy
        if 0<=x<n and 0<=y<m and grid[x,y]!=0:
            vals.append(grid[x,y])
    return np.array(vals)

def satisfaction(grid, i, j, bias):
    me = grid[i,j]
    nbrs = neighbors(grid, i, j)
    if len(nbrs)==0:
        return True, 1.0
    same = np.sum(nbrs==me)
    share = same/len(nbrs)
    return (share >= bias), share

def segregation_degree(grid, bias=None):
    shares=[]; dissatisfied=0; occ=0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i,j]==0: continue
            occ+=1
            ok,share = satisfaction(grid,i,j,bias if bias else 0)
            shares.append(share)
            if bias and not ok: dissatisfied+=1
    mean_same = np.mean(shares) if shares else 1.0
    diss_share = dissatisfied/occ if bias and occ>0 else np.nan
    return mean_same, diss_share

def step_once_random(grid, bias, rng):
    diss=[]
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i,j]==0: continue
            ok,_=satisfaction(grid,i,j,bias)
            if not ok: diss.append((i,j))
    occ=np.count_nonzero(grid)
    diss_share_before = len(diss)/occ if occ>0 else 0
    if not diss: return grid,0,diss_share_before
    empty=[(i,j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i,j]==0]
    rng.shuffle(diss); rng.shuffle(empty)
    moved=0
    for (i,j),(ei,ej) in zip(diss,empty):
        grid[ei,ej]=grid[i,j]; grid[i,j]=0; moved+=1
    return grid,moved,diss_share_before

@dataclass
class RunResult:
    final_grid: np.ndarray
    iters: int
    time_same: List[float]
    time_diss: List[float]

def run_model(n=30, m=30, vacancy=0.1, bias=0.6, seed=123, max_iters=200):
    rng=np.random.default_rng(seed)
    grid=init_board(n,m,vacancy,seed)
    time_same=[]; time_diss=[]
    for t in range(max_iters):
        mean_same, diss_share = segregation_degree(grid,bias)
        time_same.append(mean_same); time_diss.append(diss_share)
        grid,moved,_=step_once_random(grid,bias,rng)
        if moved==0: break
    return RunResult(grid,t+1,time_same,time_diss)

def plot_board(grid, title='Board'):
    vis=grid.copy().astype(float)
    vis[vis==-1]=0; vis[vis==0]=0.5; vis[vis==1]=1
    fig,ax=plt.subplots(figsize=(5,5))
    ax.imshow(vis,cmap='bwr',vmin=0,vmax=1)
    ax.set_title(title); ax.axis('off')
    st.pyplot(fig)

def plot_timeseries(rr: RunResult):
    it=np.arange(1,len(rr.time_same)+1)
    fig,ax=plt.subplots()
    ax.plot(it,rr.time_same,label='Mean same-neighbor share')
    ax.plot(it,rr.time_diss,label='Dissatisfied share')
    ax.legend(); ax.set_xlabel('Iteration')
    st.pyplot(fig)

# =============================
# --- Streamlit App
# =============================

st.title("🏙️ Schelling Segregation Model — Interactive Demo")

st.sidebar.header("Simulation Parameters")
n = st.sidebar.slider("Grid rows (n)", 10, 80, 30)
m = st.sidebar.slider("Grid cols (m)", 10, 80, 30)
vacancy = st.sidebar.slider("Vacancy rate", 0.0, 0.5, 0.1, 0.01)
bias = st.sidebar.slider("Bias (tolerance threshold)", 0.0, 1.0, 0.6, 0.01)
seed = st.sidebar.number_input("Random seed", 0, 9999, 123)
max_iters = st.sidebar.slider("Max iterations", 10, 500, 200)

if st.button("🚀 Run Simulation"):
    rr = run_model(n=n, m=m, vacancy=vacancy, bias=bias, seed=seed, max_iters=max_iters)
    st.success(f"Simulation completed in {rr.iters} iterations.")
    plot_board(rr.final_grid, "Final Segregation Pattern")
    plot_timeseries(rr)
    st.write("Final mean same-neighbor share:", round(rr.time_same[-1], 3))
    st.write("Final dissatisfied share:", round(rr.time_diss[-1], 3))
