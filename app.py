from __future__ import annotations
"""
orchard_app_v17.py – 🍒 Orchard optimiser + Dual‑side picking, kg/person & tractors
==============================================================================

### Novedades
1. **Kg/person**: al final de la simulación se muestra (y exporta) el rendimiento
   individual real = `kg_cosechados / nº pickers`.
2. **Requerimiento de tractores (forklifts)**: calcula cuántos tractores se
   necesitan para evacuar todos los bins llenos, suponiendo:
   * distancia bodega = 250 m,
   * velocidad tractor = 60 m/min,
   * descarga = 2 min.
3. **Layout X,Y**: tabla descargable con las coordenadas de cada bin.
4. **Dual‑entry picking**: checkbox “Pickers from both sides”; la mitad parte
   desde x=0 y la otra mitad desde x=L.
5. Panel lateral reorganizado.

> **Requisitos**
> ```bash
> pip install -U streamlit numpy pandas plotly ortools simpy
> ```
"""

# ────────────────────────────── IMPORTS
import io, math, itertools
from typing import List, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from ortools.linear_solver import pywraplp

# ────────────────────────────── PAGE CONFIG
#st.set_page_config(page_title="🍒 Orchard optimiser v17", layout="wide")

# ────────────────────────────── CONSTANTS
P = dict(
    dt=5,                # minutos por slot
    horizon=360,         # jornada total (min)
    picker_rate15=10,    # kg / picker / 15 min
    tree_spacing=2,      # m entre árboles
    bin_cap=300,         # kg por bin
    dist_bodega=250,     # m
    speed_fork=60,       # m/min
    unload=2,            # min descarga
)
KG_PER_SLOT_PER_PICKER = P['picker_rate15'] * P['dt'] / 15  # 3.333 kg / 5 min

# ────────────────────────────── HELPERS

def tree_coords(rows:int, L:int, spacing:int=2):
    return np.array([(r, x) for r in range(rows) for x in range(0, L+1, spacing)], float)

def equidistant_bins(rows:int, L:int, k:int):
    per_row = [k//rows]*rows
    for i in range(k%rows):
        per_row[i]+=1
    bins=[]
    for r,b in enumerate(per_row):
        for idx in range(b):
            x=(idx+1)*L/(b+1)
            bins.append((r,x))
    return np.array(bins,float)

def candidate_coords(rows:int, L:int, step:int):
    return np.array([(r, x) for r in range(rows) for x in range(0, L+1, step)], float)

# ────────────────────────────── k‑MEDIAN

def solve_k_median(trees: np.ndarray, cand: np.ndarray, k:int):
    n, m = trees.shape[0], cand.shape[0]
    d = np.linalg.norm(trees[:,None,:] - cand[None,:,:], axis=2)
    s = pywraplp.Solver.CreateSolver("SCIP")
    x={(i,j):s.BoolVar(f"x_{i}_{j}") for i in range(n) for j in range(m)}
    y={j:s.BoolVar(f"y_{j}") for j in range(m)}
    for i in range(n):
        s.Add(sum(x[i,j] for j in range(m))==1)
    for i,j in x:
        s.Add(x[i,j]<=y[j])
    s.Add(sum(y[j] for j in range(m))==k)
    s.Minimize(sum(d[i,j]*x[i,j] for i in range(n) for j in range(m)))
    s.SetTimeLimit(3000)
    s.Solve()
    chosen=[j for j in range(m) if y[j].solution_value()>0.5]
    return cand[chosen]

# ────────────────────────────── PICKERS

def sequential_assignment(rows:int,total:int,max_row:int,dual:bool)->List[int]:
    res=[0]*rows;i=0;rem=total
    while rem>0 and i<rows:
        assign=min(max_row,rem);res[i]=assign;rem-=assign;i+=1
    return res

# ────────────────────────────── ANALYSIS FUNCTIONS

def minutes_needed(target:int,pick_row:List[int])->int:
    kg_slot=sum(pick_row)*KG_PER_SLOT_PER_PICKER
    if kg_slot==0:
        return math.inf
    slots=math.ceil(target/kg_slot)
    return slots*P['dt']

def forklifts_needed(full_bins:int, L:int)->int:
    """Cálculo simple de nº tractores: ciclos vs tiempo disponible."""
    t_cycle = 2*(P['dist_bodega'] + L/2)/P['speed_fork'] + P['unload']  # min
    cycles_per_fork = P['horizon'] / t_cycle
    return math.ceil(full_bins / cycles_per_fork)

# ────────────────────────────── SIMULATION

def simulate(rows:int,L:int,pick_row:List[int],bins_xy:np.ndarray,target:int,dual:bool):
    trees_per_row=L//P['tree_spacing']
    # Build picker states
    pickers=[]
    for r,n in enumerate(pick_row):
        if n==0:
            continue
        dirs=[1]*n  # 1 -> izquierda→derecha
        if dual and n>1:
            # alternamos direcciones
            dirs=[1 if i%2==0 else -1 for i in range(n)]
        for d in dirs:
            start_tree=0 if d==1 else trees_per_row
            pickers.append(dict(row=r,tree=start_tree,t=0,dir=d))

    # Bin states
    bin_states=[dict(id=i,row=int(b[0]),x=float(b[1]),kg=0,state='Empty',full_slot=None) for i,b in enumerate(bins_xy)]
    bin_states.sort(key=lambda b:(b['row'],b['x']))

    tl_pick=[]; tl_bins=[]; kg_done=0; full_bins=0
    slots=P['horizon']//P['dt']
    for s in range(slots):
        # pickers step
        for p in pickers:
            # si salió del rango de árboles
            if p['tree']<0 or p['tree']>trees_per_row:
                continue
            p['t']+=P['dt']
            if p['t']>=15:
                p['t']-=15
                p['tree']+=p['dir']
                # deposit fruit
                row_bins=[b for b in bin_states if b['row']==p['row'] and b['state']!="Full"]
                if row_bins:
                    # elegir bin más cercano
                    b=min(row_bins,key=lambda x:abs(x['x']-p['tree']*P['tree_spacing']))
                    b['kg']+=P['picker_rate15']/3
                    if b['kg']>=P['bin_cap'] and b['state']!="Full":
                        b['state']='Full'; b['full_slot']=s; full_bins+=1
                kg_done+=P['picker_rate15']/3
            # timeline picker
            if 0<=p['tree']<=trees_per_row:
                tl_pick.append(dict(slot=s,row=p['row'],x=p['tree']*P['tree_spacing'],kind='Picker'))
        # timeline bins snapshot
        for b in bin_states:
            tl_bins.append(dict(slot=s,row=b['row'],x=b['x'],kind=b['state']))
        if kg_done>=target:
            break
    minutes=s*P['dt']
    kg_per_person=kg_done/len([p for p in pick_row for _ in range(1)]) if sum(pick_row)>0 else 0
    tractors=forklifts_needed(full_bins,L)
    return pd.DataFrame(tl_pick), pd.DataFrame(tl_bins), kg_done, minutes, kg_per_person, tractors, full_bins

# ────────────────────────────── SIDEBAR INPUTS
with st.sidebar:
    st.header('🌱 Field parameters')
    rows=st.slider('Rows',4,40,10)
    L=st.slider('Row length (m)',100,400,200,10)
    target_kg=st.number_input('Target kg',1_000,100_000,20_000,1_000)
    k_bins=max(1,int(target_kg/P['bin_cap'])+3)

    st.markdown('---')
    st.header('👷‍♂️ Workforce & strategy')
    max_row=st.slider('Max pickers / row',1,L//P['tree_spacing'],10)
    dual=st.checkbox('Pickers from both sides',value=False)

    st.markdown('---')
    st.header('📈 Scenario grid')
    scn_min=st.number_input('Min pickers',1,300,20)
    scn_max=st.number_input('Max pickers',1,500,120)
    scn_step=st.number_input('Step',1,50,20)

    st.markdown('---')
    st.header('🚜 Bins')
    place=st.radio('Placement',('Equidistant','Optimised'))
    run_grid=st.button('Run scenario grid')
    run_sim=st.button('Run detailed simulation')

# ────────────────────────────── BINS LAYOUT
if place=='Equidistant':
    bins_xy=equidistant_bins(rows,L,k_bins)
else:
    trees=tree_coords(rows,L)
    cand=candidate_coords(rows,L,10)
    bins_xy=solve_k_median(trees,cand,k_bins)

# ────────────────────────────── SCENARIO GRID
best=None
if run_grid:
    records=[]
    for pickers in range(scn_min,scn_max+1,scn_step):
        pick_row=sequential_assignment(rows,pickers,max_row,dual)
        minutes=minutes_needed(target_kg,pick_row)
        kg_pp=target_kg/pickers
        tractors=forklifts_needed(k_bins,L)
        records.append(dict(pickers=pickers,minutes=minutes,kg_pp=kg_pp,tractors=tractors))
    df=pd.DataFrame(records)
    best=df.loc[df['minutes'].idxmin()]
    st.subheader('Scenario grid')
    st.dataframe(df.style.apply(lambda row: ['background-color:lightgreen' if row.minutes==best.minutes else '' for _ in row],axis=1))
    st.success(f"Best ▶ {int(best.pickers)} pickers → {int(best.minutes)} min, {best.kg_pp:.1f} kg/person, {int(best.tractors)} tractors")
    fig=px.line(df,x='pickers',y='minutes',markers=True,title='Minutes vs Pickers')
    st.plotly_chart(fig,use_container_width=True)
    with io.BytesIO() as buf:
        df.to_csv(buf,index=False)
        st.download_button('Download grid',buf.getvalue(),'scenario_grid.csv')

# ────────────────────────────── DETAILED SIMULATION
if run_sim:
    pick_tot=best.pickers if best is not None else scn_max
    pick_row=sequential_assignment(rows,int(pick_tot),max_row,dual)
    df_pick,df_bins,kg_done,minutes,kg_pp,tractors,full_bins=simulate(rows,L,pick_row,bins_xy,target_kg,dual)
    st.success(f"{kg_done:.0f} kg in {minutes} min • {int(pick_tot)} pickers ({kg_pp:.1f} kg/person) • {tractors} tractors for {full_bins} bins")

    df_all=pd.concat([df_pick,df_bins])
    cm={'Picker':'black','Empty':'green','Filling':'orange','Full':'red'}
    fig=px.scatter(df_all,x='x',y='row',animation_frame='slot',color='kind',color_discrete_map=cm,
                   range_x=[0,L],range_y=[rows,-1],height=500)
    fig.update_yaxes(title='Row')
    st.plotly_chart(fig,use_container_width=True)

    # Layout table
    st.subheader('Bin layout (X,Y)')
    df_layout=pd.DataFrame(bins_xy,columns=['row','x'])
    st.dataframe(df_layout)
    with io.BytesIO() as buf:
        with pd.ExcelWriter(buf,engine='xlsxwriter') as xls:
            df_all.to_excel(xls,'timeline',index=False)
            df_layout.to_excel(xls,'bins_layout',index=False)
        st.download_button('Download timeline & layout',buf.getvalue(),'timeline_bins.xlsx')
else:
    st.info('Configure parameters and run the grid or the detailed simulation.')
