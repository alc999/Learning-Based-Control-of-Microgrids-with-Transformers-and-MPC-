

import numpy as np
from lpsolvers import solve_lp
import cvxopt
from scipy.optimize import linprog
import os 
import pickle
from datetime import datetime


cbuy, csell, cprod, power_load, power_res = np.load('price_load_res_profiles.npy', allow_pickle=True)




x0 = np.random.rand(1,1)*225+25 #minimum battery level is 25
N = 48


#MLD equations
Ts = 1/2 # Ts = 15m
nd = 0.99
nc = 0.9 

Amld = np.array([
    [1]
])
B1 = np.array([
    [Ts/nd,0,0,0,0]
])
B2 = np.zeros((1,5))
B3 = np.array([
    [Ts*(nc-1/nd),0]
])
B5 = np.zeros((1,1))

#parameters
H = Ts*(nc - 1/nd)
F = Ts/nd

#ESS(battery)
Mb = 100
mb = -100

#grid
Mg = 1000
mg = -1000

#dispatchable generators
Md = 150
md = 6
mdg = 6

eps = 1e-6

# state constraint

E2_sc = np.zeros((2,5))
E3_sc = np.zeros((2,2))
E1_sc = np.zeros((2,5))
E4_sc = np.array([
    [-1],
    [1]
])
E5_sc = np.array([
    [250],
    [-25]
])

# input constraints

E2_ic = np.zeros((10,5))
E3_ic = np.zeros((10,2))
E1_ic = np.array([
    [-1,0,0,0,0],
    [1,0,0,0,0],
    [0,-1,0,0,0],
    [0,1,0,0,0],
    [0,0,-1,0,0],
    [0,0,1,0,0],
    [0,0,0,-1,0],
    [0,0,0,1,0],
    [0,0,0,0,-1],
    [0,0,0,0,1]
])
E4_ic = np.zeros((10,1))
E5_ic = np.array([
    [100],
    [100],
    [1000],
    [1000],
    [150],
    [0],
    [150],
    [0],
    [150],
    [0],
])

# continuous auxiliary variables

#z_b

E2_zb = np.array([
    [-Mb,0,0,0,0],
    [mb,0,0,0,0],
    [-mb,0,0,0,0],
    [Mb,0,0,0,0]
])
E3_zb = np.array([
    [1, 0],
    [-1, 0],
    [1, 0],
    [-1, 0]
])
E1_zb = np.array([
    [0,0,0,0,0],
    [0,0,0,0,0],
    [1,0,0,0,0],
    [-1,0,0,0,0]
])
E4_zb = np.zeros((4,1))
E5_zb = np.array([
    [0],
    [0],
    [-mb],
    [Mb]
])

#z_grid

E2_zg = np.array([
    [0,-Mg,0,0,0],
    [0,mg,0,0,0],
    [0,-mg,0,0,0],
    [0,Mg,0,0,0]
])
E3_zg = np.array([
    [0,1],
    [0,-1],
    [0,1],
    [0,-1]
])
E1_zg = np.array([
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,1,0,0,0],
    [0,-1,0,0,0]
])
E4_zg = np.zeros((4,1))
E5_zg = np.array([
    [0],
    [0],
    [-mg],
    [Mg]
])

# discrete variales

#E2,E3,E1,E4,E5

#battery (ESS)
E2_db = np.array([
    [-mb,0,0,0,0],
    [-(Mb+eps),0,0,0,0]
])
E3_db = np.zeros((2,2))
E1_db = np.array([
    [1,0,0,0,0],
    [-1,0,0,0,0]
])
E4_db = np.zeros((2,1))
E5_db = np.array([
    [-mb],
    [-eps]
])

#grid
E2_dg = np.array([
    [0,-mg,0,0,0],
    [0,-(Mg+eps),0,0,0]
])
E3_dg = np.zeros((2,2))
E1_dg = np.array([
    [0,1,0,0,0],
    [0,-1,0,0,0]
])
E4_dg = np.zeros((2,1))
E5_dg = np.array([
    [-mg],
    [-eps]
])

#gen 1
E2_d1 = np.array([
    [0,0,-md,0,0],
    [0,0,-(Md+eps),0,0]
])
E3_d1 = np.zeros((2,2))
E1_d1 = np.array([
    [0,0,1,0,0],
    [0,0,-1,0,0]
])
E4_d1 = np.zeros((2,1))
E5_d1 = np.array([
    [-md],
    [-eps]
])

#gen 2
E2_d2 = np.array([
    [0,0,0,-md,0],
    [0,0,0,-(Md+eps),0]
])
E3_d2 = np.zeros((2,2))
E1_d2 = np.array([
    [0,0,0,1,0],
    [0,0,0,-1,0]
])
E4_d2 = np.zeros((2,1))
E5_d2 = np.array([
    [-md],
    [-eps]
])

#gen 3
E2_d3 = np.array([
    [0,0,0,0,-md],
    [0,0,0,0,-(Md+eps)]
])
E3_d3 = np.zeros((2,2))
E1_d3 = np.array([
    [0,0,0,0,1],
    [0,0,0,0,-1]
])
E4_d3 = np.zeros((2,1))
E5_d3 = np.array([
    [-md],
    [-eps]
])

# generator constraint

# E2 delta , E3 zed, E1 u, E4 x, E5

E2_gc = np.array([
    [0,0,6,0,0],
    [0,0,0,6,0],
    [0,0,0,0,6],
    [0,0,-150,0,0],
    [0,0,0,-150,0],
    [0,0,0,0,-150]
])
E3_gc = np.zeros((6,2))
E1_gc = np.array([
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1],
    [0,0,-1,0,0],
    [0,0,0,-1,0],
    [0,0,0,0,-1]
])
E4_gc = np.zeros((6,1))
E5_gc = np.zeros((6,1))

E1 = np.block([
    [E1_sc],
     [E1_ic],
    [E1_zb],
    [E1_zg],
    [E1_db],
    [E1_dg],
    [E1_d1],
    [E1_d2],
    [E1_d3],
    [E1_gc]
])

E2 = np.block([
    [E2_sc],
     [E2_ic],
    [E2_zb],
    [E2_zg],
    [E2_db],
    [E2_dg],
    [E2_d1],
    [E2_d2],
    [E2_d3],
    [E2_gc]
])

E3 = np.block([
    [E3_sc],
     [E3_ic],
    [E3_zb],
    [E3_zg],
    [E3_db],
    [E3_dg],
    [E3_d1],
    [E3_d2],
    [E3_d3],
    [E3_gc]
])

E4 = np.block([
    [E4_sc],
     [E4_ic],
    [E4_zb],
    [E4_zg],
    [E4_db],
    [E4_dg],
    [E4_d1],
    [E4_d2],
    [E4_d3],
    [E4_gc]
])

E5 = np.block([
    [E5_sc],
     [E5_ic],
    [E5_zb],
    [E5_zg],
    [E5_db],
    [E5_dg],
    [E5_d1],
    [E5_d2],
    [E5_d3],
    [E5_gc]
])

# MIQP parameters

import gurobipy as gp
from gurobipy import GRB
from gurobipy import *


n=Amld.shape[1]; m=B1.shape[1]; N_bin = B2.shape[1]; N_z = B3.shape[1]

xmax = 250; xmin = 25
umax = [100,1000,150,150,150]
umin = [-100,-1000,0,0,0]
zmax = [100,1000]
zmin = [-100,-1000]


def transform_delta_to_new_language(delta_matrix):
    """
    Transforms each row of the delta matrix to its decimal equivalent.
    :param delta_matrix: N x 5 matrix of binary values.
    :return: List of integers representing the decimal equivalent of binary rows.
    """
    return [int(''.join(str(int(bit)) for bit in row), 2) for row in delta_matrix]




def hybrid_fhocp(x0, N, power_res, power_load, cbuy, csell, cprod):
    mdl = gp.Model("hybridMPC")
    mdl.Params.LogToConsole = 0
    
    xmin_tile = np.tile(xmin, (N+1,1))
    xmax_tile = np.tile(xmax, (N+1,1))
    zmin_tile = np.tile(zmin, (N,1))
    zmax_tile = np.tile(zmax, (N,1))
    umin_tile = np.tile(umin, (N,1))
    umax_tile = np.tile(umax, (N,1))

    x = mdl.addMVar(shape=(N+1, n), lb=xmin_tile, ub=xmax_tile, name='x') #1*5= 5
    z = mdl.addMVar(shape=(N, N_z), lb=zmin_tile, ub=zmax_tile, name='z') #2*4= 8
    u = mdl.addMVar(shape=(N, m), lb=umin_tile, ub=umax_tile, name='u') # 5*4 = 20
    delta = mdl.addMVar(shape=(N, N_bin), vtype=gp.GRB.BINARY, name='delta') # 5*4=20, total = 53

    # 1 + 1*4 + 30*4 + 1*4= 129 (number of constraints)
    mdl.addConstr(x[0, :] == x0.reshape(Amld.shape[0],))
    for k in range(N):
        mdl.addConstr(x[k+1, :] == Amld @ x[k, :] + B1 @ u[k, :] + B2 @ delta[k, :] + B3 @ z[k,:] + B5.reshape(B1.shape[0],)) # dynamics
        mdl.addConstr(E2 @ delta[k, :] + E3 @ z[k, :] <= E1 @ u[k,:] + E4 @ x[k,:] + E5.reshape(E1.shape[0],)) # mld constraints
        mdl.addConstr(u[k,0]-u[k,1]-u[k,2]-u[k,3]-u[k,4]-power_res[k]+power_load[k] == 0) # power balance

    obj1 = sum(cbuy[k]*z[k,1] - csell[k]*z[k,1] + csell[k]*u[k,1]  for k in range(N)) # cost for power exchanged with grid
    obj2 = sum(cprod[k]*u[k,2:].sum() for k in range(N)) # cost for energy production by dispatchable generators
    mdl.setObjective(obj1 + obj2, GRB.MINIMIZE)

    mdl.optimize()
    
    return mdl




num_iterations = 16000  #128

def save_optimal_fhocp(N, cbuy, csell, cprod, power_load, power_res, num_iterations):
    all_data = []  # List to store all generated data

    for iteration in range(num_iterations):
        i = np.random.randint(cbuy.shape[0] - N)
        x0 = np.random.rand(1, 1) * 225 + 25  # Different x0 for each iteration

        cbuy_tmp = cbuy[i:i + N + 1]
        csell_tmp = csell[i:i + N + 1]
        cprod_tmp = cprod[i:i + N + 1]
        power_res_tmp = power_res[i:i + N + 1]
        power_load_tmp = power_load[i:i + N + 1]

        mdl = hybrid_fhocp(x0, N, power_res_tmp, power_load_tmp, cbuy_tmp, csell_tmp, cprod_tmp)
        mdl.optimize()
        
        # delta = []
        # for k in range(8 * N + 1, 13 * N + 1):
        #     delta.append(mdl.getVars()[k].x)
        # delta = np.array(delta).reshape((N, 5), order='C')
        
        delta = np.array([var.x for var in mdl.getVars() if var.varName.startswith('delta')]).reshape((N, 5), order='C')
        
        delta_transformed = transform_delta_to_new_language(delta)
        
        data = {
            'cbuy_tmp': cbuy_tmp,
            'csell_tmp': csell_tmp,
            'cprod_tmp': cprod_tmp,
            'power_res_tmp': power_res_tmp,
            'power_load_tmp': power_load_tmp,
            'x0': x0, 
            'delta_transformed': delta_transformed
        }

        all_data.append({
            'model': mdl,
            'data': data,
            'index': i,
        })

    # save_data_smartly(all_data, "Microgrid", "Data_Microgrid")  # Save all collected data smartly

    return all_data

all_data = save_optimal_fhocp(N, cbuy, csell, cprod, power_load, power_res, num_iterations)

# print(all_data)


# def save_data_npy(all_data, folder_name="Data_Microgrid"):
#     #check if folder exists
#     os.makedirs(folder_name,exist_ok = True)
    
#     #Current date:
#     current_date = datetime.now().strftime("%Y%m%d")
    
#     for i, data_entry in enumerate(all_data):
        
#         cbuy_tmp = data_entry['data']['cbuy_tmp']
#         csell_tmp = data_entry['data']['csell_tmp']
#         cprod_tmp = data_entry['data']['cprod_tmp']
#         power_res_tmp = data_entry['data']['power_res_tmp']
#         power_load_tmp = data_entry['data']['power_load_tmp']
#         x0 = data_entry['data']['x0']
#         delta_transformed = data_entry['data']['delta_transformed']
        
#         file_name = os.path.join(folder_name, f"Microgrid_data_{i}_{current_date}.npz")

#         np.savez(file_name, cbuy_tmp=cbuy_tmp, csell_tmp=csell_tmp, cprod_tmp=cprod_tmp,
#              power_res_tmp=power_res_tmp, power_load_tmp=power_load_tmp, x0=x0,
#              delta_transformed = delta_transformed)
#     print(f"Data saved in {folder_name}.")



def save_data_npy(all_data, folder_name="Data_Microgrid"):
    # Ensure the folder exists
    os.makedirs(folder_name, exist_ok=True)
    
    # Initialize lists to hold aggregated data for each parameter
    cbuy_all = []
    csell_all = []
    cprod_all = []
    power_res_all = []
    power_load_all = []
    x0_all = []
    delta_transformed_all = []
    
    # Aggregate data
    for data_entry in all_data:
        cbuy_all.append(data_entry['data']['cbuy_tmp'])
        csell_all.append(data_entry['data']['csell_tmp'])
        cprod_all.append(data_entry['data']['cprod_tmp'])
        power_res_all.append(data_entry['data']['power_res_tmp'])
        power_load_all.append(data_entry['data']['power_load_tmp'])
        x0_all.append(data_entry['data']['x0'])
        delta_transformed_all.append(data_entry['data']['delta_transformed'])
    
    # Get the current date for file naming
    current_date = datetime.now().strftime("%Y%m%d")
    # file_name = os.path.join(folder_name, f"Microgrid_aggregated_data_{current_date}.npz")
    file_name = os.path.join(folder_name, f"Microgrid_aggregated_data_N48_{current_date}.npz")
    
    # Save all aggregated arrays in one file
    np.savez(file_name, 
                        cbuy_all=np.array(cbuy_all), 
                        csell_all=np.array(csell_all), 
                        cprod_all=np.array(cprod_all),
                        power_res_all=np.array(power_res_all), 
                        power_load_all=np.array(power_load_all), 
                        x0_all=np.array(x0_all),
                        delta_transformed_all=np.array(delta_transformed_all))
    
    print(f"All data aggregated and saved in {file_name}.")
    return delta_transformed_all
    
save_data_npy(all_data, folder_name="Data_Microgrid")











