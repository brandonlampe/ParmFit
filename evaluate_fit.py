import sys
# sys.path.append('/Users/Lampe/GrantNo456417/Modeling/constit/FEprogram')
# # import call_fortran_loop as cfl
# import fe_module
# import call_fortran_loop as cfl  # calls FE writtin in FORTRAN

import fitting_func as func
# import time
import os
# import csv
import numpy as np
# from scipy import optimize
# import lmfit
# from matplotlib import pyplot as plt
# import matplotlib.gridspec as gridspec

CONSTIT_FIT_LIST = [
    # '175_01',
    # '175_03',
    '175_04',
    '175_09',
    '175_10',
    '175_11',
    # '175_12',
    # '175_13',
    '175_15',
    '175_16',
    # '90_02',
    # '90_03',
    # '90_04',
    # '90_08',
    # '250_03',
]

# run_lbl = '175_10-16el/03'
run_lbl = '03'

# define paths
current_dir = os.getcwd()
load_dir = 'data'
fname = run_lbl + '_parm.csv'
input_fname = run_lbl + '_input.csv'
load_parm_path = os.path.join(current_dir, load_dir, fname)
load_input_path = os.path.join(current_dir, load_dir, input_fname)

# load values for specific fit results
fit_parm = np.loadtxt(load_parm_path)
input_data = np.loadtxt(load_input_path)

PLOT_RESULTS = True
FIG_HEIGHT = 12
FIG_WIDTH = 12

VISC_TAU_BAR = fit_parm[0]  # visco-plastic paramter [1/sec]
VISC_EXP = fit_parm[1]  # visco-plastic parameter
HARD_SFT = fit_parm[2]  # initial slope of hardening curve
HARD_N = fit_parm[3]  # influences shape of hardening curve

# VISC_TAU_BAR = 2.E7  # visco-plastic paramter [1/sec]
# VISC_EXP = 1.5  # visco-plastic parameter
# HARD_SFT = 0.5  # initial slope of hardening curve
# HARD_N = 16.0  # influences shape of hardening curve

# PROBLEM INPUT DATA
POROSITY_ADJ, POROSITY_SCALE, BULK_MOD_BAR, SHEAR_MOD_BAR, \
    NEL, TINC_RATIO, TIME_SCALE, VISC_RULE, HARD_MODEL, \
    STOR_TIME_CNT = input_data
# print STOR_TIME_CNT

func.eval_parm([VISC_TAU_BAR, VISC_EXP, HARD_SFT, HARD_N], POROSITY_ADJ,
               TIME_SCALE, BULK_MOD_BAR, SHEAR_MOD_BAR, NEL, STOR_TIME_CNT,
               TINC_RATIO, VISC_RULE, HARD_MODEL, CONSTIT_FIT_LIST,
               PLOT_RESULTS, FIG_HEIGHT, FIG_WIDTH)
print fit_parm
