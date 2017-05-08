import sys
# sys.path.append('/Users/Lampe/GrantNo456417/Modeling/constit/FEprogram')
# # import call_fortran_loop as cfl
# import fe_module
# import call_fortran_loop as cfl  # calls FE writtin in FORTRAN

import fitting_func as func
import time
import os
import csv
import numpy as np
from scipy import optimize
# import lmfit
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

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
    # '175_16',
    # '90_02',
    # '90_03',
    # '90_04',
    # '90_08',
    # '250_03',
]

RUN_LBL = '02'

# initial parameter guess
VISC_TAU_BAR = 3.  # visco-plastic paramter [1/sec]
VISC_EXP = 1.  # visco-plastic parameter
HARD_SFT = 1.0  # initial slope of hardening curve
HARD_N = 12.0  # influences shape of hardening curve
BNDS = ((0.1, 10.), (1., 4.), (0.1, 10.), (10.0, 40.0))

# use reference values for large or small numbers
VISC_TAU_REF = 1E7  # multiplier of VISC_TAU_BAR
POROSITY_ADJ = True  # ADJUST TO POST-TEST MEASURED POROSITY VALUES

# parameters to calculated constant time step size
POROSITY_SCALE = 0  # MUST BE GREATER THAN ALL MEASURED POROSITY VALUES!
BULK_MOD_BAR = 7.690e9  # [pa]
SHEAR_MOD_BAR = 16.650e9  # [pa]
NEL = 16  # number of elements
TINC_RATIO = 0.95  # ratio of time increment to critical time increment
TIME_SCALE = 1E5

# viscous flow rule: visc_rule
# 0 -> F / tua
# 1 -> F^(1 + phi) / tau
# 2 -> F^(1 + visc_exp * phi) / tau
# 3 -> F^(1 + visc_exp * phi) / tau * arrh_const * exp(-Q/(RT))
VISC_RULE = int(2)  # define visco-plastic flow rule

# CHOICE OF HARDENING MODEL - THIS ARE DIMENSIONLESS TERMS
# 0 -> PERFECTLY PLASTIC: not stable with secant method
# 1 -> LINEAR
# 2 -> NONLINEAR
HARD_MODEL = int(2)

STOR_TIME_CNT = int(500)  # APPROXIMATE NUMBER TIME INCREMENTS FOR STORAGE

# define paths
CURRENT_DIR = os.getcwd()
SAVE_DIR = 'data'
PARM_FNAME = RUN_LBL + '_parm.csv'
PARM_INIT_FNAME = RUN_LBL + '_parm_init.csv'
INPUT_FNAME = RUN_LBL + '_input.csv'
RESULTS_FNAME = RUN_LBL + '_results.npy'

# CALCULATE INDEPENDENT VARIABLE (TIME)
# TDELTA_POROSITY_SCALE = func.const_time_inc(POROSITY_SCALE, NEL, TINC_RATIO,
#                                             STOR_TIME_CNT, BULK_MOD_BAR,
#                                             SHEAR_MOD_BAR, POROSITY_ADJ,
#                                             CONSTIT_FIT_LIST)
# print TDELTA_POROSITY_SCALE

# DEFINE PATHS FOR LOADING AND SAVING FILES
SAVE_PARM_PATH = os.path.join(CURRENT_DIR, SAVE_DIR, PARM_FNAME)
SAVE_INIT_PATH = os.path.join(CURRENT_DIR, SAVE_DIR, PARM_INIT_FNAME)
SAVE_INPUT_PATH = os.path.join(CURRENT_DIR, SAVE_DIR, INPUT_FNAME)
SAVE_RESULTS_PATH = os.path.join(CURRENT_DIR, SAVE_DIR, RESULTS_FNAME)
LOAD_TRACTION_PARM_NAME = 'traction_parm.npy'
LOAD_TRACTION_PATH = os.path.join(CURRENT_DIR,
                                  LOAD_TRACTION_PARM_NAME)

TRACTION_DICT = np.load(LOAD_TRACTION_PATH).item()  # fitted BCTs
WRITE_OUT = [VISC_TAU_BAR * VISC_TAU_REF, VISC_EXP, HARD_SFT, HARD_N]
np.savetxt(SAVE_INIT_PATH, WRITE_OUT)

WRITE_INPUT = [POROSITY_ADJ, POROSITY_SCALE, BULK_MOD_BAR, SHEAR_MOD_BAR,
               NEL, TINC_RATIO, TIME_SCALE, VISC_RULE, HARD_MODEL,
               STOR_TIME_CNT]
np.savetxt(SAVE_INPUT_PATH, WRITE_INPUT)

# func.objective_func([VISC_TAU_BAR, VISC_EXP, HARD_SFT, HARD_N],
#                     VISC_TAU_REF, POROSITY_ADJ,
#                     POROSITY_SCALE, TIME_SCALE,
#                     BULK_MOD_BAR, SHEAR_MOD_BAR, NEL,
#                     STOR_TIME_CNT, TINC_RATIO, VISC_RULE,
#                     HARD_MODEL,
#                     CONSTIT_FIT_LIST)

WALL_TIME_START = time.clock()
print WALL_TIME_START
FIT_RESULTS = optimize.minimize(func.objective_func,
                                [VISC_TAU_BAR, VISC_EXP, HARD_SFT, HARD_N],
                                args=(VISC_TAU_REF, POROSITY_ADJ,
                                      POROSITY_SCALE, TIME_SCALE,
                                      BULK_MOD_BAR, SHEAR_MOD_BAR, NEL,
                                      STOR_TIME_CNT, TINC_RATIO, VISC_RULE,
                                      HARD_MODEL,
                                      CONSTIT_FIT_LIST),
                                tol=1e-6,
                                method='L-BFGS-B',  # 'SLSQP',  #
                                bounds=BNDS,
                                options={'disp': True, 'maxiter': 100})
WALL_TIME_END = time.clock()
print WALL_TIME_END
print "analysis duration (sec): " + str(WALL_TIME_END - WALL_TIME_START)

PARM = FIT_RESULTS.x
VISC_TAU_BAR = PARM[0] * VISC_TAU_REF
VISC_EXP = PARM[1]
HARD_SFT = PARM[2]
HARD_N = PARM[3]
print 'visc_tau: ' + str(VISC_TAU_BAR)
print 'visc_exp: ' + str(VISC_EXP)
print 'hard_sft: ' + str(HARD_SFT)
print 'hard_n: ' + str(HARD_N)
print '--------------------------'
print 'Success: ' + str(FIT_RESULTS.success)
print 'Cause of Termination: ' + str(FIT_RESULTS.status)
print 'Message: ' + FIT_RESULTS.message
print 'Numer of Iterations: ' + str(FIT_RESULTS.nit)

print FIT_RESULTS
# print fit_results.keys()

WRITE_OUT = [VISC_TAU_BAR, VISC_EXP, HARD_SFT, HARD_N]
np.savetxt(SAVE_PARM_PATH, WRITE_OUT)
np.save(SAVE_RESULTS_PATH, FIT_RESULTS)
