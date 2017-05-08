"""
module for fitting smooth functions to measured pressure tractions
steps:
    1) load measured data stored in "PARM_FIT" folder
    2) fit Pc and Pp to smooth curve to be used as BCTs
"""

import fitting_func as func
import os
import numpy as np
import lmfit
from matplotlib import pyplot as plt
from matplotlib import ticker
import matplotlib as mpl
# mpl.rc('text', usetex=True)

meas_data_list = [
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
PLOT = True  # only works correctly when one data set is analyzed
POROSITY_ADJ = True
STORE_TRACTION_PARM_NAME = 'traction_parm.npy'
# CONSTIT_PATH = '/Users/Lampe/GrantNo456417/Modeling/constit/FEprogram/data'
STORE_PARM_DIR = '/Users/Lampe/GrantNo456417/Modeling/constit/ParmFit'
STORE_TRACTION_PATH = os.path.join(STORE_PARM_DIR,
                                   STORE_TRACTION_PARM_NAME)
# start fitting routine
meas_dict = func.load_parm_data(meas_data_list, porosity_adj=POROSITY_ADJ)

# smooth curves to measured pressure data (tractions)
traction_model = lmfit.Model(func.traction_func)
traction_dict = {}

# plotting parameters
subplot_idx = len(meas_data_list)
plot_idx = 1
FIG1 = plt.figure(figsize=(12, 12), tight_layout=True)
time_fmt = ticker.FormatStrFormatter('%1.0e')  # format figure axis
for j in meas_data_list:  # loop over each set of test data
    traction_dict[j] = {}
    meas_time = meas_dict[j]['time_sec']

    AX2 = FIG1.add_subplot(subplot_idx, 3, plot_idx)
    AX1 = FIG1.add_subplot(subplot_idx, 3, plot_idx + 1)
    AX3 = FIG1.add_subplot(subplot_idx, 3, plot_idx + 2)
    AX1.grid()
    AX1.set_ylabel('Pressure')
    AX2.grid()
    AX2.set_ylabel('Porosity')
    AX3.grid()
    AX3.set_ylabel(r'Vol. Strain Rate')
    plot_idx += 3

    for i in xrange(2):  # fit parameters to both Pc and Pp
        if i == 0:
            bc_type = 'pc_mpa'
            time_type = 'pc_time'
        elif i == 1:
            bc_type = 'pp_mpa'
            time_type = 'pp_time'

        meas_p = meas_dict[j][bc_type]
        p_start = round(meas_p[0], 0)
        start_steps = 0
        tol = 0.5  # mpa
        if p_start < tol:
            p_start = 0.

        p_end = round(meas_p[-1], 0)
        if p_end < tol:
            p_end = 0.
            start_steps = 0
        else:
            while abs(meas_p[start_steps] - p_start) <= tol:
                start_steps += 1
        if start_steps > 20:
            start_steps = start_steps - 20

        ramp_steps = (np.abs(meas_p - p_end)).argmin()

        fit_parm = traction_model.make_params()
        fit_parm['n_fact'].set(value=50, min=1.5, max=100, vary=True)
        fit_parm['ramp_fact_denom'].set(value=0.7, min=0.01, max=0.9,
                                        vary=True)
        fit_parm['start_steps'].set(value=start_steps, vary=False)
        fit_parm['ramp_steps'].set(value=ramp_steps, vary=False)
        fit_parm['p_start'].set(value=p_start, vary=False)
        fit_parm['p_end'].set(value=p_end, vary=False)

        traction_fit = traction_model.fit(meas_p, fit_parm, time_vec=meas_time)
        # store values in dictionary
        traction_dict[j][bc_type] = traction_fit.best_values
        traction_dict[j][time_type] = {
            'time_ramp_start': meas_time[start_steps],
            'time_ramp_end': meas_time[ramp_steps],
            'time_total': meas_time[-1]}

        if i == 0:
            AX1.plot(meas_time, meas_p, 'b.', label=r'Exp $P_c$')
            AX1.plot(meas_time, traction_fit.best_fit, 'r-',
                     label=r'Fit $P_c$')
        else:
            AX1.plot(meas_time, meas_p, 'g.', label=r'Exp $P_p$')
            AX1.plot(meas_time, traction_fit.best_fit, 'm-',
                     label=r'Fit $P_p$')
        AX1.xaxis.set_major_formatter(time_fmt)
        mpl.pyplot.sca(AX1)  # set current axis
        plt.xticks(rotation=20, fontsize=10)
        plt.yticks(fontsize=10)
        AX1.legend(loc=0, fontsize=8)

    AX2.plot(meas_time, meas_dict[j]['porosity'], 'k.-', label=j)
    AX2.xaxis.set_major_formatter(time_fmt)
    mpl.pyplot.sca(AX2)  # set current axis
    plt.xticks(rotation=20, fontsize=10)
    plt.yticks(fontsize=10)
    AX2.legend(loc=0, fontsize=8)

    AX3.semilogy(meas_time, meas_dict[j]['vstrn_rate'], 'k.-', label=j)
    AX3.xaxis.set_major_formatter(time_fmt)
    mpl.pyplot.sca(AX3)  # set current axis
    plt.xticks(rotation=20, fontsize=10)
    plt.yticks(fontsize=10)
    AX3.legend(loc=0, fontsize=8)
    print '---------'
    print j
    print traction_dict[j]

np.save(STORE_TRACTION_PATH, traction_dict)
# AX1.set_xlabel(r'sec $\times$ {:,.0}'.format(TIME_SCALE))
AX1.set_xlabel(r'sec')
AX2.set_xlabel(r'sec')

if PLOT:
    plt.savefig('Test_Summary.pdf')
    plt.show()

