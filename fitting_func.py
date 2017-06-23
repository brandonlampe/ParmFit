import numpy as np
import os
from lmfit import Model
import sys
sys.path.append('/Users/Lampe/GrantNo456417/Modeling/constit/FEprogram')
# import call_fortran_loop as cfl
import fe_module
import call_fortran_loop as cfl  # calls FE writtin in FORTRAN
import time
from scipy import interpolate
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from math import floor, log10, isnan
# import matplotlib as mpl
# mpl.rc('text', usetex=True)

def traction_resid(pars, x, data=None, eps=None):
    """
        pars = dictionary of parameters for fit
        x = independent variable, time [secionds]
        eps = uncertainty in measured data
        data = measured values

        model = modeled pressure / BCT
        returns -> error between modeled and measured pressure
    """
    parvals = pars.valuesdict()
    n_fact = parvals['n_fact']
    ramp_fact_denom = parvals['ramp_fact_denom']
    p_mpa = data['p_mpa']

    p_start = p_mpa[0]
    p_end = p_mpa[-1]
    const_idx = (np.abs(p_mpa - p_end)).argmin()
    ramp_steps = const_idx + 1
    const_steps = len(x) - ramp_steps

    prcnt = np.concatenate((np.linspace(0, 1, ramp_steps),
                           np.ones(const_steps)),
                           axis=0)
    model = p_start + (p_end - p_start) *\
        np.tanh((prcnt / ramp_fact_denom)**n_fact)

    if data is None:
        return model
    if eps is None:
        return model - p_mpa
    return (model - p_mpa) / eps


def traction_func(time_vec, start_steps, ramp_steps,
                  p_start, p_end,
                  n_fact, ramp_fact_denom):
    total_steps = len(time_vec)  # independent variable
    p_con_start = np.ones((start_steps)) * p_start

    prcnt = np.linspace(0, 1, ramp_steps - start_steps)
    p_ramp = p_start + (p_end - p_start) *\
        np.tanh((prcnt / ramp_fact_denom)**n_fact)
    if len(p_ramp) > 0:
        p_con_end = np.ones((total_steps - ramp_steps)) * p_ramp[-1]
    else:
        p_con_end = np.zeros((total_steps - ramp_steps))

    model = np.concatenate((p_con_start, p_ramp, p_con_end), axis=0)
    return model


def load_parm_data(data_list, porosity_adj=False):
    """
        loads test data that will be used to fit material parameters
        - returns a dict of data that can be accessed via:
            meas_dict['test_id']['parameter_id']
    """
    repo_dir = '/Users/Lampe/GrantNo456417/CurrentTesting/CrushedSaltRepo'
    folder = 'UNM_WP_HY_'
    sub_folder = 'PARM_DATA'
    file_type = '_OUT.csv'

    adj_dir = '/Porosity_Adjustment'
    adj_dict = np.load(repo_dir + adj_dir + '/porosity_adj.npy').item()

    # data column indexes
    idx_time_sec = 2
    idx_pc_mpa = 8
    idx_pp_mpa = 9
    idx_temp_c = 12
    idx_fden = 4  # INTERPOLATED FRACTIONAL DENSITY - NOT A FIT
    idx_vstrn_rate = 7  # calculated from curve fit to measured strain

    meas_dict = {}
    for i in data_list:
        test = folder + i
        file_path = os.path.join(repo_dir, test, sub_folder, i + file_type)
        print "load file: " + file_path
        all_data = np.loadtxt(file_path, dtype='float', delimiter=',',
                              skiprows=3)
        time_sec = all_data[:, idx_time_sec]
        pc_mpa = all_data[:, idx_pc_mpa]
        pp_mpa = all_data[:, idx_pp_mpa]
        temp_c = all_data[:, idx_temp_c]
        porosity = 1. - all_data[:, idx_fden]
        vstrn_rate = all_data[:, idx_vstrn_rate]

        if porosity_adj:
            # print "porosity load: " + str(porosity)
            porosity = porosity + adj_dict[i]  # apply porosity adjustment
            # print "porosity adj i: " + i
            # print "adj value: " + str(adj_dict[i])
            # print "porosity load adj: " + str(porosity)
        meas_dict[i] = {'time_sec': time_sec,
                        'pc_mpa': pc_mpa,
                        'pp_mpa': pp_mpa,
                        'temp_c': temp_c,
                        'porosity': porosity,
                        'vstrn_rate': vstrn_rate}
    return meas_dict


def objective_func(parm_list, visc_tau_ref, porosity_adj, porosity_scale,
                   time_scale, bulk_mod_bar, shear_mod_bar, nel,
                   stor_time_cnt, tinc_ratio,
                   visc_rule, hard_model, constit_fit_list):
    """ calculate residual"""

    # must be integers
    visc_rule = int(visc_rule)
    hard_model = int(hard_model)

    # parameters to be fit
    visc_tau_bar_parm = parm_list[0]  # visco-plastic paramter [1/sec]
    visc_tau_bar = visc_tau_bar_parm * visc_tau_ref  # for parm fit
    visc_exp = parm_list[1]  # visco-plastic parameter
    hard_sft = parm_list[2]  # initial slope of hardening curve
    hard_n = parm_list[3]  # influences shape of hardening curve

    # constants
    tol = 1e-6  # 1.e-8  # tolerance for +- f
    den_bar = 2.16e3  # density of solid material [kg/m^3]
    mat_type = 1
    # TEMPERATURE
    arrh_const = 488300.1  # constant factor in front of Arrhenius law

    yield_0_bar = 10.E3  # initial yield stress [pa]
    yield_model = 0  # 0=> mises stress
    yield_compare = 1  # 0=> general, 1 => uniaxial stress
    lambda_inc_0 = 5.0e-10  # initial increment size, rate ind. plastic

    load_traction_parm_name = 'traction_parm.npy'
    parm_dir = '/Users/Lampe/GrantNo456417/Modeling/constit/ParmFit'
    load_traction_path = os.path.join(parm_dir,
                                      load_traction_parm_name)

    meas_dict = load_parm_data(constit_fit_list, porosity_adj)  # lab values
    traction_dict = np.load(load_traction_path).item()  # fitted BCTs

    sum_scaled_error_norm = 0.
    for i in constit_fit_list:
        # ####################################################################
        # TEST SPECIFIC DATA
        # define tractions - dimension: pressure (MPa), time (sec)
        # ####################################################################
        pcon_start_bar = traction_dict[i]['pc_mpa']['p_start'] * 10**6
        pcon_end_bar = traction_dict[i]['pc_mpa']['p_end'] * 10**6
        pcon_ramp_start_sec = traction_dict[i]['pc_time']['time_ramp_start']
        pcon_ramp_end_sec = traction_dict[i]['pc_time']['time_ramp_end']
        pcon_n_fact = traction_dict[i]['pc_mpa']['n_fact']
        pcon_ramp_fact_denom = traction_dict[i]['pc_mpa']['ramp_fact_denom']

        ppor_start_bar = traction_dict[i]['pp_mpa']['p_start'] * 10**6
        ppor_end_bar = traction_dict[i]['pp_mpa']['p_end'] * 10**6
        ppor_ramp_start_sec = traction_dict[i]['pp_time']['time_ramp_start']
        ppor_ramp_end_sec = traction_dict[i]['pp_time']['time_ramp_end']
        ppor_n_fact = traction_dict[i]['pp_mpa']['n_fact']
        ppor_ramp_fact_denom = traction_dict[i]['pp_mpa']['ramp_fact_denom']

        meas_time = meas_dict[i]['time_sec']
        meas_porosity = meas_dict[i]['porosity']
        # meas_steps = len(meas_porosity)
        # stor_time_cnt = meas_steps  # NUMBER TIME INCREMENTS TO STORE
        phi = meas_porosity[0]  # initial porosity (already dimensionless)
        total_time_sec = meas_time[-1]  # SECONDS (86400 sec/day)
        temp_c = 170.0  # degrees celsius

        # ####################################################################
        # SCALE TIME PARAMETERS
        # ####################################################################
        scaled_total_time_sec = total_time_sec / time_scale
        scaled_visc_tau_bar = visc_tau_bar / time_scale
        scaled_pcon_ramp_start_sec = pcon_ramp_start_sec / time_scale
        scaled_pcon_ramp_end_sec = pcon_ramp_end_sec / time_scale
        scaled_ppor_ramp_start_sec = ppor_ramp_start_sec / time_scale
        scaled_ppor_ramp_end_sec = ppor_ramp_end_sec / time_scale

        # SPATIAL DIMENSIONS
        rmax_bar = 1.0  # [m]
        rmin_bar = (6. * phi * rmax_bar**3 / np.pi)**(1.0 / 3.)  # [m]

        # timer to monitor anaysis duration
        wall_time_start = time.clock()
        # ####################################################################
        # CALL FE PROGRAM
        # ####################################################################
        out_nmat, out_emat, out_smat, plot_tinc, ref_dict = cfl.call_fe_func(
            stor_time_cnt=int(stor_time_cnt),
            plot_time_cnt=int(stor_time_cnt),
            nel=int(nel),
            tinc_ratio=tinc_ratio,
            temp_c=temp_c,
            mat_type=int(mat_type),
            rmax_bar=rmax_bar,
            rmin_bar=rmin_bar,
            pcon_start_bar=pcon_start_bar,
            pcon_end_bar=pcon_end_bar,
            pcon_parm=[pcon_ramp_fact_denom, pcon_n_fact,
                       scaled_pcon_ramp_start_sec, scaled_pcon_ramp_end_sec,
                       scaled_total_time_sec],
            ppor_start_bar=ppor_start_bar,
            ppor_end_bar=ppor_end_bar,
            ppor_parm=[ppor_ramp_fact_denom, ppor_n_fact,
                       scaled_ppor_ramp_start_sec, scaled_ppor_ramp_end_sec],
            yield_model=int(yield_model),
            yield_compare=int(yield_compare),
            yield_0_bar=yield_0_bar,
            visc_rule=int(visc_rule),
            visc_tau_bar=scaled_visc_tau_bar,
            visc_exp=visc_exp,
            arr_const=arrh_const,
            tol=tol,
            lambda_inc_0=lambda_inc_0,
            hard_model=int(hard_model),
            hard_sft=hard_sft,
            hard_n=hard_n,
            bulk_mod_bar=bulk_mod_bar,
            shear_mod_bar=shear_mod_bar,
            den_bar=den_bar,
            write_test_input=0,
            debug_cnt=-1,
            debug_stps=0,
            strn_type=1,
            time_scale=time_scale,
            porosity_scale=0)
        wall_time_end = time.clock()
        print "analysis duration (sec): " + str(wall_time_end -
                                                wall_time_start)
        t_ref = ref_dict['t_ref']
        tvec_bar = out_emat[:, 48, 0] * t_ref * time_scale  # [sec]
        pred_porosity = out_smat[:, 1]
        meas_porosity_func = interpolate.interp1d(meas_time, meas_porosity)
        interp_porosity = meas_porosity_func(tvec_bar)

        # scaled p-2 norm
        error_vec = interp_porosity - pred_porosity
        scaled_error_norm = (np.sum(error_vec**2)**(0.5)) /\
            (len(tvec_bar)**(0.5))
        print "scaled p-2 error norm:" + str(scaled_error_norm)
        if isnan(scaled_error_norm):
            scaled_error_norm = 1.
        sum_scaled_error_norm = sum_scaled_error_norm + scaled_error_norm
        print "sum p-2 error norm: " + str(sum_scaled_error_norm)
    return sum_scaled_error_norm


def eval_parm(parm_list, porosity_adj,
              time_scale, bulk_mod_bar, shear_mod_bar, nel, stor_time_cnt,
              tinc_ratio, visc_rule, hard_model, constit_fit_list,
              plot, fig_height, fig_width):

    """ calculate residual"""
    print "porosity adj: " + str(porosity_adj)
    # must be integers
    visc_rule = int(visc_rule)
    hard_model = int(hard_model)

    # parameters to be fit
    visc_tau_bar = parm_list[0]  # visco-plastic paramter [1/sec]
    visc_exp = parm_list[1]  # visco-plastic parameter
    hard_sft = parm_list[2]  # initial slope of hardening curve
    hard_n = parm_list[3]  # influences shape of hardening curve

    # constants
    tol = 1e-6  # 1.e-8  # tolerance for +- f
    den_bar = 2.16e3  # density of solid material [kg/m^3]
    mat_type = 1
    # TEMPERATURE
    arrh_const = 488300.1  # constant factor in front of Arrhenius law

    yield_0_bar = 10.E3  # initial yield stress [pa]
    yield_model = 0  # 0=> mises stress
    yield_compare = 1  # 0=> general, 1 => uniaxial stress
    lambda_inc_0 = 5.0e-10  # initial increment size, rate ind. plastic

    load_traction_parm_name = 'traction_parm.npy'
    parm_dir = '/Users/Lampe/GrantNo456417/Modeling/constit/ParmFit'
    load_traction_path = os.path.join(parm_dir,
                                      load_traction_parm_name)

    meas_dict = load_parm_data(constit_fit_list, porosity_adj)  # lab values
    print "meas porosity:"
    print meas_dict['175_16']['porosity']
    traction_dict = np.load(load_traction_path).item()  # fitted BCTs

    # plotting parameters
    subplot_idx = len(constit_fit_list)
    plot_idx = 1
    FIG1 = plt.figure(figsize=([fig_width, fig_height]), tight_layout=True)
    time_fmt = mpl.ticker.FormatStrFormatter('%1.0e')  # format figure axis

    for i in constit_fit_list:
        AX1 = FIG1.add_subplot(subplot_idx, 2, plot_idx)
        AX2 = FIG1.add_subplot(subplot_idx, 2, plot_idx + 1)
        AX1.grid()
        AX1.set_ylabel(r'Porosity')
        AX2.grid()
        AX2.set_ylabel(r'Vol. Strain Rate $[sec^{-1}]$')
        plot_idx += 2

        # define tractions - dimension: pressure (MPa), time (sec)
        pcon_start_bar = traction_dict[i]['pc_mpa']['p_start'] * 10**6
        pcon_end_bar = traction_dict[i]['pc_mpa']['p_end'] * 10**6
        pcon_ramp_start_sec = traction_dict[i]['pc_time']['time_ramp_start']
        pcon_ramp_end_sec = traction_dict[i]['pc_time']['time_ramp_end']
        pcon_n_fact = traction_dict[i]['pc_mpa']['n_fact']
        pcon_ramp_fact_denom = traction_dict[i]['pc_mpa']['ramp_fact_denom']

        ppor_start_bar = traction_dict[i]['pp_mpa']['p_start'] * 10**6
        ppor_end_bar = traction_dict[i]['pp_mpa']['p_end'] * 10**6
        ppor_ramp_start_sec = traction_dict[i]['pp_time']['time_ramp_start']
        ppor_ramp_end_sec = traction_dict[i]['pp_time']['time_ramp_end']
        ppor_n_fact = traction_dict[i]['pp_mpa']['n_fact']
        ppor_ramp_fact_denom = traction_dict[i]['pp_mpa']['ramp_fact_denom']

        meas_porosity = meas_dict[i]['porosity']
        meas_time = meas_dict[i]['time_sec']
        meas_vstrn_rate = meas_dict[i]['vstrn_rate']

        # ####################################################################
        # TEST SPECIFIC DATA
        # ####################################################################
        # stor_time_cnt = meas_steps  # NUMBER TIME INCREMENTS TO STORE
        phi = meas_porosity[0]  # initial porosity (already dimensionless)
        total_time_sec = meas_time[-1]  # SECONDS (86400 sec/day)
        temp_c = 170.0  # degrees celsius

        # ####################################################################
        # SCALE TIME PARAMETERS
        # ####################################################################
        scaled_total_time_sec = total_time_sec / time_scale
        scaled_visc_tau_bar = visc_tau_bar / time_scale
        scaled_pcon_ramp_start_sec = pcon_ramp_start_sec / time_scale
        scaled_pcon_ramp_end_sec = pcon_ramp_end_sec / time_scale
        scaled_ppor_ramp_start_sec = ppor_ramp_start_sec / time_scale
        scaled_ppor_ramp_end_sec = ppor_ramp_end_sec / time_scale

        # SPATIAL DIMENSIONS
        rmax_bar = 1.0  # [m]
        rmin_bar = (6. * phi * rmax_bar**3 / np.pi)**(1.0 / 3.)  # [m]

        # timer to monitor anaysis duration
        wall_time_start = time.clock()
        # ####################################################################
        # CALL FE PROGRAM
        # ####################################################################
        out_nmat, out_emat, out_smat, plot_tinc, ref_dict = cfl.call_fe_func(
            stor_time_cnt=int(stor_time_cnt),
            plot_time_cnt=int(stor_time_cnt),
            nel=int(nel),
            tinc_ratio=tinc_ratio,
            temp_c=temp_c,
            mat_type=int(mat_type),
            rmax_bar=rmax_bar,
            rmin_bar=rmin_bar,
            pcon_start_bar=pcon_start_bar,
            pcon_end_bar=pcon_end_bar,
            pcon_parm=[pcon_ramp_fact_denom, pcon_n_fact,
                       scaled_pcon_ramp_start_sec, scaled_pcon_ramp_end_sec,
                       scaled_total_time_sec],
            ppor_start_bar=ppor_start_bar,
            ppor_end_bar=ppor_end_bar,
            ppor_parm=[ppor_ramp_fact_denom, ppor_n_fact,
                       scaled_ppor_ramp_start_sec, scaled_ppor_ramp_end_sec],
            yield_model=int(yield_model),
            yield_compare=int(yield_compare),
            yield_0_bar=yield_0_bar,
            visc_rule=int(visc_rule),
            visc_tau_bar=scaled_visc_tau_bar,
            visc_exp=visc_exp,
            arr_const=arrh_const,
            tol=tol,
            lambda_inc_0=lambda_inc_0,
            hard_model=int(hard_model),
            hard_sft=hard_sft,
            hard_n=hard_n,
            bulk_mod_bar=bulk_mod_bar,
            shear_mod_bar=shear_mod_bar,
            den_bar=den_bar,
            write_test_input=0,
            debug_cnt=-1,
            debug_stps=0,
            strn_type=1,
            time_scale=time_scale,
            porosity_scale=0)
        wall_time_end = time.clock()
        print "analysis duration (sec): " + str(wall_time_end -
                                                wall_time_start)
        t_ref = ref_dict['t_ref']
        tvec_bar = out_emat[:, 48, 0] * t_ref * time_scale
        pred_porosity = out_smat[:, 1]
        meas_porosity_func = interpolate.interp1d(meas_time, meas_porosity)
        interp_porosity = meas_porosity_func(tvec_bar)
        meas_vstrn_rate_func = interpolate.interp1d(meas_time, meas_vstrn_rate)
        interp_vstrn_rate = meas_vstrn_rate_func(tvec_bar)
        # pcon = out_smat[:, 13]
        # ppor = out_smat[:, 12]
        # pdif = pcon - ppor
        vstrn_rate = out_smat[:, 14] / t_ref

        left, width = 0.1, 0.2
        bottom, height = 1., 0.1
        right = left + width
        top = bottom + height
        AX1.plot(tvec_bar, interp_porosity, 'b.', label=r'Meas $\phi$')
        AX1.plot(tvec_bar, pred_porosity, '-r.', label=r'Pred $\phi$')
        # label=r'$P_p$',
        # linestyle='-',
        # linewidth=0.5, marker='.', markersize=5,
        # alpha=1)
        AX1.xaxis.set_major_formatter(time_fmt)
        mpl.pyplot.sca(AX1)  # set current axis
        plt.xticks(rotation=20, fontsize=10)
        plt.yticks(fontsize=10)
        AX1.legend(loc=0, fontsize=10)
        p = mpl.patches.Rectangle(
            (left, bottom), width, height, fill=False, alpha=0.7,
            edgecolor='none', facecolor='w', #  linewith='None',
            transform=AX1.transAxes, clip_on=False)
        AX1.add_patch(p)
        AX1.text(left, bottom, 'Test ID: ' + i,
                 horizontalalignment='left',
                 verticalalignment='bottom', transform=AX1.transAxes,
                 fontsize=15, color='blue')

        AX2.semilogy(tvec_bar, interp_vstrn_rate, 'b.',
                     label=r'Meas $\dot{e}_{vol}$')
        AX2.semilogy(tvec_bar, vstrn_rate, '-r.',
                     label=r'Pred $\dot{e}_{vol}$')
        AX2.xaxis.set_major_formatter(time_fmt)
        mpl.pyplot.sca(AX2)  # set current axis
        plt.xticks(rotation=20, fontsize=10)
        plt.yticks(fontsize=10)
        AX2.legend(loc=0, fontsize=10)

    AX1.set_xlabel(r'Test Duration [sec]')
    AX2.set_xlabel(r'Test Duration [sec]')
    plt.tight_layout()
    if plot:
        plt.savefig('Eval_Summary.pdf')
        plt.show()
    return


def const_time_inc(porosity_scale, nel, tinc_ratio, stor_time_cnt,
                   bulk_mod_bar, shear_mod_bar, porosity_adj,
                   constit_fit_list):
    # dimensioned constants
    den_bar = 2160.  # [kg/m3]
    rmax_bar = 1.  # [m]
    rmin_bar = (6. * porosity_scale * rmax_bar**3 / np.pi)**(1.0 / 3.)  # [m]

    # refernce values
    sigma_ref = 1e6  # [pa]
    r_ref = rmax_bar - rmin_bar  # [m]
    den_ref = 2.16e3  # [kg/m^3]
    t_ref = np.sqrt(den_ref * r_ref**2 / sigma_ref)  # [sec]

    # dimensionless values below
    rmax = rmax_bar / r_ref
    rmin = rmin_bar / r_ref
    den = den_bar / den_ref
    bulk_mod = bulk_mod_bar / sigma_ref
    shear_mod = shear_mod_bar / sigma_ref
    youngs = (9 * bulk_mod * shear_mod) / (3 * bulk_mod + shear_mod)
    el_h = (rmax - rmin) / float(nel)  # dimensionless
    wave_speed = np.sqrt(youngs / den)  # dimensionless
    tdelta_crit = el_h / wave_speed  # dimensionless
    tdelta = tdelta_crit * tinc_ratio  # time step size

    # return dimensioned time step size
    # tdelta_bar = tdelta * t_ref  # [sec]
    # meas_dict = load_parm_data(constit_fit_list, porosity_adj)  # lab values

    # stor_time_list = []
    # stor_time_list +
    # inc = 0
    # for i in constit_fit_list:
    #     meas_time = meas_dict[i]['time_sec']  # [sec]
    #     total_time = meas_time[-1] / t_ref  # dimensionless
    #     t_steps = int(total_time / tdelta)  # number of time steps

    #     stor_time_inc = int(t_steps / stor_time_cnt)
    #     stor_time_vec[inc] = np.linspace(stor_time_inc, t_steps,
    #                                         num=stor_time_cnt, dtype=int)
    #     inc += 1
    return tdelta * t_ref

