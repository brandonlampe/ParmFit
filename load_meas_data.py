"""
    functions for loading measured data
"""
import os
from scipy.interpolate import griddata
import numpy as np

def load_parm_data(data_list, max_fit_points):
    """
        data used to fit material parameters
    """
    repo_dir = '/Users/Lampe/GrantNo456417/CurrentTesting/CrushedSaltRepo'
    folder = 'UNM_WP_HY_'
    sub_folder = 'PARM_DATA'
    file_type = '_OUT.csv'
    data_col = 5

    # data column indexes
    idx_time_sec = 2
    idx_pc_mpa = 8
    idx_pp_mpa = 9
    idx_temp_c = 12
    idx_fden = 4

    meas_dict = {}
    # meas_data = np.zeros((int(max_fit_points), data_col, len(data_list)))
    print len(data_list)
    for i in data_list:
        test = folder + i
        file_path = os.path.join(repo_dir, test, sub_folder, i + file_type)
        print file_path
        all_data = np.loadtxt(file_path, dtype='float', delimiter=',',
                              skiprows=3)
        time_sec = all_data[:, idx_time_sec]
        pc_mpa = all_data[:, idx_pc_mpa]
        pp_mpa = all_data[:, idx_pp_mpa]
        temp_c = all_data[:, idx_temp_c]
        porosity = 1. - all_data[:, idx_fden]

        # data = np.array((time_sec, pc_mpa, pp_mpa, temp_c, porosity)).T
        meas_dict[i] = {'time_sec': time_sec,
                        'pc_mpa': pc_mpa,
                        'pp_mpa': pp_mpa,
                        'temp_c': temp_c,
                        'porosity': porosity}

    return meas_dict
