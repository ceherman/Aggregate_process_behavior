import numpy as np
import pandas as pd
from scipy import integrate
import math

# Specific to the Akta Explorer_________________________________________________
def cond_2_tis(cond):
    """Converts conductivity to total ionic strength for the Akta Explorer system
    Input:  Conductivity [mS/cm]
    Output:  Total ionic strength [mM]  """
    tis = (0.0229207306*(cond**2) + 9.3999636424*cond + 21.2424910935)
    return tis

def read_data_normalized(file, get_tis=True):
    """Read Akta Explorer data that has been exported in normalized format"""
    df = pd.read_excel(file)
    properties = list(df.loc[0])
    for ind, prop in enumerate(properties):
        if ind == 0:
            if df.iat[1, ind] == 'min':
                properties[ind] = 'time_min'
            elif df.iat[1, ind] == 'ml':
                properties[ind] = 'volume_ml'
            else:
                raise ValueError('Check the normalized x-axis units')
        else:
            name        = properties[ind][(properties[ind].rfind(':')+1):]
            abbrev_name = name[(name.find('_')+1):]
            abbrev_name = mod_abbrev_name(abbrev_name)
            units = df.iat[1, ind][(df.iat[1, ind].find(' ')+1):]
            units = mod_units(units)
            properties[ind] = abbrev_name + '_' + units

    df.columns = properties
    df.drop(labels=[0, 1], axis='index', inplace=True)
    df.dropna(axis='index', inplace=True)
    if get_tis:
        df['tis_mM'] = cond_2_tis(df['cond_mS_cm'])
    df.reset_index(drop=True, inplace=True)
    return df

def load_and_clean_xls(name):
    df = pd.read_excel(name)

    properties = list(df.loc[0])
    for ind, prop in enumerate(properties):
        if ind%2 == 0:
            name        = properties[ind][(properties[ind].rfind(':')+1):]
            abbrev_name = name[(name.find('_')+1):]
            abbrev_name = mod_abbrev_name(abbrev_name)
            units       = df.iat[1, ind]
        else:
            units       = df.iat[1, ind][(df.iat[1, ind].find(' ')+1):]
            units       = mod_units(units)
        properties[ind] = abbrev_name + '_' + units

    df.columns = properties
    df.drop(labels=[0, 1], axis='index', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def mod_abbrev_name(abbrev_name):
    abbrev_name = abbrev_name.replace('UV1', 'uv')
    abbrev_name = abbrev_name.replace('UV2', 'uv')
    abbrev_name = abbrev_name.replace('UV3', 'uv')
    abbrev_name = abbrev_name.replace('nm', '')
    abbrev_name = abbrev_name[0].lower() + abbrev_name[1:]
    return abbrev_name

def mod_units(units):
    units = units.replace('/', '_')
    units = units.replace('°', '')
    units = units.replace('%', '')
    return units


# Akta Pure_____________________________________________________________________

def load_and_clean_csv(name):
    df = pd.read_csv(name)
    df.dropna(axis=1, how='all', inplace=True)

    properties = list(df.loc[0])
    for ind, prop in enumerate(properties):
        if ind%2 == 0:
            name = prop.replace(' ', '_')
            name = mod_abbrev_name_pure(name)
            units = df.iat[1, ind]
        elif name != 'pH':
            units = df.iat[1, ind]
            units = mod_units(units)
        else:
            units = 'pH'
        properties[ind] = name + '_' + units

    df.columns = properties
    df.drop(labels=[0, 1], axis='index', inplace=True)
    df.reset_index(drop=True, inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    return df


def mod_abbrev_name_pure(abbrev_name):
    abbrev_name = abbrev_name.replace('UV_1', 'uv')
    abbrev_name = abbrev_name.replace('UV_2', 'uv')
    abbrev_name = abbrev_name.replace('UV_3', 'uv')
    return abbrev_name




# General_______________________________________________________________________
def correct_baseline_drift(x, y):
    """Apply a linear correction to baseline drift"""
    x = np.array(x)
    y = np.array(y)
    # Identify starting and stopping points
    y_pts = [y[0], y[-1]]
    x_pts = [x[0], x[-1]]
    # Correct for drift
    y_new = y - np.interp(x, x_pts, y_pts)
    return y_new

def get_first_moment(x, y):
    return integrate.simps(y*x, x)/integrate.simps(y, x)
















# import pandas as pd
# import numpy as np
# import numpy.polynomial.polynomial as poly
# import scipy
# from scipy import interpolate
# from scipy import stats
# import scipy.integrate as integrate
# import math
#
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# from matplotlib.ticker import NullFormatter
#
# import os
# from sklearn.linear_model import LinearRegression
# import random
#





# def instantiate_ax(x_label=None, y_label=None, font_size=20, x_size=8, y_size=6):
#     fig, ax = plt.subplots()
#     font = {'weight' : 'normal', 'size'   : font_size}
#     plt.rc('font', **font)
#     fig.set_size_inches(x_size, y_size, forward=True)
#     if x_label is not None:
#         ax.set_xlabel(x_label)
#     if y_label is not None:
#         ax.set_ylabel(y_label)
#     return fig, ax
#
# def instantiate_twinx(ax, rhs_y_label=None):
#     ax2 = ax.twinx()
#     if rhs_y_label is not None:
#         ax2.set_ylabel(rhs_y_label)
#     return ax2
#
# def align_yaxis(ax1, ax2):
#     y_lims = np.array([ax.get_ylim() for ax in [ax1, ax2]])
#
#     # force 0 to appear on both axes
#     y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
#     y_lims[:, 1] = y_lims[:, 1].clip(0, None)
#
#     # normalize both axes
#     y_mags = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
#     y_lims_normalized = y_lims / y_mags
#
#     # find combined range
#     y_new_lims_normalized = np.array([np.min(y_lims_normalized),
#                                       np.max(y_lims_normalized)])
#
#     # denormalize combined range to get new axes
#     new_lim1, new_lim2 = y_new_lims_normalized * y_mags
#     ax1.set_ylim(new_lim1)
#     ax2.set_ylim(new_lim2)
#     return
#
# def plot_twinx(ax, ax2, x, y_1, y_2, label_1=None, label_2=None, legend_loc='best',
#                align_y='no'):
#     ln1  = ax.plot(x, y_1, label=label_1)
#     ln2  = ax2.plot(x, y_2, 'k--', label=label_2)
#
#     if align_y == 'yes' or align_y == 'y':
#         align_yaxis(ax, ax2)
#
#     if label_1 is not None:
#         lns  = ln1
#         if label_2 is not None:
#             lns += ln2
#         labs = [l.get_label() for l in lns]
#         ax.legend(lns, labs, loc=legend_loc, frameon=False, handlelength=1.0)
#     return
#
# def plot_abs_and_cond(file, save_file=None):
#     df       = load_and_clean_xls_normalized_data(file)
#     fig, ax  = instantiate_ax('Time [min]', 'A215 [mAU]')
#     ax2      = instantiate_twinx(ax, 'Conductivity [mS/cm]')
#     plot_twinx(ax, ax2, df['Time_min'], df['UV2_215nm_mAU'], df['Cond_mS/cm'],
#                'A215', 'Conductivity', legend_loc='upper center', align_y='yes')
#     if save_file is not None:
#         fig.savefig(save_file, bbox_inches='tight', dpi=300)
#     return
#
# def get_salt_conc(name):
#     sub = name[(name.rfind('isocratic')+10):]
#     return sub[:sub.find(' ')]
#
# def calc_first_moment(x, y):
#     """Note:  no correction included for baseline drift."""
#     x  = np.array(x)
#     y  = np.array(y)
#     y_num = x*y
#     return np.trapz(y_num, x)/np.trapz(y, x)
#
# def calc_k_prime(t_r, t_0):
#     return (t_r - t_0)/t_0
#
# def format_loglog(ax):
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
#     ax.get_xaxis().set_minor_formatter(ticker.ScalarFormatter())
#     ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
#     ax.get_yaxis().set_minor_formatter(NullFormatter())
#     return
#
# def get_initial_tis(file, df=None):
#     if file == None:
#         pass
#     else:
#         df = load_and_clean_xls_normalized_data(file)
#     cond = []
#     for i, cont in df.iterrows():
#         if cont['Time_min'] < 10.0:
#             cond.append(cont['tis_mM'])
#     cond   = np.array(cond)
#     return np.mean(cond)
#
# def correct_baseline_drift(df, y_col):
#     """Correct baseline drift, using the y_col column"""
#
#     # Identify starting and stopping points
#     df.reset_index(drop=True, inplace=True)
#     a_base = [df.at[0, y_col], df.at[len(df)-1, y_col]]
#     t_base = [df.at[0, 'Time_min'], df.at[len(df)-1, 'Time_min']]
#
#     # Correct baseline
#     df[y_col] = df[y_col] - np.interp(df['Time_min'], t_base, a_base)
#     return df
#
# def get_B_per_from_file(file):
#     c_str = get_salt_conc(file)
#     if 'pt' in c_str:
#         val_int = float(c_str[:c_str.find('pt')])
#         val_dec = float(c_str[c_str.find('pt')+2:])/10
#         val_tot = val_int + val_dec
#     else:
#         val_tot = 100.0
#     return val_tot
#
# def order_files(files):
#     b_per = {}
#     b_val = []
#     for fil in files:
#         b_val.append(get_B_per_from_file(fil))
#         b_per[get_B_per_from_file(fil)] = fil
#     b_val.sort(reverse=True)
#     return b_per, b_val











# def get_names_normalized(df, names):
#     for i in range(len(df.iloc[0])):
#         if i != 0:
#             names.append(df.iloc[0][i][df.iloc[0][i].find(':')+4:]\
#                         + '_' + df.iloc[1][i][1:])
#     return names
#
# def load_and_clean(name):
#     df = pd.read_excel(name)
#     units = list(df.loc[1])
#
#     if list(df.iloc[1]).count('min') == 1:
#         properties = get_names_normalized(df, ['min'])
#     elif list(df.iloc[1]).count('ml') == 1:
#         properties = get_names_normalized(df, ['ml'])
#     else:
#         properties = list(df.loc[0])
#         for ind, prop in enumerate(properties):
#             if ind%2 == 0:
#                 name        = properties[ind][(properties[ind].rfind(':')+1):]
#                 abbrev_name = name[(name.find('_')+1):]
#                 units       = df.iat[1, ind]
#             else:
#                 units       = df.iat[1, ind][(df.iat[1, ind].find(' ')+1):]
#             properties[ind] = abbrev_name + '_' + units
#
#     df.columns = properties
#     df.drop(labels=[0, 1], axis='index', inplace=True)
#     df.reset_index(drop=True, inplace=True)
#     return df
#
# def align_yaxis_multiple(ax_list):
#     y_lims = np.array([ax.get_ylim() for ax in ax_list])
#
#     # force 0 to appear on both axes, comment if don't need
#     y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
#     y_lims[:, 1] = y_lims[:, 1].clip(0, None)
#
#     # normalize both axes
#     y_mags = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
#     y_lims_normalized = y_lims / y_mags
#
#     # find combined range
#     y_new_lims_normalized = np.array([np.min(y_lims_normalized),
#                                       np.max(y_lims_normalized)])
#
#     # denormalize combined range to get new axes
#     new_lim = y_new_lims_normalized * y_mags
#     for ax, lim in zip(ax_list, new_lim):
#         ax.set_ylim(lim)
#     ax.axhline(color='k')
#     return
#
# def add_fractions(ax, df, x_conversion=1, text=False, xlim_times=None):
#     if 'Fractions_cv' in df.columns:
#         frac_times = [x for x in df['Fractions_cv'] if str(x) != 'nan']
#     elif 'Fractions_min' in df.columns:
#         frac_times = [x*x_conversion for x in df['Fractions_min'] if str(x) != 'nan']
#     elif 'Fractions_ml' in df.columns:
#         frac_times = [x*x_conversion for x in df['Fractions_ml'] if str(x) != 'nan']
#     else:
#         print('No x-coordinates found')
#         return
#
#     frac_names = [x for x in df['Fractions_(Fractions)'] if str(x) != 'nan']
#
#     if xlim_times is not None:
#         xlim_min = xlim_times[0]
#         xlim_max = xlim_times[1]
#
#     for (time, name) in zip(frac_times, frac_names):
#         if name=='Waste': # hack to remove overlapping fraction cuts
#             pass
#         else:
#             ax.axvline(x=time, ymin=0, ymax=0.05, color='red')
#             if text==True and xlim_min < time < xlim_max:
#                 plt.text(time, 0.02, name, rotation=90, color='red', size=12, transform=ax.get_xaxis_transform())
#     return
