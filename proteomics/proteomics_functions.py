import pandas as pd
import numpy as np
from scipy import optimize
from scipy.stats import sem
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker
from venn import venn, pseudovenn
import seaborn as sns
from matplotlib.collections import PathCollection
from Bio import SeqIO, SeqUtils
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from addict import Dict
import json

sys.path.append("/home/chase/codes/python_functions/")
import plotting as my_plot

sys.path.append("/home/chase/my_work/correlation_aex_data/mean_field_electrostatics/v2")
import base_classes
import morbidelli as m


def get_charge(pH, seq, charge_contributions=False):
    sol = base_classes.solution(pH, ion_str=0.1)
    pro = m.protein(sol, seq=seq, dpr=2.0e-10, negative_sign=True)
    if not charge_contributions:
        return pro.net_charge
    else:
        return pro.net_charge, pro.net_neg_charge, pro.net_pos_charge, pro.charge_dens, pro.charge_dens_neg, pro.charge_dens_pos

def check_pI(pI, seq):
    if abs(get_charge(pI, seq)) < 1e-2:
        return True
    else:
        False

def get_pI(seq, guess=5.0):
    pI = optimize.fsolve(get_charge, guess, args=(seq,))[0]
    if check_pI(pI, seq):
        return pI, True
    else:
        for guess in np.linspace(0.01, 14, 15):
            pI = optimize.fsolve(get_charge, guess, args=(seq,))[0]
            if check_pI(pI, seq):
                return pI, True
        return 0, False

def get_stacked_bar_vals(std_accn, native_accn):
    intersection = len(set(std_accn).intersection(native_accn))
    std_unique = len(std_accn) - intersection
    native_unique = len(native_accn) - intersection
    return std_unique, intersection, native_unique

def add_entry(name, std_accn, native_accn, names_list, std_unique_list, intersection_list, native_unique_list):
    names_list.append(name)
    std, inter, native = get_stacked_bar_vals(std_accn, native_accn)
    std_unique_list.append(std)
    intersection_list.append(inter)
    native_unique_list.append(native)
    return names_list, std_unique_list, intersection_list, native_unique_list

# def check_against_id_map(my_df, id_map_file, map_name, n_accession=1):
#     df_map = pd.read_csv(id_map_file, sep='\t')
#     df_map.columns = ['uniprot', 'ref_seq']
#     ref_seq_vals = list(df_map.ref_seq)

#     for i, cont in my_df.iterrows():
#         for n in range(n_accession):
#             if cont[f'accession_{n}'] in ref_seq_vals:
#                 my_df.at[i, f'{map_name}_id'] = cont[f'accession_{n}']
#                 break
#     return

def get_name_matches(my_df, lit_df_path, lit_name):
    lit_df = pd.read_csv(lit_df_path)
    for i, cont in my_df.iterrows():
        for name in list(lit_df.desc_lower_2):
            if name == cont.desc_lower_2:
                my_df.at[i, 'perfect_match'] = True

            if name in cont.desc_lower_2 or cont.desc_lower_2 in name:
                if 'ubiquitin' in name:
                    my_df.at[i, 'contains_ubiquitin'] = True

                if name == 'actin' or cont.desc_lower_2 == 'actin':
                    if 'interacting' in name or 'interacting' in cont.desc_lower_2:
                        pass
                    else:
                        my_df.at[i, f'{lit_name}'] = name
                        my_df.at[i, 'contains_actin'] = True
                else:
                    my_df.at[i, f'{lit_name}'] = name
                break
    return

def map_binary_colorbar(series, color='white', row_colors=None, cmap=None):
    if cmap is None:
        cmap = matplotlib.colors.ListedColormap(['white', color])

    norm = plt.Normalize(series.min(), series.max())
    color_values = cmap(norm(series))

    if row_colors is None:
        row_colors = pd.Series(color_values.tolist(), index=series.index, name=series.name)
    else:
        row_colors = pd.concat([row_colors, pd.Series(color_values.tolist(), index=series.index, name=series.name)], axis=1)
    return row_colors
