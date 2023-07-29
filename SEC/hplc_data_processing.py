import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly
import scipy
from scipy import integrate
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from addict import Dict
import json
import sys
import os

sys.path.append("/home/chase/codes/python_functions/")
import plotting as my_plot
import akta

peak_time_bounds = {
    'large': (8.7, 10.517),
    'small': (10.517, 11.77),
    'mab': (12.25, 15.55),
    'lmw1': (15.76, 17.84),
    'lmw2': (17.84, 21.45),
    'lmw3': (21.45, 24.7),
}

class raw_hplc_data():
    def __init__(self, file_path, r_file, m_file):
        self.file_path = file_path
        self.r_file = r_file
        self.m_file = m_file

        self.df_m = pd.read_csv(f'{file_path}/{m_file}')
        assert str(self.df_m.at[0, 'Channel Id']) in m_file and str(self.df_m.at[0, 'Channel Id']) in r_file

        self.wavelength = float(self.df_m.at[0, 'Channel'][6:11])
        self.sample = self.df_m.at[0, 'SampleName']
        self.id = self.df_m.at[0, 'Channel Id']
        self.sample_id = f'{self.sample}_{self.id}'
        self.sample_set = self.df_m.at[0, 'Sample Set Name']
        self.acq_method = self.df_m.at[0, 'Acq Method Set']
        self.inj_vol = self.df_m.at[0, 'Injection Volume']
        self.acq_date = self.df_m.at[0, 'Date Acquired']

        self.df_r = pd.read_csv(f'{file_path}/{r_file}', names=['time_min', f'uv_{int(self.wavelength)}'])
        return

class clean_hplc_data():
    def __init__(self, raw_data_list, shift_baseline_time=False, shift_integral_t=0):
        for i, raw_data in enumerate(raw_data_list):
            if i == 0:
                self.sample     = raw_data.sample
                self.id         = raw_data.id # this will be different for different wavelengths, but I'm using this to differentiate replicate sample injections
                self.sample_id  = raw_data.sample_id # same as above
                self.sample_set = raw_data.sample_set
                self.acq_method = raw_data.acq_method
                self.inj_vol    = raw_data.inj_vol
                self.acq_date   = raw_data.acq_date
                self.df         = raw_data.df_r
                self.wavelengths = [raw_data.wavelength]
            else:
                assert self.sample == raw_data.sample
                assert self.acq_date == raw_data.acq_date
                self.df = pd.merge(self.df, raw_data.df_r, on=['time_min'])
                self.wavelengths.append(raw_data.wavelength)
        if shift_baseline_time is not False:
            self.shift_baseline(shift_baseline_time)
        self.shift_integral_t = shift_integral_t
        self.get_areas()
        self.get_percentages()
        return

    def shift_baseline(self, shift_baseline_time):
        df = self.df
        index = df.iloc[(df.time_min-shift_baseline_time).abs().argsort()[:1]].index[0]
        for col in list(df.columns)[1:]:
            df[col] -= df.at[index, col]
        return

    def get_areas(self):
        df = self.df
        self.areas_280 = {}
        self.areas_254 = {}

        self.areas_280['total'] = integrate.trapz(x=df.time_min.values, y=df.uv_280.values)
        self.areas_254['total'] = integrate.trapz(x=df.time_min.values, y=df.uv_254.values)
        for peak, times in peak_time_bounds.items():
            df_temp = df[(times[0] + self.shift_integral_t < df.time_min) & (df.time_min < times[1] + self.shift_integral_t)]
            self.areas_280[peak] = integrate.trapz(x=df_temp.time_min.values, y=df_temp.uv_280.values)
            self.areas_254[peak] = integrate.trapz(x=df_temp.time_min.values, y=df_temp.uv_254.values)
        return

    def get_percentages(self):
        self.get_areas()
        self.percentages_280 = {}
        self.percentages_254 = {}

        for peak in peak_time_bounds.keys():
            self.percentages_280[peak] = self.areas_280[peak]/self.areas_280['total'] * 100
            self.percentages_254[peak] = self.areas_254[peak]/self.areas_254['total'] * 100

        self.percentages_280['both_agg'] = self.percentages_280['large'] + self.percentages_280['small']
        self.percentages_254['both_agg'] = self.percentages_254['large'] + self.percentages_254['small']

        return

    def plot_chromatogram(self, wavelength_list=[280]):
        if len(wavelength_list) == 1:
            w = wavelength_list[0]
            fig, ax = my_plot.instantiate_fig(xlabel='Time [min]',
                                              ylabel=f'A{w} [AU]')
            ax.plot(self.df.time_min, self.df[f'uv_{w}'])
        else:
            fig, ax = my_plot.instantiate_fig(xlabel='Time [min]',
                                              ylabel=f'Abs. [AU]')
            for w in wavelength_list:
                ax.plot(self.df.time_min, self.df[f'uv_{w}'], label=f'A{w}')
            my_plot.set_layout(fig, ax, legend=True)
        return fig, ax


def load_hplc_data(file_path, n_wavelengths=2, remove_water=True, shift_baseline_time=False, shift_integral_t=0):
    files = os.listdir(file_path)
    files.sort()
    m_files = [f for f in files if 'meta' in f]
    r_files = [f for f in files if 'raw' in f]

    raw_data = {}
    for (r_file, m_file) in zip(r_files, m_files):
        temp = raw_hplc_data(file_path, r_file, m_file)
        raw_data[temp.sample_id] = temp

    sample_ids = list(raw_data.keys())
    sample_ids.sort()
    data = {}
    for i in range(int(len(sample_ids)/n_wavelengths)):
        # Needs to be updated if n_wavelengths > 2
        id_1, id_2 = sample_ids[n_wavelengths*i], sample_ids[n_wavelengths*i + 1]
        temp = clean_hplc_data([raw_data[id_1], raw_data[id_2]], shift_baseline_time, shift_integral_t)
        data[temp.sample_id] = temp

    sample_ids = list(data.keys())
    sample_ids.sort()
    sample_ids = [s for s in sample_ids if 'ater' not in s]
    return data, sample_ids

def get_areas_df(data, sample_ids, label_by_acq_date=False):
    """Construct a dataframe from the peak areas in each fraction"""
    areas = []
    for s in sample_ids:
        main_cols = ['total_280', 'large_280', 'small_280', 'mab_280', 'lmw1_280', 'lmw2_280', 'lmw3_280',
                     'total_254', 'large_254', 'small_254', 'mab_254', 'lmw1_254', 'lmw2_254', 'lmw3_254']

        if label_by_acq_date:
            temp = [data[s].sample, data[s].acq_date]
            columns=['frac', 'date'] + main_cols
        else:
            temp = [data[s].sample]
            columns=['frac'] + main_cols

        for peak in data[s].areas_280.keys():
            temp.append(data[s].areas_280[peak])
        for peak in data[s].areas_254.keys():
            temp.append(data[s].areas_254[peak])

        areas.append(temp)

    df_areas = pd.DataFrame(areas, columns=columns)
    return df_areas

def get_normalized_area_df(data, sample_ids, hccf_areas=False, df_areas=False):
    """Construct a dataframe with areas normalized with respect to those in hccf_areas"""
    if df_areas is False:
        df_areas = get_areas_df(data, sample_ids)
    if hccf_areas is False:
        with open("/home/chase/my_work/exp_data/2022-07-20_studies_with_raw_HPLC_data/hplc_data/hccf_areas.json") as infile:
            hccf_areas = Dict(json.load(infile))

    df_norm = pd.DataFrame(columns=list(df_areas.columns))
    df_norm.frac = df_areas.frac.copy()
    for col in list(df_areas.columns)[1:-1]:
        if col in hccf_areas:
            df_norm[col] = df_areas[col]/hccf_areas[col]
    if 'dilution_factor' in df_areas.columns:
        df_norm.dilution_factor = df_areas.dilution_factor.copy()
    return df_norm

def get_akta_name(frac, cassette=1, start_index=0):
    if cassette == 'infer':
        cassette = frac[start_index]
        letter = frac[start_index+1]
        num = int(frac[start_index+2:start_index+4])
        akta_name = f'{cassette}.{letter}.{num}'
    else:
        letter = frac[start_index]
        num = int(frac[start_index+1:start_index+3])
        akta_name = f'{cassette}.{letter}.{num}'
    return akta_name

def add_volume_midpoints(df_norm, df_akta, cassette=1, start_index=0):
    for i, cont in df_norm.iterrows():
        akta_name = get_akta_name(cont.frac, cassette=cassette, start_index=start_index)
        df_norm.at[i, 'akta_name'] = akta_name
        entry = df_akta.loc[df_akta['Fraction_Fraction'] == akta_name, 'Fraction_ml']
        index = entry.index[0]
        vol_start = entry.iloc[0] - df_akta.loc[0, 'Injection_ml']
        vol_end = df_akta.at[index + 1, 'Fraction_ml'] - df_akta.loc[0, 'Injection_ml']
        vol_mid = (vol_start + vol_end)/2.0
        df_norm.at[i, 'volume_midpoint_ml'] = vol_mid
        # df_norm.at[i, 'volume_start_ml'] = vol_start
        # df_norm.at[i, 'volume_end_ml'] = vol_end
        # df_norm.at[i, 'volume_frac_ml'] = vol_end - vol_start
    return

def get_cip(df_norm_all, conc_factor=53):
    df_cip = df_norm_all[df_norm_all.frac.str.contains('CIP')].copy()
    df_cip.drop(columns=['frac', 'total_280', 'total_254', 'mab_280', 'total_254', 'large_254',
                         'small_254', 'lmw1_254', 'lmw2_254', 'lmw3_254', 'dilution_factor'], inplace=True)
    df_cip = df_cip[['large_280', 'small_280', 'mab_254', 'lmw1_280', 'lmw2_280', 'lmw3_280']]
    df_cip /= conc_factor
    df_cip = df_cip.transpose()
    df_cip.reset_index(inplace=True)
    df_cip.columns = ['species', 'norm_conc']
    df_cip['name'] = ['Large agg.', 'Small agg.', 'mAb', 'LMW 1', 'LMW 2', 'LMW 3']
    df_cip.drop(index=[4, 5], inplace=True)
    return df_cip

def get_data(akta_path, hplc_path, cip=True, conc_factor=53, start_index=3, dil_index=-3, dil_factor=False, feed_name='PAFVIN', cassette='infer', shift_integral_t=0):
    temp = dil_factor
    df = akta.load_and_clean_csv(akta_path)
    data, sample_ids = load_hplc_data(hplc_path, shift_baseline_time=5, shift_integral_t=shift_integral_t)
    sample_ids = [s for s in sample_ids if 'buffer' not in s]
    df_areas = get_areas_df(data, sample_ids)

    # Account for dilution
    for i, cont in df_areas.iterrows():
        dil_factor = temp
        # Ad hoc
        if cont.frac == 'PAFVIN_50':
            frac = 'PAFVIN_050'
            df_areas.at[i, 'frac'] = frac
        else:
            frac = cont.frac

        if 'CIP' in frac:
            dil_factor = 100.0
        else:
            if dil_factor:
                pass
            else:
                dil_factor = float(frac[dil_index:])
        df_areas.at[i, 'dilution_factor'] = 100/dil_factor

        # if '_0' in frac:
        #     start_ind = frac.find('_0') + 1
        #     dil_factor = float(frac[start_ind : start_ind + 3])
        #     df_areas.at[i, 'dilution_factor'] = 100/dil_factor
        # else:
        #     df_areas.at[i, 'dilution_factor'] = 1

        for col in ['total', 'large', 'small', 'mab', 'lmw1', 'lmw2', 'lmw3']:
            for wavelength in [280, 254]:
                df_areas.at[i, f'{col}_{wavelength}'] = df_areas.at[i, 'dilution_factor'] * cont[f'{col}_{wavelength}']

    df_feed = df_areas[df_areas.frac.str.contains(feed_name)]
    feed_areas = df_feed.mean(axis=0)
    feed_areas = Dict(feed_areas.to_dict())

    df_norm_all = get_normalized_area_df(data, sample_ids, feed_areas, df_areas)
    df_norm = df_norm_all[(~df_norm_all.frac.str.contains(feed_name)) &
                          (~df_norm_all.frac.str.contains('CIP'))].copy()
    df_norm.reset_index(inplace=True, drop=True)
    add_volume_midpoints(df_norm, df, cassette=cassette, start_index=start_index)

    if cip:
        df_cip = get_cip(df_norm_all, conc_factor)
        return df, data, sample_ids, df_areas, df_norm_all, df_norm, df_cip, feed_areas
    else:
        return df, data, sample_ids, df_areas, df_norm_all, df_norm, feed_areas


# Helper plotting functions

def no_sec(df, fig=None, ax=None, ax2=None, ax3=None):
    if fig is None and ax is None and ax2 is None and ax3 is None:
        fig, ax = my_plot.instantiate_fig(x=9, y=7.5, xlabel='Volume [ml]', ylabel='A280 [mAU]')
        ax2 = my_plot.get_twinx(ax, ylabel='Conductivity [mS/cm]')
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.2))
        ax3.set_ylabel('pH')

    ln1, = ax.plot(df.uv_280_ml, df.uv_280_mAU)
    ln2, = ax2.plot(df.Cond_ml, df.Cond_mS_cm, linestyle='--', color='black')
    ln3, = ax3.plot(df.pH_ml, df.pH_pH, linestyle=':', color='magenta')

    ax.yaxis.label.set_color(ln1.get_color())
    ax2.yaxis.label.set_color(ln2.get_color())
    ax3.yaxis.label.set_color(ln3.get_color())
    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=ln1.get_color(), **tkw)
    ax2.tick_params(axis='y', colors=ln2.get_color(), **tkw)
    ax3.tick_params(axis='y', colors=ln3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)
    return fig, ax, ax2, ax3

def with_sec(df, df_norm, norm_vol=None, fig=None, ax=None, show_twinx=True, ax3_position=1.2, ypad=0, lmw2=False, tis=False):
    if fig is None and ax is None:
        fig, ax = my_plot.instantiate_fig(x=9, y=7.5, xlabel='Volume [ml]', ylabel='Normalized concentration')

    ln_1 = ax.plot(df_norm.volume_midpoint_ml, df_norm.large_280, 'o-', label='Large agg.')
    ln_2 = ax.plot(df_norm.volume_midpoint_ml, df_norm.small_280, 'o-', label='Small agg.')
    ln_3 = ax.plot(df_norm.volume_midpoint_ml, df_norm.mab_254, 'o-', label='mAb')
    ln_4 = ax.plot(df_norm.volume_midpoint_ml, df_norm.lmw1_280, 'o-', label='SPF 1')
    if lmw2:
        ln_5 = ax.plot(df_norm.volume_midpoint_ml, df_norm.lmw2_280, 'o-', label='SPF 2')

    if norm_vol is not None:
        uv_295_norm = df.iloc[(df['uv_295_ml']-norm_vol).abs().argsort()].iloc[0]['uv_295_mAU']
    else:
        uv_295_norm = df.uv_295_mAU.max()
    ln_7 = ax.plot(df.uv_295_ml, df.uv_295_mAU/uv_295_norm, 'k', label='A295')

    if show_twinx:
        if tis:
            ax2 = my_plot.get_twinx(ax, ylabel='TIS [mM]', ypad=ypad)
            ln_8 = ax2.plot(df.Cond_ml, akta.cond_2_tis(df.Cond_mS_cm), 'k--', label='TIS')
        else:
            ax2 = my_plot.get_twinx(ax, ylabel='Conductivity [mS/cm]', ypad=ypad)
            ln_8 = ax2.plot(df.Cond_ml, df.Cond_mS_cm, 'k--', label='Cond.')

        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", ax3_position))
        ax3.set_ylabel('pH', labelpad=ypad)
        ln9 = ax3.plot(df.pH_ml, df.pH_pH, linestyle=':', color='magenta', label='pH')
        ax3.yaxis.label.set_color('magenta')
        tkw = dict(size=6, width=1.5)
        ax3.tick_params(axis='y', colors='magenta', **tkw)
    else:
        ax2 = my_plot.get_twinx(ax)
        if tis:
            ln_8 = ax2.plot(df.Cond_ml, akta.cond_2_tis(df.Cond_mS_cm), 'k--', label='TIS')
        else:
            ln_8 = ax2.plot(df.Cond_ml, df.Cond_mS_cm, 'k--', label='Cond.')

        ax3 = ax.twinx()
        ln9 = ax3.plot(df.pH_ml, df.pH_pH, linestyle=':', color='magenta', label='pH')

    if lmw2:
        lns = ln_1 + ln_2 + ln_3 + ln_4 + ln_5 + ln_7 + ln_8 + ln9
    else:
        lns = ln_1 + ln_2 + ln_3 + ln_4 + ln_7 + ln_8 + ln9
    labs = [l.get_label() for l in lns]
    return fig, ax, ax2, ax3, lns, labs

def cip(df_cip):
    fig, ax = my_plot.instantiate_fig(x=7, y=6, ylabel='Normalized conc.')
    ticks = ax.set_xticks(df_cip.index, df_cip.name, rotation=30, ha='right', rotation_mode='anchor')
    rects_1 = ax.bar(df_cip.index, df_cip.norm_conc, width=0.75)
    my_plot.set_layout(fig, ax)
    return fig, ax

def sec_chromatograms(data, sample_ids, zoomed=False, lines=True, group='pafvin', index=3, fig=None, ax=None, xaxis='time', shift_integral_t=0):
    if fig is None and ax is None:
        if xaxis == 'time':
            fig, ax = my_plot.instantiate_fig(xlabel='Time [min]', ylabel='A280 [AU]')
        elif xaxis == 'volume':
            fig, ax = my_plot.instantiate_fig(xlabel='Volume [ml]', ylabel='A280 [AU]')

    if group == 'pafvin':
        sample_subset_ids = [s for s in sample_ids if 'PAFVIN' in s]
    elif group == 'hccf':
        sample_subset_ids = [s for s in sample_ids if 'HCCF' in s]
    elif group == 'eluate':
        sample_subset_ids = [s for s in sample_ids if s[index]=='2']
    elif group == 'cip':
        sample_subset_ids = [s for s in sample_ids if 'CIP' in s]
    elif group == 'ft':
        sample_subset_ids = [s for s in sample_ids if s[index]=='1']
    elif group == 'all':
        sample_subset_ids = sample_ids

    for s in sample_subset_ids:
        if xaxis == 'time':
            ax.plot(data[s].df.time_min, data[s].df.uv_280)
            if lines:
                for t in [8.7, 10.517, 11.77, 12.25, 15.55, 15.76, 17.84, 21.45]:
                    ax.axvline(t + shift_integral_t, linestyle='--', color='black')
            if zoomed:
                ax.set_ylim(-0.005, 0.1)
                ax.set_xlim(8.5, 18.5)
        elif xaxis == 'volume':
            ax.plot(data[s].df.time_min * 0.6, data[s].df.uv_280)
            if lines:
                for t in [8.7, 10.517, 11.77, 12.25, 15.55, 15.76, 17.84, 21.45]:
                    ax.axvline((t + shift_integral_t)* 0.6, linestyle='--', color='black')
            if zoomed:
                ax.set_ylim(-0.005, 0.1)
                ax.set_xlim(8.5 * 0.6, 18.5 * 0.6)

    my_plot.set_layout(fig, ax)
    return fig, ax
