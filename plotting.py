import matplotlib.pyplot as plt
import numpy as np

def instantiate_fig(x=7, y=6, xlabel=None, ylabel=None, rotation=90, ypad=10, xpad=10, fontsize=20):
    fig, ax = plt.subplots()
    fig.set_size_inches(x, y, forward=True)
    font = {'weight' : 'normal', 'size'   : fontsize}
    plt.rc('font', **font)
    plt.rcParams.update({"savefig.facecolor":'white'})
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=xpad)
    if ylabel is not None:
        ax.set_ylabel(ylabel, rotation=rotation, labelpad=ypad)
    return fig, ax

def instantiate_fig_roman(x=7, y=6, xlabel=None, ylabel=None, rotation=90, ypad=0, xpad=0, top=True):
    fig, ax = plt.subplots()
    fig.set_size_inches(x, y, forward=True)

    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern Roman",
    "lines.linewidth" : 0.8,
    "lines.markersize" : 6})

    plt.rcParams['font.size'] = 24
    plt.rcParams['axes.linewidth'] = 0.8

    ax.tick_params(axis='both',direction='in',length=6, bottom=True, top=top, left=True, right=True)


    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=xpad)
    if ylabel is not None:
        ax.set_ylabel(ylabel, rotation=rotation, labelpad=ypad)
    return fig, ax

def set_layout(fig, ax=None, legend=False, loc='best', fontsize=20, frameon=False, handletextpad=0.8, ncol=1, columnspacing=2, handlelength=1):
    if legend:
        ax.legend(loc=loc, handlelength=handlelength, frameon=frameon, fontsize=fontsize, handletextpad=handletextpad, ncol=ncol, columnspacing=columnspacing)
    # fig.patch.set_alpha(1)
    plt.tight_layout()
    return

def get_twinx(ax, ylabel=None, ypad=0):
    ax2 = ax.twinx()
    if ylabel is not None:
        ax2.set_ylabel(ylabel, labelpad=ypad)
    return ax2

def align_yaxis(ax_list, hline=True, hline_width=1):
    y_lims = np.array([ax.get_ylim() for ax in ax_list])
    # force 0 to appear on both axes, comment if don't need
    y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
    y_lims[:, 1] = y_lims[:, 1].clip(0, None)
    # normalize both axes
    y_mags = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
    y_lims_normalized = y_lims / y_mags
    # find combined range
    y_new_lims_normalized = np.array([np.min(y_lims_normalized), np.max(y_lims_normalized)])
    # denormalize combined range to get new axes
    new_lim = y_new_lims_normalized * y_mags
    for ax, lim in zip(ax_list, new_lim):
        ax.set_ylim(lim)

    if hline:
        ax_list[0].axhline(color='black', linewidth=hline_width)
    return

# Need to modify
def add_fractions(ax, df, x_conversion=1, text=False, xlim_times=None, hide_waste=True, text_offset=0, text_size=12, akta_pure=False, ymax=0.05):
    if akta_pure:
        frac_times = [x*x_conversion for x in df['Fraction_ml'] if str(x) != 'nan']
        frac_times = [f - df.at[0, 'Injection_ml'] for f in frac_times]
        frac_names = [x for x in df['Fraction_Fraction'] if str(x) != 'nan']
    else:
        frac_times = [x*x_conversion for x in df['fractions_ml'] if str(x) != 'nan']
        frac_names = [x for x in df['fractions_(Fractions)'] if str(x) != 'nan']

    if xlim_times is not None:
        xlim_min = xlim_times[0]
        xlim_max = xlim_times[1]

    for (time, name) in zip(frac_times, frac_names):
        if 'Waste' in name and hide_waste: # hack to remove overlapping fraction cuts
            pass
        else:
            ax.axvline(x=time, ymin=0, ymax=ymax, color='red')
            if text==True and xlim_min < time < xlim_max:
                plt.text(time+text_offset, 0.02, name, rotation=90, color='red', size=text_size, transform=ax.get_xaxis_transform())
    return
