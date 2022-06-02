import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from dtw import accelerated_dtw
from scipy.signal import hilbert, butter, filtfilt


def pearson_relation(label1, label2, window_size=5):
    # method 2 using scipy
    r, p = stats.pearsonr(df.dropna()[label1], df.dropna()[label2])
    print(f"Scipy computed Pearson r: {r} and p-value: {p}")
    return r, p


def TCLL(label1, label2):
    def crosscorr(datax, datay, lag=0, wrap=False):
        """ Lag-N cross correlation.
        Shifted data filled with NaNs

        Parameters
        ----------
        lag : int, default 0
        datax, datay : pandas.Series objects of equal length

        Returns
        ----------
        crosscorr : float
        """
        if wrap:
            shiftedy = datay.shift(lag)
            shiftedy.iloc[:lag] = datay.iloc[-lag:].values
            return datax.corr(shiftedy)
        else:
            return datax.corr(datay.shift(lag))

    d1 = df[label1]
    d2 = df[label2]
    seconds = 5
    fps = 30
    rs = [crosscorr(d1, d2, lag) for lag in range(-int(seconds * fps - 1), int(seconds * fps))]
    offset = np.ceil(len(rs) / 2) - np.argmax(rs)
    f, ax = plt.subplots(figsize=(14, 3))
    ax.plot(rs)
    ax.axvline(np.ceil(len(rs) / 2), color='k', linestyle='--', label='Center')
    ax.axvline(np.argmax(rs), color='r', linestyle='--', label='Peak synchrony')
    ax.set(title=f'Offset = {offset} frames\nS1 leads <> S2 leads', ylim=[.1, .31], xlim=[0, 300], xlabel='Offset',
           ylabel='Pearson r')
    ax.set_xticklabels([int(item - 150) for item in ax.get_xticks()])
    plt.legend()
    plt.show()


def DTW(label1, label2):
    d1 = df[label1].interpolate().values
    d2 = df[label2].interpolate().values
    d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(d1, d2, dist='euclidean')

    plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
    plt.plot(path[0], path[1], 'w')
    plt.xlabel('subject1')
    plt.ylabel('subject2')
    plt.title(f'DTW Minimum Path with minimum distance: {np.round(d, 2)}')
    plt.show()
    return np.round(d, 2)


def instantaneous_phase_synchrony(label1, label2):
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    lowcut = .01
    highcut = .5
    fs = 30.
    order = 1
    d1 = df[label1].interpolate().values
    d2 = df[label2].interpolate().values
    y1 = butter_bandpass_filter(d1, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    y2 = butter_bandpass_filter(d2, lowcut=lowcut, highcut=highcut, fs=fs, order=order)

    al1 = np.angle(hilbert(y1), deg=False)
    al2 = np.angle(hilbert(y2), deg=False)
    phase_synchrony = 1 - np.sin(np.abs(al1 - al2) / 2)
    N = len(al1)

    # 绘制结果
    f, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    ax[0].plot(y1, color='r', label=label1)
    ax[0].plot(y2, color='b', label=label2)
    ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), ncol=2)
    ax[0].set(xlim=[0, N], title='Filtered Timeseries Data')
    ax[1].plot(al1, color='r')
    ax[1].plot(al2, color='b')
    ax[1].set(ylabel='Angle', title='Angle at each Timepoint', xlim=[0, N])
    phase_synchrony = 1 - np.sin(np.abs(al1 - al2) / 2)
    ax[2].plot(phase_synchrony)
    ax[2].set(ylim=[0, 1.1], xlim=[0, N], title='Instantaneous Phase Synchrony', xlabel='Time',
              ylabel='Phase Synchrony')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = pd.read_excel('')
    cols_name = df.columns.values
    result = []
    for i in range(len(cols_name)):
        print(cols_name[i])
        result.append(pearson_relation(cols_name[i], cols_name[-1]))
        print(pearson_relation(cols_name[i], cols_name[-1]))