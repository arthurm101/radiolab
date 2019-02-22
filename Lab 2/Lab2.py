import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import signal

test_data_hist = np.load('test_50mV.npy')

def load_data(filename):
    data = np.load(filename)
    return data

def reshape_data(data):

    new_data = data.reshape(2, len(data)/2)

    return new_data


def splitting_data(data):

    samples = len(data[0])/16000

    real = np.split(data[0], samples)
    imag = np.split(data[1], samples)

    return real, imag

def complex(real, imag):

    comp = []

    for i in range(len(real)):
        comp_val = real[i] + 1j*imag[i]

        comp.append(comp_val)

    return comp

upper_band_data = load_data('1MHz_above.npy')
lower_band_data = load_data('1MHz_below.npy')

up_band_data = reshape_data(upper_band_data)
low_band_data = reshape_data(lower_band_data)

real_upper_band, imag_upper_band = splitting_data(up_band_data)
real_lower_band, imag_lower_band = splitting_data(low_band_data)

complex_num_up = complex(imag_upper_band, real_upper_band)
complex_num_low = complex(imag_lower_band, real_lower_band)

def frequency(vsamp):

    data_size = 16000
    freq = np.fft.fftfreq(data_size, d = 1/vsamp)

    return freq

def power_transform(data):

    pow = []

    vsamp = 12.5e6

    for i in data:

        fft = np.fft.fft(i)
        power = np.abs(fft)**2
        pow.append(power)

    return np.array(pow)

def avg_power(power):

    avg = 0
    N = 0

    for i in range(len(power)):
        avg += power[i]
        N += 1

    average_power = avg/N

    return average_power


online_data = load_data('online.npy')
offline_data = load_data('offline.npy')

on_data = reshape_data(online_data)
off_data = reshape_data(offline_data)

real_online, imag_online = splitting_data(on_data)
real_offline, imag_offline = splitting_data(off_data)

complex_online = complex(imag_online, real_online)
complex_offline = complex(imag_offline, real_offline)

online_power = power_transform(complex_online)
offline_power = power_transform(complex_offline)

avg_power_online = avg_power(online_power)
avg_power_offline = avg_power(offline_power)

print(len(avg_power_online))

freq = frequency(12.5e6)

def plot(freq, data, title):

    smoothed_data = signal.medfilt(data, kernel_size = 7)

    plt.figure(figsize = (8,5))

    plt.title(title)
    plt.xlim(-2,2)
    plt.ylabel(r'Power $[mV^2 \hspace{.5} s^2]$')
    plt.xlabel(r'$\nu$ (MHz)', fontsize = 15)
    plt.plot(np.fft.fftshift(freq)/1e6, np.fft.fftshift(smoothed_data))
    plt.show()

plot(freq, avg_power_online, 'Online Power Spectrum')
plot(freq, avg_power_offline, 'Offline Power Spectrum')
'''
for i in range(10):

    vsamp = 12.5e6

    fft_up = np.fft.fft(complex_num_up[i])
    fft_low = np.fft.fft(complex_num_low[i])
    freq = np.fft.fftfreq(len(complex_num_up[i]), d = 1/vsamp)

    power_up = np.abs(fft_up)**2
    power_low = np.abs(fft_low)**2

    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.set_title('Upper Band')
    ax2.set_title('Lower Band')

    ax1.set_xlim(-2, 2)
    ax2.set_xlim(-2, 2)

    ax1.plot(np.fft.fftshift(freq)/1e6, np.fft.fftshift(power_up))
    ax2.plot(np.fft.fftshift(freq)/1e6, np.fft.fftshift(power_low))

    plt.tight_layout()
    plt.show()
'''

print(dskjdh)

def plot_hist(data):

    plt.title('Histogram of 50mV test data')
    plt.ylabel('Number of Occurences')
    plt.xlabel('Measured Voltages')
    plt.hist(test_data_hist, 20)
    plt.show()

plot_hist(test_data_hist)

#all_data = data(files)

#print(all_data)
