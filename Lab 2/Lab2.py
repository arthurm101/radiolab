import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import signal

test_data_hist = np.load('test_50mV.npy')
test_data = np.load('test_100mV.npy')

list = [test_data_hist, test_data]

mean = np.mean(list, axis = 0)

waveguide_no_stop1 = np.array([8.45, 11.73, 15.10])
waveguide_stop1 = np.array([8.02, 11.3, 14.57, 17.95])

waveguide_no_stop2 = np.array([5.34, 10.34, 15.34, 20.29, 25.29, 30.30, 35.26, 40.33, 45.29, 50.31])
waveguide_stop2 = np.array([3.82, 8.82, 13.7, 18.8, 23.83, 28.8, 33.79, 38.8, 43.76, 48.72])


np.save('xband_nostop.npy', waveguide_no_stop1)
np.save('xband_stop.npy', waveguide_stop1)

np.save('cband_nostop.npy', waveguide_no_stop2)
np.save('cband_stop.npy', waveguide_stop2)

print(skjdhsj)
#print(len(mean))
#print(shsjkdh)

def load_data(filename, volt):

    data = np.load(filename)

    (data * volt)/2**16

    return data

def reshape_data(data):

    new_data = data.reshape(2, len(data)/2)

    return new_data


def splitting_data(data):

    samples = len(data[0])/16000

    print(samples)
    real = np.split(data[0], samples)
    imag = np.split(data[1], samples)

    return real, imag

def complex(real, imag):

    comp = []

    for i in range(len(real)):
        comp_val = real[i] + 1j*imag[i]

        comp.append(comp_val)

    return comp

upper_band_data = load_data('1MHz_above.npy', .05)
lower_band_data = load_data('1MHz_below.npy', .05)

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

    mean_power = np.mean(power, axis = 0)

    return mean_power

def median_power(power):

    median = np.median(power, axis = 0)

    return median

def avg_filt(data, freq):

    smoothed_data = []
    smoothed_freq = []

    split_data = np.split(data, 3200)
    split_freq = np.split(freq, 3200)

    for i in range(len(split_data)):

        mean_data = np.mean(split_data[i], axis = 0)
        mean_freq = np.mean(split_freq[i], axis = 0)

        smoothed_data.append(mean_data)
        smoothed_freq.append(mean_freq)

    return np.array(smoothed_freq), np.array(smoothed_data)

def med_filt(data, freq):

    smoothed_data = []
    smoothed_freq = []

    split_data = np.split(data, 3200)
    split_freq = np.split(freq, 3200)

    for i in range(len(split_data)):

        med_data = np.median(split_data[i], axis = 0)
        med_freq = np.median(split_freq[i], axis = 0)

        smoothed_data.append(med_data)
        smoothed_freq.append(med_freq)

    return np.array(smoothed_freq), np.array(smoothed_data)

freq = frequency(12.5e6)

online_data = load_data('online.npy', .05)
offline_data = load_data('offline.npy', .05)

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

med_power_online = median_power(online_power)
med_power_offline = median_power(offline_power)

smooth_freq_online, smooth_online = avg_filt(avg_power_online, freq)
med_freq_online, med_online = med_filt(avg_power_online, freq)


smooth_freq_offline, smooth_offline = avg_filt(avg_power_offline, freq)
med_freq_offline, med_offline = med_filt(avg_power_offline, freq)

#print(len(avg_power_online))

def plotting(freq, data, title):

    smoothed_data = signal.medfilt(data, kernel_size = 5)

    plt.figure(figsize = (8,5))

    plt.title(title, fontweight = 'bold', fontsize = 15)
    plt.xlim(-2,2)
    plt.ylabel(r'Power $[mV^2 \hspace{.5} s^2]$', fontsize = 15)
    plt.xlabel(r'$\nu$ (MHz)', fontsize = 15)
    plt.plot(np.fft.fftshift(freq)/1e6, np.fft.fftshift(smoothed_data))
    plt.show()

#def mean_val(data, window):

#plotting(freq, avg_power_online, 'Online Power Spectrum')
#plotting(freq, avg_power_offline, 'Offline Power Spectrum')

cal_data = load_data('cal_data_100mV.npy', .1)
cold_data = load_data('cold_data_100mV.npy', .1)

calibrate_data = reshape_data(cal_data)
non_calibrated_data = reshape_data(cold_data)

real_cal, imag_cal = splitting_data(calibrate_data)
real_non_cal, imag_non_cal = splitting_data(non_calibrated_data)

complex_cal = complex(imag_cal, real_cal)
complex_non_cal = complex(imag_non_cal, real_non_cal)

cal_power = power_transform(complex_cal)
non_cal_power = power_transform(complex_non_cal)

avg_power_cal = avg_power(cal_power)
avg_power_non_cal = avg_power(non_cal_power)

med_power_cal = median_power(cal_power)
med_power_non_cal = median_power(non_cal_power)

smooth_freq_cal, smooth_cal = avg_filt(avg_power_cal, freq)
med_freq_cal, med_cal = med_filt(avg_power_cal, freq)

smooth_freq_non_cal, smooth_non_cal = avg_filt(avg_power_non_cal, freq)
med_freq_non_cal, med_non_cal = med_filt(avg_power_non_cal, freq)

plotting(smooth_freq_online, smooth_online, 'Smoothed Data')
print(dhskh)

plotting(freq, avg_power_cal, 'Calibrated Average Power Spectrum')
plotting(freq, avg_power_non_cal, 'Non_calibrated Average Power Spectrum')

#plotting(freq, med_power_cal, 'Calibrated Median Power Spectrum')
#plotting(freq, med_power_non_cal, 'Non_calibrated Median Power Spectrum')


s_line = smooth_online/smooth_offline


def Gain(s_cal, s_cold):

    T_cal = 300
    T_cold = 2.73

    diff = s_cal-s_cold

    G = ((T_cal - T_cold)/(np.sum(diff))) * np.sum(s_cold)

    return G

G = Gain(smooth_cal, smooth_non_cal)
#print(G)

'''
smooth_freq, smooth_data = avg_filt(avg_power_non_cal, freq)
sm_freq, sm_data = med_filt(avg_power_non_cal, freq)

plt.plot(np.fft.fftshift(smooth_freq)/1e6, np.fft.fftshift(smooth_data), 'k',label = 'Avg Filter')
plt.plot(np.fft.fftshift(sm_freq)/1e6, np.fft.fftshift(sm_data), 'r', label = 'Median Filter', alpha = .4)
plt.xlim(-2,2)
plt.legend()
plt.show()
print(dhkhd)
'''

def T_power(line, G):

    power = line * G

    return power

final_power = T_power(s_line, G)

plotting(smooth_freq_online, final_power, 'Final Power Spectrum ?')

#plt.plot(np.fft.fftshift(freq)/1e6, np.fft.fftshift(final_power))
#plt.show()

print(hdjkdh)
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
