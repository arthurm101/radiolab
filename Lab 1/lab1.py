import ugradio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

#READING IN ALL THE DATA WE HAVE COLLECTED
data1 = np.load('data1.npy')
data2 = np.load('data2.npy')
data3 = np.load('data3.npy')
data4 = np.load('data4.npy')
data5 = np.load('data5.npy')
data6 = np.load('data6.npy')
data7 = np.load('data7.npy')
data8 = np.load('data8.npy')
data9 = np.load('data9.npy')

print(data1[:13])

#TAKING ONLY THE FIRST N FOR PLOTTING PURPOSES

firstN = data1[:100]

#PLOTTING CODE FOR FIRST N DATA POINTS

plt.xlabel('Sample Observation')
plt.ylabel('Data')
plt.title('First N Data')
plt.plot(firstN, 'o')
plt.show()















