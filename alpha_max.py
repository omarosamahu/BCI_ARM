import matplotlib.pyplot as plt 
from scipy.signal import butter, lfilter, firwin
import scipy.io as sio 
from scipy.fftpack import fft 
import numpy as np 
from scipy.signal import freqz
import pandas as pd 


def band_pass(data,lowcut,highcut,fs,order=6):
	Nyq = 0.5*fs 
	low = lowcut/Nyq
	high = highcut/Nyq
	b, a = butter(order, [low,high], btype='band')
	y = lfilter(b,a,data)
	return y 

fs = 128.0 
nyq_rate = fs/2.0
lowcut = 8
highcut = 12 
T = 1/128.0 
N = 1152

x = np.linspace(0,2*np.pi*N*T,N)
xf = np.linspace(0,1.0/(2.0*T),N/2)

#read data 
extracted_data =  sio.loadmat('/home/omar/Documents/python_programs/graduation/graz_data/dataset_BCIcomp1.mat')
data = extracted_data['x_train']
maximum_c1 = []
maximum_c3 = []
for i in range(0,139):
	t = data[:,0,i]
	x1=band_pass(t,lowcut,highcut,fs,order=6)
	f = fft(x1)
	xf1 = abs(f)
	maximum_c1.append(max(xf1))
	# plt.subplot(211)
	# plt.plot(xf,xf1[0:N/2])
	# plt.xlabel('frequncy')
	# plt.ylabel('Amplitude')
	# plt.grid()
	# plt.subplot(212)
	# plt.plot(x,x1)
	# plt.xlabel('time')
	# plt.ylabel('Amplitude')
	# plt.grid()
	# plt.show()
for i in range(0,139):
	t = data[:,2,i]
	x1=band_pass(t,lowcut,highcut,fs,order=6)
	f = fft(x1)
	xf1 = abs(f)
	maximum_c3.append(max(xf1))
dic = {'c1':maximum_c1,'c3':maximum_c3}
df1  = pd.DataFrame(dic)
df1.to_csv('/home/omar/new_features2.csv')
	# plt.subplot(211)
	# plt.plot(xf,xf1[0:N/2])
	# plt.xlabel('frequncy')
	# plt.ylabel('Amplitude')
	# plt.grid()
	# plt.subplot(212)
	# plt.plot(x,x1)
	# plt.xlabel('time')
	# plt.ylabel('Amplitude')
	# plt.grid()
	# plt.show()


# plt.figure()
# x2=band_pass(xf1,lowcut,highcut,fs,order=6)