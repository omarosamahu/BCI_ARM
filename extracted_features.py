import matplotlib.pyplot as plt 
from scipy.signal import butter, lfilter
import scipy.io as sio 
from scipy.fftpack import fft 
import numpy as np 
import pandas as pd 
from scipy.linalg import eig 
from numpy import linalg as LA 
from scipy import stats

y_train = []
time_Mean_c1 = []
freq_Mean_c1 = []
time_stanard_deviation_c1 = []
freq_stanard_deviation_c1 = []
time_variance_c1 = []
frequency_variance_c1 = []
power_time_c1 = []
time_Mean_c3 = []
freq_Mean_c3 = []
time_stanard_deviation_c3 = []
freq_stanard_deviation_c3 = []
time_variance_c3 = []
frequency_variance_c3 = []
power_time_c3 = []
p = []
maximum_c1 = []
maximum_c3 = []
fs = 128.0 
nyq_rate = fs/2.0
lowcut = 8
highcut = 12 
T = 1/128.0 
N = 1152


def band_pass(data,lowcut,highcut,fs,order=10):
	Nyq = 0.5*fs 
	low = lowcut/Nyq
	high = highcut/Nyq
	b, a = butter(order, [low,high], btype='band')
	y = lfilter(b,a,data)
	return y 



x = np.linspace(0,2*np.pi*N*T,N)
xf = np.linspace(0,1.0/(2.0*T),N/2)

#read data 
extracted_data =  sio.loadmat('/home/omar/Documents/python_programs/graduation/graz_data/dataset_BCIcomp1.mat')
data = extracted_data['x_train']

for i in range(0,139):
	t = data[:,0,i]
	xc1=band_pass(t,lowcut,highcut,fs,order=6)
	f = fft(xc1)
	xf1 = abs(f)
	Mt = np.mean(xc1)
	St = np.std(xc1)
	k=np.square(xc1)
	power_time_c1.append(np.trapz(k,axis=0))
	variance_t = np.var(xc1)
	time_Mean_c1.append(Mt)
	time_stanard_deviation_c1.append(St)
	time_variance_c1.append(variance_t)
	maximum_c1.append(max(xf1))
	Mf = np.mean(xf1)
	Sf = np.std(xf1)
	variance_f = np.var(xf1)
	freq_Mean_c1.append(Mf)
	freq_stanard_deviation_c1.append(Sf)
	frequency_variance_c1.append(variance_f)



for i in range(0,139):
	t = data[:,2,i]
	xc1=band_pass(t,lowcut,highcut,fs,order=6)
	f = fft(xc1)
	xf1 = abs(f)
	Mt = np.mean(xc1)
	St = np.std(xc1)
	k=np.square(xc1)
	power_time_c3.append(np.trapz(k,axis=0))
	variance_t = np.var(xc1)
	time_Mean_c3.append(Mt)
	time_stanard_deviation_c3.append(St)
	time_variance_c3.append(variance_t)
	maximum_c3.append(max(xf1))
	Mf = np.mean(xf1)
	Sf = np.std(xf1)
	variance_f = np.var(xf1)
	freq_Mean_c3.append(Mf)
	freq_stanard_deviation_c3.append(Sf)
	frequency_variance_c3.append(variance_f)
	


# print len(val)

#dic = {'eig':w,'val':v}
# df  = pd.DataFrame(dic)
# df.to_csv('/home/omar/Documents/python_programs/graduation/datasets/eigens.csv',index=False)


print stats.ttest_ind(time_Mean_c1,time_Mean_c3,equal_var = False)

dic = {'time_stanard_deviation_c1':time_stanard_deviation_c1,'freq_stanard_deviation_c1':freq_stanard_deviation_c1,'time_variance_c1':time_variance_c1,'frequency_variance_c1':frequency_variance_c1,'signal_power_c1':power_time_c1,'c1_max':maximum_c1,'c3_max':maximum_c3,'time_stanard_deviation_c3':time_stanard_deviation_c3,'freq_stanard_deviation_c3':freq_stanard_deviation_c3,'time_variance_c3':time_variance_c3,'frequency_variance_c3':frequency_variance_c3,'signal_power_c3':power_time_c3}
df  = pd.DataFrame(dic)
df.to_csv('/home/omar/Documents/python_programs/graduation/datasets/tkn2.csv',index=False)
