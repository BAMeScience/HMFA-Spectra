import numpy as np
from itertools import chain, combinations
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from Voigtfit import Voigtfit as VF
from WeightsFit import WeightsFit as WF
import seaborn as sns
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
import scipy
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.tools as tls
from itertools import chain, combinations
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
import sys


'''
This is an implementation of and Independant Component Analysis method proposed by
" Identification of unknown pure component spectra by indirect hard modeling - Kriesten2008"
to extract the the pure spectra from the a spectral mixture matrix.
'''

gpu = 0

run_fitting = True # when false it will load the last fitting results, should be True for new spectral matrix
Only_SharedPeaks = False # run the optimization of the portion matrix on shared peaks only and not all peaks




spectal_matrix = np.zeros((30,1000)) # load you mixed spectral matrix here
Data_array = np.array(spectal_matrix)




wl = np.zeros(1000)## your wavelength vector, must correspond to you data
Data_org = np.array(spectal_matrix)

Data_array_idx = np.hstack((np.arange(0,len(Data_array)).reshape(-1,1),Data_array))

new_wl = wl

subsets = list(combinations(Data_array_idx, 7))
corr_results = np.loadtxt('./Correlation results/correlation_results')

id_lowest = np.array(subsets[corr_results.argmin()])[:,0].astype('int')
####### Step 1 --> find x_input - maximum subset search is 7  ##################
Data_mean = Data_array[id_lowest].mean(axis=0)


Data_mean = Data_array.mean(axis=0)


# peaks fitting
if run_fitting:
    N= len(new_wl)
    x1 = Data_array[0]
    ####### Step 2 --> Voigt fitting on x_input - Noise Fitting & Peaks cutoffs at 10 ###################### 
    optimizer = VF(new_wl,Data_mean,Peak_threshold=0.15,GPU=gpu,max_itr=200)
    OPT_parameters = optimizer.FitData()
    np.savetxt('Voig_parameters_X_input',OPT_parameters)
    reconstruction = optimizer.Reconstruct_voigt(new_wl,OPT_parameters)
    mpl_fig = plt.figure(figsize=(12,12))
    plt.plot(reconstruction)
    plt.plot(Data_mean)
    plt.show()
    plotly_fig = tls.mpl_to_plotly(mpl_fig)
    plotly_fig.write_html("reconstruction.html")
    

    optimizer2 = WF(new_wl,x1,OPT_parameters,GPU=gpu,max_itr=200)
    w1 = optimizer2.FitData()
    reconstruction2 = optimizer2.Reconstruct_voigt(new_wl,OPT_parameters,w1)
    ####### Step 3 --> Weight Matrix  ###################### 

    W = np.zeros((Data_array.shape[0],OPT_parameters.shape[1]))

    for i in range(Data_array.shape[0]):
        optimizer2 = WF(new_wl,Data_array[i],OPT_parameters,GPU=gpu,max_itr=200)
        W[i] = optimizer2.FitData()

    
    np.savetxt('WeightsMatrix', W)
else:

    W = np.loadtxt('WeightsMatrix')
    OPT_parameters = np.loadtxt('Voig_parameters_X_input')

o_idx=  np.where((OPT_parameters[2,:] > 777.1) &(OPT_parameters[2,:] < 778))[0]



W_corr = np.corrcoef(W.T)
#W_corr, uncorr_p = spearmanr(W)
W_corr = np.round(W_corr,10)

W_corr99 = W_corr>=.99
W_corr99_s = W_corr99[:100,:100]


plt.figure(figsize=(7,7))
sns.heatmap(W_corr99_s)
plt.show()

ox_corr = W_corr[o_idx[0]:o_idx[-1]+1,o_idx[0]:o_idx[-1]+1]
print('Oxygen peaks correlation')
print(ox_corr)



def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True 
    else:
        return False
def Union(a,b):
    return sorted(list(set(a)) + list(set(b)))


####### Step 4 --> Peak2Peak-Correlation Matrix 1 - (Correlation threshold has huge effect on Delta Matrix)  
#######                                    2 - Correlation Threshold (0.98 or 0.99) might be biased to large peaks when very small peaks appear their also 
#######                                    3 - Correlation Threshold might affect the number of components (K-components)
#######                                    4 - Highly Correlated does not necceserly mean the peaks are totally pure
#######                                    5 -  


corr_threshold = 0.98#np.min(ox_corr) # takes the minimum correlation of the oxygen fingerprints as a reference 

cor99_list  = []
for i in range(W_corr.shape[0]):

    cor99_list.append(np.where(W_corr[i]>=corr_threshold)[0])

def flatten(l):
    return [item for sublist in l for item in sublist]

# find correlated peaks, if peak 1 and peak 2 are correlated and peak 2 and peak 3 are correlated then peak 1, 2 and 3 are correlated 
K_list=[]
U_peaks_length = []
for i in range(W_corr.shape[0]):
    spec=cor99_list[i]
    for j in range(W_corr.shape[0]):
        if common_member(list(spec),list(cor99_list[j])):
            spec = set(spec).union(set(cor99_list[j]))
    if not common_member(list(spec),flatten(K_list)) and len(spec)>1:
        K_list.append(list(spec))
        U_peaks_length.append(len(spec))



K_components = len(U_peaks_length)

unique_peaks_n = sum(U_peaks_length)
distinctive_peaks = [item for sublist in K_list for item in sublist]

distinctive_peaks_1 =[]

print('number of components found:',K_components)
# find distinctive peaks, defined as the peaks that have the minimum correlation with other peaks
for i,k in enumerate(K_list):
    min_corr_id = W_corr[k].mean(axis=1).argmin()
    distinctive_peaks_1.append(k[min_corr_id])

all_peaks_idx = list(np.arange(0,W.shape[1]))
shared_peak_idx = list(set(all_peaks_idx) - set(distinctive_peaks_1))


optimizer = VF(new_wl,Data_mean,Peak_threshold=0.15,GPU=gpu,max_itr=200)
reconstruction = 0 

mpl_fig = plt.figure(figsize=(12,12))
leg =[]
for id,k in enumerate(K_list):
    reconstruction = optimizer.Reconstruct_voigt(new_wl,OPT_parameters[:,k])# + reconstruction
    plt.plot(new_wl,reconstruction,label=str(id))

    
plt.legend()

plt.show()
plotly_fig = tls.mpl_to_plotly(mpl_fig)
plotly_fig.write_html('correlation threshold 0.99.html')
    
    
plt.figure(figsize=(9,9))
plt.plot(new_wl,reconstruction)
plt.plot(new_wl,Data_mean,alpha=0.3)
plt.xlim([760,780])
plt.show()

plt.figure(figsize=(9,9))
plt.plot(new_wl,Data_mean-reconstruction,alpha=0.3)
plt.show()


####### Step 5 --> Delta Matrix - (Very Vunrable to Peak2Peak Correlation Matrix, the Selected X_input and threshold of voigt fitting)
#######                            Peaks chosen as distinctive by the correlation threshold can be interpreted as not shared by the portion optimization
#######                            Therefore Peaks must be for certain distincive before running the portion optimization
######                             The problem needs to be checked for ill-conditioning
 ################## for all shared peaks #################


error_list= []
if Only_SharedPeaks:
    delta_matrix_shared = np.zeros((len(shared_peak_idx),len(distinctive_peaks_1)))
    delta_matrix = np.zeros((len(W.T),len(distinctive_peaks_1)))
    for i,K in enumerate(K_list):
        delta_matrix[K,i]= 1

    
    for i in range(len(shared_peak_idx)):
        def cost_fun(x):
            y = np.linalg.norm(W[:,shared_peak_idx[i]] - np.sum(W[:,distinctive_peaks_1] * x , axis=1)) + 100 * np.power(1-sum(x),2)
            return y

        x0 = np.random.uniform(0,1,len(distinctive_peaks_1))#.reshape(1,-1)
        lb= np.zeros(len(distinctive_peaks_1))#.reshape(1,-1)
        ub= np.ones(len(distinctive_peaks_1))#.reshape(1,-1)

        sol = minimize(cost_fun, x0, method='SLSQP',
                            bounds=scipy.optimize.Bounds(lb, ub, keep_feasible=False))

        x1 = np.copy(sol.x)
        delta_matrix_shared[i] = x1
        delta_matrix[shared_peak_idx[i],:] = x1

        error = sol.fun - 100 * np.power(1-sum(sol.x),2)
        error_list.append(error)
############################For All peaks  ##################################
else:
    error_list= []
    RMSE = []
    #distinctive_peaks_1=np.arange(0,W.shape[1],1)#np.array([33])
    delta_matrix = np.zeros((len(W.T),len(distinctive_peaks_1)))
    Lambda=100
    for i in range(len(delta_matrix)):
        def cost_fun(x):
            y = np.linalg.norm(W[:,i] - np.sum(W[:,distinctive_peaks_1] * x , axis=1)) + Lambda * np.power(1-sum(x),2) #- 1*np.linalg.norm(x)
            return y
        peak_error_list =[]
        sol_list = []
        for trial in range(1):
            #print(trial)
            #distinctive_peaks_1 = np.array(list(set(distinctive_peaks_1) - set(list([i]))))
            x0 = np.random.uniform(0,1,len(distinctive_peaks_1))#.reshape(1,-1)
            lb= np.zeros(len(distinctive_peaks_1))#.reshape(1,-1)
            ub= np.ones(len(distinctive_peaks_1))#.reshape(1,-1)

            sol = minimize(cost_fun, x0, method='SLSQP',
                                bounds=scipy.optimize.Bounds(lb, ub, keep_feasible=False))
            error = sol.fun - Lambda * np.power(1-sum(sol.x),2)
            peak_error_list.append(error)
            sol_list.append(sol.x)

        error_list.append(peak_error_list[np.argmin(peak_error_list)])

        x1 = sol_list[np.argmin(peak_error_list)]
        #x1 = np.copy(sol.x)
        delta_matrix[i] = x1
        f = W[:,i] - np.sum(W[:,distinctive_peaks_1] * x1 , axis=1)
        RMSE.append(np.abs(f).mean())
        #print(i)
    error_list = np.array(error_list)
    RMSE= np.array(RMSE)
    print(np.mean(error_list))
    print(np.median(error_list))    


error_factor = 1- ((1/RMSE) / np.max(1/RMSE)) 

delta_matrix = delta_matrix * error_factor[:,None]

plt.figure(figsize=(7,7))
sns.heatmap(delta_matrix)
plt.show()


#optimizer2 = WF(new_wl,x1,OPT_parameters,GPU=gpu,max_itr=200)
#reconstruction2 = optimizer2.Reconstruct_voigt(new_wl,OPT_parameters,delta_matrix[:,0])
#plt.plot(reconstruction2)


reconstruction_all = 0
Pure_spectra = np.zeros((delta_matrix.shape[1],len(new_wl)))
optimizer2 = WF(new_wl,x1,OPT_parameters,GPU=gpu,max_itr=200) # init of the fitting module
for i in range(delta_matrix.shape[1]):
    reconstructionk = optimizer2.Reconstruct_voigt(new_wl,OPT_parameters,delta_matrix[:,i])
    reconstruction_all = reconstruction_all + reconstructionk
    Pure_spectra[i] = reconstructionk
plt.plot(reconstruction_all)
plt.plot(Data_mean)

