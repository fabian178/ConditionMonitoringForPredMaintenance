

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from sklearn.ensemble import RandomForestClassifier
import random
import pandas as pd
from sklearn.externals import joblib
from sklearn.svm import SVC
import pickle
import data_extraction as extraction
from sklearn.model_selection import GridSearchCV
import librosa as lr
from matplotlib.pyplot import subplots_adjust

def extractFeatures(df, s_rate, percentage_testing = 0):
    
    ind = df.index
   
    feature_list = []
    label_list = []
    group_list=[] #should at the end be: [j2,j3,j4], then reset for each group
    
   
    new_index = []
    for e in df['group']:
        if e not in new_index:
            new_index.append(e)
    
    new_dataframe = pd.DataFrame(index = new_index, columns=['feature vector', 'label','feature names', 'type', 'name'])
    
    
    prev_group = 0
    feature_names = []
    for i in ind:
        
        group = df.get_value(i,'group')
        data = df.get_value(i,'data')
        label = df.get_value(i,'label')
        type1 = df.get_value(i, 'type')
        name1 = df.get_value(i,'name')
        position_name = df.get_value(i,'position')
        
        #if we have the same group, 
        if prev_group==group: #belongs to same time stamp
            print('length: ' + str(len(data[0])))
            dd,feature_names = getFeature(data,s_rate)
            #for ii in range(len(dd)):
            #    print(str(dd[ii]) + '\t' + feature_names[ii])
            #exit()
            
            for x in range(len(dd)):
                try: #this tests to see if it is an array instead of a value, will do nothing if everything Ok.
                    print(str(len(dd[x])))
                    print(feature_names[x])
                except:
                    continue
            
            group_list.append(dd)
            
            
        else:
            
            #next time stamp encountered, load the full group_list into the feature_list
            #flatten the group list first
            #next time stamp encountered, load the full group_list into the feature_list
            #flatten the group list first
            feature_list.append(np.concatenate(group_list))
            label_list.append(label)
            print('group done: ' + str(group))
        
            #now append to the feature dataframe, need to take i-1 to get prev group's values
            new_dataframe.at[prev_group,'feature vector'] = list(np.concatenate(group_list))
            new_dataframe.at[prev_group, 'label'] = df.get_value(i-1,'label')
            new_dataframe.at[prev_group, 'type'] = df.get_value(i-1, 'type')
            new_dataframe.at[prev_group,'name'] = df.get_value(i-1,'name')
        
            print(str(prev_group))
            prev_group=group #new time stamp
            while(len(group_list)) > 0: 
                group_list.pop() #reset the group feature list, delete all previous entries
            group_list.append(getFeature(data, s_rate)[0]) #and append the first one
    
    #append the last data entry to the new df
    new_dataframe.at[prev_group,'feature vector'] = list(np.concatenate(group_list))
    new_dataframe.at[prev_group, 'label'] = label
    new_dataframe.at[prev_group, 'type'] = type1
    new_dataframe.at[prev_group,'name'] = name1 
    
    complete_feature_names = []
    po = ['j2','j3','j4'] #we have 3 positions
    for entry in po:
        for x in feature_names:
            complete_feature_names.append(x+ '-' + entry)
    
    for ii in new_dataframe.index:
        new_dataframe.at[ii, 'feature names'] = complete_feature_names #just put it for every column
    
    return new_dataframe
        
        
def getFeature(data, s_rate):
    #this is a private function, used by the extratFeatures function only
    
    position_name=''
    ar = []
    feature_label = []
    #data is a 2d array, eg [[xj2],[yj2],[zj2]]
    for i in range(len(data)):
            
        d = data[i]
        t_array =[] #time array
        for ttt in range(len(d)):
            t_array.append(ttt*1.0/s_rate)
            
        d = np.array(d)
        
        ll = ""
        if len(data) == 3:
            if i ==0:
                ll='x'
            elif i ==1:
                ll='y'
            elif i==2:
                ll='z'
        
            
                
        ar.append(np.mean(d)) #mean
        feature_label.append('t_mean-' + ll + '-' + position_name)
        ar.append(max(d))
        feature_label.append('t_max-' + ll+ '-' + position_name)
        ar.append(min(d))
        feature_label.append('t_min-' + ll+ '-' + position_name)
        ar.append(np.var(d)) #variance
        feature_label.append('t_variance-' + ll+ '-' + position_name)
        ar.append(sp.stats.skew(d)) #skew
        feature_label.append('t_skew-' + ll+ '-' + position_name)
        ar.append(sp.stats.kurtosis(d))#kurtosis
        feature_label.append('t_kurtosis-' + ll+ '-' + position_name)
        ar.append(((d[:-1] * d[1:]) < 0).sum()/float((len(d)))) #number of zero crossings, normalized by length of 
                                                                #data,ie ratio of zero crossing
        feature_label.append('t_zero-crossing-rate-' + ll+ '-' + position_name)
        
        #linear correlation coefficient = pearson corr corefficient, if more than one data entry - mars and bma
        
        for j in range(len(data)):
            if j ==i:
                continue
            
            corr_label = ''
            if len(data) == 3: #these labels only for bma
                if j ==0:
                    corr_label='x'
                elif j ==1:
                    corr_label='y'
                elif j==2:
                        corr_label='z'
                        
            v = sp.stats.pearsonr(data[i],data[j])
            ar.append(v[0]) #pearson linear correlation coefficient
            feature_label.append('t_lin-corr-coeff-' + ll+ corr_label+ '-' + position_name)
            ar.append(v[1]) #2-tailed p value
            feature_label.append('t_2-tailed-p-value-' + ll+corr_label+ '-' + position_name)
        print('finished extracting time features')
        #now frequency domain
        #location of peak, value of peak ie magnitude at that frequency, 
        f = 10*np.log10(np.abs(np.fft.rfft(data[i]))**2) #log magnitude spectrum
        f[0] = 0 #take out the dc component.
        freqs = np.fft.rfftfreq(len(data[i]),1.0/s_rate)#one sided real fft
        
        idx = np.argsort(freqs)
        freqs = freqs[idx]
        f = f[idx]
       
        ar.append(sum(freqs*f)/sum(f)) #spectral centroid
        feature_label.append('f_spectral-centroid-' + ll + '-' + position_name)
        ar.append(np.max(f)) #max value of spectrum
        feature_label.append('f_max-amplitude-' + ll+ '-' + position_name)
        ar.append(freqs[np.argmax(f)]) #frequency of max value
        feature_label.append('f_f-of-max-amplitude-' + ll+ '-' + position_name)
        ar.append(np.mean(f)) #mean of the spectrum
        feature_label.append('f_mean-' + ll+ '-' + position_name)
        ar.append(np.var(f))#variance
        feature_label.append('f_var-' + ll+ '-' + position_name)
        ar.append(sp.stats.skew(f)) #skewness
        feature_label.append('f_skew-' + ll+ '-' + position_name)
        ar.append(sp.stats.kurtosis(f)) #kurtosis
        feature_label.append('f_kurtosis-' + ll+ '-' + position_name)
        #spectral roll off - the frequency below which some fraction, 
        #k (typically 0.85, 0.9 or 0.95 percentile), of the cumulative spectral power resides
        #choose k = 85, done in paper 3
        k_constant = 0.85
        powersum = np.sum(f)
        rolloff_f = freqs[-1]
        for q in range(len(f)):
            csum = np.sum(f[:q])
            if csum >= powersum*k_constant:
                rolloff_f = freqs[q]
                break
        if rolloff_f==0:
            print('could not find roll off frequency, taking last frequency valie')
            
        
        ar.append(rolloff_f)
        feature_label.append('f_rolloff-f--k='+str(k_constant)+'-' + ll+ '-' + position_name)
        print('finished extracting frequency features')
        
        #now the spectrogram
        
        fs,ts,ss = sp.signal.spectrogram(d,s_rate)
        
        
        rmse = lr.feature.rmse(S = ss) #root mean sqaure energy for each time bin, rmse takes no s_rate as argument
        rolloffs = lr.feature.spectral_rolloff(S=ss, sr = s_rate) #roll of f, for each time bin
        centroids = lr.feature.spectral_centroid(S=ss, sr = s_rate) #centroid for each time bin
        
        #THIS IS TO REORDER THE SPECTROGRAM INTO TIME BINS TO GET THE MOST PROMINENT AND LEAST PROMINENT FREQUENCY IN A TIME BIN
        magnitudes = []
        for s1 in range(len(ss[0])):
            time_bin = []
            for s2 in range(len(ss)):
                time_bin.append(ss[s2][s1])
            magnitudes.append(time_bin)
        #now we have split the spectrogram into time bins. the magnitudes list holds the values
        max_fs = []
        min_fs = []
        for xx in range(len(magnitudes)):
            l = magnitudes[xx]
            max_val = max(l)
            min_val = min(l)
            min_index = l.index(min_val)
            max_index = l.index(max_val)
            min_fs.append(fs[min_index])
            max_fs.append(fs[max_index])
        
        #print('len of all: ' +str(len(max_fs)) + '-' + str(len(min_fs)) + '-' + str(len(rolloffs[0])))
        #print(max_fs)
        #print(min_fs)
        
        # the length of these varies depending on the length of the time signal since bins of time samples are taken for 
        #the short time fourier transform, default 256 with scipy.
        #therefore, features need to be extracted in such a way so that the same number of features exists for every time signal
        spectrogram_features = [rmse[0],rolloffs[0],centroids[0], max_fs, min_fs] #these all have the same size = num of time bins
        
        sg_f_names = ['rmse', 'spectral-rolloff', 'spectral-centroid', 'highest-f-per-bin', 'lowest-f-per-bin']
        #get statistical features of each:
        for x1 in range(len(spectrogram_features)):
            d1 = spectrogram_features[x1] #get the actual data
            name = sg_f_names[x1]
            
            ar.append(np.mean(d1))
            feature_label.append('tf_'+name+'-mean-' + ll+ '-' + position_name)
            ar.append(max(d1))
            feature_label.append('tf_'+name+'-max-' + ll+ '-' + position_name)
            ar.append(min(d1))
            feature_label.append('tf_'+name+'-min-' + ll+ '-' + position_name)
            ar.append(np.var(d1))
            feature_label.append('tf_'+name+'-variance-' + ll+ '-' + position_name)
            ar.append(sp.stats.skew(d1))
            feature_label.append('tf_'+name+'-skew-' + ll+ '-' + position_name)
            ar.append(sp.stats.kurtosis(d1))
            feature_label.append('tf_'+name+'-kurtosis-' + ll+ '-' + position_name)
      
        
            
            
            
        print('finishd extracting spectrogram features')
        '''
        print('rolloff: ' + str(rolloffs))
        print('len rolloff: ' + str(len(rolloffs)))
        print('len rolloff data: ' + str(len(rolloffs[0])))
        print('len s: ' + str(len(ss[0])) + ', len f: ' + str(len(fs)) + ', len t: ' + str(len(ts)))
        ener = lr.feature.rmse(S=ss)
        print(ener)
        print('len ener: ' + str(len(ener[0])))
        '''
        
        '''
        plt.figure()
        
        plt.subplot(4,1,1)
        plt.plot(t_array,d)
        plt.title('Zeitsignal')
        
        plt.subplot(4,1,2)
        #plt.xscale('log')
        plt.yscale('log')
        plt.plot(freqs,f)
        plt.title('Frequenzspektrum')
        
        plt.subplot(4,1,3)
        plt.pcolormesh(ts,fs,ss)
        plt.title('Spektrogramm')
        #plt.xscale('log')
        #plt.yscale('log')
        
        plt.subplot(4,1,4)
        plt.plot(ts,centroids[0])
        plt.yscale('log')
        
        plt.tight_layout()
        plt.show()
        exit()
        '''
    
    
    return ar, feature_label #ar is a 1d array containing the features for one pos [xj2f1,xj2f2...,yj2f1,...]
    #feature_label is  a list containing the names of the features
       