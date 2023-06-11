
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as signal
import os
from sklearn.ensemble import RandomForestClassifier
import random
import pandas as pd
from sklearn.externals import joblib
from sklearn.svm import SVC
import wave
from scipy.io.wavfile import read
import pickle


def getFiles(directory, use_sensors = ['bma','ks', 'mars']):
    '''
    returns the extracted data with training and testing subsets
    '''
    
    #SORT THE FILES INTO TIMESTAMPS: {time1: [j2,j3,j4], time2:....}
    
    used_sensors = ''
    for s in use_sensors:
        used_sensors = used_sensors + '-' + s
    
    print('gathering data for directory: ' + directory)
    #we have posotions j2, j3, j4
    MARSj2 = []
    BMAj2 = []
    KSj2 = []
    
    MARSj3 = []
    BMAj3 = []
    KSj3 = []
    
    MARSj4 = []
    BMAj4 = []
    KSj4 = []
    
    
    
    files = os.listdir(directory)
    print('total length of files before removal: ' + str(len(files)))
    
    c1 = 0
    c2 = 0
    
    for f in files[:]:
        
        #this is just to remove all files that have no Trigger1
        if not 'trigger1' in f.lower() and not 'trigger2' in f.lower() and not 'trigger3' in f.lower() and not 'trigger4' in f.lower():
            c2 +=1
            files.remove(f)
            continue
        c1 +=1
        
        if not '.csv' in f:
            if not '.wav' in f:
                    files.remove(f)
                    continue
        
        if 'mars' in f.lower() and 'mars' in use_sensors:
            if "j2" in f.lower():
                MARSj2.append(f)
                continue
            elif 'j3' in f.lower():
                MARSj3.append(f)
                continue
            elif "j4" in f.lower():
                MARSj4.append(f)
                continue
                
        if 'sbss' in f.lower():
            if "bma" in f and 'bma' in use_sensors:
                if 'j2' in f.lower():
                    BMAj2.append(f)
                    continue
                elif 'j3' in f.lower():
                    BMAj3.append(f)
                    continue
                elif 'j4' in f.lower():
                    BMAj4.append(f)
                    continue
            if not 'bma' in f and 'ks' in use_sensors:
                if not 'ks' in use_sensors:
                    continue
                if 'j2' in f.lower():
                    KSj2.append(f)
                    continue
                elif 'j3' in f.lower():
                    KSj3.append(f)
                    continue
                elif 'j4' in f.lower():
                    KSj4.append(f)
                    continue
    print('bma')
    print(BMAj2)
    print(BMAj3)
    print(BMAj4)
    
    print('KS')
    print(KSj2)
    print(KSj3)
    print(KSj4)
    
    print('MARS')
    print(MARSj2)
    print(MARSj3)
    print(MARSj4)
    #just to check: if c1 and c2 are both equal, it means we have all Trigger1
    print('number of files removed: '+ str(c2))
    
    
    sorted_files = [BMAj2,BMAj3,BMAj4,KSj2,KSj3,KSj4,MARSj2,MARSj3,MARSj4]
    
    length_of_longest_entry = 0
    #all entries in sorted1 should have the same length, if not catch which sensor has a file missing
    for entry in sorted_files:
        if len(entry) > length_of_longest_entry:
            length_of_longest_entry = len(entry)        
        
    for entry in sorted_files:
        if len(entry) == 0:
            continue
        if not len(entry) == length_of_longest_entry:
            print('this entry does not have the same length as others:' + str(length_of_longest_entry) + 'length of this position: ' + str(len(entry)))
            print(entry)
            print(directory)
            exit()
    
    #now to sort the files
    sorted_list = []
    for i in range(length_of_longest_entry):
        f = []
        for entry in sorted_files:
            
            if len(entry)==0:
                continue
            f.append(entry[i])
            
        sorted_list.append(f)
        
    #file list is now: [[j2bma,j3bma,j4bma,j2ks,j3ks,j4ks,j2mrs,j3mars,j4mars],[j2bma,j3bma....]]
    return sorted_list
    
def getDataFromFiles(sortedFiles,directory,group_ind=0):
    #sortedFiles is a list of lists containing all files 
   
    print(sortedFiles)
    
    
    
    print('extracting data')
    print(sortedFiles)
    
    num_of_files = 0
    for e in sortedFiles:
        num_of_files += len(e)
         
    cols = ['data','position','type', 'name', 'group', 'trigger','label']
    ind = range(num_of_files)
    
    df = pd.DataFrame(columns = cols, index=ind)
    print('dataframe: ')
    
    dataframe_index = 0
    group_index = group_ind
    
    for fs in sortedFiles:
        
        for f in fs:
                
            df.at[dataframe_index,'name'] = f 
            df.at[dataframe_index,'group'] = group_index
            df.at[dataframe_index,'position'] =  f.split('_')[2]
            df.at[dataframe_index,'trigger'] = f.split('_')[3]
            
            if '.' in f:
                df.at[dataframe_index,'trigger'] = f.split('_')[3].split('.')[0]
            
            #to get sensor type
            stype = ''
            if 'sbss' in f.lower():
                if 'bma' in f.lower():
                    stype = 'bma'
                else:
                    stype = 'ks'
            elif 'mars' in f.lower():
                stype = 'mars'
                
            if stype =='':
                print('Error: sensor type could not be determined')
                exit()
                
            df.at[dataframe_index,'type'] = stype
            
            #read out contents of files
           
            #if not used, skip it
            
            print('extracting data from: ' + f)
                
            data_list_or_bma_x = []
            data_bma_y = []
            data_bma_z = []
            mars_ch1 = []
            mars_ch2 = []
        
            #if c ==5:
            #   break
            
            
            try:
                if '.csv' in f.lower():#the other one is a wave file and cannot be opened like this
                    data = open(directory+f , 'r')
                elif '.wav' in f.lower():
                    #this returns 3 objects
                    #[0] is the sampling rate, [1] is the amplitude values. set of 2 values, left and right channel
                    #[3] is the bit resolution of the amplitude.
                    data = read(directory+f,'r')[1] 
                    
            except Exception as e:
                print(e)
                print('error with: ' + f)
                continue
            if 'bma' in f.lower():
                
                #first clear all line breaks in data
                #clearExtraSymbols(data)
                wrongentries = False
                for line in data:
                    
                    line = line.split(' ')
                    if '\n' in line[:]:
                        line.remove('\n')
                    
                    
                    xyz = [line[i:i+3] for i in range(0, len(line), 3)]
                    
                    #remove all entries that do not have full x,y,z values
                    for xyzEntry in xyz[:]:
                                
                        if not len(xyzEntry) == 3:
                            xyz.remove(xyzEntry)
                            print('this xyz value did not have 3 entries and is removed: ')
                            print(xyzEntry)
                            wrongentries = True
                        
                        if len(xyzEntry)==3:
                            for eee in xyzEntry:
                                try:
                                    int(eee)
                                except:
                                    print('not all values were ints, removing: ')
                                    print(xyzEntry)
                                    xyz.remove(xyzEntry)
                                    
                    for xyzEntry in xyz:
                        try:
                            data_list_or_bma_x.append(int(xyzEntry[0]))
                            data_bma_y.append(int(xyzEntry[1]))
                            data_bma_z.append(int(xyzEntry[2]))
                        except:
                            print('error with bma, xyz set: ')
                            print(xyzEntry)
                            print(' in ' + f)
                            continue
                            
                if wrongentries:
                    print('file name: ' + directory+f)
                    '''
                    plt.subplot(3,1,1)
                    plt.plot(data_list_or_bma_x)
                    plt.title('x')
                    plt.subplot(3,1,2)
                    plt.plot(data_bma_y)
                    plt.title('y')
                    plt.subplot(3,1,3)
                    plt.plot(data_bma_z)
                    plt.title('z')
                    
                    plt.show()
                    '''
                    wrongentries= False
                
                aaa = False
                if not len(data_list_or_bma_x)==len(data_bma_y):
                    print('lengths not equal, exiting: ' + directory + f)
                    aaa=True
                if not len(data_list_or_bma_x)==len(data_bma_z):
                    print('lengths not equal, exiting: ' + directory + f)
                    aaa=True
                if not len(data_bma_y)==len(data_bma_z):
                    print('lengths not equal, exiting: ' + directory + f)
                    aaa=True
                    
                if aaa == True:
                    exit()
                    
                df.at[dataframe_index,'data'] = [data_list_or_bma_x, data_bma_y, data_bma_z]
                
               
                    
            elif 'mars' in f.lower(): 
                
                #data has 2 channels
                #[[1,2],[1,2]....]]
                #each data 
                for frame1 in data:
                    try:   
                        mars_ch1.append(frame1[0])
                        mars_ch2.append(frame1[1])
                    except:
                        print('error with: '+ f + ' line: ' + line)
                        continue
                
                #take the mean of the two channels as pre processing step
                #this takes the elementwise mean of each entry
                avg_channel = np.mean([mars_ch1,mars_ch2], axis = 0)
                
                '''
                print(str(len(mars_ch1)))
                print(str(len(mars_ch2)))
                plt.subplot(3,1,1)
                plt.plot(mars_ch1)
                plt.subplot(3,1,2)
                plt.plot(mars_ch2)
                plt.subplot(3,1,3)
                plt.plot(avg_channel)
                plt.show()
                exit()
                '''
                df.at[dataframe_index,'data'] = [avg_channel]
                
                
            else: #must be KS
                
                #print('ks: ' + f)
                for line in data:
                    line = line.split(' ')
                    if '\n' in line[:]:
                        line.remove('\n')
                        
                     
                    try:
                        for entry in line:
                            data_list_or_bma_x.append(int(entry))
                            
                    except Exception as e:
                        
                        print('error with: ' + f + ' line: ')
                        print(line)
                        print('entry: ' + repr(entry))
                        print(e)
                        continue
                
                df.at[dataframe_index,'data'] = [data_list_or_bma_x]
                
                
            dataframe_index += 1               
        group_index += 1
    
    #now go through the dataframe again and discard all groups where there was no data in the file
    groups_to_remove = []
    for i in range(len(df)):
        if df.at[i,'data'][0] == []:
            groups_to_remove.append(df.at[i,'group'])
    print('groups to remove: ')
    print(groups_to_remove)
    
    for i in range(len(df)):
        if df.at[i,'group'] in groups_to_remove:
            for x in df.columns:
                df.at[i,x] = np.nan #set the whole row where data is empty to nan
    
            
    df = df.dropna(how = 'all')
    df = df.reset_index(drop=True)
    
    return df