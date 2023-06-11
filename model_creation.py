import numpy as np
from matplotlib import pyplot as plt

import scipy as sp
from sklearn.ensemble import RandomForestClassifier
import random
import pandas as pd
from sklearn.externals import joblib
from sklearn.svm import SVC
import pickle
from sklearn.model_selection import GridSearchCV
from matplotlib.pyplot import subplots_adjust
import json

from sklearn.feature_selection import chi2,SelectKBest, RFECV, RFE, SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

        
import data_extraction, feature_extraction
import MerkmalsSelektion as ms
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
   
def classification_(df, selected_features, writeFile, iterations = 1,folds=10):
    '''This function takes as input a feature matrix X and a target variable array label and conducts crossvalidation
    for multiple classifiers specified in "classifiers" (NNK, SVM - lin, SVM - gauss, Random Forest, Naive Bayes) which multiple
    parameters sets as specified in "poss_params".
   
    This evaluation can be done multiple time by controlling the "iterations" variable which is by default set to one.
    
    As an output, the best crossvalidation classification accuracy is returned.
   
    This function has been used to evaluate sensors regarding their ability for classification in AMELI.
    For each single sensor, the feature matrix is computed and then fed to this function'''
    used_features_list = []
    label = []
    for i in range(len(df)):
        feature_vector = df['feature vector'][i]
        feature_names = df['feature names'][i]
        
        
        
        used_features_for_this_entry = []
        
        for xx in range(len(feature_names)): #can use feature names or feature vector, they both have the same length
            current_feature_name = feature_names[xx]
            current_feature = feature_vector[xx]
            
            if current_feature_name in selected_features:
                used_features_for_this_entry.append(current_feature)
                
        used_features_list.append(used_features_for_this_entry)
        
        label.append(df['label'][i])
      
    
    X = used_features_list
    X = np.array(X)
    num_feats = X.shape
   
    prediction = []
    params_ = []
 
    classifiers = [ [KNeighborsClassifier(),'NNK'],
                    [svm.LinearSVC(), 'SVM - lin'],
                    [svm.SVC(), 'SVM _ gaus'],
                    [RandomForestClassifier(), 'RF'],
                    [GaussianNB(), 'NB']
                    ]
   
    
    C_conventional = [2**k for k in range(-5,16)]
    gamma_conventional = [2**k for k in range(-15, 4)]
   
    poss_params = [{'p':[1,2,4,8], 'leaf_size':[1,2,5,10,15,20,25,30]},
                   {'C': C_conventional, 'dual':[False, True], 'tol':[1e-5,1e-4,1e-3,1e-2,1e-1], 'fit_intercept':[True,False], 'intercept_scaling':[0.5,1.0,1.5], },
                   {'C': C_conventional, 'gamma': gamma_conventional,  'kernel': ['linear', 'poly'], 'tol':[1e-4,1e-3,1e-2]},
                   {'max_depth':[1,2,3, None], 'n_estimators':[5, 10, 20], 'bootstrap': [True, False], 'min_samples_split':[1,2,3], 'min_samples_leaf':[1,2,3], 'warm_start':[True,False], },
                   {'priors':[None]}
                  ]
   
    for idx,i in enumerate(classifiers):
       
        if idx ==2: #skip gaussian svm
            continue
        
        print('evaluating ', i[1])
        clf_score = []
        clf_params = []
        for count in range(iterations):
            clf = GridSearchCV(i[0], poss_params[idx], return_train_score = True, cv = folds)
            clf.fit(X, label)           
            clf_score.append(clf.best_score_)
            clf_params.append(clf.best_params_)
           
        print(i[1], clf_score)
        prediction.append((i[1], np.mean(clf_score)))
        params_.append((i[1], clf_params))
        
    #now write the results to file
    writeFile.write('Accuracies and optimal parameters: \n')
    for x in prediction:
        writeFile.write(x[0] +' : ' + str(x[1]) + '\n')
    for x in params_:
        writeFile.write(json.dumps(x) + '\n')
        
    print(prediction)   
   
    
        
def extractAndSaveData(dir_read_gut,dir_read_schlecht, dir_save, label= '', sensors = []):
    
    dirs_gut = []
    dirs_schlecht = []
    
    if 'riemenspannung' in dir_read_gut.lower():
        dirs_gut.append(dir_read_gut)
        dirs_schlecht.append(dir_read_schlecht)
        
    elif 'spektrum' in dir_read_gut.lower():
        g1 = dir_read_gut + '01_Werkstueck_1//01_Scheibe_scharf//'
        g2 = dir_read_gut + '02_Werkstueck_2//01_Scheibe_scharf//'
        s1 = dir_read_schlecht + '01_Werkstueck_1//02_Scheibe_stumpf//'
        s2 = dir_read_schlecht + '02_Werkstueck_2//02_Scheibe_stumpf//'
        
        dirs_gut.append(g1 + '01_Vorschubverringerung//')
        dirs_gut.append(g1 + '01b_Vorschubverringerung_2//')
        dirs_gut.append(g1 + '02_Keine_Vorschubverringerung//')
        dirs_gut.append(g2 + '01_Vorschubverringerung//')
        dirs_gut.append(g2 + '02_Keine_Vorschubverringerung//')
        
        dirs_schlecht.append(s1 + '01_Vorschubverringerung//')
        dirs_schlecht.append(s1 + '02_Keine_Vorschubverringerung//')
        dirs_schlecht.append(s2 + '01_Vorschubverringerung//')
        dirs_schlecht.append(s2 + '02_Keine_Vorschubverringerung//')
        
    elif 'luftschleifen' in dir_read_gut.lower():
        g1 = dir_read_gut + '01_Werkstueck_1//01_Scheibe_scharf_vorschub_minimal//'
        g2 = dir_read_gut + '02_Werkstueck_2//01_Scheibe_scharf_vorschub_minimal//'
        s1 = dir_read_schlecht + '01_Werkstueck_1//02_Scheibe_stumpf_vorschub_minimal//'
        s2 = dir_read_schlecht + '02_Werkstueck_2//02_Scheibe_stumpf_vorschub_minimal//'
        
        dirs_gut.append(g1)
        dirs_gut.append(g2)
        dirs_schlecht.append(s1)
        dirs_schlecht.append(s2)
    
    elif 'abrichten' in dir_read_gut.lower():
        g1 = dir_read_gut + '01_Nominaldrehzahl//'
        g2 = dir_read_gut + '02_Drehzahl_minus_30_Prozent//'
        g3 = dir_read_gut + '03_Drehzahl_plus_30_Prozent//'
        
        s1 = dir_read_schlecht + '01_Nominaldrehzahl//'
        s2 = dir_read_schlecht + '02_Drehzahl_minus_30_Prozent//'
        s3 = dir_read_schlecht + '03_Drehzahl_plus_30_Prozent//'
        
        for g in [g1,g2,g3]:
            dirs_gut.append(g + '01_Nominalvorschub_200//')
            dirs_gut.append(g + '02_Vorschub_minus_50_prozent_100//')
        for ss in [s1,s2,s3]:
            dirs_schlecht.append(ss + '01_Nominalvorschub_800//')
            dirs_schlecht.append(ss + '02_Vorschub_minus_50_prozent_400//')
            
            
    for s in sensors:
       
        lastgroupindex = 0
        cols = ['data','position','type', 'name', 'group', 'trigger','label']
        gut_df = pd.DataFrame(columns = cols)
        schlecht_df = pd.DataFrame(columns = cols)
        
        for dirr in dirs_gut:
            files_gut = data_extraction.getFiles(dirr,use_sensors=[s] )
            gut_df = gut_df.append(data_extraction.getDataFromFiles(files_gut, dirr,group_ind=lastgroupindex), ignore_index = True)
            lastgroupindex = gut_df['group'].iat[-1] + 1
            
        gut_df['label'] = 'gut' #when the whole dataframe is built, label the elements
        
        for dirr in dirs_schlecht:
            files_schlecht = data_extraction.getFiles(dirr, use_sensors=[s])
            schlecht_df = schlecht_df.append(data_extraction.getDataFromFiles(files_schlecht, dirr, group_ind = lastgroupindex),ignore_index = True)
            lastgroupindex = schlecht_df['group'].iat[-1] + 1
            
        schlecht_df['label'] = 'schlecht'
        
        
        complete_df = gut_df.append(schlecht_df, ignore_index=True)
        print(complete_df)
        
        print('finished extracting data for: ' + s)
        complete_df.to_pickle(dir_save+s)
        print('finished saving')
    
def f_ratio(feature_dataframe, num = 5):
    
    feature_names = feature_dataframe['feature names'][0]
    cols = feature_dataframe['feature names'][0]
    cols.append('label')
    
    df = pd.DataFrame(index = range(len(feature_dataframe)), columns = cols)
    
    #switch the indexes
    feature_dataframe = feature_dataframe.reset_index(drop = True)
   
    #print(feature_dataframe.isnull().any().any())
    #exit()
    
    for i in range(len(feature_dataframe)):
        try:
            df.loc[i,'label'] = feature_dataframe.loc[i,'label']
        except:
            print(feature_dataframe)
            exit()
        for j in range(len(feature_dataframe['feature vector'][0])): 
            df.loc[i,feature_names[j]] = feature_dataframe['feature vector'][i][j]
    
    
    for i in range(len(feature_dataframe)):
        if df.loc[i,'label'] =='gut':
            df.loc[i,'label'] = 1
        elif df.loc[i,'label'] =='schlecht':
            df.loc[i,'label']=0
    
    #CONVERT ALL TO FLOAT OR INT
    for i in range(0,len(df.columns)):
       df.iloc[:,i] = pd.to_numeric(df.iloc[:,i])
    
    a,b = ms.addon(df,abbr=num)
    print(a)
    print('f ratio features: ')
    print(b)
    return b #these are the feature names
    
def remove_nan(df_feature):
    
    df_feature = df_feature.dropna(how='any')
    c = np.isfinite(list(df_feature['feature vector']))
    feature_names = df_feature['feature names'][0]
    groupCount = 0
    count = 0
    for i in range(len(c)):
        suba = c[i]
        written = False
        for x in range(len(suba)):
            
            entry = suba[x]
            if entry ==False:
                #print('indexes: ' + str(i) + ', ' + str(x))
                aaa = df_feature['feature vector'][i]
                #print(str(aaa[x]) + ' nameoffeature: ' + feature_names[x], 'file: ' + df_feature['name'][i])
                df_feature['feature names'][i] = np.NaN
                written = True
                count +=1
                
        if written == True:
            groupCount+=1
                
    df_feature=df_feature.dropna(how = 'any')
    df_feature = df_feature.reset_index(drop=True)
    
    print('number of removed groups: ' + str(groupCount) + ' total removed: ' + str(count))
    
    return df_feature
    
    
def extraTreeFeatureSelection(df_feature, feature_names):
    
    print('feature importance with extra tree classifier:-----------')
    counter = 0
    used_featres_fi = []
    
    while 1: #repeat forever until no new feature is added
        counter +=1
        #print('going for time: ' + str(counter))
        model = ExtraTreesClassifier()
            
        model.fit(list(df_feature['feature vector']), list(df_feature['label']))
        #print(model.feature_importances_)
        
        #check if there are new features, if not break
        selected_features = []
        for i in range(len(model.feature_importances_)):
            if not model.feature_importances_[i] ==0:
                selected_features.append(feature_names[i])
                
        keepgoing = False
        for entry in selected_features:
            if not entry in used_featres_fi:
                keepgoing = True
                #print('new one is: ' + entry)
                break
        
        if keepgoing==False:
            print('no new features, stopping')
            break
                
        for i in range(len(model.feature_importances_)):
            if not model.feature_importances_[i] ==0:
                used_featres_fi.append(feature_names[i])
                    
    highest_count = 0
    for feature in used_featres_fi:
        if used_featres_fi.count(feature) > highest_count:
            highest_count = used_featres_fi.count(feature)
    print('highest count is: ' + str(highest_count))
    
    highest_features_fi =[]
    for feature in used_featres_fi:
        if used_featres_fi.count(feature) == highest_count: #or used_featres_fi.count(feature) == highest_count -1:
            highest_features_fi.append(feature)
    print('highest features for feature importance: ' + str(len(set(highest_features_fi))))
    
    print(set(highest_features_fi))
    
    return set(highest_features_fi)
    
    
    
    
def rfeSelection(df_feature, feature_names, feature_number):
    print('recursive feature elimination')
    
    used_features_rfe = []
    counter = 0
    while 1:
        counter +=1
        #print('going for time: ' + str(counter))
        clf = RandomForestClassifier()
        rfe = RFE(clf,feature_number) #the same num of features determined in feature importance
        fit = rfe.fit(list(df_feature['feature vector']), list(df_feature['label']))
        #print(str(fit.support_))
        #print('feature ranking: ')
        #print(fit.ranking_)
        selected_features = []

        
        for i in range(len(fit.ranking_)):
            if fit.ranking_[i] == 1:
                selected_features.append(feature_names[i])
        
        keepgoing = False
        for entry in selected_features:
            if not entry in used_features_rfe:
                keepgoing = True
        if keepgoing==False:
            print('no new features, stopping')
            break
                
        for i in range(len(fit.ranking_)):
            if fit.ranking_[i] == 1:
                used_features_rfe.append(feature_names[i])
    
    highest_count = 0
    for feature in used_features_rfe:
        if used_features_rfe.count(feature) > highest_count:
            highest_count = used_features_rfe.count(feature)
    print('highest count is: ' + str(highest_count))
    
    highest_features_rfe =[]
    for feature in used_features_rfe:
        if used_features_rfe.count(feature) == highest_count:# or used_features_rfe.count(feature) == highest_count -1:
            highest_features_rfe.append(feature)
    print('highest features for rfe: ' + str(len(set(highest_features_rfe))))
    
    print(set(highest_features_rfe))
    
    return set(highest_features_rfe)
    
    
def makeFeatureDF(load_dir, save_dir, sr):
    
    df = pd.read_pickle(load_dir)    
    feature_df = feature_extraction.extractFeatures(df, sr)
    
    feature_df.to_pickle(save_dir)
    print('finished saving: ' + save_dir)
        
        
if __name__=='__main__':
    
    '''
    df = pd.read_pickle('Messdaten//03_Spektrum//dataframe_features_bma')
    df = pd.read_pickle('Messdaten//05_Abrichten//dataframe_mars')
    print(df)
    print(df.info())
    print(str(len(df)))
    exit()
    '''
   
    
    #ks: 62500 Hz
    #bma: 2000 Hz
    #wav: 44100 Hz
    f_ks = 62500
    f_bma = 2000
    f_mars = 44100
    
    
    ds_gut = ['Messdaten//02_Riemenspannung//01_gut_gespannt//',
              'Messdaten//03_Spektrum//',
              'Messdaten//04_Luftschleifen//',
              'Messdaten//05_Abrichten//01_Scheibe_scharf_800//']
    ds_schlecht = ['Messdaten//02_Riemenspannung//02_schlecht_gespannt//',
                   'Messdaten//03_Spektrum//',
                   'Messdaten//04_Luftschleifen//',
                   'Messdaten//05_Abrichten//02_Scheibe_stumpf_200//']
    save_dirs = ['Messdaten//02_Riemenspannung//dataframe_', 
                 'Messdaten//03_Spektrum//dataframe_',
                 'Messdaten//04_Luftschleifen//dataframe_',
                 'Messdaten//05_Abrichten//dataframe_',]
    
    
    '''
    #EXTRACTING AND SAVING DATA DATAFRAME
    dir_gut = ds_gut[3]
    dir_schlecht = ds_schlecht[3]
    save_dir = save_dirs[3]
    
    sensors = ['mars']#,'ks', 'mars']
    extractAndSaveData(dir_gut, dir_schlecht, save_dir, sensors=sensors)
    
    exit()
    '''
    
    
    #MAKING THE FEATURE DATAFRAME
    s = 'bma'
    #save_dir = 'Messdaten//03_Spektrum//dataframe_features_'+s#'Messdaten//02_Riemenspannung//dataframe_features_' + s
    load_dir_ks = 'Messdaten//05_Abrichten//dataframe_ks'#'Messdaten//02_Riemenspannung//dataframe_ks'
    load_dir_bma = 'Messdaten//05_Abrichten//dataframe_bma'#'Messdaten//02_Riemenspannung//dataframe_bma'
    load_dir_mars = 'Messdaten//05_Abrichten//dataframe_mars'#'Messdaten//02_Riemenspannung//dataframe_mars'
    
    
    
    '''
    df = pd.read_pickle(load_dir_ks)
    df1 = pd.DataFrame(index = range(len(df)), columns = df.columns)
    for i in range(len(df)):
        row = df.loc[i,:]
        
        print('doing :' + str(i))
        original_data = df.loc[i,'data'][0]
        
        group = df.loc[i,'group']
        new_data = []
        for x in range(len(original_data)):
            if x % 2 ==0:
                new_data.append(original_data[x])
                
        
        row[0] = [new_data]
        df1.loc[i,:] = row
        print('group: ' + str(group))
        print(str(len(original_data)))
        print(str(len(new_data)))
    print(df1)
    df1.to_pickle('Messdaten//05_Abrichten//dataframe_ks_shortened')
    print('finsihed')
    exit()
    '''
    
    #makeFeatureDF(load_dir_ks, 'Messdaten//05_Abrichten//dataframe_features_ks', f_ks)
    #makeFeatureDF(load_dir_bma, 'Messdaten//05_Abrichten//dataframe_features_bma', f_bma)
    #makeFeatureDF(load_dir_mars, 'Messdaten//05_Abrichten//dataframe_features_mars', f_mars)
    #exit()
    
    '''
    x = pd.read_pickle('Messdaten//03_Spektrum//dataframe_features_bma')
    y = pd.read_pickle('Messdaten//02_Riemenspannung//dataframe_features_bma')
    print(x)
    print('length spektrum: ' + str(len(x)))
    print('length riemen: ' + str(len(y)))
    
    print(str(len(x['feature vector'][0])))
    print(str(len(y['feature vector'][0])))
    print(x['feature vector'][0])
    print(x['feature names'][0])
    print(y['feature vector'][0])
    print(y['feature names'][0])
    
    print('finished extracting all features, exiting')
    exit()
    '''
    
    
   
    #DOING THE MODELS
    sbma= 'bma'
    smars = 'mars'
    sks = 'ks'
    
    sd2 = 'Messdaten//02_Riemenspannung//dataframe_features_' 
    sd3 = 'Messdaten//03_Spektrum//dataframe_features_' 
    sd4 = 'Messdaten//04_Luftschleifen//dataframe_features_' 
    sd5 = 'Messdaten//05_Abrichten//dataframe_features_'
    #save_dir = 'Messdaten//03_Spektrum//dataframe_features_' + s
    sds = [sd2+sbma,sd2+sks, sd2+smars, sd3+sbma,sd3+sks,sd3+smars,sd4+sbma,sd4+sks,sd4+smars]#sd2+sbma,sd2+sks,sd2+smars
    sds = [sd5+sbma]
    for save_dir in sds:
        
        core_dir = save_dir.split('dataframe')[0]
        
        df_feature = pd.read_pickle(save_dir)
        df_feature = df_feature.reset_index(drop=True) #set the index properly if it hasnt been set
        df_feature = remove_nan(df_feature.copy()) 
        feature_names = df_feature['feature names'][0]
        print('doing: ' + save_dir)
                
        #classification_(df_feature,[fff[0]], open('test.txt','a'), folds = 2)
        #exit()
        
        
        all_features_fi = []
        while 1:
            highest_features_fi = extraTreeFeatureSelection(df_feature, feature_names)
            doMore = False
            for x in highest_features_fi:
                if not x in all_features_fi:
                    all_features_fi.append(x)
                    doMore = True
            if doMore == False:
                print('finished extra trees, stopping')
                break
                
        print('all features extra trees:')
        print(all_features_fi)
        
                   
        all_features_rfe = []
        while 1:
            highest_features_rfe = rfeSelection(df_feature, feature_names, len(all_features_fi))
            doMore = False
            for x in highest_features_rfe:
                if not x in all_features_rfe:
                    all_features_rfe.append(x)
                    doMore = True
            if doMore == False:
                print('finished rfe, stopping')
                break
        print('all features rfe:')
        print(all_features_rfe)
        
      
        features_fratio = f_ratio(df_feature.copy(),num= max([len(all_features_fi), len(all_features_rfe)]) )
        
        #LOAD IT AGAIN IN CASE IT WAS MODIFIED BY THE FEATURE EXTRACTION METHODS
        df_feature = pd.read_pickle(save_dir)
        df_feature = df_feature.reset_index(drop=True) #set the index properly if it hasnt been set
        df_feature = remove_nan(df_feature.copy()) 
        feature_names = df_feature['feature names'][0]
        
        
        #NOW DO CLASSIFICATION: with different features
        
        features_t = []
        features_f = []
        features_tf = []
        
        for x in feature_names:
            if 't' == x.split('_')[0]:
                features_t.append(x)
            elif 'f' == x.split('_')[0]:
                features_f.append(x)
            elif 'tf' == x.split('_')[0]:
                features_tf.append(x)
                
        features = [feature_names, features_t, features_f, features_tf, all_features_fi, all_features_rfe, features_fratio]
        names = ['All features', 'Only time features', 'Only frequency features', 'Only time-freq features', 'Extratree features', 'Recursive elimination features', 'F-Ratio features']
        wf = open(core_dir+'results_' + save_dir.split('_')[-1], 'a')   
        
        for i in range(len(features)):
            print('doing feture: ' + names[i])
            wf.write(names[i] + ':\n')
            wf.write(json.dumps(features[i]) + '\n') #write the whole list 
            
            
            classification_(df_feature, features[i], wf)
            wf.write('\nNext Feature Set:\n')
            
    
        wf.close()
    
    '''
    both = []
    highest_features_fi = set(highest_features_fi)
    highest_features_rfe = set(highest_features_rfe)
    for i in highest_features_fi:
        if i in highest_features_rfe:
            both.append(i)
    for i in highest_features_rfe:
        if i in highest_features_fi:
            if i not in both:
                both.append(i)
    print('features selected in rfe and feature importance: '+str(len(both)))
    print(both)
    
    exit()
    '''

    
    
    '''
    
    #EXAMPLE PLOTS FOR BMA
    s1='bma'
    save_dir = 'Messdaten//02_Riemenspannung//dataframe_' 
    df = pd.read_pickle(save_dir + s1)
    f1 = 0
    
    if s1 =='ks':
        f1 = f_ks
    elif s1=='bma':
        f1 = f_bma
    else:
        f1 = f_mars
    
    #manche x,y,z sind verschoben weil falsch eingetragen in der csv datei
    gut = df['data'][1][0] #x daten vom dritten eintrag
    schlecht = df['data'][10][0] #x daten vom 4. schlechten eintrag
    
    tgut = []
    tsch = []
    for i in range(len(gut)):
        tgut.append(i*1.0/f1)
    for i in range(len(schlecht)):
        tsch.append(i*1.0/f1)
    
    f_gut = 10*np.log10(np.abs(np.fft.rfft(gut))**2) #log magnitude spectrum
    f_gut[0] = 0 #take out the dc component.
    freqs_gut = np.fft.rfftfreq(len(gut),1.0/f1)#one sided real fft
    idx_gut = np.argsort(freqs_gut)
    freqs_gut = freqs_gut[idx_gut]
    f_gut = f_gut[idx_gut]
    
    f_sch = 10*np.log10(np.abs(np.fft.rfft(schlecht))**2) #log magnitude spectrum
    f_sch[0] = 0 #take out the dc component.
    freqs_sch = np.fft.rfftfreq(len(schlecht),1.0/f1)#one sided real fft
    idx_sch = np.argsort(freqs_sch)
    freqs_sch = freqs_gut[idx_sch]
    f_sch = f_gut[idx_sch]
    
    fs_gut,ts_gut,ss_gut = sp.signal.spectrogram(gut,f1)
    fs_sch,ts_sch,ss_sch = sp.signal.spectrogram(schlecht,f1)
    
    
    
    plt.subplot(3,2,1)
    plt.plot(tgut,gut)
    plt.title('Zeitbereich- Gut')
    plt.subplot(3,2,3)
    plt.plot(freqs_gut,f_gut)
    plt.title('Frequenzbereich - Gut')
    plt.subplot(3,2,5)
    plt.pcolormesh(ts_gut,fs_gut,ss_gut)
    plt.title('Spektrogramm - Gut')
    
    plt.subplot(3,2,2)
    plt.plot(tsch,schlecht)
    plt.title('Zeitbereich - Schlecht')
    plt.subplot(3,2,4)
    plt.plot(freqs_sch,f_sch)
    plt.title('Frequenzbereich - Schlecht')
    plt.subplot(3,2,6)
    plt.title('Spektrogramm - Schlecht')
    plt.pcolormesh(ts_sch,fs_sch,ss_sch)
    
    plt.tight_layout()
    plt.show()
    '''

    
    '''
    clf = RandomForestClassifier() #init a new one
    print('recursive feature elminiation with cross validation')
    rfecv = RFECV(estimator=clf,step = 1, cv = StratifiedKFold(10), scoring = 'accuracy')
    rfecv.fit(list(df_feature['feature vector']), list(df_feature['label']))
    print('feature ranking: ')
    print(rfecv.ranking_)
    print('num of features: ' + str(rfecv.n_features_))
    plt.figure()
    plt.plot(range(1,len(rfecv.grid_scores_) +1), rfecv.grid_scores_ )
    plt.xlabel('num of features')
    plt.ylabel('score')
    plt.show()
    '''
    #print(f)
    #print(l)
    
   
   
    
    
    
    
    
