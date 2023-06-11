# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:56:55 2017

@author: leehees
"""
import pandas as pd
import numpy as np

def fratio(featMat, classLabel):
    clb = np.sort(classLabel.unique())
    classanz=len(clb)   # Anzahl der Klassen
    #unter=1                # Untergrenze intialisieren
    mju_i=pd.DataFrame()               # Mittelwertmatrix initialisieren
    var_i=pd.DataFrame()               # Streumatrix initialisieren

    for ii in clb:
        #ober=sum(classLabel[0:ii])          # Obergrenze berechnen
        teilmat=featMat.loc[classLabel==ii,:]        # Aktuelle Klasse
        
        #mju_i(ii,:)=np.mean(teilmat)       # Mittelwerte der Klasse pro Merkmal
        mju_i = mju_i.append(teilmat.mean(), ignore_index = True) # ignoring index is 
        
        
        var_i=var_i.append(pd.Series(np.diag(teilmat.cov()), index=teilmat.columns),ignore_index=True)
        #diag(cov(teilmat)) # Varianzen der Merkmale innerhalb 
                                        # der aktuellen Klasse
        #unter=ober+1;                    # Untergrenze neu berechnen

    # Spaltenvektor mit den Varianzen der Mittelwerte pro Merkmal
    num=pd.Series(np.diag(mju_i.cov()), index=mju_i.columns)

    # Spaltenvektor der Mittelwerte der Varianzen pro Merkmal
    den=var_i.mean()

    # F ratio berechnen
    F=num / den
    return F

def sepblty2(featMat, classLabel):
    [must, merk]  = featMat.shape
    #[must,merk]=size(featMat)   # Anzahl Merkmale
    #classanz=length(classLabel)   # Anzahl Klassen
    clb = np.sort((classLabel).unique())
    classanz=len(clb)   # Anzahl der Klassen
    #unter=1                # Untergrenze initialisieren
    mju_i=pd.DataFrame()               # Mittelwertmatrix initialisieren
    cov_i=pd.DataFrame(0, index=featMat.columns, columns=featMat.columns)
    
    # Kovarianzmatrizen der Klassen
    #==============================
    for ii in clb:
        #ober=sum(anz(1:ii))         # Obergrenze Berechnen
        teilmat=featMat.loc[classLabel==ii,:]        # Aktuelle Klasse
        #print(teilmat)
        #print(teilmat.cov())
        
        #teilmat=featMat(unter:ober,:)    # Matrix auf Klassengroesse reduzieren
        #mju_i(ii,:) = mean(teilmat) # Mittelwerte der Klasse pro Merkmal
        #print(mju_i)
        #print(teilmat.mean())
        #print(mju_i)
        mju_i = mju_i.append(teilmat.mean(), ignore_index = True) # ignoring index is optional
        cov_i=cov_i+teilmat.cov()    # Kovarianz + andere Kovarianzen
        #unter=ober+1                # Untergrenze neu berechnen
    #print(cov_i)
    # Intra- und Interkovarianzmatrix
    #================================
    W=cov_i*(1.0/classanz) # Mittlere Kovarianzmatrix aller Klassen
    B=mju_i.cov()       # Kovarianzmatrix der Klassenmittelwerte
    # Trennungswirksamkeit
    #=====================
    F= np.trace(B)/ float(np.trace(W))
        
    return F,W,B

def addon(featMat, abbr=5):# V

    #Erstes Merkmal waehlen:
    #======================
    [must, merk]=(featMat.iloc[:,:-1]).shape            # Anzahl Merkmale bzw. Muster
    
    #print("nMuster:",must," nMerkmale:",merk)
    fratioList = fratio(featMat.iloc[:,:-1], featMat["label"])
    x = fratioList.max()
    maxi = fratioList.idxmax()
    #print("First:",x," ",maxi)
    #maxi = np.ndarray.argmax(np.array(fratioList))
    #[x, maxi]=max()      # Trennungswirksamkeit der Einzelmerkmale
    r = []
    w = []
    r.extend([x])
    w.extend([maxi])
    #r(count)=x                        # Spurkriterium = max. Fratio
    #w(count)=maxi                    # Zugehoerige Merkmalsnummer an 1. Stelle
    #verbleib=1:merk
    verbleib = list((featMat.iloc[:,:-1]).columns)
    verbleib= [x for x in verbleib if x != maxi] # Alle uebrigen Merkmale
    #print("Verbleibende Merkmale:",verbleib)
    #Pro Runde ein Merkmal mehr pruefen
    #=================================

    for dummy in range(1,abbr):                  # Laufindex Kombinationen
        xx= []
        x=0
        #inc=0
        featList = []
        
        for ii in verbleib:            # Kombination mit je einem anderen,
                                       # verbliebenen Merkmal
            featCh = list(w)
            featCh.extend([ii])
            
            teilMat = (featMat.loc[:,featCh])
            '''
            if typ==1:
                (xx1,w2,b2) = sepblty1(teilMat, featMat["label"])
            elif typ==2:
                (xx1,w2,b2) = sepblty2(teilMat, featMat["label"])
            '''
            #inc=inc+1
            #print(teilMat)
            (xx1,w2,b2) = sepblty2(teilMat, featMat["label"])
            featList.extend([ii])
            xx.extend([xx1])         # Vektor der Spurkriterien

        x = max(xx)
        #print(xx)
        maxIdx = np.ndarray.argmax(np.array(xx))
        maxi = featList[maxIdx]

        #[x, maxi]=max(xx)             # Groesstes Spurkrit. an Stelle 'maxi'
        r.extend([x])                  # Spurkriteriumsverlauf ergaenzen
        #w(count)=verbleib(maxi)       # Reihenfolge ergaenzen
        
        w.extend([maxi])
        verbleib=[x for x in verbleib if x != maxi] # Alle uebrigen Merkmale

        #verbleib(verbleib!=verbleib(maxi))
                                       # Restvektor neu definieren
    return r,w