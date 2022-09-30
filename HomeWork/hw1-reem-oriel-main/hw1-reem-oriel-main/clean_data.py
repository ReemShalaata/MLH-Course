# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer 

def rm_ext_and_nan(CTG_features, extra_feature):
    """
    
    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    c_ctg=CTG_features.copy()
    c_ctg.drop(columns=[extra_feature],axis=1)
    c_ctg=c_ctg.apply(pd.to_numeric, errors='coerce')
    c_ctg.dropna(axis=0,inplace=True)
    c_ctg.to_dict()
    return c_ctg

def rand_sampling(col_ser,feature_no_nan):

  col_ser=col_ser.replace(np.nan,feature_non_nan[np.random.choice(len(feature_no_nan))])
  return col_ser

def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas dataframe of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe c_cdf containing the "clean" features
    """
    df_nan=CTG_features.copy()
    df_nan.drop(columns=[extra_feature],inplace=True)
    df_nan=df_nan.apply(pd.to_numeric, errors='coerce')
    # for feature in c_cdf.columns:
    #   feature_no_nan=c_cdf[feature].dropna(axis=0)
    #   c_cdf=c_cdf.apply(lambda col_ser: rand_sampling(col_ser,feature_no_nan),axis=1)
    imputer = SimpleImputer(strategy="median") # method 2, mostly preferred due to it's generalized form
    p = imputer.fit(df_nan)
    X = imputer.transform(df_nan)  # Could also simply skip the above line and use X = imputer.fit_transform(df_nan) instead
    c_cdf = pd.DataFrame(X, columns=df_nan.columns)
    return c_cdf


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_samp
    :return: Summary statistics as a dictionary of dictionaries (called d_summary) as explained in the notebook
    """
    d_summary={}
    dict_feature={}
    for feature in c_feat.columns:
      dict_feature={}
      dict_feature["min"]=c_feat[feature].min()
      dict_feature["Q1"]=c_feat[feature].quantile(0.25)
      dict_feature["median"] =c_feat[feature].median()
      dict_feature["Q3"]=c_feat[feature].quantile(0.75)
      dict_feature["max"]=c_feat[feature].max()
      d_summary[feature]=dict_feature
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_samp
    :param d_summary: Output of sum_stat
    :return: Dataframe containing c_feat with outliers removed
    """

    c_no_outlier = c_feat.copy()
    for feature in c_feat.columns:
      minimum=d_summary[feature]["Q1"]-1.5*(d_summary[feature]["Q3"]-d_summary[feature]["Q1"])
      maximum=d_summary[feature]["Q3"]+1.5*(d_summary[feature]["Q3"]-d_summary[feature]["Q1"])
      c_no_outlier[feature]=c_no_outlier.loc[(c_no_outlier[feature] <=maximum) & (c_no_outlier[feature]>=minimum),feature]
      
    return c_no_outlier


def phys_prior(c_samp, feature, thresh):
    """

    :param c_samp: Output of nan2num_samp
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    mask=(c_samp[feature]<=thresh)
    x=c_samp[feature]
    filt_feature=x[mask]

    return np.array(filt_feature)


class NSD:

    def __init__(self):
        self.max = np.nan
        self.min = np.nan
        self.mean = np.nan
        self.std = np.nan
        self.fit_called = False
    
    def fit(self, CTG_features):
        self.max = CTG_features.apply(np.max)
        self.min =CTG_features.apply(np.min)
        self.mean =CTG_features.apply(np.mean)
        self.std = CTG_features.apply(np.std)   
        
        self.fit_called = True

    def transform(self, CTG_features, mode='none', selected_feat=('LB', 'ASTV'), flag=False):
        """
        Note: x_lbl should only be either: 'Original values [N.U]', 'Standardized values [N.U.]', 'Normalized values [N.U.]' or 'Mean normalized values [N.U.]'
        :param mode: A string determining the mode according to the notebook
        :param selected_feat: A two elements tuple of strings of the features for comparison
        :param flag: A boolean determining whether or not plot a histogram
        :return: Dataframe of the normalized/standardized features called nsd_res
        """
        ctg_features = CTG_features.copy()
        if self.fit_called:
            if mode == 'none':
                nsd_res = ctg_features
                x_lbl = 'Original values [N.U]'
            elif mode=='standard':
                nsd_res = (ctg_features-self.mean)/self.std
                x_lbl = 'Standardized values [N.U.]'
            elif mode=='MinMax':
                nsd_res = (ctg_features-self.min)/(self.max-self.min)
                x_lbl =  'Normalized values [N.U.]'
            elif mode=='mean':
                nsd_res = (ctg_features-self.mean)/(self.max-self.min)
                x_lbl = 'Mean normalized values [N.U.]'           
            if flag:
                self.plot_hist(nsd_res, mode, selected_feat, x_lbl)
            return nsd_res
        else:
            raise Exception('Object must be fitted first!')

    def fit_transform(self, CTG_features, mode='none', selected_feat=('LB', 'ASTV'), flag=False):
        self.fit(CTG_features)
        return self.transform(CTG_features, mode=mode, selected_feat=selected_feat, flag=flag)

    def plot_hist(self, nsd_res, mode, selected_feat, x_lbl):
        x, y = selected_feat
        if mode == 'none':
            bins = 50
        else:
            bins = 80  

      
        axarr = nsd_res.hist(column=[x,y] ,bins=bins,xlabelsize=10,ylabelsize=10, figsize=(15,5))  
        for idx, ax in enumerate(axarr.flatten()):
            ax.set_xlabel(x_lbl)
            ax.set_ylabel("Count")  

        plt.show()