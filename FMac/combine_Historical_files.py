
# coding: utf-8

# In[1]:


import requests
import re
import os
import time
import datetime
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import seaborn as sns


# In[2]:


def fillNAN(df):
    df['fico'] = df['fico'].fillna(9999)
    df['flag_fthb']=df['flag_fthb'].fillna('X')
    df['cd_msa']=df['cd_msa'].fillna(0)
    df['mi_pct']=df['mi_pct'].fillna(999)
    df['cnt_units']=df['cnt_units'].fillna(99)
    df['occpy_sts']=df['occpy_sts'].fillna('X')
    df['cltv']=df['cltv'].fillna(999)
    df['dti']=df['dti'].fillna(999)
    df['ltv']=df['ltv'].fillna(999)
    df['channel']=df['channel'].fillna('X')
    df['ppmt_pnlty']=df['ppmt_pnlty'].fillna('X')
    df['prod_type']= df['prod_type'].fillna('X')
    df['prop_type']=df['prop_type'].fillna('XX')
    df['zipcode']=df['zipcode'].fillna('999999')
    df['loan_purpose']=df['loan_purpose'].fillna('X')
    df['cnt_borr']=df['cnt_borr'].fillna('99')
    df['flag_sc']=df['flag_sc'].fillna('X')
    
    return df


# In[3]:


def changedatatype(df):
    #Change the data types for all column
    df[['fico','cd_msa','mi_pct','cnt_units','cltv','dti','orig_upb','ltv','orig_loan_term']] = df[['fico','cd_msa','mi_pct','cnt_units','cltv','dti','orig_upb','ltv','orig_loan_term']].astype('float64')
    df[['flag_sc','flag_fthb','cnt_borr','occpy_sts','channel','ppmt_pnlty','zipcode','servicer_name','id_loan','loan_purpose','seller_name']] = df[['flag_sc','flag_fthb','cnt_borr','occpy_sts','channel','ppmt_pnlty','zipcode','servicer_name','id_loan','loan_purpose','seller_name']].astype('str')
    return df


# In[4]:


def fillNA(df):
    df['delq_sts'] = df['delq_sts'].fillna('XX')
    df['loan_age'] = df['loan_age'].fillna(999)
    df['mths_remng'] = df['mths_remng'].fillna('XX')
    df['repch_flag']=df['repch_flag'].fillna('X')
    df['flag_mod']=df['flag_mod'].fillna('X')
    df['cd_zero_bal']=df['cd_zero_bal'].fillna('00')
    df['dt_zero_bal']=df['dt_zero_bal'].fillna('189901')
    df['non_int_brng_upb']=df['non_int_brng_upb'].fillna(0)
    df['dt_lst_pi']=df['dt_lst_pi'].fillna('189901')
    df['mi_recoveries']=df['mi_recoveries'].fillna(0)
    df['net_sale_proceeds']=df['net_sale_proceeds'].fillna('0')
    df['non_mi_recoveries']=df['non_mi_recoveries'].fillna(0)
    df['expenses']=df['expenses'].fillna(0)
    df['legal_costs']=df['legal_costs'].fillna(0)
    df['maint_pres_costs']=df['maint_pres_costs'].fillna(0)
    df['taxes_ins_costs']=df['taxes_ins_costs'].fillna(0)
    df['misc_costs']=df['misc_costs'].fillna(0)
    df['actual_loss']=df['actual_loss'].fillna(0)
    df['modcost']=df['modcost'].fillna(0)
    df['stepmod_ind']=df['stepmod_ind'].fillna('X')
    return df


# In[5]:


def changedtype(df):
    #Change the data types for all column
    df[['current_upb','loan_age','mths_remng','current_int_rt','non_int_brng_upb','mi_recoveries','net_sale_proceeds','non_mi_recoveries','expenses', 'legal_costs',
    'maint_pres_costs','taxes_ins_costs','misc_costs','actual_loss', 'modcost']] = df[['current_upb','loan_age','mths_remng','current_int_rt','non_int_brng_upb','mi_recoveries','net_sale_proceeds','non_mi_recoveries','expenses', 'legal_costs',
    'maint_pres_costs','taxes_ins_costs','misc_costs','actual_loss', 'modcost']].astype('float64')
    df[['id_loan','svcg_cycle','delq_sts','repch_flag','flag_mod', 'cd_zero_bal']] = df[['id_loan','svcg_cycle','delq_sts','repch_flag','flag_mod', 'cd_zero_bal']].astype('str')
    return df


# In[6]:


def chnge_code_zero(x):
    if x=='00':
        return 'C'
    elif x=='01':
        return 'P'
    elif x=='06':
        return 'R'
    elif x=='03':
        return 'S'
    elif x=='09':
        return 'F'
def chnge_delinquecy(x):
    if x=='0':
        return 'C'
    elif x not in list(map(str,list(range(1,9))))+['R']:
        return '9+'
    else:
        return x


# In[7]:


def orig_summary_statistics(x):
    names = {
        'loan_count': x['id_loan'].count(),
        'total_orig_UPB ($B)':  x['orig_upb'].sum()/1000000000,
        'average_orig_UPB':  x['orig_upb'].mean(),
        'weighted_average_fico':  (x.loc[x.fico!=9999,'orig_upb'] * x.loc[x.fico!=9999,'fico']).sum()/x.loc[x.fico!=9999,'orig_upb'].sum(),
        'weighted_average_cltv':  (x.loc[x.cltv!=999,'orig_upb'] * x.loc[x.cltv!=999,'cltv']).sum()/x.loc[x.cltv!=999,'orig_upb'].sum(),
        'weighted_average_ltv':  (x.loc[x.ltv!=999,'orig_upb'] * x.loc[x.ltv!=999,'ltv']).sum()/x.loc[x.ltv!=999,'orig_upb'].sum(),
        'weighted_average_dti':  (x.loc[x.dti!=999,'orig_upb'] * x.loc[x.dti!=999,'dti']).sum()/x.loc[x.dti!=999,'orig_upb'].sum(),
        }

    return pd.Series(names, index=['loan_count', 'total_orig_UPB ($B)', 'average_orig_UPB',
                                   'weighted_average_fico', 'weighted_average_cltv', 'weighted_average_ltv','weighted_average_dti'])


# In[ ]:


def performance_summary_statistics(x):
    names = {
        'loan_count': x['id_loan'].count(),
        'prepay_loan(%)':  x.loc[x.cd_zero_bal.isin(['P','R']),'id_loan'].count()*100/x['id_loan'].count(),
        'default_loan(%)' :  x.loc[x.default==1,'id_loan'].count()*100/x['id_loan'].count(),
        'active_loan(%)':  x.loc[x.cd_zero_bal=='C','id_loan'].count()*100/x['id_loan'].count(),
        'cumulative_post‐default_event_repurchase(%)':  x.loc[(x.cd_zero_bal.isin(['S','F']))&(x.repch_flag=='Y'),'id_loan'].count()*100/x['id_loan'].count(),
        'ever_D180(%)':  x.loc[(x.delq_sts_180_upb!=0)|(x.delq_sts=='R'),'id_loan'].count()*100/x['id_loan'].count(),
        'D180_and_pre‐D180_credit_event(%)':  x.loc[(x.delq_sts_180_upb!=0)|(x.delq_sts=='R')|(x.default==1),'id_loan'].count()*100/x['id_loan'].count(),
        'cumulative_modification(%)' : x.loc[x.flag_mod=='Y','id_loan'].count()*100/x['id_loan'].count(),
        'actuall_loss($M)':  -x.actual_loss.sum()/1000000,
        'total_orig_UPB($B)':  x.orig_upb.sum()/1000000000,
        'prepaid_upb(%)':  x.prepaid_upb.sum()*100/x.orig_upb.sum(),
        'defaulted_upb(%)': x.defaulted_upb.sum()*100/x.orig_upb.sum(),
        'current_upb(%)':  x.current_upb.sum()*100/x.orig_upb.sum(),
        'post-defaulted_upb(%)':  x.loc[(com_df.default==1)&(x.repch_flag=='Y'),'defaulted_upb'].sum()*100/x.orig_upb.sum(),
        'original_UPB_ever_D180(%)':  (x.delq_sts_180_upb.sum()+x.loc[(x.cd_zero_bal=='F'),'defaulted_upb'].sum()+x.loc[(x.delq_sts=='R')&(x.delq_sts_180_upb==0),'current_upb'].sum())*100/x.orig_upb.sum(),
        'original_UPB_D180_and_pre_D180_credit_event(%)':  (x.delq_sts_180_upb.sum()+x.loc[(x.default==1),'defaulted_upb'].sum()+x.loc[(x.delq_sts=='R')&(x.delq_sts_180_upb==0),'current_upb'].sum())*100/x.orig_upb.sum(),
        'UPB_weighted_cumulative_modification(%)':  x.loc[x.flag_mod=='Y','current_upb'].sum()*100/x.orig_upb.sum()
    }

    return pd.Series(names, index=['loan_count', 'prepay_loan(%)', 'default_loan(%)',
                                   'active_loan(%)', 'cumulative_post‐default_event_repurchase(%)', 'ever_D180(%)'
                                   ,'D180_and_pre‐D180_credit_event(%)','actuall_loss($M)','total_orig_UPB($B)',
                                  'prepaid_upb(%)','defaulted_upb(%)','current_upb(%)','post-defaulted_upb(%)','original_UPB_ever_D180(%)'
                                  ,'original_UPB_D180_and_pre_D180_credit_event(%)','UPB_weighted_cumulative_modification(%)'])


# In[8]:


def createOriginationCombined(str):
    #print(str)
    writeHeader1 = True
    if "sample" in str:
        filename= "SampleOriginationCombined.csv"
    else:
        filename= "HistoricalOriginationCombined.csv"
    
    abc = tqdm(glob.glob(str))
      
    with open(filename, 'w',encoding='utf-8',newline="") as file:
        for f in abc: 
            abc.set_description("Working on  {}".format(f.split('\\')[-1]))
            sample_df = pd.read_csv(f ,sep="|", names=['fico','dt_first_pi','flag_fthb','dt_matr','cd_msa',"mi_pct",'cnt_units','occpy_sts','cltv','dti','orig_upb','ltv','int_rt','channel','ppmt_pnlty','prod_type','st', 'prop_type','zipcode','id_loan','loan_purpose', 'orig_loan_term','cnt_borr','seller_name','servicer_name','flag_sc'],skipinitialspace=True,dtype='unicode') 
            sample_df.flag_fthb[sample_df.flag_fthb=='9'] = 'X'
            sample_df = fillNAN(sample_df)
            sample_df = changedatatype(sample_df)
            sample_df.dt_first_pi = pd.to_datetime(sample_df.dt_first_pi.apply(lambda x: x[:4] +'/'+x[4:]))
            sample_df.dt_matr = pd.to_datetime(sample_df.dt_matr.apply(lambda x: x[:4] +'/'+x[4:]))
            sample_df['fico_bins'] = pd.cut(sample_df.fico,[0,600,680,720,760,780,850,900,9999],include_lowest=True)
            sample_df['cltv_bins'] = pd.cut(sample_df.cltv,[0,6,50,70,80,90,110,150,200,999],include_lowest=True)
            sample_df['dti_bins'] = pd.cut(sample_df.dti,[0,27,36,46,65,999],include_lowest=True)
            sample_df['ltv_bins'] = pd.cut(sample_df.ltv,[6,50,70,80,90,105,999],include_lowest=True)
            sample_df['mi_pct_bins'] = pd.cut(sample_df.mi_pct,[0,20,30,40,55,999],include_lowest=True)

            
            sample_df['Year'] = ['19'+x if x=='99' else '20'+x for x in (sample_df['id_loan'].apply(lambda x: x[2:4]))]
            sample_df.to_csv(file, mode='a', header=writeHeader1,index=False,encoding='utf-8')
            writeHeader1=False
    return filename


# In[9]:


def createPerformanceCombined(str): 
#     print(str)
    writeHeader2 = True
    if "sample" in str:
        filename= "SamplePerformanceCombinedSummary.csv"
    else:
        filename= "HistoricalPerformanceCombinedSummary.csv"
    
    abc = tqdm(glob.glob(str))
 
    with open(filename, 'w',encoding='utf-8',newline="") as file:
        for f in abc: 
            abc.set_description("Working on  {}".format(f.split('\\')[-1]))
            perf_df = pd.read_csv(f,sep="|",header=None,skipinitialspace=True,dtype='unicode')
            perf_df.columns =['id_loan','svcg_cycle','current_upb','delq_sts','loan_age','mths_remng', 'repch_flag',
                                          'flag_mod','cd_zero_bal', 'dt_zero_bal','current_int_rt','non_int_brng_upb','dt_lst_pi',
                                          'mi_recoveries','net_sale_proceeds','non_mi_recoveries','expenses', 'legal_costs', 
                                          'maint_pres_costs','taxes_ins_costs','misc_costs','actual_loss', 'modcost','stepmod_ind']
            #             perf_df['delq_sts'] = [ 999 if x=='R' else x for x in (perf_df['delq_sts'].apply(lambda x: x))]
            #             perf_df['delq_sts'] = [ 0 if x=='XX' else x for x in (perf_df['delq_sts'].apply(lambda x: x))]
            perf_df.loc[(perf_df.net_sale_proceeds=='U')|(perf_df.net_sale_proceeds=='C'),'net_sale_proceeds'] = '0'
            #             perf_df['net_sale_proceeds'] = [ '0.0' if x=='C' else x for x in (perf_df['net_sale_proceeds'].apply(lambda x: x))]

            #             perf_df.cd_zero_bal = perf_df.cd_zero_bal.apply(lambda x : chnge_code_zero(x))

            perf_df = fillNA(perf_df)
            perf_df = changedtype(perf_df)
            perf_df.cd_zero_bal = perf_df.cd_zero_bal.apply(lambda x : chnge_code_zero(x))

            ve =perf_df.drop(perf_df[(perf_df.cd_zero_bal=='S')|(perf_df.cd_zero_bal=='F')].index)
            h = ve.groupby(by='id_loan').last().reset_index()
            defauled_upb = h.loc[h.id_loan.isin(perf_df[(perf_df.cd_zero_bal=='S')|(perf_df.cd_zero_bal=='F')].id_loan.values),['id_loan','current_upb']]

            ve1 =perf_df.drop(perf_df[(perf_df.cd_zero_bal=='P')|(perf_df.cd_zero_bal=='R')].index)
            h1 = ve1.groupby(by='id_loan').last().reset_index()
            prepaid_upb = h1.loc[h1.id_loan.isin(perf_df[(perf_df.cd_zero_bal=='P')|(perf_df.cd_zero_bal=='R')].id_loan.values),['id_loan','current_upb']]

            lpi = perf_df.loc[(perf_df.delq_sts=='0'),['id_loan','svcg_cycle']]
            lpi =lpi.groupby(by='id_loan').last().reset_index()
            delq_sts_180 = perf_df.loc[(perf_df.delq_sts=='6'),['id_loan','svcg_cycle','current_upb']]
            delq_sts_180 = delq_sts_180.groupby(by='id_loan').last().reset_index()

            perf_df = perf_df.groupby(by='id_loan').last().reset_index()

            perf_df['delq_sts_180_date'] = '189901'
            perf_df['last_payment_date'] = '189901'
            perf_df['delq_sts_180_upb'] = 0
            perf_df['defaulted_upb'] = 0
            perf_df['prepaid_upb'] = 0
            perf_df.loc[perf_df.id_loan.isin(delq_sts_180.id_loan.values),'delq_sts_180_date'] = delq_sts_180.svcg_cycle.values
            perf_df.loc[perf_df.id_loan.isin(delq_sts_180.id_loan.values),'delq_sts_180_upb'] = delq_sts_180.current_upb.values
            perf_df.loc[perf_df.id_loan.isin(lpi.id_loan.values),'last_payment_date'] = lpi.svcg_cycle.values
            perf_df.loc[perf_df.id_loan.isin(defauled_upb.id_loan.values),'defaulted_upb'] = defauled_upb.current_upb.values
            perf_df.loc[perf_df.id_loan.isin(prepaid_upb.id_loan.values),'prepaid_upb'] = prepaid_upb.current_upb.values

            #             perf_df = perf_df.drop(perf_df[perf_df.cd_zero_bal=='06'].index)
            #             perf_df = perf_df.drop('repch_flag',axis=1)

            perf_df.dt_lst_pi = pd.to_datetime(perf_df.dt_lst_pi.astype('str').apply(lambda x: x[:4] +'/'+x[4:]))
            perf_df.dt_zero_bal = pd.to_datetime(perf_df.dt_zero_bal.astype('str').apply(lambda x: x[:4] +'/'+x[4:]))
            perf_df.delq_sts_180_date = pd.to_datetime(perf_df.delq_sts_180_date.astype('str').apply(lambda x: x[:4] +'/'+x[4:]))
            perf_df.last_payment_date = pd.to_datetime(perf_df.last_payment_date.astype('str').apply(lambda x: x[:4] +'/'+x[4:]))

            perf_df['GT_90_days_delinquecy'] = perf_df.delq_sts.values
            perf_df['GT_90_days_delinquecy'] = perf_df['GT_90_days_delinquecy'].apply(lambda x: 0 if (x=='0') | (x=='1') | (x=='2')|(x=='XX')  else 1)

            perf_df['default'] = perf_df.cd_zero_bal.values
            perf_df['default'] = perf_df['default'].apply(lambda x: 1 if (x=='S') | (x=='F') else 0)

            perf_df['prepayment']=0
            perf_df.loc[(perf_df.cd_zero_bal=='P')&(perf_df.mths_remng!=0),'prepayment'] = 1 
            #             de = perf_df[perf_df.default==1]
            #             months_delinquecny = (pd.to_datetime(de.dt_zero_bal.values).year - pd.to_datetime(de.last_payment_date.values).year)*12 + (pd.to_datetime(de.dt_zero_bal.values).month - pd.to_datetime(de.last_payment_date.values).month)

            perf_df['lpi2zero'] = 0
            perf_df['delinquent_interest'] = 0
            perf_df['net_loss'] = perf_df.actual_loss.copy()
            perf_df['loss_severity'] = 0

            c = perf_df[(perf_df.dt_lst_pi!='1899-01-01')&(perf_df.dt_zero_bal!='1899-01-01')]
            if c.shape[0]>0:
                o = (pd.to_datetime(c.dt_zero_bal.values).year - pd.to_datetime(c.dt_lst_pi.values).year)*12 + (pd.to_datetime(c.dt_zero_bal.values).month - pd.to_datetime(c.dt_lst_pi.values).month)

                perf_df.loc[(perf_df.dt_lst_pi!='1899-01-01')&(perf_df.dt_zero_bal!='1899-01-01'),'lpi2zero'] = o

                de_i = perf_df.loc[(perf_df.lpi2zero!=0)&(perf_df.default==1)] 
                perf_df.loc[(perf_df.lpi2zero!=0)&(perf_df.default==1),'delinquent_interest'] = (de_i.lpi2zero) * (de_i.defaulted_upb - de_i.non_int_brng_upb) * (de_i.current_int_rt - 0.35) / 1200

            perf_df['total_proceeds'] = perf_df[['mi_recoveries','net_sale_proceeds', 'non_mi_recoveries']].sum(axis=1)

            if perf_df[perf_df.actual_loss!=0].shape[0]>0:
                perf_df.loc[(perf_df.actual_loss!=0),'net_loss'] = perf_df.loc[(perf_df.actual_loss!=0),['actual_loss','modcost']].T.apply(lambda x: x[0]-x[1])

            perf_df.loc[perf_df.net_loss!=0,'loss_severity'] = perf_df.loc[perf_df.net_loss!=0,['defaulted_upb','net_loss']].T.apply(lambda x: x[1]/x[0])

            perf_df.delq_sts = perf_df.delq_sts.apply(lambda x : chnge_delinquecy(x))
            perf_df.to_csv(file, mode='a', header=writeHeader2,index=False,encoding='utf-8')
            writeHeader2=False
    return filename


# In[15]:


def main():
    ts = time.time()
    foldername1= 'SampleInputFiles'
    foldername2= 'HistoricalInputFiles' 
    
    sampleOrigFiles=str(os.getcwd())+"/"+foldername1+"/sample_orig_*.txt"
    samplePerfFiles=str(os.getcwd())+"/"+foldername1+"/sample_svcg_*.txt"
    
    historical_Files=str(os.getcwd())+"/"+foldername2+"/historical_data1_Q*.txt"
    historical_timeFiles=str(os.getcwd())+"/"+foldername2+"/historical_data1_time_Q*.txt"
    
    
    orig1_file = createOriginationCombined(sampleOrigFiles)
    per1_file = createPerformanceCombined(samplePerfFiles)
    
    orig1_df = pd.read_csv(orig1_file)
    per1_df = pd.read_csv(per1_file,dtype={'delq_sts':'str'})
    combined_df = orig1_df.merge(per1_df,on='id_loan')
    combined_df.to_csv('combined_SF_smaple_data.csv', index=False)
    
    com1_df = pd.read_csv('combined_SF_smaple_data.csv')

    orig_summary_statistic1 = com1_df.groupby("Year").apply(orig_summary_statistics).round(1)
    performance_summary_statistic1 = com1_df.groupby("Year").apply(performance_summary_statistics).round(1)
   
    orig_summary_statistic1.to_csv("sample_SF_orig_summary_Statistics.csv",index=False)
    performance_summary_statistic1.to_csv("sample_SF_performance_summary_Statistics.csv",index=False)
    
#     orig2_file = createOriginationCombined(historical_Files)
#     per2_file = createPerformanceCombined(historical_timeFiles)

#     orig2_df = pd.read_csv(orig2_file)
#     per2_df = pd.read_csv(per2_file,dtype={'delq_sts':'str'})
    
#     combined2_df = orig2_df.merge(per2_df,on='id_loan')
#     combined2_df.to_csv('combined_SF_historical_all_data.csv', encoding='utf-8', index=False)
    
#     com2_df = pd.read_csv('combined_SF_historical_all_data.csv')
    
#     orig_summary_statistic2 = com2_df.groupby("Year").apply(orig_summary_statistics).round(1)
#     performance_summary_statistic2 = com2_df.groupby("Year").apply(performance_summary_statistics).round(1)

#     orig_summary_statistic2.to_csv("full_SF_orig_summary_Statistics.csv",index=False)
#     performance_summary_statistic2.to_csv("full_SF_performance_summary_Statistics.csv",index=False)


# In[16]:


if __name__ == '__main__':
    main()

