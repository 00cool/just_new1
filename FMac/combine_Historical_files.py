
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
    df['net_sale_proceeds']=df['net_sale_proceeds'].fillna('-999.0')
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


# In[7]:


def createOriginationCombined(str):
    #print(str)
    writeHeader1 = True
    if "sample" in str:
        filename= "SampleOriginationCombined_1.csv"
    else:
        filename= "HistoricalOriginationCombined_1.csv"
    
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
            sample_df['mi_pct'] = pd.cut(sample_df.mi_pct,[1,20,30,40,55,999],include_lowest=True)

            
            sample_df['Year'] = ['19'+x if x=='99' else '20'+x for x in (sample_df['id_loan'].apply(lambda x: x[2:4]))]
            sample_df.to_csv(file, mode='a', header=writeHeader1,index=False,encoding='utf-8')
            writeHeader1=False
    return sample_df


# In[8]:


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
            perf_df.loc[(perf_df.net_sale_proceeds=='U')|(perf_df.net_sale_proceeds=='C'),'net_sale_proceeds'] = '0.0'
#             perf_df['net_sale_proceeds'] = [ '0.0' if x=='C' else x for x in (perf_df['net_sale_proceeds'].apply(lambda x: x))]

            perf_df = fillNA(perf_df)
            perf_df = changedtype(perf_df)

            ve =perf_df.drop(perf_df[(perf_df.cd_zero_bal=='03')|(perf_df.cd_zero_bal=='09')].index)
            h = ve.groupby(by='id_loan').last().reset_index()
            defauled_upb = h.loc[h.id_loan.isin(perf_df[(perf_df.cd_zero_bal=='03')|(perf_df.cd_zero_bal=='09')].id_loan.values),['id_loan','current_upb']]

            ve1 =perf_df.drop(perf_df[(perf_df.cd_zero_bal=='01')].index)
            h1 = ve1.groupby(by='id_loan').last().reset_index()
            prepaid_upb = h1.loc[h1.id_loan.isin(perf_df[(perf_df.cd_zero_bal=='01')].id_loan.values),['id_loan','current_upb']]

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

            perf_df['GT_90_days_deliquecy'] = perf_df.delq_sts.values
            perf_df['GT_90_days_deliquecy'] = perf_df['GT_90_days_deliquecy'].apply(lambda x: 0 if (x=='0') | (x=='1') | (x=='2')|(x=='XX')  else 1)

            perf_df['default'] = perf_df.cd_zero_bal.values
            perf_df['default'] = perf_df['default'].apply(lambda x: 1 if (x=='03') | (x=='09') else 0)

            perf_df['prepayment']=0
            perf_df.loc[(perf_df.cd_zero_bal=='01')&(perf_df.mths_remng!=0),'prepayment'] = 1 

            de = perf_df[perf_df.default==1]
            months_deliquecny = (pd.to_datetime(de.dt_zero_bal.values).year - pd.to_datetime(de.last_payment_date.values).year)*12 + (pd.to_datetime(de.dt_zero_bal.values).month - pd.to_datetime(de.last_payment_date.values).month)

            c = perf_df[(perf_df.dt_lst_pi!='1899-01-01')&(perf_df.dt_zero_bal!='1899-01-01')]
            o = (pd.to_datetime(c.dt_zero_bal.values).year - pd.to_datetime(c.last_payment_date.values).year)*12 + (pd.to_datetime(c.dt_zero_bal.values).month - pd.to_datetime(c.last_payment_date.values).month)

            perf_df['lpi2zero'] = 0
            perf_df.loc[(perf_df.dt_lst_pi!='1899-01-01')&(perf_df.dt_zero_bal!='1899-01-01'),'lpi2zero'] = o

            de_i = perf_df.loc[(perf_df.lpi2zero!=0)&(perf_df.default==1)] 

            perf_df['deliquent_interest'] = 0
            perf_df.loc[(perf_df.lpi2zero!=0)&(perf_df.default==1),'deliquent_interest'] = (de_i.lpi2zero) * (de_i.defaulted_upb - de_i.non_int_brng_upb) * (de_i.current_int_rt - 0.35) / 1200
            
            perf_df['total_costs'] = perf_df[['legal_costs','maint_pres_costs', 'taxes_ins_costs',
                                  'misc_costs']].sum(axis=1)

            perf_df['total_proceeds'] = perf_df[['mi_recoveries','net_sale_proceeds', 'non_mi_recoveries', 'expenses']].sum(axis=1)
            perf_df['net_loss'] = perf_df['actual_loss']-perf_df['total_costs']

            perf_df['loss_severity'] = 0

            perf_df.loc[perf_df.net_loss!=0,'loss_severity'] = perf_df.loc[perf_df.net_loss!=0,['defaulted_upb','net_loss']].T.apply(lambda x: x[1]/x[0])
            
            perf_df.to_csv(file, mode='a', header=writeHeader2,index=False,encoding='utf-8')
            writeHeader2=False
    return perf_df


# In[9]:


def main():
    ts = time.time()
    foldername1= 'SampleInputFiles'
    foldername2= 'HistoricalInputFiles' 
    
    sampleOrigFiles=str(os.getcwd())+"/"+foldername1+"/sample_orig_*.txt"
    samplePerfFiles=str(os.getcwd())+"/"+foldername1+"/sample_svcg_*.txt"
    
    historical_Files=str(os.getcwd())+"/"+foldername2+"/historical_data1_*.txt"
    historical_timeFiles=str(os.getcwd())+"/"+foldername2+"/historical_data1_*.txt"
    
    
    orig1_df = createOriginationCombined(sampleOrigFiles)
    per1_df = createPerformanceCombined(samplePerfFiles)
    
    combined_sample_df = orig1_df.join(per1_df,on='id_loan',lsuffix='_')
    combined_sample_df.drop('id_loan_',axis=1,inplace=True)
    combined_sample_df.to_csv('combined_SF_smaple_data.csv', encoding='utf-8', index=False)
    
#     orig2_df = createOriginationCombined(historical_Files)
#     per2_df = createPerformanceCombined(historical_timeFiles)
    
#     combined2_df = orig_df.merge(per_df,on='id_loan')
#     combined_df.to_csv('combined_SF_historical_all_data.csv', encoding='utf-8', index=False)


# In[244]:


if __name__ == '__main__':
    main()

