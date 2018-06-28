
# coding: utf-8

# In[1]:


# Prgramitically Log in to Freddie Mac Webisite and download all the files based on request
import requests
import re
import os
from bs4 import BeautifulSoup
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import time
import datetime
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob


# In[2]:


url='https://freddiemac.embs.com/FLoan/secure/auth.php'
postUrl='https://freddiemac.embs.com/FLoan/Data/download3.php'


# In[3]:


def payloadCreation(user, passwd):
    creds={'username': user,'password': passwd}
    return creds

def assure_path_exists(path):
    if not os.path.exists(path):
            os.makedirs(path)

def extracrtZip(s,monthlistdata,path,more=False):
    abc = tqdm(monthlistdata)
    for month in abc:
        if more:
            abc.set_description("Downloading {}".format(month[54:75]))
        else:
            abc.set_description("Downloading {}".format(month[54:65]))
        r = s.get(month)
        z = ZipFile(BytesIO(r.content))
        z.extractall(path)
        if more:
            p1 = glob.glob(path+"/historical_data1*.zip")[0]
            with ZipFile(p1,mode='r') as zip:
                zip.extractall(path)


# In[4]:


def getsampleFilesFromFreddieMac(payload,st,en,foldername):
    with requests.Session() as s:
        preUrl = s.post(url, data=payload)  
        payload2={'accept': 'Yes','acceptSubmit':'Continue','action':'acceptTandC'}
        finalUrl=s.post(postUrl,payload2)
        linkhtml =finalUrl.text 
        allzipfiles=BeautifulSoup(linkhtml, "html.parser")
        ziplist=allzipfiles.find_all('td')

        Samplepath=str(os.getcwd())+"/"+foldername
        assure_path_exists(Samplepath)
        sampledata=[]
        historicaldata=[]
        count=0
        slist=[]
        for i in range(int(st),int(en)+1):
            #print(i)
            slist.append(i)
        for li in ziplist:
            zipatags=li.findAll('a')
            for zipa in zipatags:
                for yr in slist:
                    if str(yr) in zipa.text:
                        if re.match('sample',zipa.text):
                            link = zipa.get('href')
                            finallink ='https://freddiemac.embs.com/FLoan/Data/' + link
                            sampledata.append(finallink) 
        extracrtZip(s,sampledata,Samplepath)


# In[5]:


def getFilesFromFreddieMacPeryear(payload,st,en,foldername):
    with requests.Session() as s:
        preUrl = s.post(url, data=payload)
        payload2={'accept': 'Yes','acceptSubmit':'Continue','action':'acceptTandC'}
        finalUrl=s.post(postUrl,payload2)
        linkhtml =finalUrl.text
        allzipfiles=BeautifulSoup(linkhtml, "html.parser")
        ziplist=allzipfiles.find_all('td')
        sampledata=[]
        historicaldata=[]
        count=0
        hlist=[]
        for i in range(int(st),int(en)+1):
            #print(i)
            hlist.append(i)
        Historicalpath=str(os.getcwd())+"/"+foldername
        assure_path_exists(Historicalpath)
        #q =quarter[2:6]
        #t =testquarter[2:6]
        for li in ziplist:
            zipatags=li.findAll('a')
            for zipa in zipatags:
                fetchFile='historical_data1_'
                for yr in hlist:
                    if (fetchFile in zipa.text) and (str(yr) in zipa.text):
                        link = zipa.get('href')
                        finallink ='https://freddiemac.embs.com/FLoan/Data/' + link
    #                     print(finallink)
                        historicaldata.append(finallink)
        extracrtZip(s,historicaldata,Historicalpath,more=True)


# In[6]:


def main():
    ts = time.time()
    foldername1= 'SampleInputFiles'
    foldername2= 'HistoricalInputFiles'
    startYear = 1999
    endYear = 2016
    user = 'eagle11061997@gmail.com'
    password = '>w@<6J=^'
    
    payload=payloadCreation(user,password)
    getsampleFilesFromFreddieMac(payload,startYear,endYear,foldername1)
    getFilesFromFreddieMacPeryear(payload,startYear,endYear,foldername2)


# In[7]:


if __name__ == '__main__':
    main()

