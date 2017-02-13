#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
# project: Insight-Churn
# date: 25/01/2017
# author: Deniz Ustebay
# description: Connect to POSTGRES database, create features 
			   from time series and static data,
               save train and test datasets for classification
               (making sure there's no leakage over time)
"""

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib.dates as dates
import matplotlib.gridspec as gridspec
import os
from collections import defaultdict
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import seaborn as sns
import statsmodels.api as sm
from datetime import timedelta, datetime, date
import random
from sklearn.model_selection import train_test_split
import matplotlib.ticker as plticker


os.chdir('/Users/deniz/Research/Insight_Churn/')
plt.style.use('ggplot')

# Connect to POSTGRES database
dbname = 'CHURN'
username = 'deniz'
engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
print engine.url
engine.table_names()

# connect:
con = None
con = psycopg2.connect(database = dbname, user = username)


#%%
initial_period = 28
target_period = 28*3
# compute statistics from the initial_period, and classify if they will quit in target_period


# this function creates time series dataframe from dictionary of id:dates
def make_timeSeries_pd(s):
    df_final = pd.DataFrame()
    for w in s.keys():
        times = s[w]
        dateIndex1 = pd.to_datetime(times,format='%a %b %d %H:%M:%S +0000 %Y').sort_values()

        dateIndex1 = sorted(dateIndex1)

        df = pd.DataFrame(data=1, index = dateIndex1, columns=[w])
        df = df.resample('D').sum()
        df_final = df_final.join(df,how='outer')
    return df_final

# query for PROVIDERS
sql_query = """
SELECT p.id, p.started_datetime, p.finished_datetime, p.birthday, p.provider_hometown_id, p.provider_source_id,
p.city_id, p.years_of_experience, p.average_rating, s.resigned FROM providers p
left join (SELECT id, finish_type, reason_finished, False as resigned FROM terminations
where finish_type='Terminated in probation period' 
or finish_type = 'Dismissal with reason' 
or finish_type= 'Dismissal without reason'
UNION
SELECT id, finish_type, reason_finished, True as resigned FROM terminations
where finish_type='Resigned'
UNION
SELECT id, finish_type, reason_finished, True as resigned FROM terminations
where finish_type='missing' and (reason_finished='Found another job' 
or reason_finished='Exhausted' or reason_finished='Illness')
UNION
SELECT id, finish_type, reason_finished, False as resigned FROM terminations
where finish_type='missing' and reason_finished='Breach'
UNION
SELECT id, finish_type, reason_finished, Null as resigned FROM terminations
where finish_type='missing' and reason_finished='missing') s
on p.id = s.id
"""
provider_df = pd.read_sql_query(sql_query,con)
print str(provider_df.shape[0]) + ' rows in the original' 
provider_df = provider_df.join(pd.DataFrame(data=np.array(provider_df['finished_datetime'].isnull()==False),index=provider_df.index, columns=['churned']),how='outer')

# replace nan's in finished dates with the date data was accquired, this is to count the days with the company
# for those providers who are still working for the company
provider_df['finished_datetime'].fillna(pd.to_datetime('01/20/2017'), inplace=True)

# one provider without start date
provider_df['started_datetime'].fillna(provider_df.started_datetime.min(), inplace=True)

# remove hour from dates
provider_df['started_datetime'] = pd.DatetimeIndex(provider_df['started_datetime']).normalize()
provider_df['finished_datetime'] = pd.DatetimeIndex(provider_df['finished_datetime']).normalize()

# add days in company as a column
provider_df = provider_df.join(pd.DataFrame(
        data=((provider_df.finished_datetime-provider_df.started_datetime)
        / np.timedelta64(1, 'D')), 
        index=provider_df.index, columns=['days_in_company']),how='outer')

# add age as a column
provider_df = provider_df.join(pd.DataFrame(
        data=((provider_df.started_datetime-provider_df.birthday)
        / np.timedelta64(1, 'Y')),
        index=provider_df.index, columns=['age']),how='outer')
        
        
# Plot number of days in company
sns.set_style("whitegrid", {'axes.grid' : False})
fig = plt.figure(figsize=(10,6))
plot = fig.add_subplot(111)
provider_df[provider_df['churned']==True]['days_in_company'].hist(bins=30)
plt.ylabel("Count", fontsize=24)
plt.xlabel('Days in company', fontsize=24)
plt.grid(False)
plot.spines['right'].set_visible(False)
plot.spines['top'].set_visible(False)
plot.yaxis.set_ticks_position('left')
plot.xaxis.set_ticks_position('bottom')
loc = plticker.MultipleLocator(base=15.0) # this locator puts ticks at regular intervals
plot.yaxis.set_major_locator(loc)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.tight_layout()
plt.savefig('distributionTenure.png')

# remove current emplyees that are in the company for less than target_period
ix = ((provider_df.days_in_company<target_period) & (provider_df.churned==False))==False
provider_df = provider_df.ix[ix]
print str(sum(~ix)) + ' removed for they joined less than target period ago' 

# remove churned employees that were in the company for less than initial_period
ix = ((provider_df.days_in_company<initial_period) & (provider_df.churned==True))==False
provider_df = provider_df.ix[ix]
print str(sum(~ix)) + ' removed for they churned before initial period' 

# flag for churned in the target period
provider_df = \
provider_df.join(pd.DataFrame(data=np.array((provider_df.days_in_company < target_period) &\
                                            (provider_df.churned==True)), \
        index=provider_df.index, columns=['churned_in_target']),how='outer')
                      
print str(provider_df.shape[0]) + ' rows in the final' 


a =  ((provider_df['churned_in_target']==True).sum())*100
b =  ((provider_df['churned']==True).sum())

print str(a/b) + '% of all that left churned in the target period'

# %%

# JOIN SERVICES and PROVIDERS tables, ONLY FINISHED SERVICES, ONE ROW PER SHIFT:
sql_query = """
SELECT s.provider_id, s.id as service_id, s.date as service_date, s.morning, False as afternoon, s.status
FROM services s
where (status ='Carried out' or status='In action') and (morning=True and afternoon=True)
UNION ALL
SELECT s.provider_id, s.id as service_id, s.date as service_date, False as morning, s.afternoon, s.status
FROM services s
where (status ='Carried out' or status='In action') and (morning=True and afternoon=True)
UNION ALL
SELECT s.provider_id, s.id as service_id, s.date as service_date, s.morning, s.afternoon, s.status
FROM services s
where (status ='Carried out' or status='In action') and not(morning=True and afternoon=True)
order by service_id
"""
services_df = pd.read_sql_query(sql_query,con)


# CLIENT SCORES AND WORK TYPE

sql_query = """
select s.provider_id,s.service_id, c.office, s.service_date, s.morning, s.afternoon, s.status, sr.nps_score from
(SELECT s.provider_id,s.id as service_id, s.client_id, s.date as service_date, s.morning, False as afternoon, s.status
FROM services s
where (s.status ='Carried out' or s.status='In action') and (s.morning=True and s.afternoon=True)
UNION ALL
SELECT s.provider_id,s.id as service_id,s.client_id, s.date as service_date, False as morning, s.afternoon, s.status
FROM services s
where (s.status ='Carried out' or s.status='In action') and (s.morning=True and s.afternoon=True)
UNION ALL
SELECT s.provider_id,s.id as service_id,s.client_id, s.date as service_date, s.morning, s.afternoon, s.status
FROM services s
where (s.status ='Carried out' or s.status='In action') and not(s.morning=True and s.afternoon=True)) s
left join service_reviews sr
on s.service_id = sr.service_id
left join clients c
on s.client_id = c.id
order by s.service_id
"""
services_and_ratings = pd.read_sql_query(sql_query,con)

#%% TIME OFF
sql_query = """
select date, provider_id from holidays
"""
holidays_df = pd.read_sql_query(sql_query,con)

#%% PUBLIC HOLIDAYS
sql_query = """
select * from public_holidays
"""
public_holidays = pd.read_sql_query(sql_query,con)

# %% remove those without services data
remove_ids = set(provider_df['id'])-set(services_df['provider_id'])
for i in remove_ids:
    provider_df = provider_df[provider_df['id']!=i]
provider_ids = provider_df['id']


#%% dictionary of all service times: keys=providers
all_s = defaultdict(int)
for item in provider_ids:
    if np.isnan(item)==False and len(services_df.service_date[services_df.provider_id==item].values)>0:
        all_s[item] = services_df.service_date[services_df.provider_id==item].values

# convert to time series data frame
services_time_series = make_timeSeries_pd(all_s)


# Generate time series for scores and work type
startTime = time.time()
nps_scores = pd.DataFrame(columns=services_time_series.columns, index=services_time_series.index)
office_shifts = pd.DataFrame(data = 0, columns=services_time_series.columns, index=services_time_series.index)

for i in services_time_series.columns:
   for j in services_time_series.index:
       ix = (services_and_ratings['provider_id']== i) & (services_and_ratings.service_date==j)
       if ix.sum()>0:
           nps_scores.loc[nps_scores.index==j,i] = services_and_ratings.loc[ix,'nps_score'].mean()
           office_shifts.loc[nps_scores.index==j,i] = services_and_ratings.loc[ix,'office'].sum()

endTime = time.time()
print str((endTime - startTime)/60) + ' minutes'


#%% dictionary of all service times: keys=providers
all_s = defaultdict(int)
for item in provider_ids:
    if np.isnan(item)==False and len(holidays_df.date[holidays_df.provider_id==item].values)>0:
        all_s[item] = holidays_df.date[holidays_df.provider_id==item].values

# convert to time series data frame
holidays_time_series = make_timeSeries_pd(all_s)

# make sure time series data frames are same size
remove_ix = list(set(holidays_time_series.index)-set(services_time_series.index))
holidays_time_series = holidays_time_series[~holidays_time_series.index.isin(remove_ix)]
add_ix = list(set(services_time_series.columns)-set(holidays_time_series.columns))
holidays_time_series = pd.concat([holidays_time_series,pd.DataFrame(columns=add_ix)])

# %%
# compute features in the initial period

provider_df = provider_df.join(
    pd.DataFrame(data=np.nan, index = provider_df.index, columns=['occupation_ratio']))
provider_df = provider_df.join(
    pd.DataFrame(data=np.nan, index = provider_df.index, columns=['double_shift_ratio']))
provider_df = provider_df.join(
    pd.DataFrame(data=np.nan, index = provider_df.index, columns=['total_holidays_initial']))
provider_df = provider_df.join(
    pd.DataFrame(data=np.nan, index = provider_df.index, columns=['office_shifts_ratio']))
provider_df = provider_df.join(
    pd.DataFrame(data=np.nan, index = provider_df.index, columns=['average_client_score']))


for i in provider_ids:
    start_time = np.abs(services_time_series.index - provider_df[provider_df['id']==i]['started_datetime'].values).argmin()
    end_time = start_time + initial_period
    public_holiday_shifts = [j for j in public_holidays.date if j in services_time_series[i].index[start_time:end_time]]
    total_shifts_available = (initial_period*12/7.0)-len(public_holiday_shifts)*2
    number_days_worked = (~services_time_series[i].ix[start_time:end_time].isnull()).sum()    
    total_shifts_done = services_time_series[i].ix[start_time:end_time].sum()
    
    provider_df.loc[provider_df['id']==i,'occupation_ratio'] = total_shifts_done/total_shifts_available
    provider_df.loc[provider_df['id']==i,'double_shift_ratio'] = (services_time_series[i].ix[start_time:end_time]==2).sum() * 1.0 / number_days_worked
    if total_shifts_done>0:
        provider_df.loc[provider_df['id']==i,'office_shifts_ratio'] = office_shifts[i].ix[start_time:end_time].sum()/total_shifts_done
    provider_df.loc[provider_df['id']==i,'average_client_score'] = nps_scores[i].ix[start_time:end_time].mean()
    provider_df.loc[provider_df['id']==i,'total_holidays_initial'] = holidays_time_series[i].ix[start_time:end_time].sum()/total_shifts_available
    if total_shifts_done>0 and np.isnan(provider_df.loc[provider_df['id']==i,'total_holidays_initial'].values):
        provider_df.loc[provider_df['id']==i,'total_holidays_initial'] = 0


# %%
    

provider_df = provider_df[['started_datetime', 'finished_datetime', 'birthday','average_rating',\
'days_in_company','provider_hometown_id', 'provider_source_id', 'id','churned','resigned',\
'churned_in_target','city_id', 'age','years_of_experience','occupation_ratio',\
'double_shift_ratio',\
'total_holidays_initial','average_client_score',\
'office_shifts_ratio']]

provider_df = provider_df[provider_df['office_shifts_ratio'].isnull()==False]
provider_df = provider_df[provider_df['years_of_experience'].isnull()==False]
provider_df = provider_df[provider_df['average_client_score'].isnull()==False]
provider_df = provider_df[provider_df['total_holidays_initial'].isnull()==False]

data_train, data_test, y_train, y_test = train_test_split(provider_df,range(provider_df.shape[0]), test_size=0.2, random_state=8557)

data_train.to_pickle('dataset_forLogisticRegression_TRAIN.pkl') 
data_test.to_pickle('dataset_forLogisticRegression_TEST.pkl') 

print str(provider_df.shape[0]) + ' rows in the full' 
print str(data_train.shape[0]) + ' rows in the train' 
print str(data_test.shape[0]) + ' rows in the test' 

# %% 
print 'Is there class imbalance?'
print 'churned samples: ' +  str((provider_df['churned_in_target']==True).sum())
print 'not churned samples: ' +  str((provider_df['churned_in_target']==False).sum())