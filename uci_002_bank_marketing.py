#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib
import http
import io
import zipfile
import numpy
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'

def download_file(url):
    components = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(components.hostname, port = components.port)
    conn.request('GET', components.path)
    resp = conn.getresponse()

    if resp.status != 200:
        resp.close()
        conn.close()
        raise ValueError('Error: {0}'.format(resp.reason))

    print('\rStarted', end = '\r')
    content_length = resp.getheader('Content-Length')
    if content_length is None:
        content_length = '(total: unknown)'
    else:
        content_length = int(content_length)
        if content_length < 1024:
            content_length_str = '(total %.0f Bytes)' % content_length
        elif content_length < 1024 * 1024:
            content_length_str = '(total %.0f KB)' % (content_length / 1024)
        else:
            content_length_str = '(total %.1f MB)' % (content_length / 1024 / 1024)

    total = bytes()
    while not resp.isclosed():
        total += resp.read(10 * 1024)
        if len(total) < 1024:
            print(('\rDownloaded: %.0f Bytes ' % len(total)) + content_length_str + '  ', end = '\r')
        if len(total) < 1024 * 1024:
            print(('\rDownloaded: %.0f KB ' % (len(total) / 1024)) + content_length_str + '  ', end = '\r')
        else:
            print(('\rDownloaded: %.1f MB ' % (len(total) / 1024 / 1024)) + content_length_str + '  ', end = '\r')

    print()
    conn.close()
    return io.BytesIO(total)

data_train = download_file(url_data_train)

# unzip the downloaded file, get df_train and df_test
with zipfile.ZipFile(data_train) as myzip:
    with myzip.open('bank-full.csv') as myfile:
        df_train = pandas.read_csv(io.BytesIO(myfile.read()), delimiter = ';')

    with myzip.open('bank.csv') as myfile:
        df_test = pandas.read_csv(io.BytesIO(myfile.read()), delimiter = ';')

# 2 dataframes will be concatenated for data processing, a "train_or_test" label makes it easier for separating them later
df_train['train_or_test'] = 'train'
df_test['train_or_test'] = 'test'
df_total = pandas.concat([df_train, df_test]).reset_index(drop = True)

# drop day and month, because they are not appropriate for modeling
df_total = df_total.drop(['day', 'month'], axis = 1)

# create dummy variables for job, then drop the original job variable
df_total['job_blue-collar'  ] = df_total['job'].apply(lambda x: 1 if x == 'blue-collar'   else 0)
df_total['job_management'   ] = df_total['job'].apply(lambda x: 1 if x == 'management'    else 0)
df_total['job_technician'   ] = df_total['job'].apply(lambda x: 1 if x == 'technician'    else 0)
df_total['job_admin.'       ] = df_total['job'].apply(lambda x: 1 if x == 'admin.'        else 0)
df_total['job_services'     ] = df_total['job'].apply(lambda x: 1 if x == 'services'      else 0)
df_total['job_retired'      ] = df_total['job'].apply(lambda x: 1 if x == 'retired'       else 0)
df_total['job_self-employed'] = df_total['job'].apply(lambda x: 1 if x == 'self-employed' else 0)
df_total['job_entrepreneur' ] = df_total['job'].apply(lambda x: 1 if x == 'entrepreneur'  else 0)
df_total['job_unemployed'   ] = df_total['job'].apply(lambda x: 1 if x == 'unemployed'    else 0)
df_total['job_housemaid'    ] = df_total['job'].apply(lambda x: 1 if x == 'housemaid'     else 0)
df_total['job_student'      ] = df_total['job'].apply(lambda x: 1 if x == 'student'       else 0)
df_total = df_total.drop('job', axis = 1)

# set pdays = -1 to missing, avoid binning -1 with other scalar values when modeling
df_total['pdays'] = df_total['pdays'].apply(lambda x: numpy.nan if x == -1 else x)

# the target variable, y = 1 (yes) and y = 0 (no)
df_total['y'] = df_total['y'].apply(lambda x: 1 if x == 'yes' else 0)

# re-order the columns so that y becomes the first column
df_total = df_total[[
    'y',
    'age',
    'marital',
    'education',
    'default',
    'balance',
    'housing',
    'loan',
    'contact',
    'duration',
    'campaign',
    'pdays',
    'previous',
    'poutcome',
    'train_or_test',
    'job_blue-collar',
    'job_management',
    'job_technician',
    'job_admin.',
    'job_services',
    'job_retired',
    'job_self-employed',
    'job_entrepreneur',
    'job_unemployed',
    'job_housemaid',
    'job_student']]

# separate train and test data samples, and drop the train_or_test label
bank_marketing_train = df_total[df_total['train_or_test'] == 'train'].drop('train_or_test', axis = 1)
bank_marketing_test = df_total[df_total['train_or_test'] == 'test'].drop('train_or_test', axis = 1)

# save the 2 dataframes as CSV files, you can zip them respectively, upload them to t1modeler.com, and build a model
bank_marketing_train.to_csv('bank_marketing_train.csv', index = False)
bank_marketing_test.to_csv('bank_marketing_test.csv', index = False)
