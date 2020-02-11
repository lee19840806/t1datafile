#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib
import http
import io
import numpy
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Credit+Approval

url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'

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

# download data from UCI Machine Learning Repository
#data_train = download_file(url_data_train)
data_train = '/home/lee1984/Desktop/UCI_Data/crx.data'

# A16 is the original target variable, which will be converted into 0 or 1 later
columns = [
    'A1',
    'A2',
    'A3',
    'A4',
    'A5',
    'A6',
    'A7',
    'A8',
    'A9',
    'A10',
    'A11',
    'A12',
    'A13',
    'A14',
    'A15',
    'A16']

# convert flat file into pandas dataframe 
df_train = pandas.read_csv(data_train, header = None, names = columns, index_col = False)

# set "?" to numpy.nan and convert column into float
df_train['A2'] = df_train['A2'].apply(lambda x: numpy.nan if x == '?' else x).astype(numpy.float64)

# set "?" to numpy.nan and convert column into float
df_train['A14'] = df_train['A14'].apply(lambda x: numpy.nan if x == '?' else x).astype(numpy.float64)

# convert target A16 into 0 (-) and 1 (+)
df_train['target_A16'] = df_train['A16'].apply(lambda x: 1 if x == '+' else 0).astype(numpy.int64)
df_train = df_train.drop('A16', axis = 1)

# re-order the columns so that target_A16 becomes the first column
df_train = df_train[[
    'target_A16',
    'A1',
    'A2',
    'A3',
    'A4',
    'A5',
    'A6',
    'A7',
    'A8',
    'A9',
    'A10',
    'A11',
    'A12',
    'A13',
    'A14',
    'A15']]

# save the dataframes as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('credit_approval.csv', index = False)
