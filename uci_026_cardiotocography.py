#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib
import io
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Cardiotocography

url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00193/CTG.xls'

def download_file(url):
    resp = urllib.request.urlopen(url)
    if resp.status != 200:
        resp.close()
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
    return io.BytesIO(total)

# download data from UCI Machine Learning Repository
data_train = download_file(url_data_train)

# convert flat file into pandas dataframe
df_train = pandas.read_excel(data_train, sheet_name = 'Raw Data')

# drop the first data row because it's an empty row
df_train = df_train.drop(0)

# drop columns which are inappropriate for modeling
df_train = df_train.drop(['FileName', 'Date', 'SegFile', 'CLASS'], axis = 1)

# the target variable, we insert target_NSP into the dataframe as the first column and drop the original NSP column
df_train.insert(0, 'target_NSP', df_train['NSP'].apply(lambda x: 1 if x == 1 else 0))
df_train = df_train.drop('NSP', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_026_cardiotocography.csv', index = False)
