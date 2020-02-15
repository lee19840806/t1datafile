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
# https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset

url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00224/Dataset.zip'

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
data_train = download_file(url_data_train)

# generate column names
base_name = ['IR_', '|IR|_', 'EMAi0.001_', 'EMAi0.01_', 'EMAi0.1_', 'EMAd0.001_', 'EMAd0.01_', 'EMAd0.1_']
columns = ['target_gas_substance']
for i in range(1, 17):
    suffix_added = [j + str(i) for j in base_name]
    columns.extend(suffix_added)

# unzip the downloaded file, and get data files
df_train = pandas.DataFrame()
with zipfile.ZipFile(data_train) as myzip:
    file_names = ['Dataset/batch{0}.dat'.format(i) for i in range(1, 11)]
    for file in file_names:
        with myzip.open(file) as myfile:
            single_df = pandas.read_csv(myfile, delimiter = '\s+', header = None, names = columns, index_col = False)
            df_train = pandas.concat([df_train, single_df])

# get the numeric part of variables by splitting the values with ':' and converting them into float
for col in df_train.columns:
    if col != 'target_gas_substance':
        df_train[col] = df_train[col].str.split(':').str[-1].astype(numpy.float64)

# the target variable, we binarize it as 1 = 1 (Ethanol) and 0 = other values
df_train['target_gas_substance'] = df_train['target_gas_substance'].apply(lambda x: 1 if x == 1 else 0)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_028_gas_sensor_array_drift.csv', index = False)
