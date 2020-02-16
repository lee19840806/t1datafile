#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib
import http
import io
import zipfile
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Urban+Land+Cover

url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00295/Urban%20land%20cover.zip'

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

# unzip the downloaded file, and get data files
with zipfile.ZipFile(data_train) as myzip:
    with myzip.open('training.csv') as myfile:
        df_training = pandas.read_csv(myfile, header = 0, index_col = False)

    with myzip.open('testing.csv') as myfile:
        df_testing = pandas.read_csv(myfile, header = 0, index_col = False)

# concatenate df_training and df_testing so as to have more data for modeling
df_total = pandas.concat([df_training, df_testing])

# watch out, values of column class have a trailing space so we remove spaces
df_total['class'] = df_total['class'].str.replace(' ', '')

# the target variable, inserted into the dataframe as the first column, and drop the original class variable
# see if we can distinguish 'car' from the other classes, so we set 1 = car and 0 = other classes
df_total.insert(0, 'target_class', df_total['class'].apply(lambda x: 1 if x == 'car' else 0))
df_total = df_total.drop('class', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_total.to_csv('uci_033_urban_land_cover.csv', index = False)