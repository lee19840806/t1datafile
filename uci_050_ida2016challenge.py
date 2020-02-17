#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib
import http
import io
import zipfile
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/IDA2016Challenge

url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00414/to_uci.zip'

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
    with myzip.open('to_uci/aps_failure_training_set.csv') as myfile:
        df_train = pandas.read_csv(myfile, header = 0, skiprows = 19, na_values = 'na', low_memory = False)

# the class variable, set class = pos as 1 and class = neg as 0
df_train['class'] = df_train['class'].apply(lambda x: 1 if x == 'pos' else 0)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_050_ida2016challenge.csv', index = False)
