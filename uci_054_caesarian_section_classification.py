#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib
import http
import io
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Caesarian+Section+Classification+Dataset

url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00472/caesarian.csv.arff'

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

columns = [
    'Age',
    'Delivery number',
    'Delivery time',
    'Blood of Pressure',
    'Heart Problem',
    'Caesarian']

# convert flat file into pandas dataframe
df_train = pandas.read_csv(data_train, header = None, names = columns, skiprows = 17)

# the target variable, inserted into the dataframe as the first column, and drop the original Caesarian variable
# set 1 = 'stable' and 0 = 'unstable'
df_train.insert(0, 'target_Caesarian', df_train['Caesarian'])
df_train = df_train.drop('Caesarian', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_054_caesarian_section_classification.csv', index = False)