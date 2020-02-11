#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib
import http
import io
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Letter+Recognition

url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'

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

# lettr is the original target variable, which will be converted into 0 or 1 later
columns = [
    'lettr',
    'x-box',
    'y-box',
    'width',
    'high ',
    'onpix',
    'x-bar',
    'y-bar',
    'x2bar',
    'y2bar',
    'xybar',
    'x2ybr',
    'xy2br',
    'x-ege',
    'xegvy',
    'y-ege',
    'yegvx']

# convert flat file into pandas dataframe
df_train = pandas.read_csv(data_train, header = None, names = columns, index_col = False)

# the target variable, there are 26 letters (from A to Z)
# here we set A to 1 and other letters to 0. with this example we merely want to distinguish A from Non-A letters
df_train['target_lettr'] = df_train['lettr'].apply(lambda x: 1 if x == 'A' else 0)
df_train = df_train.drop('lettr', axis = 1)

# re-order the columns so that target becomes the first column
df_train = df_train[[
    'target_lettr',
    'x-box',
    'y-box',
    'width',
    'high ',
    'onpix',
    'x-bar',
    'y-bar',
    'x2bar',
    'y2bar',
    'xybar',
    'x2ybr',
    'xy2br',
    'x-ege',
    'xegvy',
    'y-ege',
    'yegvx']]

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('letter_recognition.csv', index = False)