#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib
import http
import io
import numpy
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# http://biostat.mc.vanderbilt.edu/wiki/Main/DataSets

url_data_train = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls'

def download_file(url):
    hds = {'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'}
    components = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(components.hostname, port = components.port)
    conn.request('GET', components.path, headers = hds)
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
data_train = '/home/lee1984/Desktop/titanic3.xls'

# convert Excel file into pandas dataframe
df_train = pandas.read_excel(data_train)

# drop variables which are not for modeling
df_train = df_train.drop(['name', 'ticket', 'boat', 'home.dest'], axis = 1)

# get cabin category by extracting the first letter
df_train['cabin'] = df_train['cabin'].str[0]

# there are very few cabin G and cabin T records, so we set them to missing
df_train['cabin'] = df_train['cabin'].apply(lambda x: numpy.nan if x in ['G', 'T'] else x)

# the target variable, inserted into the dataframe as the first column, and drop the original survived variable
df_train.insert(0, 'target_survived', df_train['survived'])
df_train = df_train.drop('survived', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('vanderbilt_001_titanic.csv', index = False)
