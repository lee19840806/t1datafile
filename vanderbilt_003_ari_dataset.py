#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib
import http
import io
import zipfile
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/abm.html

url_data_train = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/acath.xls.zip'

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

# download data from website
data_train = download_file(url_data_train)

# unzip the downloaded file, and get data files
with zipfile.ZipFile(data_train) as myzip:
    with myzip.open('ari/ari.csv') as myfile:
        df_train = pandas.read_csv(myfile, header = 0)

    with myzip.open('ari/Sc.csv') as myfile:
        df_sc = pandas.read_csv(myfile, header = 0)

    with myzip.open('ari/Y.csv') as myfile:
        df_y = pandas.read_csv(myfile, header = 0)

# merge y lable with features
df_y = df_y.drop('Unnamed: 0', axis = 1).rename(columns = {'x': 'target_death'})
df_sc = df_sc.drop(['age', 'rr', 'hrat', 'temp', 'waz'], axis = 1)
df_train = df_train.merge(df_sc, how = 'left', on = 'Unnamed: 0')
df_train = df_y.merge(df_train, how = 'left', left_index = True, right_index = True)

# drop variables which are not for modeling
df_train = df_train.drop(['Unnamed: 0', 'stno'], axis = 1)

# the target variable, set target_death > 0 to 1 and target_death = other values to 0
df_train['target_death'] = df_train['target_death'].apply(lambda x: 1 if x > 0 else 0)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('vanderbilt_003_ari_dataset.csv', index = False)
