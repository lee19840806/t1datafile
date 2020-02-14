#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib
import http
import io
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/seismic-bumps

url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff'

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

# locate data rows by using '@data\n' as a split word
content_data = io.BytesIO(data_train.read().decode().split('@data')[-1].encode())

# class is the original target variable, which will be converted into 0 or 1 later
columns = [
    'seismic',
    'seismoacoustic',
    'shift',
    'genergy',
    'gpuls',
    'gdenergy',
    'gdpuls',
    'ghazard',
    'nbumps',
    'nbumps2',
    'nbumps3',
    'nbumps4',
    'nbumps5',
    'nbumps6',
    'nbumps7',
    'nbumps89',
    'energy',
    'maxenergy',
    'class']

# convert flat files into pandas dataframes
df_train = pandas.read_csv(content_data, header = None, names = columns, index_col = False)

# convert seismic to numeric values, because the underlying categorical values can be binned when modeling
seismic_dict = {
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4}
df_train['seismic'] = df_train['seismic'].apply(lambda x: seismic_dict[x])

# convert seismoacoustic to numeric values, because the underlying categorical values can be binned when modeling
seismoacoustic_dict = {
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4}
df_train['seismoacoustic'] = df_train['seismoacoustic'].apply(lambda x: seismoacoustic_dict[x])

# convert ghazard to numeric values, because the underlying categorical values can be binned when modeling
ghazard_dict = {
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4}
df_train['ghazard'] = df_train['ghazard'].apply(lambda x: ghazard_dict[x])

# the target variable, we insert target_class into the dataframe as the first column and drop the original class column
df_train.insert(0, 'target_class', df_train['class'])
df_train = df_train.drop('class', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_016_seismic_bumps.csv', index = False)
