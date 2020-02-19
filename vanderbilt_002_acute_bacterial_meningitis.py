#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib
import http
import io
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

# convert Excel file into pandas dataframe
df_train = pandas.read_excel(data_train)

# records with missing sex values have very high missing rate for other variables, which are inappropriate for modeling
# retain records with non-missing abm values and non-missing sex values
df_train = df_train[df_train['abm'].notnull() & df_train['sex'].notnull()]

# drop variables which are not for modeling
df_train = df_train.drop(['casenum', 'year', 'month', 'subset'], axis = 1)

# the target variable, inserted into the dataframe as the first column, and drop the original abm variable
df_train.insert(0, 'target_abm', df_train['abm'])
df_train = df_train.drop('abm', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('vanderbilt_002_acute_bacterial_meningitis.csv', index = False)
