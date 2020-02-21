#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib
import io
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# http://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits

url_data_train = 'http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra'

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

# generate column names
columns = ['variable_{0}'.format(str(i + 1).zfill(3)) for i in range(64)] + ['class_code']

# convert flat file into pandas dataframe
df_train = pandas.read_csv(data_train, header = None, names = columns)

# the target variable, inserted into the dataframe as the first column, and drop the original class_code variable
# set class_code = 9 to 1 and class = other digits to 0, see if we can distinguish 9 from other handwritten digits
df_train.insert(0, 'target_class_code', df_train['class_code'].apply(lambda x: 1 if x == 9 else 0))
df_train = df_train.drop('class_code', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_059_optical_recognition_of_handwritten_digits.csv', index = False)
