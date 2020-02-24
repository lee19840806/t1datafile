#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pathlib

workdir = pathlib.Path().absolute()
files_in_folder = [f for f in os.listdir(workdir) if os.path.isfile(os.path.join(workdir, f))]
py_files = sorted([f for f in files_in_folder if os.path.splitext(f)[1] == '.py' and f != os.path.basename(__file__)])

total_items = []
item_num = 0
for file in py_files:
    item_num += 1
    with open(file) as f:
        line = f.readline()
        while line:
            if line == '# data set repository\n':
                total_items.append({
                    'item_num': item_num,
                    'dataset_name': os.path.splitext(file)[0],
                    'source_page': f.readline().strip()[2:]
                    })
                break
            line = f.readline()

template_number = '<td>{0}</td>'
template_dataset = '<td><a href="https://t1-public.oss-cn-hangzhou.aliyuncs.com/{0}.zip" class="text-info">{0}.zip</a></td>'
template_desc = '<td><a href="{0}" class="text-info" target="_blank" rel="noopener noreferrer">Description</a></td>'
template_code = '<td><a href="https://github.com/t1modeler/data_script/blob/master/{0}.py"' + \
    ' class="text-info" target="_blank" rel="noopener noreferrer">Link</a></td>'

html_table = ''
for i in total_items:
    html_table += '<tr>\n'
    html_table += '    ' + template_number.format(i['item_num']) + '\n'
    html_table += '    ' + template_dataset.format(i['dataset_name']) + '\n'
    html_table += '    ' + template_desc.format(i['source_page']) + '\n'
    html_table += '    ' + template_code.format(i['dataset_name']) + '\n'
    html_table += '</tr>\n'

print(html_table)
