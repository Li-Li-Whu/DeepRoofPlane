import numpy as np
import csv
import os
import glob

feature_path = 'result/feature/'
result_path = 'result/feature_tsv/'
os.makedirs(result_path, exist_ok=True)
files_all = glob.glob(os.path.join(feature_path, '*.txt'))

for file in files_all:
    scan_name = file.split('.')[0].split('/')[-1]
    txtfile_path = file
    csvfile_path = result_path + str(scan_name) + '_meta.csv'
    tsvfile_path = result_path + str(scan_name) + '_meta.tsv'
    csvFile = open(csvfile_path, 'w',newline='',encoding='utf-8') # 我的数据是中文数据集，所以用utf-8
    writer = csv.writer(csvFile)
    csvRow = []
    f = open(txtfile_path,'r',encoding='utf-8')
    for line in f:
        csvRow = line.split()
        writer.writerow(csvRow)
    f.close()
    csvFile.close()

    with open(csvfile_path,encoding='utf-8') as f:
        data = f.read().replace(',', '\t')
    with open(tsvfile_path,'w',encoding='utf-8') as f:
        f.write(data)
    f.close()
    print('txt2tsv finished!')

