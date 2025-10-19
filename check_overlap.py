import pandas as pd
import os

root = r'D:\uaam-phish\data\processed'
train = pd.read_csv(os.path.join(root,'train.csv'))
val   = pd.read_csv(os.path.join(root,'val.csv'))
test  = pd.read_csv(os.path.join(root,'test.csv'))

def overlap(a, b):
    return len(set(a['url_text']) & set(b['url_text']))

print('URL overlap train–val :', overlap(train, val))
print('URL overlap train–test:', overlap(train, test))
print('URL overlap val–test  :', overlap(val, test))
