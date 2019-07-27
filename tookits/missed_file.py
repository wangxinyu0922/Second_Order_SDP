import sys
import os
import pdb
import time
from datetime import datetime
from argparse import ArgumentParser
argparser = ArgumentParser('train sentence generator for dozat')
argparser.add_argument('--gpu', default='3')
argparser.add_argument('--name', default='')
argparser.add_argument('--dozat', action='store_true')
args = argparser.parse_args()

filePath='./missedlist.txt'
f = open(filePath,'r')
train_cfg=f.readlines()
filelist=[]
for cfg in train_cfg:
	cfg=cfg.strip()
	if cfg == '':
		continue
	if cfg[0]=='#':
		continue
	data=cfg.split()
	filePath=data[-1][:-1]
	if os.path.exists(filePath):
		continue
	else:
		#print(filePath)
		filelist.append(filePath)
	#pdb.set_trace()

path ='./shell/'
files=os.listdir(path)
for file in files:
	if not os.path.isdir(file):
		filePath=path+file
		if 'swp' in filePath:
			continue
		ftime = os.path.getmtime(filePath)
		ftime = datetime.fromtimestamp(ftime)
		now = datetime.now()
		duration = now-ftime
		duration_in_s = duration.total_seconds()
		hours = divmod(duration_in_s, 3600)[0]
		if hours>=30:
			f.close()
			continue
		f = open(filePath,'r');
		data=f.read()
		#pdb.set_trace()
		removelist=[]
		for file in filelist:
			if file in data:
				removelist.append(file)
		for file in removelist:
			filelist.remove(file)


for file in filelist:
	print(file)
		
		