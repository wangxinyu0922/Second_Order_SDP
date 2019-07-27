import sys
import os
from argparse import ArgumentParser
import time
from datetime import datetime
import pdb
import shutil
argparser = ArgumentParser('test sentence generator')
argparser.add_argument('--path', default='./log/')
argparser.add_argument('--time',type=int, default=10)
argparser.add_argument('--keyword', default='')
args = argparser.parse_args()
path =args.path
files=os.listdir(path)
keyword=args.keyword
for file in files:
	filePath=path+file
	ftime = os.path.getmtime(filePath)
	ftime = datetime.fromtimestamp(ftime)
	now = datetime.now()
	duration = now-ftime
	duration_in_s = duration.total_seconds()
	hours = divmod(duration_in_s, 3600)[0]
	days=hours/24
	#pdb.set_trace()
	if days>=args.time:
		#pdb.set_trace()
		#os.removedirs(filePath)
		if keyword!='' and keyword in filePath:
			shutil.rmtree(filePath)
		elif keyword=='':
			shutil.rmtree(filePath)