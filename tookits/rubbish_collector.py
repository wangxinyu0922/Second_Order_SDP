import os, shutil
import time
from datetime import datetime
from argparse import ArgumentParser

argparser = ArgumentParser('test sentence generator')
argparser.add_argument('--dir', default='log')
argparser.add_argument('--dst', default='rubbish')
argparser.add_argument('--days', type=int,default=5)
args = argparser.parse_args()
def get_FileModifyTime(filePath):
	filePath = unicode(filePath,'utf8')
	t = os.path.getmtime(filePath)
	return TimeStampToTime(t)

path ='./'+args.dir+'/'
files=os.listdir(path)

args.time=args.days*24+8
dir='./'+args.dst+'/'+args.dir
if not os.path.exists(dir):
	os.makedirs(dir)
for file in files:
	if not os.path.isdir(file):
		filePath=path+file
		ftime = os.path.getmtime(filePath)
		ftime = datetime.fromtimestamp(ftime)
		now = datetime.now()
		duration = now-ftime
		duration_in_s = duration.total_seconds()
		hours = divmod(duration_in_s, 3600)[0]
		if hours>=args.time:
			try:
				shutil.move(filePath,dir)
			except:
				os.remove(os.path.join(dir,file))
				shutil.move(filePath,dir)
			
