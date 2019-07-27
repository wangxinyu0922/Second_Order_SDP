import sys
import os
from argparse import ArgumentParser
import time
from datetime import datetime
import pdb
argparser = ArgumentParser('File printer')
argparser.add_argument('--length', default=20000)
argparser.add_argument('--kill', action='store_true')
argparser.add_argument('--dontkill', action='store_true')
argparser.add_argument('--keyword', default='')
argparser.add_argument('--time', default=120)
args = argparser.parse_args()
def TimeStampToTime(timestamp):
	timeStruct = time.localtime(timestamp)
	return time.strftime('%Y-%m-%d %H:%M:%S',timeStruct)

def get_FileModifyTime(filePath):
	#filePath = unicode(filePath,'utf8')
	t = os.path.getmtime(filePath)
	# return TimeStampToTime(t)
	return t

def find_second_step(lists):
	i = len(lists)-1
	pos1 = 0
	pos2 = 0
	find = 0
	while i>=0:
		if "Epoch" in lists[i] and find == 1:
			pos2 = i
			find = 2
			return pos1,pos2
		if "Epoch" in lists[i] and find == 0:
			pos1 = i
			find = 1
		i-=1
	return [-1,-1]


path ='./log/'
files=os.listdir(path)

file_time = {}

for i in range(len(files)):
	filename = path+files[i]
	fsize = os.path.getsize(filename)
	#pdb.set_trace()
	if fsize/1024<50:
		#f.close()
		continue
	timing = get_FileModifyTime(filename)
	file_time[filename] = timing

sort_files = sorted(file_time.items(), key=lambda d: d[1],reverse=True) 
# w=open('result', 'w')
# pdb.set_trace()
#pdb.set_trace()
limited_files = sort_files
first_file_time = sort_files[0][1]
count=0
print_len=int(args.length)
for i in range(len(limited_files)):

	if (first_file_time - limited_files[i][1]) > float(args.time):
		break
	l_file = limited_files[i][0]
	# pdb.set_trace()
	f = open(l_file,'r')
	flines = f.readlines()
	if args.keyword not in flines[0]:
		continue
	pos1, pos2 = find_second_step(flines)
	pos = pos1-pos2
	piece = flines[pos2:]
	last_piece = piece[pos:]
	first_piece = piece[:pos]
	# print first_piece
	# print last_piece
	#pdb.set_trace()
	steps_since_improvement = int(last_piece[2].split(':')[1][:-1])
	#steps_since_improvement = 0
	if args.kill:
		if steps_since_improvement>=print_len:	#steps since improvement
			name = flines[0]	#name
			print (name[:-1])	
	elif args.dontkill:
		if steps_since_improvement<=print_len:	#steps since improvement
			name = flines[0]	#name
			print (name[:-1])		
	else:
		name = flines[0]
		print (name[:-1])
		#print (last_piece[3][1:-1])
		for j in range(0,len(first_piece)):
			print (first_piece[j][:-1])
		for j in range(0,len(last_piece)):
			print (last_piece[j][:-1])
		print("\n")
	
	count+=1
print (count)


		
