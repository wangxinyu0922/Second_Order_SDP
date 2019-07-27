from argparse import ArgumentParser
import numpy as np
argparser = ArgumentParser('test sentence generator')
argparser.add_argument('--keyword', default='')
argparser.add_argument('--switch', action='store_true')
argparser.add_argument('--best', action='store_true')
argparser.add_argument('--cfg', default='')
argparser.add_argument('--avg', action='store_true')
argparser.add_argument('--avoid', action='store_true')
argparser.add_argument('--byset', action='store_true')
argparser.add_argument('--printcount', action='store_true')
argparser.add_argument('--unlabel', action='store_true')
argparser.add_argument('--std', action='store_true')

args = argparser.parse_args()
keyword=args.keyword
cfg=args.cfg
results={}
f = open('evaluate_results.txt','r')
#writer=open('multipletest_gen_dozat.sh','w')
import pdb
res=f.readlines()
results=[]
for line in res:

	if keyword not in line:
		continue
	if args.switch and 'switch' not in data:
		continue
	line=line.strip()
	if line=='':
		continue
	data=line.split()
	if cfg!='' and cfg not in data[0]:
		continue
	if 'no_lc' in line:
		target='basic'
	elif 'no_char' in line:
		target='no_char'
	elif 'no_lemma' in line:
		target='no_lemma'
	else:
		target='all'
	if 'pas' in line:
		#pdb.set_trace()
		dataset='pas'
	elif 'psd' in line:
		dataset='psd'
	else:
		dataset='dm'
	#id=data[-4]
	#try:
	results.append([float(data[1]),float(data[3]),float(data[4]),float(data[6])])
pdb.set_trace()
total=0
count=0
results=np.array(results)
for i in range(len(results)):
	for j in range(len(results)):
		if i==j:
			continue
		dat=(results[i]-results[j])>=0
		if dat[0]==dat[2] or dat[1]==dat[3]:
			count+=1
		total+=1
pdb.set_trace()