import sys
import os
from argparse import ArgumentParser
import time
from datetime import datetime
def get_FileModifyTime(filePath):
	filePath = unicode(filePath,'utf8')
	t = os.path.getmtime(filePath)
	return TimeStampToTime(t)
argparser = ArgumentParser('test sentence generator')
argparser.add_argument('--gpu', default='3')
argparser.add_argument('--keyword', default='')
argparser.add_argument('--rejectword', default='')
argparser.add_argument('--dozat', action='store_true')
argparser.add_argument('--in_node', action='store_true')
argparser.add_argument('--node', type=str, default='17')
argparser.add_argument('--time', type=int,default=24)
argparser.add_argument('--argmax', default=True)
args = argparser.parse_args()
path ='./log/'
files=os.listdir(path)
if args.dozat:
	bash_sent='bash test_auth.sh '
else:
	bash_sent='bash test.sh '
gpu_id= ' '+args.gpu+' '
key_word=args.keyword
keyword=key_word
rejectword=args.rejectword
#if key_word=='':
#	assert False, 'key word not defined!'
def get_type(data_dir):
	if 'DM' in data_dir:
		return 'DM'
	if 'PAS' in data_dir:
		return 'PAS'
	if 'PSD' in data_dir:
		return 'PSD'
	if 'ptb' in data_dir:
		return 'ptb'
	if 'ctb' in data_dir:
		return 'ctb'

import pdb
writer=open('multipletest_gen'+'_'+keyword+'.sh','w')
writer.write('#!/usr/bin/env bash\n')
for file in files:
	if 'node' not in file and not args.in_node:
		continue
	if 'node' in file and args.in_node:
		continue
	if 'ptb' in file or 'ctb' in file:
		bash_sent='bash test_ptb.sh '
	elif 'dozat' in file:
		bash_sent='bash test_auth.sh '
	else:
		bash_sent='bash test.sh '
	if not os.path.isdir(file):
		filePath=path+file
		f = open(filePath,'r');
		directory=f.readline()
		if rejectword in directory and rejectword !='':
			f.close()
			continue
		if key_word not in directory:
			f.close()
			continue
		fsize = os.path.getsize(filePath)
		#pdb.set_trace()
		if fsize/1024<50:
			f.close()
			continue
		ftime = os.path.getmtime(filePath)
		ftime = datetime.fromtimestamp(ftime)
		now = datetime.now()
		duration = now-ftime
		duration_in_s = duration.total_seconds()
		hours = divmod(duration_in_s, 3600)[0]
		if hours>=args.time:
			f.close()
			continue

		#pdb.set_trace()
		directory=directory.strip()
		try:
			datatype=directory.split('/')[2]
		except:
			#pdb.set_trace()
			print(file)
			continue
		data_type=get_type(datatype)
		if 'ptb' in file :
			writer.write('echo -ne "' + file+' " \n')
			if not args.argmax:
				runsentence='CUDA_VISIBLE_DEVICES='+args.gpu+' python main.py --save_dir '+directory+' run data/SemEval15/ptb_modified/test.en.ptb.conllu --output_dir results --testing --gen_tree \n'
			else:
				runsentence='CUDA_VISIBLE_DEVICES='+args.gpu+' python main.py --save_dir '+directory+' run data/SemEval15/ptb_modified/test.en.ptb.conllu --output_dir results --testing --get_argmax\n'
			writer.write(runsentence)
			if args.argmax:
				writer.write('rm results/test.en.ptb_modified.conllu\n')
				writer.write('python results/converter.py --name test.en.ptb.conllu\n')
			writer.write('sleep 30\n')
			if not args.argmax:
				writer.write('./conll06eval.pl -g data/SemEval15/ptb_modified/test.en.ptb.conllu -s results/test.en.ptb.conllu -q\n')
			else:
				writer.write('./conll06eval.pl -g data/SemEval15/ptb_modified/test.en.ptb.conllu -s results/test.en.ptb_modified.conllu -q\n')
		elif 'ctb' in file:
			writer.write('echo -ne "' + file+' " \n')
			#runsentence='bash test_as.sh '+ filename +' '+args.gpu+' '+ dataset+' '+'ctb'+'\n' 
			if not args.argmax:
				runsentence='CUDA_VISIBLE_DEVICES='+args.gpu+' python main.py --save_dir '+directory+' run data/SemEval15/ctb/test.en.ctb.conllu --output_dir results --testing --gen_tree \n'
			else:
				runsentence='CUDA_VISIBLE_DEVICES='+args.gpu+' python main.py --save_dir '+directory+' run data/SemEval15/ctb/test.en.ctb.conllu --output_dir results --testing --get_argmax\n'
			writer.write(runsentence)
			if args.argmax:
				writer.write('rm results/test.en.ctb_modified.conllu\n')
				writer.write('python results/converter.py --name test.en.ctb.conllu\n')
			writer.write('sleep 30\n')
			if not args.argmax:
				writer.write('./conll06eval.pl -g data/SemEval15/ctb/test.en.ctb_modified.conllu -s results/test.en.ctb.conllu -q\n')
			else:
				writer.write('./conll06eval.pl -g data/SemEval15/ctb/test.en.ctb_modified.conllu -s results/test.en.ctb_modified.conllu -q\n')
		else:
			res_sent=bash_sent+directory+gpu_id+data_type+'New'+' '+data_type.lower()+'\n'
			writer.write(res_sent)
		f.close()

writer.close()
writer=open('run_test.sh','w')
writer.write('#!/usr/bin/env bash\n')
writer.write('vsub multipletest_gen_'+keyword+'.sh'+' --name evaluation_test_'+keyword+' --node '+args.node+' --shell')
writer.close()
