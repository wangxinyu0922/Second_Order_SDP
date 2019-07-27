import sys
import os
import pdb
from argparse import ArgumentParser
argparser = ArgumentParser('train sentence generator for dozat')
argparser.add_argument('--gpu', default='1')
argparser.add_argument('--name', default='')
argparser.add_argument('--dozat', action='store_true')
argparser.add_argument('--tb', action='store_true')
argparser.add_argument('--argmax', action='store_true')
args = argparser.parse_args()

filePath='./file_name_list.txt'
f = open(filePath,'r')
train_cfg=f.readlines()
if args.dozat:
	args.name+='dozat'
writer=open('finded_file'+args.name+'.sh','w')
datasetsid=''
for cfg in train_cfg:
	
	if 'tiny' not in cfg:
		tiny_name='modified'
	else:
		tiny_name='tiny'
	if '40set' in cfg:
		tiny_name+='40'
	elif '70set' in cfg:
		tiny_name+='70'
	elif '10set' in cfg:
		tiny_name+='10'
	cfg=cfg.strip()
	if cfg == '':
		continue
	if cfg[0]=='#':
		continue
	if 'dm' in cfg:
		dataset='DM'
	elif 'psd' in cfg:
		dataset='PSD'
	elif 'pas' in cfg:
		dataset='PAS'
	elif 'ptb' in cfg:
		dataset='ptb_modified'
	elif 'ctb' in cfg:
		dataset='ctb'
	if 'tiny' in cfg:		
		try:
			datasetsid=int(cfg.split('tiny')[1][0])
			datasetsid=cfg.split('tiny')[1][0]
		except:
			datasetsid=''
	elif 'ptb' in cfg or 'ctb' in cfg:
		pass
	else:
		dataset+=''
	if 'dozat' in cfg:
		additional='GraphParserNetwork'
	else:
		additional=''
	if os.path.exists('log/'+cfg):
		#pdb.set_trace()
		reader=open('log/'+cfg,'r')
		filename=reader.readline().strip()
		reader.close()
	elif 'ptb' in cfg or 'ctb' in cfg:
		filename='saves/SemEval15/'+dataset+'/'+additional+cfg
	elif 'tiny' not in cfg:
		filename='saves/SemEval15/'+dataset+'New'+'_'+tiny_name+'/'+additional+cfg
	else:
		filename='saves/SemEval15/'+dataset+'_'+tiny_name+'_h'+datasetsid+'/'+additional+cfg
		if not os.path.exists(filename):
			filename='saves/SemEval15/'+dataset+'_'+tiny_name+'_new'+datasetsid+'/'+additional+cfg
	#print(filename)
	if os.path.exists(filename):
		pass
	else:
		print(cfg)
		continue
	if 'ptb' in cfg or 'ctb' in cfg:
		writer.write('echo -ne "' + cfg+' " \n')
		
		if 'ptb' in cfg:
			#runsentence='bash test_as.sh '+ filename +' '+args.gpu+' '+ dataset+' '+'ptb'+'\n' 
			if not args.argmax:
				runsentence='CUDA_VISIBLE_DEVICES='+args.gpu+' python main.py --save_dir '+filename+' run data/SemEval15/ptb_modified/test.en.ptb.conllu --output_dir results --testing --gen_tree \n'
			else:
				runsentence='CUDA_VISIBLE_DEVICES='+args.gpu+' python main.py --save_dir '+filename+' run data/SemEval15/ptb_modified/test.en.ptb.conllu --output_dir results --testing --get_argmax\n'
			writer.write(runsentence)
			if args.argmax:
				writer.write('rm results/test.en.ptb_modified.conllu\n')
				writer.write('python results/converter.py --name test.en.ptb.conllu\n')
			writer.write('sleep 30\n')
			if not args.argmax:
				writer.write('./conll06eval.pl -g data/SemEval15/ptb_modified/test.en.ptb.conllu -s results/test.en.ptb.conllu -q\n')
			else:
				writer.write('./conll06eval.pl -g data/SemEval15/ptb_modified/test.en.ptb.conllu -s results/test.en.ptb_modified.conllu -q\n')

		else:
			#runsentence='bash test_as.sh '+ filename +' '+args.gpu+' '+ dataset+' '+'ctb'+'\n' 
			if not args.argmax:
				runsentence='CUDA_VISIBLE_DEVICES='+args.gpu+' python main.py --save_dir '+filename+' run data/SemEval15/ctb/test.en.ctb.conllu --output_dir results --testing --gen_tree \n'
			else:
				runsentence='CUDA_VISIBLE_DEVICES='+args.gpu+' python main.py --save_dir '+filename+' run data/SemEval15/ctb/test.en.ctb.conllu --output_dir results --testing --get_argmax\n'
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
		#pdb.set_trace()
		runsentence='bash test_length_auth.sh '+ filename +' '+args.gpu+' '+ dataset+'New '+dataset.lower()+'\n' 
		writer.write(runsentence)
