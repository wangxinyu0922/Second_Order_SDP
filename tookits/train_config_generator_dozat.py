from parser.config import Config
from argparse import ArgumentParser
import pdb
argparser = ArgumentParser('train sentence generator for dozat')
argparser.add_argument('--nornn', action='store_true')
args = argparser.parse_args()

config_file='config/DM18.cfg'

filePath='./train_list_dozat.txt'

configwords=['iter','unary','token','binary','batch','lr','decay','reg','init']
floatwords=['lr','decay','init']

f = open(filePath,'r')
train_cfg=f.readlines()
writer=open('multipletrain_gen_dozat.sh','w')
index=0
for cfg in train_cfg:
	#pdb.set_trace()

	cfg=cfg.strip()
	if cfg=='':
		continue
	if index%4==0 and index!=0:
		writer.write('\n')
	kwargs={}
	kwargs['CoNLLUTrainset']={}
	kwargs['Optimizer']={}
	kwargs['BaseNetwork']={}
	kwargs['DEFAULT']={}
	kwargs['GraphParserNetwork']={}
	kwargs['FormMultivocab']={}
	kwargs['CoNLLUDataset']={}
	split_cfg=cfg.split('_')
	for parameter in split_cfg:
		if 'batch' in parameter:
			kwargs['CoNLLUTrainset']['batch_size']=parameter.split('batch')[0]
			continue
		elif 'lr' in parameter:
			kwargs['Optimizer']['learning_rate']='.'+parameter.split('lr')[0]
			continue
		elif 'decay' in parameter:
			kwargs['Optimizer']['decay_rate']='.'+parameter.split('decay')[0]
			continue
		elif 'reg' in parameter:
			kwargs['BaseNetwork']['l2_reg']=parameter.split('reg')[0]
			continue
		'''
		elif 'init' in parameter:
			init_value=parameter.split('init')[0]
			if init_value[0]=='0':
				kwargs['SecondOrderGraphIndexVocab']['tri_std']='.'+parameter.split('init')[0]
			else:
				kwargs['SecondOrderGraphIndexVocab']['tri_std']=int(parameter.split('init')[0])/10
			continue
		'''
	if 'psd' in cfg:
		parameter='psd'
	elif 'pas' in cfg:
		parameter='pas'
	else:
		parameter='dm'
	
	kwargs['BaseNetwork']['switch_optimizers']=True
	if 'switch' in cfg:
		kwargs['BaseNetwork']['switch_iter']=5000
	else:
		kwargs['BaseNetwork']['switch_iter']=500
	kwargs['BaseNetwork']['max_steps']=150000
	kwargs['BaseNetwork']['max_steps_without_improvement']=15000
	kwargs['DEFAULT']['TREEBANK']=parameter.upper()+'New'
	kwargs['DEFAULT']['TB']=parameter
	kwargs['CoNLLUDataset']['batch_size']=1000
	if '40set' in cfg:
		dset='40'
		rate=4
	elif '70set' in cfg:
		dset='70'
		rate=7
	elif 'tiny' in cfg:
		dset=''
		rate=1
	else:
		rate=10
	if 'tiny' in cfg:
		#pdb.set_trace()
		try:
			datasetsid=int(cfg.split('tiny')[1][0])
			datasetsid=cfg.split('tiny')[1][0]
		except:
			datasetsid=''
		kwargs['DEFAULT']['TREEBANK']=parameter.upper()+'_tiny'+dset+'_h'+datasetsid
		kwargs['DEFAULT']['train_conllus']='data/SemEval15/${TREEBANK}/train.${LC}.${TB}.tiny.conllu'
		kwargs['BaseNetwork']['max_steps_without_improvement']=10000
		kwargs['Optimizer']['decay_steps']=1000*rate
	if 'no_lc' in cfg:
		kwargs['GraphParserNetwork']['input_vocab_classes']='FormMultivocab:XPOSTokenVocab'
		kwargs['FormMultivocab']['use_subtoken_vocab']='False'
	elif 'no_lemma' in cfg:
		kwargs['GraphParserNetwork']['input_vocab_classes']='FormMultivocab:XPOSTokenVocab'
		kwargs['FormMultivocab']['use_subtoken_vocab']='True'
	elif 'no_char' in cfg:
		kwargs['GraphParserNetwork']['input_vocab_classes']='FormMultivocab:XPOSTokenVocab:LemmaTokenVocab'
		kwargs['FormMultivocab']['use_subtoken_vocab']='False'
	else:
		kwargs['GraphParserNetwork']['input_vocab_classes']='FormMultivocab:XPOSTokenVocab:LemmaTokenVocab'
		kwargs['FormMultivocab']['use_subtoken_vocab']='True'
	#pdb.set_trace()
	if 'decay' not in cfg:
		kwargs['Optimizer']['decay_rate']=1
	kwargs['DEFAULT']['modelname']=cfg
	config = Config(config_file=config_file, **kwargs)
	cfg_dir='config_gen/'+cfg+'.cfg'
	if args.nornn:
		nornn=' --nornn'
	else:
		nornn=''
	with open(cfg_dir, 'w') as f:
		config.write(f)
	train_case='CUDA_VISIBLE_DEVICES='+str(index%4)+' nohup python3 -u main.py train GraphParserNetwork --config_file '\
	+cfg_dir+nornn+' --noscreen > log/'+cfg+'&'+'\n'
	writer.write(train_case)
	index+=1
writer.close()