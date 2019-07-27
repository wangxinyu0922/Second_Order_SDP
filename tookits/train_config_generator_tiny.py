from parser.config import Config
from argparse import ArgumentParser
import pdb
argparser = ArgumentParser('train sentence generator')
argparser.add_argument('--LBP', action='store_true')
argparser.add_argument('--twogpu', action='store_true')
argparser.add_argument('--autorun', action='store_true')
args = argparser.parse_args()
if args.LBP:
	print('LBP generation have not been checked! Please check it first')
	config_file='config/sec_order_LBP.cfg'
	#vocab_sent='SecondOrderGraphLBPVocab'
else:
	config_file='config/sec_order.cfg'
	#vocab_sent='SecondOrderGraphIndexVocab'
filePath='./train_list.txt'

configwords=['iter','unary','token','binary','batch','lr','decay','reg','init']
floatwords=['lr','decay','init']

f = open(filePath,'r')
train_cfg=f.readlines()
writer=open('multipletrain_gen.sh','w')
index=0
if args.autorun:
	args.twogpu=True
for cfg in train_cfg:
	#pdb.set_trace()

	cfg=cfg.strip()
	if cfg=='':
		continue
	if args.twogpu:
		if index%2==0 and index!=0:
			writer.write('\n')
	else:
		if index%3==0 and index!=0:
			writer.write('\n')
	kwargs={}
	kwargs['SecondOrderVocab']={}
	kwargs['SecondOrderGraphIndexVocab']={}
	kwargs['SemrelGraphTokenVocab']={}
	kwargs['CoNLLUTrainset']={}
	kwargs['Optimizer']={}
	kwargs['BaseNetwork']={}
	kwargs['DEFAULT']={}
	kwargs['GraphParserNetwork']={}
	kwargs['FormMultivocab']={}
	
	split_cfg=cfg.split('_')
	for parameter in split_cfg:
		if 'iter' in parameter:
			kwargs['SecondOrderGraphIndexVocab']['num_iteration']=parameter.split('iter')[0]
			continue
		elif 'unary' in parameter:
			kwargs['SecondOrderVocab']['unary_hidden']=parameter.split('unary')[0]
			continue
		elif 'token' in parameter:
			kwargs['SemrelGraphTokenVocab']['hidden_size']=parameter.split('token')[0]
			continue
		elif 'binary' in parameter:
			kwargs['SecondOrderVocab']['hidden_size']=parameter.split('binary')[0]
			continue
		elif 'batch' in parameter:
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
		elif 'inter' in parameter:
			kwargs['SemrelGraphTokenVocab']['loss_interpolation']='.'+parameter.split('inter')[0]
			continue
		elif 'rnn' in parameter:
			kwargs['GraphParserNetwork']['recur_size']=parameter.split('rnn')[0]
		elif 'init' in parameter:
			if '.' in parameter:
				kwargs['SecondOrderGraphIndexVocab']['tri_std']=parameter.split('init')[0]
				continue
			init_value=parameter.split('init')[0]
			if init_value[0]=='0':
				kwargs['SecondOrderGraphIndexVocab']['tri_std']='.'+parameter.split('init')[0]
				if init_value=='0':
					kwargs['SecondOrderGraphIndexVocab']['tri_std_unary']='.0'
			else:
				kwargs['SecondOrderGraphIndexVocab']['tri_std']=int(parameter.split('init')[0])/10
			continue
		elif 'mu' in parameter:
			#init_value=parameter.split('mu')[0]
			kwargs['Optimizer']['mu']='.'+parameter.split('mu')[0]
			continue
		elif 'nu' in parameter:
			kwargs['Optimizer']['nu']='.'+parameter.split('nu')[0]
	if '2gpu' in cfg or args.twogpu:
		kwargs['GraphParserNetwork']['two_gpu']=True
	else:
		kwargs['GraphParserNetwork']['two_gpu']=False
	if 'test' in cfg:
		kwargs['SecondOrderGraphIndexVocab']['test_new_potential']=True
	else:
		kwargs['SecondOrderGraphIndexVocab']['test_new_potential']=False
	kwargs['BaseNetwork']['max_steps']=150000
	kwargs['BaseNetwork']['max_steps_without_improvement']=20000
	kwargs['SecondOrderGraphIndexVocab']['as_score']=True
	if 'sib_only' in cfg:
		kwargs['SecondOrderGraphIndexVocab']['use_sib']=True
		kwargs['SecondOrderGraphIndexVocab']['use_gp']=False
		kwargs['SecondOrderGraphIndexVocab']['use_cop']=False
	elif 'cop_only' in cfg:
		kwargs['SecondOrderGraphIndexVocab']['use_sib']=False
		kwargs['SecondOrderGraphIndexVocab']['use_gp']=False
		kwargs['SecondOrderGraphIndexVocab']['use_cop']=True
	elif 'gp_only' in cfg:
		kwargs['SecondOrderGraphIndexVocab']['use_sib']=False
		kwargs['SecondOrderGraphIndexVocab']['use_gp']=True
		kwargs['SecondOrderGraphIndexVocab']['use_cop']=False
	else:
		kwargs['SecondOrderGraphIndexVocab']['use_sib']=True
		kwargs['SecondOrderGraphIndexVocab']['use_gp']=True
		kwargs['SecondOrderGraphIndexVocab']['use_cop']=True

	if 'psd' in cfg:
		parameter='psd'
	elif 'pas' in cfg:
		parameter='pas'
	else:
		parameter='dm'
	if 'head_dep' in cfg:
		kwargs['SecondOrderGraphIndexVocab']['separate_embed']=False
	if 'switch' in cfg:
		kwargs['BaseNetwork']['switch_optimizers']=True
		kwargs['BaseNetwork']['switch_iter']=5000
	if '45' in cfg:
		kwargs['DEFAULT']['TREEBANK']=parameter.upper()+'New'+'_'+'45'
	else:
		kwargs['DEFAULT']['TREEBANK']=parameter.upper()+'_tiny_h'
	kwargs['DEFAULT']['TB']=parameter
	kwargs['DEFAULT']['train_conllus']='data/SemEval15/${TREEBANK}/train.${LC}.${TB}.tiny.conllu'
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
	with open(cfg_dir, 'w') as f:
		config.write(f)
	if args.twogpu:
		if args.autorun:
			train_case='CUDA_VISIBLE_DEVICES='+str(index%2*2)+','+str(index%2*2+1)+' nohup python3 -u main.py train GraphParserNetwork --nowarning --config_file '\
			+cfg_dir+' --noscreen > log/'+cfg+'\n'
		else:
			train_case='CUDA_VISIBLE_DEVICES='+str(index%2*2)+','+str(index%2*2+1)+' nohup python3 -u main.py train GraphParserNetwork --config_file '\
			+cfg_dir+' --noscreen > log/'+cfg+'&'+'\n'
	else:
		train_case='CUDA_VISIBLE_DEVICES='+str(index%3)+',3'' nohup python3 -u main.py train GraphParserNetwork --config_file '\
		+cfg_dir+' --noscreen > log/'+cfg+'&'+'\n'
	writer.write(train_case)
	index+=1
writer.close()