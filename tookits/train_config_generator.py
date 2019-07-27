from parser.config import Config
from argparse import ArgumentParser
from tookits.cfg_parser import cfg_parser
import pdb
argparser = ArgumentParser('train sentence generator')
argparser.add_argument('--LBP', action='store_true')
argparser.add_argument('--twogpu', action='store_true')
argparser.add_argument('--autorun', action='store_true')
argparser.add_argument('--dozat', action='store_true')
argparser.add_argument('--nornn', action='store_true')
argparser.add_argument('--setname', default='new')
argparser.add_argument('--stop_iter', default=10000)
args = argparser.parse_args()
if args.LBP:
	#LBP
	#print('LBP generation have not been checked! Please check it first')
	config_file='config/sec_order_LBP.cfg'
	vocab_sent='SecondOrderGraphLBPVocab'
	hyper_vocab='SecondOrderLBPVocab'
elif args.dozat:
	config_file='config/DM18.cfg'
	vocab_sent='SecondOrderGraphIndexVocab'
	hyper_vocab='SecondOrderVocab'
else:
	#Mean Field
	config_file='config/sec_order.cfg'
	vocab_sent='SecondOrderGraphIndexVocab'
	hyper_vocab='SecondOrderVocab'
filePath='./train_list.txt'

configwords=['iter','unary','token','binary','batch','lr','decay','reg','init']
floatwords=['lr','decay','init']

f = open(filePath,'r')
train_cfg=f.readlines()
writer=open('multipletrain_gen.sh','w')
index=0
if args.autorun:
	args.twogpu=True
parameter_parser=cfg_parser(args)
for cfg in train_cfg:
	#pdb.set_trace()
	if 'LBP' in cfg:
		#LBP
		config_file='config/sec_order_LBP.cfg'
		parameter_parser.vocab_sent='SecondOrderGraphLBPVocab'
		parameter_parser.hyper_vocab='SecondOrderLBPVocab'
		vocab_sent=parameter_parser.vocab_sent
		hyper_vocab=parameter_parser.hyper_vocab
	elif 'dozat' in cfg:
		config_file='config/DM18.cfg'
		parameter_parser.vocab_sent='SecondOrderGraphIndexVocab'
		parameter_parser.hyper_vocab='SecondOrderVocab'
		vocab_sent=parameter_parser.vocab_sent
		hyper_vocab=parameter_parser.hyper_vocab
	else:
		#Mean Field
		config_file='config/sec_order.cfg'
		parameter_parser.vocab_sent='SecondOrderGraphIndexVocab'
		parameter_parser.hyper_vocab='SecondOrderVocab'
		vocab_sent=parameter_parser.vocab_sent
		hyper_vocab=parameter_parser.hyper_vocab
	cfg=cfg.strip()
	if cfg=='':
		continue
	if args.dozat:
		if index%4==0 and index!=0:
			writer.write('\n')
	elif args.twogpu:
		if index%2==0 and index!=0:
			writer.write('\n')
	else:
		if index%3==0 and index!=0:
			writer.write('\n')
	kwargs=parameter_parser.parse(cfg)
	'''
	split_cfg=cfg.split('_')
	for parameter in split_cfg:
		if 'iter' in parameter:
			kwargs[vocab_sent]['num_iteration']=parameter.split('iter')[0]
			continue
		elif 'unary' in parameter:
			kwargs[hyper_vocab]['unary_hidden']=parameter.split('unary')[0]
			continue
		elif 'token' in parameter:
			kwargs['SemrelGraphTokenVocab']['hidden_size']=parameter.split('token')[0]
			continue
		elif 'binary' in parameter:
			kwargs[hyper_vocab]['hidden_size']=parameter.split('binary')[0]
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
				kwargs[vocab_sent]['tri_std']=parameter.split('init')[0]
				continue
			init_value=parameter.split('init')[0]
			if init_value[0]=='0':
				kwargs[vocab_sent]['tri_std']='.'+parameter.split('init')[0]
				if init_value=='0':
					kwargs[vocab_sent]['tri_std_unary']='.0'
			else:
				kwargs[vocab_sent]['tri_std']=int(parameter.split('init')[0])/10
			continue
		elif 'mu' in parameter:
			#init_value=parameter.split('mu')[0]
			kwargs['Optimizer']['mu']='.'+parameter.split('mu')[0]
			continue
		elif 'nu' in parameter:
			kwargs['Optimizer']['nu']='.'+parameter.split('nu')[0]
	'''
	if '2gpu' in cfg or args.twogpu:
		kwargs['GraphParserNetwork']['two_gpu']=True
	else:
		kwargs['GraphParserNetwork']['two_gpu']=False
	if 'nornn' in cfg:
		kwargs['GraphParserNetwork']['nornn']=True
	else:
		kwargs['GraphParserNetwork']['nornn']=False
	if 'test' in cfg:
		kwargs[vocab_sent]['test_new_potential']=True
		kwargs[vocab_sent]['hidden_k']=kwargs[hyper_vocab]['hidden_size']
	else:
		kwargs[vocab_sent]['test_new_potential']=False
	kwargs['BaseNetwork']['max_steps']=100000
	kwargs['BaseNetwork']['max_steps_without_improvement']=args.stop_iter
	kwargs[vocab_sent]['as_score']=True
	if 'sib_only' in cfg:
		kwargs[vocab_sent]['use_sib']=True
		kwargs[vocab_sent]['use_gp']=False
		kwargs[vocab_sent]['use_cop']=False
	elif 'cop_only' in cfg:
		kwargs[vocab_sent]['use_sib']=False
		kwargs[vocab_sent]['use_gp']=False
		kwargs[vocab_sent]['use_cop']=True
	elif 'gp_only' in cfg:
		kwargs[vocab_sent]['use_sib']=False
		kwargs[vocab_sent]['use_gp']=True
		kwargs[vocab_sent]['use_cop']=False
	elif 'no_cop' in cfg:
		kwargs[vocab_sent]['use_sib']=True
		kwargs[vocab_sent]['use_gp']=True
		kwargs[vocab_sent]['use_cop']=False
	else:
		kwargs[vocab_sent]['use_sib']=True
		kwargs[vocab_sent]['use_gp']=True
		kwargs[vocab_sent]['use_cop']=True
	if '40set' in cfg:
		dset='40'
		rate=4
	elif '70set' in cfg:
		dset='70'
		rate=7
	elif '10set' in cfg:
		dset='10'
		rate=7
	elif 'tiny' in cfg:
		dset=''
		rate=1
	else:
		rate=10
	if 'psd' in cfg:
		parameter='psd'
	elif 'pas' in cfg:
		parameter='pas'
	elif 'ptb' in cfg:
		parameter='ptb'
	elif 'ctb' in cfg:
		parameter='ctb'
	else:
		parameter='dm'
	if 'head_dep' in cfg:
		kwargs[vocab_sent]['separate_embed']=False
	if 'switch' in cfg:
		kwargs['BaseNetwork']['switch_optimizers']=True
		if 'tiny' in cfg:
			kwargs['BaseNetwork']['switch_iter']=500*rate
		else:
			kwargs['BaseNetwork']['switch_iter']=5000
	if '45' in cfg:
		kwargs['DEFAULT']['TREEBANK']=parameter.upper()+'New'+'_'+'45'
	elif 'ptb' in cfg:
		kwargs['DEFAULT']['TREEBANK']=parameter+'_modified'
		kwargs['DEFAULT']['test_conllus']='data/SemEval15/${TREEBANK}/test.${LC}.${TB}.conllu'
	elif 'ctb' in cfg:
		kwargs['PretrainedVocab']={}
		kwargs['FormPretrainedVocab']={}
		kwargs['DEFAULT']['TREEBANK']=parameter
		kwargs['DEFAULT']['test_conllus']='data/SemEval15/${TREEBANK}/test.${LC}.${TB}.conllu'
		kwargs['FormPretrainedVocab']['pretrained_file']='data/cn_vecs/cn_embeddings.txt'
		kwargs['DEFAULT']['lang']='Chinese'
		#kwargs['DEFAULT']['lc']='cn'
		kwargs['PretrainedVocab']['vocab_loadname']=''
		kwargs['FormPretrainedVocab']['name']='Chinese'
	else:
		kwargs['DEFAULT']['TREEBANK']=parameter.upper()+'New'+'_'+'modified'
	kwargs['BaseNetwork']['parse_devset']='False'
	kwargs['DEFAULT']['TB']=parameter
	kwargs['Optimizer']['decay_steps']=10000
	if 'tiny' in cfg:
		#pdb.set_trace()
		try:
			datasetsid=int(cfg.split('tiny')[1][0])
			datasetsid=cfg.split('tiny')[1][0]
		except:
			datasetsid=''
		kwargs['DEFAULT']['TREEBANK']=parameter.upper()+'_tiny'+dset+'_'+args.setname+datasetsid
		kwargs['DEFAULT']['train_conllus']='data/SemEval15/${TREEBANK}/train.${LC}.${TB}.tiny.conllu'
		kwargs['BaseNetwork']['max_steps_without_improvement']=args.stop_iter
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
	if 'no_pos' in cfg:
		input_vocabs=kwargs['GraphParserNetwork']['input_vocab_classes'].split(':')
		input_vocabs.remove('XPOSTokenVocab')
		kwargs['GraphParserNetwork']['input_vocab_classes']=':'.join(input_vocabs)
	# set remove root child to be false
	if 'no_rm' in cfg:
		kwargs[vocab_sent]['remove_root_child']=False
	#pdb.set_trace()
	if 'decay' not in cfg:
		kwargs['Optimizer']['decay_rate']=1
	kwargs['DEFAULT']['modelname']=cfg
	config = Config(config_file=config_file, **kwargs)
	cfg_dir='config_gen/'+cfg+'.cfg'
	with open(cfg_dir, 'w') as f:
		config.write(f)
	if args.nornn or 'nornn' in cfg:
		nornn=' --nornn'
	else:
		nornn=''
	if args.dozat:
		train_case='CUDA_VISIBLE_DEVICES='+str(index%4)+' nohup python3 -u main.py train GraphParserNetwork --config_file '\
		+cfg_dir+nornn+' --noscreen > log/'+cfg+'&'+'\n'
	elif args.twogpu:
		if args.autorun:
			train_case='CUDA_VISIBLE_DEVICES='+str(index%2*2)+','+str(index%2*2+1)+' nohup python3 -u main.py train GraphParserNetwork --nowarning --config_file '\
			+cfg_dir+nornn+' --noscreen > log/'+cfg+'\n'
		else:
			train_case='CUDA_VISIBLE_DEVICES='+str(index%2*2)+','+str(index%2*2+1)+' nohup python3 -u main.py train GraphParserNetwork --config_file '\
			+cfg_dir+nornn+' --noscreen > log/'+cfg+'&'+'\n'
	else:
		train_case='CUDA_VISIBLE_DEVICES='+str(index%3)+',3'' nohup python3 -u main.py train GraphParserNetwork --config_file '\
		+cfg_dir+nornn+' --noscreen > log/'+cfg+'&'+'\n'
	writer.write(train_case)
	index+=1
writer.close()