from argparse import ArgumentParser
from tookits.cfg_parser import cfg_parser
import pdb
argparser = ArgumentParser('test sentence generator')
argparser.add_argument('--keyword', default='')
argparser.add_argument('--name', default='')
argparser.add_argument('--cfg', default='')
argparser.add_argument('--dozat', action='store_true')
argparser.add_argument('--test', action='store_true')
argparser.add_argument('--LBP', action='store_true')
argparser.add_argument('--set', default='')
args = argparser.parse_args()
#filePath='./trial_list.txt'

#regs=['3e-6','3e-7','3e-8','3e-9']
#regs=['3e-9','3e-8']
regs=['3e-9']
#regs=[['BaseNetwork']['l2_reg']]
#regs=['3e-9']
#tiny_sets=['tiny1','tiny2','tiny3','tiny4','tiny5']
#tiny_sets=['tiny3','tiny4','tiny5']
#tiny_sets=['tiny1','tiny2','tiny3','tiny4']
tiny_sets=['']
#tiny_sets=['tiny5']
#tiny_sets=['tiny1','tiny3','tiny4']
#iterations=['1','2','3']
iterations=['3']
#iterations=['1','2','3']
#inters=['01','03','05','09','10','11']
#inters=['07']
inters=['07']
#inters=['03','05','07','09','10','11']
#inits=['10','12','15']
#inits=['01','1','0.25','5','0.75','10']
inits=['1','2','0.25','3','4','5']
#inits=['01','05','1','2','3','4','5','0.75','10']
#inits=['1','0.25','0.75','10']
#inits=['1','0.25','5','10']
#inits=['0.25']
#inits=['3']
#datasets=['dm_','pas_','psd_']
#datasets=['dm_']
#datasets=['ptb_','ctb_']
datasets=['ctb_']
augments=['no_lemma_']
#augments=['no_lc_','no_char_','no_lemma_','']
#augments=['no_char_','no_lemma_']
#augments=['no_lc_','']
#augments=['']
#augments=['no_lc_']
#binary=['100','150','200','250','300','350','400','450','500','550','600']
#binary=['100','150','200','250','300']
#binary=['100','200','250','300']
#binary=['200','600']
binary=['150']
#batch_size='3000'
batch_size='6000'

#lrs=['05','01']
#decays=['9','7','5','3','1']
lrs=['01']
decays=['5']
#lrs=['05']
#decays=['7']
if args.cfg!='':
	parameter_parser=cfg_parser()
	
	kwargs=parameter_parser.parse(args.cfg)
	regs=[kwargs['BaseNetwork']['l2_reg']]
	inters=[kwargs['SemrelGraphTokenVocab']['loss_interpolation']]
	for index, inter in enumerate(inters):
		if inter[0]=='.':
			inters[index]=inter[1:]
	binary=[kwargs[parameter_parser.hyper_vocab]['hidden_size']]
	inits=[str(kwargs[parameter_parser.vocab_sent]['tri_std'])]
	for index, init in enumerate(inits):
		if len(init)==3 and init[0:2]=='0.':
			inits[index]=init[2:]
		if len(init)==3 and init[0:1]=='.':
			inits[index]=init[1:]
	iterations=[kwargs[parameter_parser.vocab_sent]['num_iteration']]
	batch_size=kwargs['CoNLLUTrainset']['batch_size']

#pdb.set_trace()
if args.LBP:
	LBP='_LBP'
else:
	LBP=''
if args.test:
	if len(tiny_sets)==1 and tiny_sets[0]=='':
		test='test'
	else:		
		test='_test'
else:
	test=''
if len(inters)==1 and inters[0]=='':
	intersent=''
else:
	intersent='inter_'
startid=50
endid=60
for tiny_set in tiny_sets:
	for reg in regs:
		for lr in lrs:
			for decay in decays:
				if args.dozat:
					for i in range(startid,endid):
						for augment in augments:
							for dataset in datasets:
								modelname='dozat_'+lr+'lr_'+decay+'decay_'+batch_size+'batch_'+reg+'reg_'+augment+dataset+tiny_set+args.set+'new'+str(i)
								print(modelname)
				else:
					'''
					for i in range(2):
						for iteration in iterations:
							for inter in inters:
								for init in inits:
									modelname=iteration+ 'iter_600unary_600token_200binary_6000batch_01lr_adam_5decay_'+reg+'reg_'+init+'init_'+inter+'inter_no_lc_dm_switch_'+tiny_set+'_new'+str(i)
									print(modelname)
					'''
					for i in range(startid,endid):
						for iteration in iterations:
							for init in inits:
								for augment in augments:
									for dataset in datasets:
										for inter in inters:
											for bin in binary:
												modelname=iteration+ 'iter_600unary_600token_'+bin+'binary_'+batch_size+'batch_'+lr+'lr_'+'adam_'+decay+'decay_'+reg+'reg_'+init+'init_'+inter+'inter_'+augment+dataset+'switch_'+tiny_set+args.set+test+LBP+'_new'+str(i)+args.name
												print(modelname)




'''
for i in range(30):
  print('3iter_600unary_600token_200binary_6000batch_01lr_adam_5decay_3e-8reg_5init_10inter_no_lc_psd_switch_new'+str(6+i)+'again')
'''
  
'''
>>> f=open('jobids','r')
>>> data=f.readlines()
>>> for dat in data:
...     print(dat.split()[0],end=' ')
'''