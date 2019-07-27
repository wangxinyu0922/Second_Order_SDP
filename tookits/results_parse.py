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
argparser.add_argument('--top5', action='store_true')

args = argparser.parse_args()
keyword=args.keyword
cfg=args.cfg
results={}
f = open('evaluate_results.txt','r')
#writer=open('multipletest_gen_dozat.sh','w')
import pdb
res=f.readlines()
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
	try:
		if args.unlabel:
			id=data[-8]
			ood=data[-3]
			dev=data[3]
		else:
			id=data[-6]
			ood=data[-1]
			dev=data[5]
	except:
		continue
	
	#modelname=data[0].split('_')[10]
	#modelname=data[0].split('GraphParserNetwork')[-1]
	#modelname=data[0].split('_')[10]
	if 'dozat' in data[0]:
		#pdb.set_trace()
		#modelname='dozat_01lr_5decay_6000batch_3e-8reg'
		modelname='_'.join(data[0].split('_')[:5])
	elif 'iter' in data[0]:
		#pdb.set_trace()
		modelname='_'.join(data[0].split('_')[:11])
		#modelname='3iter_600unary_600token_200binary_6000batch_01lr_adam_5decay_3e-8reg_5init_10inter'
		#modelname='3iter_600unary_600token_200binary_6000batch_01lr_adam_5decay_3e-8reg_5init_10inter'
	else:
		continue

	#pdb.set_trace()
	if 'test' in data[0]:
		modelname=modelname+'_test'
	if 'tiny' in data[0] and not args.byset:
		try:
			datasetsid=int(data[0].split('tiny')[1][0])
			datasetsid=data[0].split('tiny')[1][0]
		except:
			datasetsid=''
		modelname+='_tiny'+datasetsid
	#modelname=data[0].split('_')[0]
	#pdb.set_trace()
	if 'switch' in line: modelname=modelname+'_switch'
	if 'LBP' in data[0]: modelname=modelname+'_LBP'
	if '10set' in data[0]: modelname=modelname+'_10set'
	if '40set' in data[0]: modelname=modelname+'_40set'
	if '70set' in data[0]: modelname=modelname+'_70set'
	if modelname not in results:
		results[modelname]={}
		results[modelname]['psd']={}
		results[modelname]['pas']={}
		results[modelname]['dm']={}
		results[modelname]['psd_id']={}
		results[modelname]['psd_ood']={}
		results[modelname]['dm_id']={}
		results[modelname]['dm_ood']={}
		results[modelname]['pas_id']={}
		results[modelname]['pas_ood']={}
		results[modelname]['dm_dev']={}
		results[modelname]['pas_dev']={}
		results[modelname]['psd_dev']={}
		
		results[modelname]['dm_dev']['basic_best']=[]
		results[modelname]['dm_dev']['no_char_best']=[]
		results[modelname]['dm_dev']['no_lemma_best']=[]
		results[modelname]['dm_dev']['all_best']=[]
		results[modelname]['pas_dev']['basic_best']=[]
		results[modelname]['pas_dev']['no_char_best']=[]
		results[modelname]['pas_dev']['no_lemma_best']=[]
		results[modelname]['pas_dev']['all_best']=[]
		results[modelname]['psd_dev']['basic_best']=[]
		results[modelname]['psd_dev']['no_char_best']=[]
		results[modelname]['psd_dev']['no_lemma_best']=[]
		results[modelname]['psd_dev']['all_best']=[]

		results[modelname]['psd_id']['basic']=0
		results[modelname]['psd_id']['no_char']=0
		results[modelname]['psd_id']['no_lemma']=0
		results[modelname]['psd_id']['all']=0
		results[modelname]['pas_id']['basic']=0
		results[modelname]['pas_id']['no_char']=0
		results[modelname]['pas_id']['no_lemma']=0
		results[modelname]['pas_id']['all']=0
		results[modelname]['dm_id']['basic']=0
		results[modelname]['dm_id']['no_char']=0
		results[modelname]['dm_id']['no_lemma']=0
		results[modelname]['dm_id']['all']=0
		results[modelname]['psd_ood']['basic']=0
		results[modelname]['psd_ood']['no_char']=0
		results[modelname]['psd_ood']['no_lemma']=0
		results[modelname]['psd_ood']['all']=0
		results[modelname]['pas_ood']['basic']=0
		results[modelname]['pas_ood']['no_char']=0
		results[modelname]['pas_ood']['no_lemma']=0
		results[modelname]['pas_ood']['all']=0
		results[modelname]['dm_ood']['basic']=0
		results[modelname]['dm_ood']['no_char']=0
		results[modelname]['dm_ood']['no_lemma']=0
		results[modelname]['dm_ood']['all']=0
		results[modelname]['dm_dev']['basic']=0
		results[modelname]['dm_dev']['no_char']=0
		results[modelname]['dm_dev']['no_lemma']=0
		results[modelname]['dm_dev']['all']=0
		results[modelname]['pas_dev']['basic']=0
		results[modelname]['pas_dev']['no_char']=0
		results[modelname]['pas_dev']['no_lemma']=0
		results[modelname]['pas_dev']['all']=0
		results[modelname]['psd_dev']['basic']=0
		results[modelname]['psd_dev']['no_char']=0
		results[modelname]['psd_dev']['no_lemma']=0
		results[modelname]['psd_dev']['all']=0
		if args.avg:
			results[modelname]['psd_id']['basic'+'count']=np.array([])
			results[modelname]['psd_id']['no_char'+'count']=np.array([])
			results[modelname]['psd_id']['no_lemma'+'count']=np.array([])
			results[modelname]['psd_id']['all'+'count']=np.array([])
			results[modelname]['pas_id']['basic'+'count']=np.array([])
			results[modelname]['pas_id']['no_char'+'count']=np.array([])
			results[modelname]['pas_id']['no_lemma'+'count']=np.array([])
			results[modelname]['pas_id']['all'+'count']=np.array([])
			results[modelname]['dm_id']['basic'+'count']=np.array([])
			results[modelname]['dm_id']['no_char'+'count']=np.array([])
			results[modelname]['dm_id']['no_lemma'+'count']=np.array([])
			results[modelname]['dm_id']['all'+'count']=np.array([])
			results[modelname]['psd_ood']['basic'+'count']=np.array([])
			results[modelname]['psd_ood']['no_char'+'count']=np.array([])
			results[modelname]['psd_ood']['no_lemma'+'count']=np.array([])
			results[modelname]['psd_ood']['all'+'count']=np.array([])
			results[modelname]['pas_ood']['basic'+'count']=np.array([])
			results[modelname]['pas_ood']['no_char'+'count']=np.array([])
			results[modelname]['pas_ood']['no_lemma'+'count']=np.array([])
			results[modelname]['pas_ood']['all'+'count']=np.array([])
			results[modelname]['dm_ood']['basic'+'count']=np.array([])
			results[modelname]['dm_ood']['no_char'+'count']=np.array([])
			results[modelname]['dm_ood']['no_lemma'+'count']=np.array([])
			results[modelname]['dm_ood']['all'+'count']=np.array([])
			results[modelname]['dm_dev']['basic'+'count']=np.array([])
			results[modelname]['dm_dev']['no_char'+'count']=np.array([])
			results[modelname]['dm_dev']['no_lemma'+'count']=np.array([])
			results[modelname]['dm_dev']['all'+'count']=np.array([])
			results[modelname]['pas_dev']['basic'+'count']=np.array([])
			results[modelname]['pas_dev']['no_char'+'count']=np.array([])
			results[modelname]['pas_dev']['no_lemma'+'count']=np.array([])
			results[modelname]['pas_dev']['all'+'count']=np.array([])
			results[modelname]['psd_dev']['basic'+'count']=np.array([])
			results[modelname]['psd_dev']['no_char'+'count']=np.array([])
			results[modelname]['psd_dev']['no_lemma'+'count']=np.array([])
			results[modelname]['psd_dev']['all'+'count']=np.array([])
		if args.std:
			results[modelname]['psd_id']['basic'+'list']=np.array([])
			results[modelname]['psd_id']['no_char'+'list']=np.array([])
			results[modelname]['psd_id']['no_lemma'+'list']=np.array([])
			results[modelname]['psd_id']['all'+'list']=np.array([])
			results[modelname]['pas_id']['basic'+'list']=np.array([])
			results[modelname]['pas_id']['no_char'+'list']=np.array([])
			results[modelname]['pas_id']['no_lemma'+'list']=np.array([])
			results[modelname]['pas_id']['all'+'list']=np.array([])
			results[modelname]['dm_id']['basic'+'list']=np.array([])
			results[modelname]['dm_id']['no_char'+'list']=np.array([])
			results[modelname]['dm_id']['no_lemma'+'list']=np.array([])
			results[modelname]['dm_id']['all'+'list']=np.array([])
			results[modelname]['psd_ood']['basic'+'list']=np.array([])
			results[modelname]['psd_ood']['no_char'+'list']=np.array([])
			results[modelname]['psd_ood']['no_lemma'+'list']=np.array([])
			results[modelname]['psd_ood']['all'+'list']=np.array([])
			results[modelname]['pas_ood']['basic'+'list']=np.array([])
			results[modelname]['pas_ood']['no_char'+'list']=np.array([])
			results[modelname]['pas_ood']['no_lemma'+'list']=np.array([])
			results[modelname]['pas_ood']['all'+'list']=np.array([])
			results[modelname]['dm_ood']['basic'+'list']=np.array([])
			results[modelname]['dm_ood']['no_char'+'list']=np.array([])
			results[modelname]['dm_ood']['no_lemma'+'list']=np.array([])
			results[modelname]['dm_ood']['all'+'list']=np.array([])
			results[modelname]['dm_dev']['basic'+'list']=np.array([])
			results[modelname]['dm_dev']['no_char'+'list']=np.array([])
			results[modelname]['dm_dev']['no_lemma'+'list']=np.array([])
			results[modelname]['dm_dev']['all'+'list']=np.array([])
			results[modelname]['pas_dev']['basic'+'list']=np.array([])
			results[modelname]['pas_dev']['no_char'+'list']=np.array([])
			results[modelname]['pas_dev']['no_lemma'+'list']=np.array([])
			results[modelname]['pas_dev']['all'+'list']=np.array([])
			results[modelname]['psd_dev']['basic'+'list']=np.array([])
			results[modelname]['psd_dev']['no_char'+'list']=np.array([])
			results[modelname]['psd_dev']['no_lemma'+'list']=np.array([])
			results[modelname]['psd_dev']['all'+'list']=np.array([])
	if args.avg:
		results[modelname][dataset+'_dev'][target+'count']=np.append(results[modelname][dataset+'_dev'][target+'count'],float(dev))
		results[modelname][dataset+'_id'][target+'count']=np.append(results[modelname][dataset+'_id'][target+'count'],float(id))
		results[modelname][dataset+'_ood'][target+'count']=np.append(results[modelname][dataset+'_ood'][target+'count'],float(ood))
		results[modelname][dataset+'_dev'][target+'_best'].append(data[0])
		#pdb.set_trace()
		if args.top5:
			indices=np.argsort(results[modelname][dataset+'_dev'][target+'count'])[::-1][:5]
			results[modelname][dataset+'_dev'][target]=np.sum(results[modelname][dataset+'_dev'][target+'count'][indices])/len(results[modelname][dataset+'_dev'][target+'count'][indices])
			results[modelname][dataset+'_id'][target]=np.sum(results[modelname][dataset+'_id'][target+'count'][indices])/len(results[modelname][dataset+'_id'][target+'count'][indices])
			results[modelname][dataset+'_ood'][target]=np.sum(results[modelname][dataset+'_ood'][target+'count'][indices])/len(results[modelname][dataset+'_ood'][target+'count'][indices])
			#np.argsort()
			pass
		else:
			results[modelname][dataset+'_dev'][target]=np.sum(results[modelname][dataset+'_dev'][target+'count'])/len(results[modelname][dataset+'_dev'][target+'count'])
			results[modelname][dataset+'_id'][target]=np.sum(results[modelname][dataset+'_id'][target+'count'])/len(results[modelname][dataset+'_id'][target+'count'])
			results[modelname][dataset+'_ood'][target]=np.sum(results[modelname][dataset+'_ood'][target+'count'])/len(results[modelname][dataset+'_ood'][target+'count'])
		# results[modelname][dataset+'_dev'][target]=(results[modelname][dataset+'_dev'][target]*results[modelname][dataset+'_dev'][target+'count']+float(dev))/(results[modelname][dataset+'_dev'][target+'count']+1)
		# results[modelname][dataset+'_id'][target]=(results[modelname][dataset+'_id'][target]*results[modelname][dataset+'_id'][target+'count']+float(id))/(results[modelname][dataset+'_id'][target+'count']+1)
		# results[modelname][dataset+'_ood'][target]=(results[modelname][dataset+'_ood'][target]*results[modelname][dataset+'_ood'][target+'count']+float(ood))/(results[modelname][dataset+'_ood'][target+'count']+1)
		# results[modelname][dataset+'_dev'][target+'count']+=1
		# results[modelname][dataset+'_id'][target+'count']+=1
		# results[modelname][dataset+'_ood'][target+'count']+=1
	elif args.std:
		results[modelname][dataset+'_dev'][target+'list'].append(float(dev))
		results[modelname][dataset+'_id'][target+'list'].append(float(id))
		results[modelname][dataset+'_ood'][target+'list'].append(float(ood))
		#results[modelname][dataset+'_dev'][target+'_best']=data[0]
	elif results[modelname][dataset+'_dev'][target]<float(dev):
		results[modelname][dataset+'_dev'][target]=float(dev)
		results[modelname][dataset+'_id'][target]=float(id)
		results[modelname][dataset+'_ood'][target]=float(ood)
		results[modelname][dataset+'_dev'][target+'_best']=data[0]
#pdb.set_trace()
for modelname in results:
	if results[modelname]['pas_id']['basic']==0 and args.avoid:
		continue
	if args.best:
		#pdb.set_trace()
		#if cfg=='':
		#	print('Current model: ',modelname)
		indices=np.argsort(results[modelname][dataset+'_dev'][target+'count'])[::-1][:5]
		for id in indices:
			print(results[modelname]['dm_dev']['basic_best'][id])
			#print(results[modelname]['dm_dev']['no_char_best'])
			#print(results[modelname]['dm_dev']['no_lemma_best'])
			#print(results[modelname]['dm_dev']['all_best'])
			print(results[modelname]['pas_dev']['basic_best'][id])
			#print(results[modelname]['pas_dev']['no_char_best'])
			#print(results[modelname]['pas_dev']['no_lemma_best'])
			#print(results[modelname]['pas_dev']['all_best'])
			print(results[modelname]['psd_dev']['basic_best'][id])
		#print(results[modelname]['psd_dev']['no_char_best'])
		#print(results[modelname]['psd_dev']['no_lemma_best'])
		#print(results[modelname]['psd_dev']['all_best'])
	elif args.std:
		print(modelname,'DM_id','DM_ood','PAS_id','PAS_ood','PSD_id','PSD_ood')
		print('basic',np.std(results[modelname]['dm_id']['basic'+'list']),np.std(results[modelname]['dm_ood']['basic'+'list']),np.std(results[modelname]['pas_id']['basic'+'list']),np.std(results[modelname]['pas_ood']['basic'+'list']),np.std(results[modelname]['psd_id']['basic'+'list']),np.std(results[modelname]['psd_ood']['basic'+'list']))
		print('+char',np.std(results[modelname]['dm_id']['no_lemma'+'list']),np.std(results[modelname]['dm_ood']['no_lemma'+'list']),np.std(results[modelname]['pas_id']['no_lemma'+'list']),np.std(results[modelname]['pas_ood']['no_lemma'+'list']),np.std(results[modelname]['psd_id']['no_lemma'+'list']),np.std(results[modelname]['psd_ood']['no_lemma'+'list']))
		print('+lemma',np.std(results[modelname]['dm_id']['no_char'+'list']),np.std(results[modelname]['dm_ood']['no_char'+'list']),np.std(results[modelname]['pas_id']['no_char'+'list']),np.std(results[modelname]['pas_ood']['no_char'+'list']),np.std(results[modelname]['psd_id']['no_char'+'list']),np.std(results[modelname]['psd_ood']['no_char'+'list']))
		print('all',np.std(results[modelname]['dm_id']['all'+'list']),np.std(results[modelname]['dm_ood']['all'+'list']),np.std(results[modelname]['pas_id']['all'+'list']),np.std(results[modelname]['pas_ood']['all'+'list']),np.std(results[modelname]['psd_id']['all'+'list']),np.std(results[modelname]['psd_ood']['all'+'list']))
	else:
		print(modelname,'DM_id','DM_ood','PAS_id','PAS_ood','PSD_id','PSD_ood')
		print('basic',results[modelname]['dm_id']['basic'],results[modelname]['dm_ood']['basic'],results[modelname]['pas_id']['basic'],results[modelname]['pas_ood']['basic'],results[modelname]['psd_id']['basic'],results[modelname]['psd_ood']['basic'])
		print('+char',results[modelname]['dm_id']['no_lemma'],results[modelname]['dm_ood']['no_lemma'],results[modelname]['pas_id']['no_lemma'],results[modelname]['pas_ood']['no_lemma'],results[modelname]['psd_id']['no_lemma'],results[modelname]['psd_ood']['no_lemma'])
		print('+lemma',results[modelname]['dm_id']['no_char'],results[modelname]['dm_ood']['no_char'],results[modelname]['pas_id']['no_char'],results[modelname]['pas_ood']['no_char'],results[modelname]['psd_id']['no_char'],results[modelname]['psd_ood']['no_char'])
		print('all',results[modelname]['dm_id']['all'],results[modelname]['dm_ood']['all'],results[modelname]['pas_id']['all'],results[modelname]['pas_ood']['all'],results[modelname]['psd_id']['all'],results[modelname]['psd_ood']['all'])
	if args.printcount:
		print(modelname,'DM_id','DM_ood','PAS_id','PAS_ood','PSD_id','PSD_ood')
		print('basic',len(results[modelname]['dm_id']['basic'+'count']),len(results[modelname]['pas_id']['basic'+'count']),len(results[modelname]['psd_id']['basic'+'count']))
		print('+char',len(results[modelname]['dm_id']['no_lemma'+'count']),len(results[modelname]['pas_id']['no_lemma'+'count']),len(results[modelname]['psd_id']['no_lemma'+'count']))
		print('+lemma',len(results[modelname]['dm_id']['no_char'+'count']),len(results[modelname]['pas_id']['no_char'+'count']),len(results[modelname]['psd_id']['no_char'+'count']))
		print('all',len(results[modelname]['dm_id']['all'+'count']),len(results[modelname]['pas_id']['all'+'count']),len(results[modelname]['psd_id']['all'+'count']))
	#pdb.set_trace()

