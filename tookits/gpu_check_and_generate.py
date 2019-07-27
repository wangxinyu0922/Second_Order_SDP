from goodluck.goodluck.main import Luck
from parser.config import Config
from argparse import ArgumentParser
import pdb
argparser = ArgumentParser('train sentence generator')
argparser.add_argument('--dozat', action='store_true')
argparser.add_argument('--shell', action='store_true')
argparser.add_argument('--self_design', action='store_true')
argparser.add_argument('--fix', type=str, default='',dest='fix_gpu')
#argparser.add_argument('--extra', action='store_true')
args = argparser.parse_args()
#pdb.set_trace()
fix_gpu=False
if args.fix_gpu!='':
	fix_gpu_list=args.fix_gpu.split(',')
	fix_gpu_list=[int(x) for x in fix_gpu_list]
	fix_gpu=True
def node_analysis(gpus):
	types=[]
	memories=[]
	#type 1 can run 1.1gpu, type2 can run dozat or twogpu ours, type3 used for 1.1 gpu, type4 for nothing
	typedicts={}
	typedicts[1]=[]
	typedicts[2]=[]
	typedicts[3]=[]
	typedicts[4]=[]
	for gpu in gpus:
		if gpus[gpu]>11:
			gputype=1
		elif gpus[gpu]>7.5:
			gputype=2
		elif gpus[gpu]>4:
			gputype=3
		else:
			gputype=4
		typedicts[gputype].append(str(gpu))
		#if typedicts[gputype].append(gpu):
		#types.append(gputype)
		#memories.append(gpus[gpu])
	analy={}
	analy[1]=[]
	analy[2]=[]
	analy[3]=[]
	if args.dozat:
		if len(typedicts[2])>=1:
			for i in range(len(typedicts[2])):
				analy[1].append(typedicts[2][i])
		if len(typedicts[1])>=1:
			for i in range(len(typedicts[1])):
				analy[1].append(typedicts[1][i])
	else:
		if len(typedicts[1])>=1:
			if len(typedicts[2])>0:
				another_gpu=typedicts[2][0]
			elif len(typedicts[3])>0:
				another_gpu=typedicts[3][0]	
			else:
				another_gpu=typedicts[1][0]
			for gpuid in typedicts[1]:
				if gpuid==another_gpu:
					continue
				analy[3].append([gpuid,another_gpu])
		elif len(typedicts[2])>=2:
			gpuid,another_gpu=typedicts[2][0],typedicts[2][1]
			analy[2].append([gpuid,another_gpu])
			if len(typedicts[2])==2:
				pass
			elif len(typedicts[2])==3:
				analy[1].append(typedicts[2][2])
			else:
				gpuid,another_gpu=typedicts[2][2],typedicts[2][3]
				analy[2].append([gpuid,another_gpu])
		elif len(typedicts[2])==1:
			analy[1].append(typedicts[2][0])
		#pdb.set_trace()
	return analy

def set_config(config_file, two_gpu=False):
	kwargs={}
	kwargs['GraphParserNetwork']={}
	if two_gpu:
		kwargs['GraphParserNetwork']['two_gpu']=True
	else:
		kwargs['GraphParserNetwork']['two_gpu']=False
	config = Config(config_file=config_file, **kwargs)
	with open(config_file, 'w') as f:
		config.write(f)

def parse_running_sentence(file):
	sentences=open(file,'r').readlines()
	to_run=[]
	for sentence in sentences:
		sentence=sentence.strip()
		if sentence=='':
			continue
		if sentence[0]=='#':
			continue
		to_run.append(sentence)
	return to_run

def pop_next(to_run):
	if to_run==[]:
		return False,[]
	else:
		sent=to_run[0]
		if len(to_run)>1:
			to_run=to_run[1:]
		else:
			to_run=[]
	return sent,to_run
def parse_run_config(sentence,gpu_set):
	#data=sentence.split()
	cases=sentence.split()
	cases[0]='CUDA_VISIBLE_DEVICES='+','.join(gpu_set)
	sentence=' '.join(cases)
	return sentence
ourfile='multipletrain_gen.sh'
our_run=parse_running_sentence(ourfile)
#our_iter=0
dozatfile='multipletrain_gen_dozat.sh'
dozat_run=parse_running_sentence(dozatfile)
#dozat_iter=0
gputester=Luck()
free_node, qualified_gpu_list=gputester.mrun()
#1:dozat, 2:ours twogpu, 3:ours 1.1gpu
runtype=[1,2,3]

if args.dozat:
	use_dozat='_dozat'
else:
	use_dozat=''
banned=['13','14','15','16','23']
#TODO: automaticly comment train files
if fix_gpu:
	new_free_node={}
	for node in free_node:
		if int(node) in fix_gpu_list:
			new_free_node[node]=free_node[node]
	free_node=new_free_node
#pdb.set_trace()
if args.self_design:
	#pdb.set_trace()
	free_node={}
	free_node['temp1']={1:11}
	free_node['temp2']={2:11}
if not args.shell:
	writer=open('gen_runner.sh','w')
	for node in free_node:
		if node in banned:
			continue
		writer.write('#node'+str(node)+'\n')
		analy=node_analysis(free_node[node])
		#to_train=[]
		for run in runtype:
			gpu_sets=analy[run]
			#pdb.set_trace()
			for gpu_set in gpu_sets:
				if run==1:
					sentence,dozat_run=pop_next(dozat_run)
					if sentence==False:
						continue
					train_case=parse_run_config(sentence,gpu_set)
				elif run==2 or run==3:
					sentence,our_run=pop_next(our_run)
					if sentence==False:
						continue
					set_config(sentence.split()[8],1-(run-2))
					train_case=parse_run_config(sentence,gpu_set)

				writer.write(train_case+'\n')
	writer.close()
else:

	total_run=len(our_run)
	each_node=int(total_run/len(free_node))
	if each_node>10:
		each_node=10
	god_writer=open('gen_runner'+'_node_starter'+'.sh','w')
	for node in free_node:
		#pdb.set_trace()	
		writer=open('shell/gen_runner'+'_node'+str(node)+'.sh','w')
		analy=node_analysis(free_node[node])
		if len(analy[1])==0 and len(analy[2])==0 and len(analy[3])==0:
			continue
		#pdb.set_trace()
		total_writer=len(analy[1])+len(analy[2])+len(analy[3])
		writelist=[open('shell/gen_runner'+'_node'+str(node)+'_'+str(i)+use_dozat+'.sh','w') for i in range(total_writer)]
		if node in banned:
			continue
		writer.write('#node'+str(node)+'\n')
		#to_train=[]
		for k in range(each_node):
			current=0
			for run in runtype:
				gpu_sets=analy[run]
				#pdb.set_trace()
				for gpu_set in gpu_sets:
					if run==1:
						sentence,dozat_run=pop_next(dozat_run)
						if sentence==False:
							continue
						train_case=parse_run_config(sentence,gpu_set)
					elif run==2 or run==3:
						sentence,our_run=pop_next(our_run)
						if sentence==False:
							continue
						set_config(sentence.split()[8],1-(run-2))
						train_case=parse_run_config(sentence,gpu_set)
					if train_case[-1]=='&':
						train_case=train_case[:-1]
					writelist[current].write('echo "'+train_case.split()[-1].split('/')[1]+'"\n')
					writelist[current].write(train_case+'\n')
					current+=1
		for i in range(total_writer):
			writelist[i].close()
			writer.write('bash '+'shell/gen_runner'+'_node'+str(node)+'_'+str(i)+use_dozat+'.sh&'+'\n')
		writer.close()
		god_writer.write('nohup bash '+'shell/gen_runner'+'_node'+str(node)+'.sh'+' >shell_log/shell'+str(node)+use_dozat+'\n')
god_writer.close()
sentence,dozat_run=pop_next(dozat_run)
print('dozat:',sentence)
sentence,our_run=pop_next(our_run)
print('ours:',sentence)



