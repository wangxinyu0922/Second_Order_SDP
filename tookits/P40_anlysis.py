from parser.config import Config
from argparse import ArgumentParser
from goodluck.goodluck.main import Luck
import pdb


argparser = ArgumentParser('train sentence generator')
argparser.add_argument('--dozat', action='store_true')
argparser.add_argument('--extra', action='store_true')
argparser.add_argument('--shell', action='store_true')
argparser.add_argument('--runtwo', action='store_true')
argparser.add_argument('--P40', action='store_true')
argparser.add_argument('--self_design', action='store_true')
argparser.add_argument('--cluster', type=str, default='P40')
argparser.add_argument('--fix', type=str, default='',dest='fix_gpu')
argparser.add_argument('--nowarning',action='store_true')
argparser.add_argument('--name',type=str,default='')
argparser.add_argument('--max',type=int,default=10)
argparser.add_argument('--startsize',type=int,default=17000)
argparser.add_argument('--twogpu', action='store_true')
argparser.add_argument('--singlegpu', action='store_true')
argparser.add_argument('--LBP', action='store_true')

#argparser.add_argument('--')
args = argparser.parse_args()


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

class Train_generation:
	def __init__(self, startsize, runfile, args, myusers, cluster='P40',statefile='P40stat.txt', shell_mode=True, self_design=False, banned=[],fix_gpu=False, fix_gpu_list=None):
		self.our_run=self.parse_running_sentence(runfile)
		self.startsize=startsize
		self.args=args
		self.cluster=cluster
		self.shell_mode=shell_mode
		self.myusers=myusers
		self.avail_gpu=0
		if cluster=='P40':
			statefile='P40stat.txt'
			self.gpulist=self.parse_gpu_state(statefile)
			self.write_head='P40_runner'+args.name
			if self.shell_mode:
				self.program=self.create_shell_P40
			else:
				self.program=self.create_template
		if cluster=='AI':
			gputester=Luck()
			self.gpulist, _=gputester.mrun()
			self.write_head='gen_runner'+args.name
			#self.avail_gpu=len(self.gpulist)
			if self.shell_mode:
				self.program=self.create_shell_AI
			else:
				self.program=self.create_template
		if fix_gpu:
			new_gpu_list={}
			for node in self.gpulist:
				if int(node) in fix_gpu_list:
					new_gpu_list[node]=self.gpulist[node]
			self.gpulist=new_gpu_list
		if self_design:
			self.gpulist={}
			# self.gpulist['TitanX']={0:11,1:11}
			#self.gpulist['TitanV']={1:11,2:11}
			#self.gpulist['temp13']={0:11,1:11,2:10,3:10,4:10,5:10,6:10,7:10}
			#self.gpulist['temp14']={2:11,3:11,4:11,5:11,6:11,7:11}
			#self.gpulist['temp26']={1:10,2:10,3:11,4:11,5:10,6:10,7:10}
			#self.gpulist['temp27']={0:11,1:11,3:11,3:11}
			#self.gpulist['temp07']={0:8,1:8,2:8}
			#self.gpulist['temp08']={2:8}
			#self.gpulist['temp10']={0:11,1:11}
			#self.gpulist['temp27']={0:11,1:11,2:11,3:11}
			#self.gpulist['temp32']={1:11}
			#self.gpulist['temp15']={0:8,1:8,2:8,3:8}
			#self.gpulist['temp16']={1:8,2:8}
			#self.gpulist['temp13']={3:11}
			#self.gpulist['temp18']={2:11}
			self.gpulist['temp13']={0:22,1:22,2:22,3:22}
			self.gpulist['temp16']={1:22}
			# self.gpulist['temp15']={2:22,3:22}
			# self.gpulist['temp16']={1:22,3:22}
			self.gpulist['temp17']={0:22,1:22}
			#self.gpulist['temp18']={2:22}
			# self.gpulist['temp23']={0:22,1:22,2:22,3:22}
			self.gpulist['temp25']={0:22,1:22,2:22,3:22}
			self.gpulist['temp26']={0:22,1:22,2:22,3:22}
			self.gpulist['temp27']={0:22,1:22,2:22,3:22}
			self.gpulist['temp28']={0:22,1:22,2:22,3:22}

		if cluster=='P40':
			for gpu in self.gpulist:
				self.avail_gpu+=len(self.gpulist[gpu])
		if cluster=='AI':
			self.analy={}
			for node in self.gpulist:
				if node not in banned:
					self.analy[node]=self.node_analysis(self.gpulist[node])
					self.avail_gpu+=len(self.analy[node][1])+len(self.analy[node][2])+len(self.analy[node][3])
		#pdb.set_trace()
		total_run=len(self.our_run)
		self.each_node=int(total_run/self.avail_gpu)+1

		if self.each_node>self.args.max:
			self.each_node=self.args.max
		if args.dozat and args.runtwo:
			self.tworunning=2
		else:
			self.tworunning=1
		self.god_writer=open(self.write_head+'_node_starter'+'.sh','w')
		self.banned=banned
		if args.dozat:
			self.use_dozat='_dozat'
		else:
			self.use_dozat=''

		
	def parse_gpu_state(self, file):
		sentences=open(file,'r').readlines()
		gpulist={}
		#pdb.set_trace()
		for sentence in sentences:
			sentence=sentence.strip()
			if sentence=='':
				continue
			if sentence[0]=='#':
				continue
			if 'sist-gpu' in sentence:
				blocks=sentence.split()
				nodeid=blocks[0].split('sist-gpu')[1]
				gpulist[nodeid]=[]
				continue
			#pdb.set_trace()
			blocks=sentence.split('|')
			gpuid=blocks[0].split()[0][1]
			memory=int(blocks[2].split()[2])-int(blocks[2].split()[0])
			users=blocks[-1].strip(' ')
			users=users.split()
			flag=0
			for user in users:
				curuser=user.split(':')[0]
				if curuser not in self.myusers:
					flag=1
					break
			if flag==1:
				continue
			if nodeid=='16' and gpuid=='2':
				continue
			if memory>startsize:
				gpulist[nodeid].append(gpuid)
		return gpulist

	def parse_running_sentence(self, file):
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

	def pop_next(self):
		if self.our_run==[]:
			return False
		else:
			#pdb.set_trace()
			sent=self.our_run[0]
			if len(self.our_run)>1:
				self.our_run=self.our_run[1:]
			else:
				self.our_run=[]
		return sent

	def node_analysis(self,gpus):
		types=[]
		memories=[]
		#type 1 can run 1.1gpu, type2 can run dozat or twogpu ours, type3 used for 1.1 gpu, type4 for nothing
		typedicts={}
		typedicts[1]=[]
		typedicts[2]=[]
		typedicts[3]=[]
		typedicts[4]=[]
		for gpu in gpus:
			if gpus[gpu]>=9:
				gputype=1
			elif gpus[gpu]>7:
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

		if self.args.twogpu:
			
			res=list(gpus.keys())
			if self.args.LBP:
				able=[]
				for i in range(len(res)):
					if gpus[res[i]]>=9:
						able.append(res[i])
				if len(able)>=2:
					#pdb.set_trace()
					res=able	
					for i in range(int(len(res)/2)):
						gpuid,another_gpu=str(res[i*2]),str(res[i*2+1])
						analy[2].append([gpuid,another_gpu])	
			else:
				for i in range(int(len(res)/2)):
					gpuid,another_gpu=str(res[i*2]),str(res[i*2+1])
					analy[2].append([gpuid,another_gpu])
			#pdb.set_trace()
			return analy
		if args.dozat:
			#pdb.set_trace()
			if len(typedicts[2])>=1:
				for i in range(len(typedicts[2])):
					analy[1].append(typedicts[2][i])
			if len(typedicts[1])>=1:
				for i in range(len(typedicts[1])):
					analy[1].append(typedicts[1][i])
		elif args.singlegpu:
			#if len(typedicts[2])>=1:
			#	for i in range(len(typedicts[2])):
			#		analy[1].append(typedicts[2][i])
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
				#pdb.set_trace()
			elif len(typedicts[2])>=2:
				gpuid,another_gpu=typedicts[2][0],typedicts[2][1]
				analy[2].append([gpuid,another_gpu])
				if len(typedicts[2])==2:
					pass
				elif len(typedicts[2])==3:
					pass
				else:
					gpuid,another_gpu=typedicts[2][2],typedicts[2][3]
					analy[2].append([gpuid,another_gpu])
			#pdb.set_trace()
		return analy


	def parse_run_config(self, sentence,node,gpu_set):
		#data=sentence.split()
		#pdb.set_trace()
		cases=sentence.split()
		cases[1]='"CUDA_VISIBLE_DEVICES='+gpu_set
		cases[-1]=node
		cases[2]=running
		sentence=' '.join(cases)
		return sentence
	def parse_run_config2(self, sentence,gpu_set):
		#data=sentence.split()
		cases=sentence.split()
		cases[0]='CUDA_VISIBLE_DEVICES='+','.join(gpu_set)
		#pdb.set_trace()
		if args.nowarning:
			cases.insert(-2,'--nowarning')
		sentence=' '.join(cases)
		return sentence
	def create_template(self):
		for node in self.gpulist:
			#total_writer=len(gpulist[node])*2
			#writelist=[open('shell/gen_runner'+'_node'+str(node)+'_'+str(i)+self.use_dozat'.sh','w') for i in range(total_writer)]
			#for k in range(each_node):

			for gpuid in self.gpulist[node]:
				sentence=self.pop_next()
				if sentence==False:
					continue
				train_case=self.parse_run_config(sentence,node,gpuid)
				self.god_writer.write(train_case+'\n')
		self.god_writer.close()
	def create_shell_AI(self):
		runtype=[1,2,3]
		writelister={}
		writerls={}
		for k in range(self.each_node):
			for node in self.gpulist:
				if node in self.banned:
					continue
				if len(self.analy[node][1])==0 and len(self.analy[node][2])==0 and len(self.analy[node][3])==0:
					continue
				if k==0:
					writerls[node]=open('shell/'+self.write_head+'_node'+str(node)+'.sh','w')
					#pdb.set_trace()
					total_writer=len(self.gpulist[node])
					writelister[node]=[open('shell/'+self.write_head+'_node'+str(node)+'_'+str(i)+self.use_dozat+'.sh','w') for i in range(total_writer)]
				writelist=writelister[node]
				writer=writerls[node]
				#pdb.set_trace()
				
				
				#writer.write('#node'+str(node)+'\n')
				#to_train=[]
			
				current=0
				for run in runtype:
					gpu_sets=self.analy[node][run]
					#pdb.set_trace()
					for gpu_set in gpu_sets:
						sentence=self.pop_next()
						if sentence==False:
							continue
						if run==2 or run==3:
							set_config(sentence.split()[8],1-(run-2))
						
						train_case=self.parse_run_config2(sentence,gpu_set)
						if train_case[-1]=='&':
							train_case=train_case[:-1]
						writelist[current].write('echo "'+train_case.split()[-1].split('/')[1]+'"\n')
						writelist[current].write(train_case+'\n')
						current+=1
		for node in self.gpulist:
			if node in self.banned:
					continue
			if len(self.analy[node][1])==0 and len(self.analy[node][2])==0 and len(self.analy[node][3])==0:
					continue
			total_writer=len(self.gpulist[node])
			for i in range(total_writer):
				writelist=writelister[node]
				writer=writerls[node]
				writelist[i].close()
				writer.write('bash '+'shell/'+self.write_head+'_node'+str(node)+'_'+str(i)+self.use_dozat+'.sh&'+'\n')
			writer.close()
			self.god_writer.write('nohup bash '+'shell/'+self.write_head+'_node'+str(node)+'.sh'+' >shell_log/shell'+str(node)+'\n')
	def create_shell_P40(self):
		writelister={}
		writerls={}
		for k in range(self.each_node):
			for node in self.gpulist:
				if k==0:
					#pdb.set_trace()
					writerls[node]=open('shell/'+self.write_head+'_node'+str(node)+'.sh','w')
					#analy=node_analysis(free_node[node])
					total_writer=len(self.gpulist[node])
					writelister[node]=[open('shell/'+self.write_head+'_node'+str(node)+'_'+str(i)+self.use_dozat+'.sh','w') for i in range(total_writer)]
				writelist=writelister[node]
				writer=writerls[node]
				current=0
				for i in range(self.tworunning):
					for gpuid in self.gpulist[node]:
						sentence=self.pop_next()
						if sentence==False:
							continue
						train_case=self.parse_run_config2(sentence,[str(gpuid)])
						#pdb.set_trace()
						if train_case[-1]=='&':
							train_case=train_case[:-1]
						writelist[current].write('echo "'+train_case.split()[-1].split('/')[1]+'"\n')
						writelist[current].write(train_case+'\n')
						current+=1
		for node in self.gpulist:
			writelist=writelister[node]
			writer=writerls[node]
			total_writer=len(self.gpulist[node])
			for i in range(total_writer):
				writelist[i].close()
				writer.write('bash '+'shell/'+self.write_head+'_node'+str(node)+'_'+str(i)+self.use_dozat+'.sh&'+'\n')
			writer.close()
			self.god_writer.write('nohup bash '+'shell/'+self.write_head+'_node'+str(node)+'.sh'+' >shell_log/shell'+str(node)+args.name+'\n')
	def start_create(self):
		self.program()
		self.god_writer.close()
		sentence=self.pop_next()
		print('ours:',sentence)


if not args.shell and args.cluster=='P40':
	if args.dozat:
		runfile='multipletrain_gen_dozat_P40.sh'	
	else:
		runfile='multipletrain_gen_P40.sh'
else:
	
	runfile='multipletrain_gen.sh'

fix_gpu=False
if args.fix_gpu!='':
	fix_gpu_list=args.fix_gpu.split(',')
	fix_gpu_list=[int(x) for x in fix_gpu_list]
	fix_gpu=True
else:
	fix_gpu_list=None

myusers=['wangxy1','lijn']

if args.dozat:
	startsize=6000
else:
	startsize=args.startsize
if args.extra and not args.dozat:
	startsize=22000


analizer=Train_generation(startsize, runfile, args, myusers, cluster=args.cluster,statefile='P40stat.txt', shell_mode=args.shell, self_design=args.self_design, fix_gpu=fix_gpu, fix_gpu_list=fix_gpu_list)
analizer.start_create()

#if args.extra:	
#	running='/public/sist/home/lijn/anaconda2/envs/parser/bin/python'
#else:
#	running='/public/sist/home/wangxy1/anaconda3/envs/parser/bin/python'
#statefile='P40stat.txt'
#gpulist=parse_gpu_state(statefile)
#pdb.set_trace()

#our_run=parse_running_sentence(runfile)

#pdb.set_trace()
#total_run=len(our_run)
#avail_gpu=0

#for gpu in gpulist:
#	avail_gpu+=len(gpulist[gpu])
'''
if not args.shell:
	each_node=int(total_run/avail_gpu)
	for node in gpulist:
		#total_writer=len(gpulist[node])*2
		#writelist=[open('shell/gen_runner'+'_node'+str(node)+'_'+str(i)+'.sh','w') for i in range(total_writer)]
		#for k in range(each_node):

		for gpuid in gpulist[node]:
			sentence,our_run=pop_next(our_run)
			if sentence==False:
				continue
			train_case=parse_run_config(sentence,node,gpuid)
			god_writer.write(train_case+'\n')
	god_writer.close()
else:
	each_node=int(total_run/avail_gpu)+1
	for node in gpulist:
		#pdb.set_trace()	
		if int(node)>18:	
			running='/public/sist/home/lijn/anaconda2/envs/parser/bin/python'
		else:
			running='/public/sist/home/wangxy1/anaconda3/envs/parser/bin/python'
		writer=open('shell/P40_runner'+'_node'+str(node)+'.sh','w')
		#analy=node_analysis(free_node[node])
		total_writer=len(gpulist[node])
		writelist=[open('shell/P40_runner'+'_node'+str(node)+'_'+str(i)+'.sh','w') for i in range(total_writer)]
		#if node in banned:
		#	continue
		writer.write('#node'+str(node)+'\n')
		#to_train=[]
		for k in range(each_node):
			current=0
			for i in range(tworunning):
				for gpuid in gpulist[node]:
					sentence,our_run=pop_next(our_run)
					if sentence==False:
						continue
					train_case=parse_run_config2(sentence,gpuid)
					#pdb.set_trace()
					if train_case[-1]=='&':
						train_case=train_case[:-1]
					writelist[current].write('echo "'+train_case.split()[-1].split('/')[1]+'"\n')
					writelist[current].write(train_case+'\n')
					current+=1
		for i in range(total_writer):
			writelist[i].close()
			writer.write('bash '+'shell/P40_runner'+'_node'+str(node)+'_'+str(i)+'.sh&'+'\n')
		writer.close()
		god_writer.write('nohup bash '+'shell/P40_runner'+'_node'+str(node)+'.sh'+' >shell_log/shell'+str(node)+'\n')
god_writer.close()
#sentence,dozat_run=pop_next(dozat_run)
#print('dozat:',sentence)
sentence,our_run=pop_next(our_run)
print('ours:',sentence)
'''