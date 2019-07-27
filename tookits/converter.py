import pdb
from argparse import ArgumentParser
argparser = ArgumentParser('File printer')
argparser.add_argument('--file', default='multipletrain_gen.sh')
argparser.add_argument('--user', default='wangxy')
args = argparser.parse_args()
filename=args.file.split('.')
writer=open(filename[0]+'_P40.'+filename[1],'w')
f = open(args.file,'r')
flines = f.readlines()
index=-1
user=args.user
startnode=13
if user=='livc':
	running='/public/sist/home/lizhao/.anaconda3/envs/parser/bin/python'
elif user=='lijn':
	running='/public/sist/home/lijn/anaconda2/envs/parser/bin/python'
	startnode=23
else:
	running='/public/sist/home/wangxy1/anaconda3/envs/parser/bin/python'
for line in flines:
	line=line.strip()

	if line=='':
		continue
	if line[0]=='#':
		continue
	index+=1
	tokens=line.split()
	try:
		tokens.pop(1)
	except:
		pdb.set_trace()
	logname=tokens.pop(-1)
	logname=logname.split('/')[1][:-1]
	tokens.pop(-1)
	tokens[1]=running#'/public/sist/home/lizhao/.anaconda3/envs/parser/bin/python'
	tokens[0]='CUDA_VISIBLE_DEVICES='+str(index%4)
	runsent=' '.join(tokens)
	writesent='vsub '+'"'+runsent+'" '+'--name log/'+logname+' --node '+str(startnode+int(index/4))
	#pdb.set_trace()
	#sent="vsub "+"CUDA_VISIBLE_DEVICES="+std(index)+" /public/sist/home/lizhao/.anaconda3/envs/parser2/bin/python -u main.py train GraphParserNetwork --config_file config_gen/dozat_no_lc_dm_new0.cfg --noscreen" --name dozat_no_lc_dm_new0 --node 13
	writer.write(writesent)
	writer.write('\n')

writer.close()