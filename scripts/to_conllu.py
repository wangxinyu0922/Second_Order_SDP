
def to_conllu(sdp_filename,cpn_filename):
	with open(sdp_filename) as sdp_f:
		with open(cpn_filename) as cpn_f:
			train_conllu_filename = 'train.' + sdp_filename.strip('.sdp')+'.conllu'
			dev_conllu_filename = 'dev.' + sdp_filename.strip('.sdp')+'.conllu'
			with open(train_conllu_filename,'w') as train_conllu_f:
				with open(dev_conllu_filename,'w') as dev_conllu_f:
					pred = set()
					dep = dict()
					words = []
					cnt = 0
					dev = 0
					for line in sdp_f:
						if line == '#SDP 2015\n':
							continue
						line_cpn = cpn_f.readline()
						if '#' == line[0]:
							cnt += 1
							print(line,cnt)
							if line[2:4] == '20':
								dev = 1
								dev_conllu_f.write(line)
							else:
								dev = 0
								train_conllu_f.write(line)

						elif line == '\n':
							if dev == 1:
								conllu_f = dev_conllu_f
							else:
								conllu_f = train_conllu_f
							pred = sorted(list(pred))
							for i,everyword in enumerate(words):
								if (i+1) in dep.keys():
									semheadBuff = []
									for everyPred in dep[(i+1)]:
										pred_idx = everyPred[0]
										pred_label = everyPred[1]
										semheadBuff.append(str(pred[pred_idx])+':'+pred_label)
									everyword[8]='|'.join(semheadBuff)
								conllu_f.write('\t'.join(everyword)+'\n')
							conllu_f.write('\n')
							pred = set()
							dep = dict()
							words = []

						else:
							fields = []
							line = line.strip('\n')
							line = line.split('\t')
							line_cpn = line_cpn.strip('\n')
							line_cpn = line_cpn.split('\t')
							# id, form, lemma, upos, xpos
							fields+=[line[0],line[1],line[2],'_',line[3]]
							# feats,dephead,deprel,semhead,semrel
							fields+=['_',line_cpn[1],line_cpn[2],'_','_']

							# compute semantic dependencies
							if line[5] == '+': # a pred
								pred.add(int(line[0]))

							for everyPred in range(7,len(line)):
								if line[everyPred] != '_':
									# everyPred - 7 : the index of predicate in pred
									if int(line[0]) in dep.keys():
										dep[int(line[0])].append((everyPred-7,line[everyPred]))
									else:
										dep[int(line[0])] = [(everyPred-7,line[everyPred])]

							words.append(fields)

			




if __name__ == '__main__':
	''' translate sdp-format to conllu-format.
	'''
	import sys
	to_conllu(sys.argv[1],sys.argv[2])