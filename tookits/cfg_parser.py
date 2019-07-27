
class cfg_parser(object):
	"""docstring for cfg_parser"""
	def __init__(self, args=None):
		super(cfg_parser, self).__init__()
		self.args = args
		if self.args==None:
			self.vocab_sent='SecondOrderGraphIndexVocab'
			self.hyper_vocab='SecondOrderVocab'
		elif args.LBP:
			self.vocab_sent='SecondOrderGraphLBPVocab'
			self.hyper_vocab='SecondOrderLBPVocab'
		else:
			self.vocab_sent='SecondOrderGraphIndexVocab'
			self.hyper_vocab='SecondOrderVocab'
	def parse(self,cfg,kwargs=None):
		if kwargs==None:
			kwargs={}
			kwargs[self.hyper_vocab]={}
			kwargs[self.vocab_sent]={}
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
				kwargs[self.vocab_sent]['num_iteration']=parameter.split('iter')[0]
				continue
			elif 'unary' in parameter:
				kwargs[self.hyper_vocab]['unary_hidden']=parameter.split('unary')[0]
				continue
			elif 'token' in parameter:
				kwargs['SemrelGraphTokenVocab']['hidden_size']=parameter.split('token')[0]
				continue
			elif 'binary' in parameter:
				kwargs[self.hyper_vocab]['hidden_size']=parameter.split('binary')[0]
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
					kwargs[self.vocab_sent]['tri_std']=parameter.split('init')[0]
					continue
				init_value=parameter.split('init')[0]
				if init_value[0]=='0':
					kwargs[self.vocab_sent]['tri_std']='.'+parameter.split('init')[0]
					if init_value=='0':
						kwargs[self.vocab_sent]['tri_std_unary']='.0'
				else:
					kwargs[self.vocab_sent]['tri_std']=int(parameter.split('init')[0])/10
				continue
			elif 'mu' in parameter:
				#init_value=parameter.split('mu')[0]
				kwargs['Optimizer']['mu']='.'+parameter.split('mu')[0]
				continue
			elif 'nu' in parameter:
				kwargs['Optimizer']['nu']='.'+parameter.split('nu')[0]

		
		return kwargs