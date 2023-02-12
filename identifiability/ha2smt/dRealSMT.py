from ha2smt.utilFunc import *
import re

class cond:
	def __init__(self, op, var):
		self.op = op
		self.operands = var
		
class SMT:
	def __init__(self, model, p, ltype, delta):
		self.model = model.clone()
		self.path = p
		self.ltype = ltype
		self.precision = delta		
		self.variables = {}
		self.odes = {}
		self.invariants = []
		self.Initials = []
		self.asserts = []
		self.goal = ''
		
	def getModel(self):
		model = self.model.clone()
		return model
	
	def setPath(self, p):
		self.path = p
	
	def getPath(self):
		path = []
		for p in self.path:
			path.append(p)
		return path
	
	def getPrecision(self):
		return self.precision
	
	def getType(self):
		return self.ltype
	
	def setType(self, t):
		self.ltype = t
	
	def setPrecision(self, t):
		self.precision = t
		
	def addVariable(self, var, value):
		if(self.getVariable(var) == None):
			self.variables.update({var:value})
		
	def getVariable(self, var):
		if var in self.variables.keys():
			return var
		else:
			return None
		
	def addODEs(self, mode, odes):
		odes1 = []
		for ode in odes:
			odes1.append(ode.clone())
		#~ for od in odes1:
			#~ print(str(od))		
		self.odes.update({mode:odes1})
	
	def getODEs(self):
		odes1 = {}
		for key in self.odes.keys:
			m = key
			ode = []
			for o in self.odes[key]:
				ode.append(o.clone())
		#~ for od in odes1:
			#~ print(str(od))		
			odes1.update({m:ode})
		return odes1
		
	def addODE(self, mode, ode):
		odes = self.getODE(mode)
		#odes1 = []
		#~ print('before', mode)
		#~ for od in odes:
			#~ print(od.var, str(od))
		#~ #if(ode not in odes):
		flag = 0
		for od in odes:
			#odes1.append(od.clone())
			#print('match ', od.var, ode.var)
			if(od.var == ode.var):
				flag = 1
				break;
		if(flag == 0):	
			#print('no match ', ode.var)
			odes.append(ode.clone())
		#~ print('after', mode)
		#~ for od in odes:
			#~ print(str(od))					
		self.odes.update({mode:odes})
	
	def getODE(self, mode):
		if(mode in self.odes.keys()):
			odes = self.odes[mode]
			odes1 = []
			for ode in odes:
				odes1.append(ode.clone())
			return odes1
		else:
			return []
		
	def addAssert(self, op, var1, var2):
		cond = cond(op, [var1, var2])
		self.asserts.append(cond)
	
		
	def addGoal(self, chk):
		self.goal = chk
		
	#def addModelPath(self, model):
	
	def toString(self, neg = False):
		smt = '; SMT for path '+ str(self.path) +'\n'
		smt += '(set-logic '+self.ltype+')\n'
		smt += '(set-info :precision ' + str(self.precision) +')\n'
		
		for var in self.variables.keys():			
			#print(var, self.variables[var])
			smt += '(declare-fun ' + var + ' () Real '+ self.variables[var].to_prefix() +' )\n'
		
		for mode in self.odes.keys():
			smt += '(define-ode flow_'+ mode +' ('
			variables = ''
			for ode in self.odes[mode]:
				smt += ode.to_prefix() + '\n';
				variables += ode.var+' ';
			smt += '))\n';
			
		for asrt in self.asserts:
			smt  += '(assert ('+op+' '+ asrt.var1 +' ('+ asrt.var2.to_prefix() + ')))\n'
		
		smt += self.generatePATHencoding(neg)			
					
		smt +='(check-sat)\n(exit)\n';
		return smt
	
	def __str__(self):
		smt = '; SMT for path '+ str(self.path) +'\n'
		smt += '(set-logic '+self.ltype+')\n'
		smt += '(set-info :precision ' + str(self.precision) +')\n'
		
		for var in self.variables.keys():			
			#print(var, self.variables[var])
			smt += '(declare-fun ' + var + ' () Real '+ self.variables[var].to_prefix() +' )\n'
		
		for mode in self.odes.keys():
			smt += '(define-ode flow_'+ mode +' ('
			variables = ''
			for ode in self.odes[mode]:
				smt += ode.to_prefix() + '\n';
				variables += ode.var+' ';
			smt += '))\n';
			
		for asrt in self.asserts:
			smt  += '(assert ('+op+' '+ asrt.var1 +' ('+ asrt.var2.to_prefix() + ')))\n'
		
		smt += self.generatePATHencoding()			
		
		smt +='(check-sat)\n(exit)\n';
		return smt

		
	def generatePATHencoding(self, neg = False):	
		#print('generatePATHencoding')
		
		smt = '\n(assert'
		
		if(neg == True):
			smt += ' (not (and \n'
		else:
			smt += ' (and \n'
		#smt += ' (and \n'
		
		smt += ' (and \n'	
		smt += generateInitCondition(self)		
		
		m = len(self.path)
		for i in range(m):		
			loc = self.path[i]
			state = findMode(self.model, loc)
			smt+= '\n; Mode '+ loc+ '\n'
			smt+= '\t(= ' +getVar_at_depth('mode', i)+ ' '+ loc+  ')\n'		
			smt += generateInvariants(self.model, state, i)			
			smt += generateFlows(self, state, i)			
			if i < m-1:
				smt += generateJumps(self, state, i)
			smt += generateConstraints(self.model, state, i)
		
		smt += ')\n'
		
		# print(self.goal)
		# smt +=  self.goal
		
		if(neg == True):
			smt += '( '+ self.goal+' ))\n'
		else:
			smt += '( '+ self.goal+' )\n'	
		
		smt+= '))\n'
		return smt
	
def generateInvariants(model, state, depth):	
	#print('generateInvariants '+state.mode+'\n')
	mode = state.mode
	#smt = ''
	smt = '; generate invariants for mode '+mode+' \n'	
	for inv in state.invariants:
		index0 = var_0_index(depth)
		indext = var_t_index(depth)
		smt +='\t( forall_t '+ mode+ ' [0 '+ getVar_at_depth('time', depth)+']' +'('+ inv.to_prefix(indext)	+'))\n'
		smt+= '\t('+ inv.to_prefix(index0)	+')' +'('+ inv.to_prefix(indext)	+')\n'
	return smt

def generateFlows(smtEncode, state, depth):
	#print('generateFlows'+state.mode)
	#model = smtEncode.getModel()
	mode = state.mode
	odes = smtEncode.getODE(mode)
	variables = ''
	#smt = ''
	smt = '; generate flow for mode '+ mode+' \n'
	for ode in odes:
		variables += ode.var+' ';
	index0 = var_0_index(depth)+' '
	indext = var_t_index(depth)+' '
	smt += '\t(= [' + variables.replace(' ', indext) + '] (integral 0. time_'+str(depth)+' ['\
						+ variables.replace(' ', index0) + '] flow_'+mode+'))\n';
	return smt

def generateJumps(smtEncode, state, i):
	#print('generateJumps'+state.mode)
	model = smtEncode.getModel()
	mode = state.mode
	#smt = ''
	smt = '; generate jumps \n'	
	index0 = var_0_index(i+1)
	indext = var_t_index(i)
	if len(state.jumps) > 1:
		smt += '\t( or \n'
	for jump in state.jumps:
		#smt += '\n; from mode '+ mode+' \n'
		smt += '\t( and\n'	
		#smt += '(= mode_'+str(i)+ ' ' +mode+')\n'
		smt += '\t\t(= mode_'+str(i+1)+ ' ' +jump.toMode+')\n'
		
		for gd in jump.guard:
			smt += '\t\t('+gd.to_prefix(indext)+')\n'		
			#smt +=  '('+(jump.guard)+ ')'
		for reset in jump.reset:
			smt += '\t\t'+ reset.to_prefix(index0, indext)+ '\n'
			#~ smt +=  '\t\t( = '+replace_var_0_index(model, reset.var, i+1)
			#~ expr1 = reset.expr.to_prefix(indext)
			#~ smt+= ' ('+ expr1 +'))\n' 
		for key in smtEncode.variables.keys():			
			#print(var, self.variables[var])
			m1 = bool(re.match(r'tm_l[\d]?_[\d]?_t', key))			
			m2 = bool(re.match(r'tm_[\d]?_t', key))
			md = i+1
			if(m1 or m2):
				md = int(key.split('_')[-2])+1
				if(md <= i+1):
					var = '_'.join(key.split('_')[:-2])+ '_'+str(md)+'_0'				
					smt += '( = ' + var +' '+ key +' ) '
			
		smt += '\t)\n'
	#smt += '\t)\n'				
	if len(state.jumps) > 1:
		smt += '\t)\n' 
	return smt

def generateConstraints(model, state, depth):	
	#print('generateConstraints '+state.mode+'\n')
	mode = state.mode
	#smt = ''
	smt = '; add constraints for mode '+mode+' \n'	
	for inv in model.getConstraints():
		index0 = var_0_index(depth)
		indext = var_t_index(depth)
		smt +='\t( forall_t '+ mode+ ' [0 '+ getVar_at_depth('time', depth)+']' +'('+ inv.to_prefix(indext)	+'))\n'
		smt+= '\t('+ inv.to_prefix(index0)	+')' +'('+ inv.to_prefix(indext)	+')\n'
	return smt
	
def generateInitCondition(smtEncode):
	#print('generateInitCondition')
	model = smtEncode.getModel()
	smt = '\n; initial condition \n'
	smt += '(and '
	smt += '(= '+ getVar_at_depth('mode', 0) +' '+ model.init.mode+') '
	for condition in model.init.condition:
		index = var_0_index(0)
		smt += '('+ condition.to_prefix(index) +') '
	for key in smtEncode.variables.keys():			
		#print(var, self.variables[var])
		m1 = bool(re.match(r'tm_l[\d]?_0_0', key))
		if(m1):
			smt += '( = ' + key +' '+ smtEncode.variables[key].getleftPre() +' ) '
		
	smt += ')\n'
	return smt
