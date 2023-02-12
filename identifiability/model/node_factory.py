from model.node import *
from ha2smt.dRealSMT import *
from ha2smt.utilFunc import *

def to_SMT(node, smtEncode):
	model = smtEncode.getModel()
	path = smtEncode.getPath()
	delta = smtEncode.getPrecision()
	depth = len(path)-1
		
	smt = ''
	if node.is_terminal() :
		# print('to_SMT- is_terminal', node.value)
		if(node.value == 'mode'):
			index = index_at_depth(depth)
		else:
			index = var_t_index(depth)			
		if(not node.is_number() and not node.isLit()):
			smt += node.value+index	
		else: 
			smt += node.value			
			
		#for n in node.operands:
		#	smt += to_SMT(n, smtEncode, i, t, level+1, depth)
	else:	
		# print('to_SMT- is_terminal', node.value, node.operands)
		if(node.value  == '&'):
			smt += '(and '
			for n in node.operands:
				smt2 = to_SMT(n, smtEncode)
				smt += smt2
			smt+= ')'
		elif(node.value  == '|'):
			smt += '(not (and '
			for n in node.operands:
				smt2 = to_SMT(n.negate().to_cnf(), smtEncode)
				smt += smt2
				smt+= '))'
				
		elif(node.value  == '!'):
			#~ smt += '(not '
			#~ for n in node.operands:
				#~ smt += to_SMT(n, depth)
			#~ smt+= ')'
			smt += '(not '
			for n in node.operands:
				smt2 =  to_SMT(n, smtEncode)
				smt += smt2
			smt+= ')'
			
		elif(node.value  == '='):
			smt += '(= '
			n0 = node.operands[0]
			n1 = node.operands[1]
			smt21 = to_SMT(n0, smtEncode)
			smt22 =  to_SMT(n1, smtEncode)
			smt += smt21 + ' '+ smt22
			smt+= ')'
			
		elif(node.value  == '<'):
			smt += '( < '
			n0 = node.operands[0]
			n1 = node.operands[1]
			smt21 = to_SMT(n0, smtEncode)
			smt22 =  to_SMT(n1, smtEncode)
			smt += smt21 + ' '+ smt22
			smt+= ')'
		elif(node.value  == '>'):
			smt += '( > '
			n0 = node.operands[0]
			n1 = node.operands[1]
			smt21 = to_SMT(n0, smtEncode)
			smt22 =  to_SMT(n1, smtEncode)
			smt += smt21 + ' '+ smt22
			smt+= ')'
		elif(node.value  == '<='):
			smt += '(<= '
			n0 = node.operands[0]
			n1 = node.operands[1]
			smt21 = to_SMT(n0, smtEncode)
			smt22 =  to_SMT(n1, smtEncode)
			smt += smt21 + ' '+ smt22
			smt+= ')'
		elif(node.value  == '>='):				
			smt += '(>= '
			n0 = node.operands[0]
			n1 = node.operands[1]
			smt21 = to_SMT(n0, smtEncode)
			smt22 =  to_SMT(n1, smtEncode)
			smt += smt21 + ' '+ smt22
			smt+= ')'
		
		else: 
			#smt += '(and ( = '+getVar_t_index('tm_l'+str(level), i)+ ' '+ getVar_t_index('tm', i)+')'
			if(node.value == 'mode'):
				smt += node.to_prefix(index_at_depth(depth))
			else:
				smt+= node.to_prefix(var_t_index(depth)) #+ ')'
	return smt

def addTimeVariable(smt, var, i):
	timeRange = getTimeRange(smt.model)
	smt.addVariable(var, timeRange)
	smt.addVariable(getVar_0_index(var, i), timeRange)
	smt.addVariable(getVar_t_index(var, i), timeRange)
	ode = ODE(var,Node('1.0'))
	for mode in smt.odes.keys():
		smt.addODE(mode, ode)
