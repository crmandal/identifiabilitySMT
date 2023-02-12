from model.haModel import *

def getTimeRange(model):
	timeRange = model.variables['time'].clone()
	return timeRange

def findMode(model, loc):
	for st in model.states:
		if st.mode == loc:
			return st #.clone()
	return None
	
def addTimeVar(model, var):
	timeRange = getTimeRange(model)
	flag = 0
	for varble in model.variables.keys():
		#print(var, varble)
		if(var == varble):
			flag = 1
			break;
	if(flag ==0):
		model.variables.update({var: timeRange})
	#~ else:
		#~ print(var+' variable exists')
		
	timeode = Node('1.0')
	for state in model.states:
		ode = ODE(var, timeode)
		flag = 0
		#~ for od in state.flow:
			#~ print(od.var, str(od)  + ';')	
		for od in state.flow:
			#print('match ',od.var, ode.var)
			if(od.var == ode.var):
				flag = 1
				break
		if(flag == 0):
			#print('no match ', ode.var)
			state.flow.append(ode)
		#~ else:
			#~ print(var+' ode exists')	
	return model
		
def generateTimeCheck(T, delta, depth):
	smt= '(and (<= '
	if(depth > 0):
		smt += '(+ '
		for k in range(depth -1):
			smt += 'time_'+ str(k)+' '
		
		smt += 'time_'+ str(depth-1)+' '
		smt += ' )'			
	else:
		smt += '0'
	#smt += ' ' +str(T)+')'
	smt += '( + '+ str(T) + ' '+ str(delta)+'))'
	smt+= '(>= '
	if(depth > 0):
		smt += '(+ '
		for k in range(depth):
			smt += 'time_'+ str(k)+' '
		smt += 'time_'+ str(depth-1)+' '
		smt += ' )'			
	else:
		smt += 'time_'+ str(depth)+' '
	#smt += ' ' +str(T)+')'	
	smt += '( - '+ str(T) + ' '+ str(delta)+')))'
	return smt
	
def generateTimeRange(model, v, value, i, T, delta):
	smt = '( and '
	var = getVar_t_index(v, i)
	(op, t1, t2) = getTimeandOp(model, value)
	
	op1 = op[0] if len(op)> 1 else '>='
	op2 = op[1] if len(op)> 1 else '<='
	
	T1 = '( + '+ T+ ' '+ t1+ ')'
	T2 = '( + '+ T+ ' '+ t2+ ')'
	#~ if(op1 == '>' or op1 == '>='):
		#~ pd1 = '( - '+ T1 + ' '+ str(delta)+')'
	#~ else:
		#~ pd1 = '( + '+ T1 + ' '+ str(delta)+')'
		
	#~ if(op2 == '>' or op2 == '>='):
		#~ pd2 = '( - '+ T2 + ' '+ str(delta)+')'
	#~ else:
		#~ pd2 = '( + '+ T2 + ' '+ str(delta)+')'
	#~ #print(t1, t2)
	pd1 = T1
	pd2 = T2
	smt+= '( '+op1+' '+var+' '+ pd1+') '+ '( '+op2+' '+var+' '+pd2+')'
	smt += ')'
	#print('generateTimeRange', v, var , )
	return smt

def getTimeandOp(model, value):
	timeRange = getTimeRange(model)
	op = []
	t = []
	if('[' in value and ']' in value):
		t = value.split('[')[1][:-1].split(',')		
		op = ['>=', '<=']		
	elif('[' in value and ')' in value):
		t = value.split('[')[1][:-1].split(',')	
		op = ['>=', '<']							
	elif('(' in value and ')' in value):
		t = value.split('(')[1][:-1].split(',')	
		op = ['>', '<']						
	elif('(' in value and ']' in value):
		t = value.split('(')[1][:-1].split(',')
		op = ['>', '<']					

	t1 = t[0] if len(t)> 1 else str(0)
	t2 = t[1] if len(t)> 1 else timeRange.getrighttPre()
	
	return (op, t1, t2)

def index_at_depth(depth):
	return '_'+ str(depth)
	
def var_t_index(depth):
	return index_at_depth(depth)+ '_t'

def var_0_index(depth):
	return index_at_depth(depth)+ '_0'
	
def getVar_at_depth(var, depth):
	return var + index_at_depth(depth)
	
def getVar_0_index(var, depth):
	return var+ var_0_index(depth)

def getVar_t_index(var, depth):
	return var+ var_t_index(depth)
	
