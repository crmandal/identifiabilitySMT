import getopt
from collections import OrderedDict
#from ha.util.exprEval import *

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#from util.exprEval import *
from util.graph import *
from model.range import *
from model.condition import *
from model.node import *
from model.haModel import *
from model.phaModel import *
from parser.parseModel import *

def convertHA2PHA(ha, params):
	ha1 = ha.clone()
	macros = ha1.macros
	variables = ha1.variables	
	states = ha1.states
	init = ha1.init
	goals = ha1.goals
	constraints = ha1.constraints
	
	parameters = {}
	for it in params:
		rng = params[it].clone()
		parameters.update({it: rng})
		
	phaStates = []
	for state in states:
		pflows = []
		for ode in state.flow:
			pf = ode.clone()
			pflows.append(pf)
			
		#for param in parameters.keys():
		#	pf = ODE(param, Node('0'))
		#	pflows.append(pf)
			
		pmode = state.mode #.clone()
		
		invariants = []
		for invt in state.invariants:
			invariants.append(invt.clone())

		jumps = []
		for jump in state.jumps:
			jumps.append(jump.clone())

		phaState = State(state.mode, invariants, pflows, jumps)
		phaStates.append(phaState)
		
	pha = PHA(macros, parameters, variables, phaStates, init, goals) 
	pha.addConstraints(constraints)
	
	return pha

def replaceMacros(model):
	model2 = model.clone()
	for key in model2.macros.keys():
		if(key is None):
			continue;
		value = model2.macros[key]
		for key1 in model2.macros.keys():
			#val = model2.macros[key]	
			expr = model2.macros[key1]
			#print('replace ', key, value.to_infix())
			expr1 = expr.replace(key, value)
			#print(key1, expr.to_infix(), key1, expr1.to_infix())
			
			model2.macros[key1]= expr1 
			#model2.macros[key1]= model2.macros[key1].replace(key, val)

	model3 = model2.clone()
	model3.macros = {}
	st = str(model3)
	for key in model2.macros.keys():
		if(key is None):
			continue
		#print(key)
		value = str(eval(model2.macros[key].to_infix()))
		st = st.replace(key, value)
	#print(st)
	#with open('temp_model.drh', 'w') as of:
	#	of.write(st)
	model1 = getHA(st)
	return model1
