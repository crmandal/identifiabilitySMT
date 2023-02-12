from __future__ import print_function
import os
import subprocess
import re
import sys, getopt
from ha.model.parseModel import getModel, Goal
from ha.model.parseProperty import getProperty, Properties, Property
from ha.util.reach import getReachability
import collections
from collections import OrderedDict
from decimal import Decimal
from ha.sage.sageCal import *

INT_MAX = 10000
INT_MIN = 1/10000
cmd = "dReach";
SAT = 51
UNSAT = 52
ERROR = -1
#TNAME = 'temp_hyper18.drh'
#tNAME = 'try_hyper18.drh'
tempfile = "temp.drh"
tryfile = "try.drh"
tempPrefix = "pc_"
folder = 'thyroid'
tempfolder = 'temp'
folder = 'thyroid'
N = 20
#f = 8
p = 1
f = 8
tg = 0
ka = 0
frac = 0

class Dosage:
	def __init__(self):
		self.result = -1
		self.value = 0	 
	def __init__(self, a, b):
		self.result = a
		self.value = b	
	def __str__(self):
		return str(self.result) + " "+ str(self.value)

class Stack:
	def __init__(self):
		self.items = []

	def isEmpty(self):
		return self.items == []

	def push(self, item):
		self.items.append(item)
		#print('Stack pushed ', item)

	def pop(self):
		item = self.items.pop()
		#print('Stack poped ', item)
		return item

	def peek(self):
		return self.items[len(self.items)-1]

	def size(self):
		return len(self.items)

def main(argv):
	filter = True
	fourier = True
	
	try:
		opts, args = getopt.getopt(argv,"hi:p:k:l:u:d:n:v:",["modelfile=","prop=","bound=", "lrange=", "urange", "delta","temp","value"])
	except getopt.GetoptError:
			print("estimateDosage.py -i <modelfile> -p <propertyFile> [-b <bound>] [-d <precision>] [-l <initial lower range>] [-u <initial upper range>]")
			sys.exit(2)
	filename  = ""
	propertyfile = ""
	k = 5
	dval = 0.0
	d = 0.001
	global tempfile, tempPrefix
	#tempfile = TNAME
	tempPrefix1 = tempPrefix

	for opt, arg in opts:
		print(opt, arg)
		if opt == '-h':
			print("\t -i <modelfile> \n\t-p <propertyFile>  \n\t-b <bound> (optional, default 10) \n\t-d <precision> (optional , default 0.01) \n\t-l <initial lower range> (optional, default 0) \n\t-u <initial upper range> (optional, default 2)")
			sys.exit()
		elif opt in ("-i", "--modelfile"):
			filename = arg
			print(arg)
		elif opt in ("-p", "--prop"):
			propertyfile = arg
		elif opt in ("-k", "--bound"):
			k = int(arg)
			print('k ',arg)
		elif opt in ("-d", "--delta"):
			d = arg			
		elif opt in ("-l", "--lrange"):
			l = float(arg)
		elif opt in ("-u", "--urange"):
			u = float(arg)
		elif opt in ("-n", "--temp"):
			print('temp ', arg)
			tempPrefix1 = tempPrefix+arg+'_'
		elif opt in ("-v", "--value"):
			print('dval ', arg)
			dval = float(arg)

	print("Model file is :"+filename)

	tempfile = tempPrefix1 + filename
	print("temp file is :", tempPrefix1,tempfile)

	fname = os.path.join(folder, filename)
	modelfile = filename;
	model = getModel(fname)

	global tg, ka, frac, p, f
	mValues  = getMacroValues(model, ['t_gap','k_a', 'eps', 'frac'])
	print(mValues)
	tg = eval(mValues.get('t_gap'))
	ka  = eval(mValues.get('k_a'))
	#p  = eval(mValues.get('eps'))
	frac = eval(mValues.get('frac'))

	print(tg, ka, p, frac)
	
	pfr = ""
	defs = ""
	initf = ""

	(off, ffs, p, f, tm1) = optimizeSmootherfunction(N, tg, ka, dval)

	if(fourier):
		(fn, f1, f2, a0) = getFourier(N)
		#dfn = getFourierDerivative(fn)

		pfr = fn(t_gap = tg, k_a = ka, eps = p, pi=math.pi, delta= f, off = off)
		defs = getFourierDerivative(pfr)
		#defs2 = dfn(t_gap = tg, k_a = ka, eps = p, pi=math.pi, f= f, off = 0)
		#inita = a0(t_gap = tg, k_a = ka, eps = p, pi=math.pi, f= f)
		initf = pfr(tm = 0)
		#print("four ", pfr, "\n derivative: ",toString(defs),"\n der: ", toString(defs2), "\n init: ", toString(inita) , "\n initf: ", toString(initf))
		print("four ", pfr, "\n derivative: ",toString(defs),"\n initf: ", toString(initf))
	else:
		(der, init) = getSteadyStateDerivative()
		defs = der(t_gap = tg, k_a = ka, eps = p, pi=math.pi)
		initf = init(t_gap = tg, k_a = ka, eps = p, pi=math.pi)
		print("\n derivative: ",toString(defs),"\n initf: ", toString(initf))

	# p2 = getIntegral(defs) + inita
	# plt1 = plot(pfr(t), t, 0, 20)
	# plt2 = plot(p2(t), t, 0, 20, color = 'red')
	# plt = plt1+plt2
	# plt.save('fourier1.png')
	#dv = 0.8

	# offset = float(getOffset(pfr(d0 = dv), defs(d0 = dv), N, tg, p))
	# print('offset: ', offset, inita(d0 = dv))

	# p1 = pfr(d0 = dv)
	# p2 = getIntegral(defs(d0 = dv)) + inita(d0 = dv)
	# plt1 = plot(p1(t) + offset, t, 0, 20)
	# plt2 = plot(p2(t) + offset, t, 0, 20, color = 'red')
	# plt3 = plot(p2(t), t, 0, 20, color = 'green')

	# plt = plt1+plt2+plt3
	# plt.save('fourier1.png')

	for state in model.states:
		for ode in state.flow:
			if(ode.var == 'drug_abs'):
				ode.expr = toString(defs)
	
	for cond in model.init.condition:
		if(cond.literal1 == 'drug_abs'):
			cond.literal2 = toString(initf) + '+ offset'

	#print(str(model))

	print(propertyfile)
	pname = os.path.join(folder, propertyfile)
	properties = getProperty(pname)
	print(str(properties))

#	print("main: temp name ", tempfile)

	print("In verify")

	#u = rmin - d
	#v = l

	for prop in properties.min:
		print("Estimating Min dosage")
		print(str(prop))
		print(' applied dose: ', dval)
		res = updateDosage(model, pfr, dval, fourier)
		if(res.r): 
			out = checkResult(res.m, prop, k, d, tempfile, filter)
			print('checkResult: ', dval, out)
		else:
			print('')

def toString(expr):
	exprString = str(expr).replace('e^', 'exp')#.replace('t', 'tm')
	return exprString

def updateDosage(model, fn, din, fourier = True):
	global N, p, f, tg, frac, ka

	offset = 0.0
	if (fourier):
		dval = frac * din
		print(dval, frac, din)

		(off, ffs, p1, f1, tm) = optimizeSmootherfunction(N, tg, ka, dval)
		p = p1
		f = f1
		print(p, f)
		model.macros['eps'] = '{:.5f}'.format(float(p))
		model.macros['delta'] = '{:.5f}'.format(float(f))

		pfr = fn(d0=dval, eps=p, delta=f)
		# defs = dfn(d0 = dval, eps = p, delta = f)

		offset = getOptimizedOffset(N, tg, ka, p, f, dval)
		# (offset, tm) = getOffset(pfr, tg - p, (tg/N))
		print('offset: ', offset, din)

	update = collections.namedtuple('update', ['r', 'm'])

	# val1  = getMacroKey(model, "drug_dosage");
	# val2  = getMacroKey(model, "offset");

	val1 = 'drug_dosage'
	val2 = 'offset'

	if len(val1) > 0 and len(val2) > 0 :
		#print(model.macros[val])
		model.macros[val1] = '{:.10f}'.format(din)
		model.macros[val2] = str(offset)#'{:.5f}'.format(offset)
		#print(model.macros[val])
		#print(str(model))
		model1 = replaceDefines(model)
		#print(model1.macros[val])
		#print(str(model1))		
		return update(r = True, m = model1)
	else:
		return update(r = False, m = model)

def getMacroKey(model, macro):
	val = ""
	if macro in model.macros.keys():
		#print(key)
	#	if (key == macro):
		val = macro
	#break;
	print(val)

def getMacroValues(model, names):
	values = {}
	val = 0
	for key in model.macros.keys():
		#print(key)
		if key in names:
			val = model.macros[key]
			values.update({key:val})

	model2 = model.clone()
	for key in model2.macros.keys():
		value = model2.macros[key]
		for key1 in model2.macros.keys():
			val = model2.macros[key]	
			model2.macros[key1]= model2.macros[key1].replace(key, val)

		# for key1 in model2.macros.keys():	
		# 	model2.macros[key1]= model2.macros[key1].replace(key, value)
	
	# for key1 in model2.macros.keys():
	#  	print(key1, model2.macros[key1])

	for key in values.keys():
		#for key1 in model2.macros.keys():
		val = model2.macros[key]
		#	val= values[key].replace(key1, value1)
		values.update({key:val})
	
	return values

def replaceDefines(model):
	global tryfile, tempfile
	tryfile = "try1_"+tempfile
	print(tryfile)	
	model2 = model.clone()
	for key in model2.macros.keys():
		if(key is None):
			continue;
		value = model2.macros[key]
		if key == 'drug_dosage' or key == 'd0':
			print("orig_model", key, value)
		for key1 in model2.macros.keys():
			val = model2.macros[key]	
			model2.macros[key1]= model2.macros[key1].replace(key, val)

	model3 = model2.clone()
	model3.macros = {}

	st = str(model3)
	#print('model3\n',st)

	for key in model2.macros.keys():
		if(key is None):
			continue
		#print(key)
		value = model2.macros[key]
		#if key.startswith("drug_dosage") or key.startswith("d0"):
		#	print("replaced_macro",key, value)
		st = st.replace(key, value)
	#print('replaced\n',st)
	#fname = os.path.join(tempfolder, tryfile)
	fname = tryfile
	with open(fname, 'w') as of:
	 	of.write(st)
	model1 = getModel(fname)
	#print(model1.init)		
	return model1

def checkResult(model, props, k, d, temp, filter = False):
	prefixTable = OrderedDict()
	out = ERROR
	stack = Stack()
	# for prop in reversed(props.goals):
	# 	print(prop)
	(tco,tcp)  = (0,0)
	#print("checkResult: temp name ", temp)
	for prop in reversed(props.goals):
		#print('checkResult ',prop)
		#pre_out = out
		if(not (prop == '!' or prop == '&' or prop == '|')):
			stack.push(prop)
		else:
			op = prop
			#print(op)

			if(stack.size() > 0):
				goal1 = stack.pop()
			else:
				return ERROR
			#print(goal1)
			(out1, tco1, tcp1) = getReachability(model, goal1, k, d, temp, prefixTable,(tco,tcp), filter)
			if(out1 == ERROR):
				print("checkResult failed")
			else:	
				(tco, tcp) = (tco1, tcp1)
				if(op == '!'):
					out = neg(out1)
					#print("Not", out1, out)
					stack.push(out)

				else: #if(op != '!'):
					if(stack.size() > 0):
						goal2 = stack.pop()
					else:
						return ERROR

					(out2, tco, tcp) = getReachability(model, goal2, k, d, temp, prefixTable,(tco1, tcp1), filter)
					if(out2 == ERROR):
						print("checkResult failed")
					else:	
						if(op == '&'):
							out = And(out1, out2)
							#print("And", out1, out2, out)
							stack.push(out)
						elif(op == '|'):
							out = Or(out1, out2)
							#print("Or", out1, out2, out)
							stack.push(out)


	if(stack.size() > 0):
		goal = stack.pop()
		(out, tc1, tc2) = getReachability(model, goal, k, d, temp, prefixTable, (tco, tcp), filter)
	#	print(out)
	else:
		out = ERROR
		(tc1, tc2) = (tco, tcp)

	print('Number of paths visited (without pruning) for k=', k, ' are ', tc1)
	print('Number of paths visited (with pruning) for k=',k,' are ',tc2)
	return out

def neg(sat):
	if sat == SAT:
		return UNSAT
	elif sat == UNSAT:
		return SAT
	else:
		return ERROR

def And(a, b):
	if(a == ERROR):
		return b
	elif(b == ERROR):
		return a
	elif(a == SAT and b == SAT):
		return SAT
	else:
		return UNSAT

def Or(a, b):
	if(a == ERROR):
		return b
	elif(b == ERROR):
		return a
	elif(a == SAT or b == SAT):
		return SAT
	else:
		return UNSAT

if __name__ == "__main__": 
   main(sys.argv[1:])



