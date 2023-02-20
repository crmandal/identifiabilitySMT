
from __future__ import print_function
import os
import queue
import subprocess
import re
import sys, getopt, gc
import csv, math

import multiprocessing
# from multiprocessing import Pool#, Queue, Process
from pathos.multiprocessing import ProcessingPool
from scipy.spatial import ConvexHull

import collections
from collections import OrderedDict
from decimal import Decimal

import warnings
# with warnings.catch_warnings():
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from model.property import *
from model.haModel import *
from model.phaModel import *
from model.ha_factory import *
from model.node import *
from parser.parseSTL import *
from parser.parseParameters import *
from parser.parseEquations import *

#from model.interval import *
#from model.box import *
#from model.condition import *
import numpy as np
import random as rnd
#from util.reach import *
from util.graph import *
from util.stack import *
from util.heap import *
from ha2smt.smtEncoder import *
from paramUtil.interval import *
from paramUtil.box import *
import paramUtil.box_factory as bfact
from util.parseOutput import *
from util.smtOutput import *
#from paramUtil.readDataBB import *
from model.node_factory import *
from ha2smt.smtEncoder import *
import ha2ode.HA2ode as hode

import numpy
import time

PLOT = hode.PLOT
if PLOT:
	import matplotlib
	# matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_pdf import PdfPages
	from matplotlib.patches import Rectangle

import itertools

import gp_regres_grad_test as gp_grad

# fig = plt.figure()
import random
#random.seed(10)

tempfolder = os.path.join('temp','testBox')
SAT = 51
UNSAT = 52
UNKNOWN = -1
TRUE = 1
FALSE = 0
UNDET = 2
ONLYMARKED = 1
NOMARK = 0

# EPS = 0.001
# N_GRIDS = 0.001
# E_GRIDS = 0.001
# MIN_EPS = 0.001
# MIN_DELTA = 0.00001
# DATA_NOISE = 0.1 #0.015
# todo: read these setup and filenames from config file ---json file

# DEBUG = True
DEBUG = False
dRealCmd = "dReal"
# Optimizer = collections.namedtuple('Optimizer', ['x', 'fun', 'success'])
Instance = collections.namedtuple('Instance', ['p1', 'p2'])
BoxEvaluate = collections.namedtuple('BoxEvaluate', ['ip', 'op'])
one_time = 0
PropRow = 0
rank = 0
RESUME = False
# RESUME = True
TEST = False
# TEST = True
IVT = False
# IVT = True
OPT = False
# OPT = True

# INSTANCE = False
INSTANCE = True
MAX_clusters = 30
FREQUENT = 50 #0
CLEAN_GEN = 300

import json

DELTA = 0.01
N_GRIDS = 0.1
E_GRIDS = 0.2

EPS = 0.001

# from enum import Enum
class BoxTyp(object):
	APPROX = 1
	EXT = 2


class BoxInfo(object):
	"""docstring for ClassName"""
	def __init__(self, *tuple):
		super(BoxInfo, self).__init__()
		self.pri = tuple[0]
		self.box = tuple[1]
		self.delta = tuple[2]
		self.prop = tuple[3]
		self.ttyp = tuple[4]

	def __lt__(self, nxt):
		return self.pri < nxt.pri

	def __gt__(self, nxt):
		return self.pri > nxt.pri

	def __eq__(self, nxt):
		return self.pri == nxt.pri

	def __le__(self, nxt):
		return self.pri <= nxt.pri

	def __ge__(self, nxt):
		return self.pri >= nxt.pri

	def getInfo(self):
		return self.pri, self.box, self.delta, self.prop, self.ttyp
		
	def __repr__(self):
		sk = str(self.pri) + ', ' + str(self.box) + ', '+ str(self.delta)+ ', '+ str(self.prop)+', '+ str(self.ttyp)
		return sk

	def __str__(self):
		return self.__repr__()
		

def findsubsets(S,m):
	return set(itertools.combinations(S, m))

def plotBox(currentAxis, b, combs, boxType= FALSE):
	#if boxType = TRUE:
	# print(boxType)
	col = 'white'
	if boxType == TRUE:
		col = 'blue'
	elif boxType == UNDET:
		col = 'white'
	elif boxType == FALSE:
		col = 'black'
	# plt.figure()  
	if b.size() == 2:
		b_edges = b.get_map()
		x = []
		w = []
		for it in combs:
			intrvl = b_edges[it]
			x.append(intrvl.leftBound())
			w.append(intrvl.width())
		currentAxis.add_patch(Rectangle((x[0], x[1]), w[0], w[1], facecolor=col, alpha=1))
	# plotBox.show()
	# return plt


def main(argv):
	#global EPS
	#EPS = 1 * d
	one_time = 0

	inputfile = ''
	datafile = ''
	paramfile = ''
	paramdefaultfile = ''
	connectedfile = ''
	configfile = ''
	
	try:
		opts, args = getopt.getopt(argv,"hc:", ["cfile="])
	except getopt.GetoptError:
			print("base_param_smt.py -c config_file.json> logs/log.txt ")
			sys.exit(2)
			
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print("base_param_smt.py -c config_file.json > logs/log.txt ")
			sys.exit()
		elif opt in ("-c", "--cfile"):
			configfile = arg
		print("Config file is :" + configfile)

	print("Config file is :" + configfile)

	with open(configfile, "r") as jsonfile:
		configuration = json.load(jsonfile)

	print(configuration)
	inputfile = configuration["model"]
	outputfile =configuration["out"] 
	datafile= configuration["data"]
	paramdefaultfile = configuration["default"]
	connectedfile = configuration["connected"]

	print("Input file is :" + inputfile)
	print("Output file is :" + outputfile)
	print("Data file is :" + datafile)
	print("paramdefault file is :" + paramdefaultfile)
	print("Connected components file is :" + connectedfile)

	# EPS = configuration["EPS"]
	data_tp = configuration["data_points"] # number of data points to be considered 
	FNAME = configuration["FNAME"]  # example code 'ex1': for example 1, 'th': for thyrotropic regulation, 'bb': for bouncing ball model , 'pc': for prostate cancer model
	samples_per_dim = configuration["samples_per_dim"]# if FNAME == 'pc' --> 3, samples per dimension to choose during approximation in GPR and Gradient based approach
	DATA_NOISE = configuration["DATA_NOISE"]  # if FNAME == 'bb' else 0.01 
	TIME_NOISE = configuration["TIME_NOISE"]  # if FNAME == 'bb'or FNAME == 'pc' else 0.08
	MIN_DELTA = configuration["MIN_DELTA"]  # if FNAME == 'pc' else 0.00001 # minimum precision used in dReal SMT solver
	MIN_EPS = configuration["MIN_EPS"]  # minimum box size to be considered
	PATH_LEN = configuration["PATH_LEN"]  #if FNAME == 'bb'-> 12, if FNAME == 'pc' -> 10, else 2  
	EB_N = configuration["EB_N"]  #if FNAME == 'pc' else 0.15, else 0.1

	DATA_tp = data_tp

	TB_N = 0.04 #8*MIN_EPS #if FNAME == 'pc' else 8*MIN_EPS
	MAX_EPS = 30*MIN_EPS

	MIN_EPS_2 = 2*MIN_EPS
	MIN_EPS_3 = 0.5*MIN_EPS
	
	k_length = PATH_LEN
	d = DELTA   

	pName = 'plotBox_grad_no_slice_'+FNAME+'.pdf'
	satName = 'satbox_grad_no_slice_'+FNAME+'.csv'

	prefix = inputfile.split('.')[0]+'_'
	outfile = prefix+satName
	# ha = getModel(inputfile)
	# print('model parsed')
	#print(str(ha))

	plotName = prefix+pName

	# initialisation = {}
	if len(paramfile) > 0:
		init_params = getEquationsFile(paramfile)
		print('Initialisation')
		for var in init_params:
			print(var + ' : '+ str(init_params[var]))

	default_params_all = []
	default_param_boxes = []
	with open(paramdefaultfile, 'r') as csvfile:
		# creating a csv reader object
		csvreader = csv.reader(csvfile)

		# extracting field names through first row
		param_names = next(csvreader)
		print('from_default_file', param_names)
		sys.stdout.flush()
		# extracting each data row one by one
		for row in csvreader:
			f_row = []
			for i in range(len(row)):
				f_row.append(float(row[i]))
			default_params_all.append(f_row)
			# print(f_row)
			#sys.stdout.flush()
			default_param_box = {}
			i = 0
			for it in param_names:
				#b_edges.update({it:PyInterval(default_params[i]*(1-MIN_DELTA), default_params[i]*(1+MIN_DELTA))})
				# print(i, it)
				default_param_box.update({it:f_row[i]})
				i+= 1
			print('default_params', param_names, f_row, default_param_box)

			default_param_boxes.append(default_param_box)

	def ifSubset(s1, s2):
		# if set s2 is a subset of set s1
		flag = True
		for s in s2:
			if s not in s1:
				flag = False
				break
		return flag
	
	def intersection(s1, s2):
		# ifntersection of set s1, set s2
		res = []
		for s in s2:
			if s in s1:
				res.append(s)
		return res

	print(param_names)
	connectedComponents = [param_names]
	
	print('connectedComponents', connectedComponents)

	default_param_box = default_param_boxes[0]
	default_params = default_params_all[0]

	all_sat_file = prefix+'sat_box_npg_'+FNAME+'.csv' #.format(rank)
	all_q_file = prefix+'queued_box_npg_'+FNAME+'.csv' #.format(rank)
	if not RESUME:
		fp = open(all_sat_file, 'w')
		fp.close()
		fp = open(all_q_file, 'w')
		fp.close()
	fp = open(outfile, 'w+')
	fp.close()


	def getPropSMT(tp, pid, i, neg):
		fn ='temp_npg_'+str(FNAME)+'_p_'+str(tp)+'_'+str(pid)+'_'+str(i)
		if neg :
			fn = 'tempC_npg_'+FNAME+'_p_'+str(tp)+'_'+str(pid)+'_'+str(i)
		return fn
		
	def checkproperty(model, prop, sbox, delta, klen, tp, pid = 0, neg = False):
		g = model.getGraph()
		st = model.init.mode
		s = ''
		for it in model.variables:
			s += it + ':' + str(model.variables[it])+ ' , '
		# print('##checkproperty', 'model.variables: ', s, 'model.init', str(model.init))
		if DEBUG:
			print('delta: ', delta)
		smts = []
		i = 0
		for path in g.getKPaths(st, klen):
			# print('checkproperty', path)
			depth = len(path)
			#smt = ''
			smtEncode = generateSMTforPath(model, path, delta)  
			propertySmt = to_SMT(prop, smtEncode)
			# print('checkproperty', propertySmt)
			#propertySmtneg = Node(prop).negate().toSMT()
			#if(neg):
			#   smtEncode.addGoal(propertySmtneg)
			#else:
			smtEncode.addGoal(propertySmt)          
			smt = smtEncode.toString(neg)
			# print('2.######')
			#if(i == 1):
			#   break
			fn = getPropSMT(tp, pid, i, neg)
			fname = os.path.join(tempfolder, fn+'.smt2')
			with open(fname, 'w') as of:
				of.write(smt)   
			#sys.stdout.flush()
			
			#st = [dRealCmd, fname, "--precision", str(delta), "--ode-step", str(0.05), "--model"]
			st = [dRealCmd, fname, "--precision", str(delta), "--model"]
			
			if DEBUG:
				print('\t----- '+str(st))
			# p = subprocess.Popen(st, stdout=subprocess.PIPE)
			# (output, err) = p.communicate(timeout=1200)  
			'''This makes the wait possible'''
			# p_status = p.wait()   
			# out = p_status

			# p = subprocess.run(st) #, capture_output=True)
			try:
				gc.collect()
				output =  subprocess.check_output(st)#, timeout=6*3600)
				out = 0
			except subprocess.CalledProcessError as e:
				out = e.returncode
				output = e.stdout
			except Exception as e:
				print('Running call again....') 
				if DEBUG:
					print('\t----- '+str(st))
				try:
					output =  subprocess.check_output(st)
					out = 0
				except subprocess.CalledProcessError as e:
					out = e.returncode
					output = e.stdout
			if DEBUG:
				print('dReal res:', out, output)

			# start_time = time.time()
			
			# end_time = time.time()

			# if end_time - start_time:
			#   p.kill()
			#   print('Killed following call to dReal... ')#, st)
			#   print('Running call again....') 
			#   print('\t----- '+str(st))
			#   (output, err) = p.communicate()
			#   '''This makes the wait possible'''
			#   p_status = p.wait() 

			'''This will give you the output of the command being executed'''
			if DEBUG:
				print ("\t----- Output: " + str(out), 'depth ', depth, i) #,  'out', output)       
			
			sys.stdout.flush()
			
			if(out == 0 and b'delta-sat' in output):
				if DEBUG:
					print('delta-sat', SAT)
				return (SAT, i, output)
			elif(out == 0 and b'unsat' in output):
				if DEBUG:
					print('unsat')
				res = (UNSAT, i, '')
			else:               
				res = (UNKNOWN, i, '')
				if DEBUG:
					print('ERROR')
				return res

			i +=1    
		return res  
			
	def evaluate(args):
		model, all_props, bi, klen, pid = args
		(ip, sbox, d, tp, bttyp) = bi.getInfo()
		prop = all_props[tp]

		params = []

		if DEBUG:
			print(prop, sbox, d, klen, pid)
		''' delta is a fraction of min dimension of the box'''
		# delta = d #
		delta = d #max(sbox.min_side_width() * 0.1, EPS)
		sk = 'Evaluate - Pid: ' + str(pid) +' Checking box : ' + str(sbox) + 'delta: '+ str(delta) + ' klen: '+ str(klen)
			
		updateModel(model, sbox)
		#propNode = getSTL(prop)[0]
		(propNode, propNeg1) = prop
		propNeg = getNegatedProperty(propNeg1) #, instance1)
		if DEBUG:
			print('In evaluate') #,\t prop : '+str(propNode)+'\n\t negProp : '+str(propNeg1) )
		sk += '\n -- @@ prop: '+ str(propNode) +', ' + str(propNeg1)
		(res1, i1, out1) = checkproperty(model, propNode, sbox, delta, klen, tp, pid)
		(res2, i2, out2) = checkproperty(model, propNeg, sbox, delta, klen, tp, pid, neg = True)    
		sk += '\n -- 1. ###### {0}, {1}'.format(res1, res2)
		# print(sk)
		# sys.stdout.flush()
		flag = False
		if (res1 == UNSAT):
			ret = (FALSE, None)
			sk += ' -- FALSE, None'
		elif (res2 == UNSAT):
			ret = (TRUE, None)
			sk += ' -- TRUE, None'
		else:
			instance1 = getSAT(model, i1, tp, pid, False) #;getSATPoint(i1)
			instance2 = getSAT(model, i2, tp, pid, True) #.getSATPoint(i2)
			instance = Instance(p1 = instance1, p2 = instance2)
			ret = (UNDET, instance)
			sk += ' -- UNDET, Instance'
		#
		# if DEBUG:
		#print(sk)
		sys.stdout.flush()
		if flag:
			exit()
		return ret
				
	def getSAT(model, i, tp, pid = 0, neg = False):
		# if neg:
		#   fn = 'tempC_p_'+str(rank)+'_'+str(pid)+'_'+str(i)
		# else:
		#   fn = 'temp_p_'+str(rank)+'_'+str(pid)+'_'+str(i)
		# fname = os.path.join(tempfolder, fn+'.smt2.model')

		fn = getPropSMT(tp, pid, i, neg)
		fname = os.path.join(tempfolder, fn+'.smt2.model')
		if DEBUG:
			print('Reading sat instance :', fname)
		sys.stdout.flush()
		satinstance = parseInstance(fname)
		#print(satinstance.variables[0])
		satinstance.addModel(model)
		satinstance.addDepth(i)
		return satinstance
			
	def updateModel(model, sbox):
		#print('updateModel', box, 'model.params: ')
		#for it in model.parameters:
		#   print(it, str(model.parameters[it]))
		edges = sbox.get_map()
		for it in edges:
			intrvl = edges[it]
			if DEBUG:
				print('InUpdate Model: ', intrvl.leftBound(), str(intrvl.leftBound()))
			param = Range(Node(intrvl.leftBound()), Node(intrvl.rightBound()))
			if it in model.parameters:
				model.parameters.update({it: param})
			if it in model.variables:
				model.variables.update({it: param})
		s = ''
		for it in model.parameters:
			s += it + ':' + str(model.parameters[it])+ ' , '
		if DEBUG:
			print('updatedModel', 'model.params: ', s)
		#return model
		
	'''
	In case of adjacent sat instances.. divide the box through the adjacent dimension
	intersecting boxes -- error box -- remove it
	smaller box -- cover it and remove
	otherwise, divide along the middle point

	'''
	def heuristicPartition(sbox, instance, delta):
		boxesWPriority = []
		boxes = []
		
		inst1 = instance.p1
		inst2 = instance.p2
		
		point = inst1.getSATPoint()
		negpoint = inst2.getSATPoint()

		if DEBUG:
			print(sbox, 'point:', point, 'negpoint:', negpoint)
			sys.stdout.flush()
		
		if point.empty() or negpoint.empty():
			return []

		# global noise
		emap = {}
		b_map = sbox.get_map()
		for it in b_map.keys():     
			# emap.update({it: PyInterval(0.01 * noise[it])})
			emap.update({it: PyInterval(delta, delta)})

		if DEBUG:
			print('heuristicPartition: Box = ', str(sbox)) 

		boxes = bfact.bisect(sbox, emap, NOMARK) #, mPoint) #negpoint)

		# b3 = bfact.boxIntersection(satb, negsatb)
		for b in boxes:
			if(not b.empty()):
				if len(boxes) == 1:
					item = (3, b)
				else:
					item = (1, b)
				boxesWPriority.append(item)
		#print('heuristicPartition - boxes', len(boxes), len(boxesWPriority))
		return boxesWPriority
			
	def getNegatedProperty(prop, instance = None):  
		#print('getNegatedProperty:'+'prop: ', prop.to_prefix())
		propneg = prop
		#print('prop: ', prop.to_prefix())
		if DEBUG:
			print('negated prop: ', propneg.to_prefix())
		return propneg
			
	def getProperties(propstr): #, dtype):  
		(p, pc) = propstr
		pn = getSTL(p)[0]
		negpn =  (getSTL(pc)[0]).to_cnf() #pc #pn.negate().to_cnf() #(getSTL(pc)[0]).to_cnf() #pn.negate().to_cnf()
		# print(pn, negpn)
		# print(type(pn), type(negpn))
		#clause = Node('=', [Node('mode'), Node(str(dtype))])
		
		propneg = negpn #Node('&', [clause, negpn])
		prop = pn # Node('&', [clause, pn])
		#prop = pn
		#propneg = negpn
		if DEBUG:
			print('prop: ', prop.to_prefix())
			print('negated prop: ', propneg.to_prefix())
		return (prop, propneg)

	def getBox(params):
		edges = {}
		for par in params:
			rng = params[par]
			left = rng.leftVal()
			right = rng.rightVal()
			it = PyInterval(left, right)
			#it.mark()
			edges.update({par: it}) 
			
		sbox = Box(edges)
		return sbox

	def getPropertyFromData(data, outputEqns):
		print('getPropertyFromData', data)
		# Y = data
		ffl = 0
		if FNAME == 'bb' or FNAME == 'pc': # or FNAME == 'eg':
			ffl = 1
		mode = int(data[-1]) if ffl > 0 else 1
		tm = data[0]
		data_noise = DATA_NOISE 
		time_noise = DATA_NOISE*TIME_NOISE# * 0.11
		if tm >=0:
			tm_0, tm_1  = tm*(1-time_noise), tm*(1+time_noise)
		else:	
			tm_0, tm_1  = tm*(1+time_noise), tm*(1-time_noise)
		prop = '((mode = {0}) & (tm > {1}) & (tm < {2}) & ('.format(mode, tm_0, tm_1)
		propn = '((mode = {0}) & (tm > {1}) & (tm < {2}) & ! ('.format(mode, tm_0, tm_1)	
		# prop = '((tm > {0}) & (tm < {1}) & ('.format(tm_0, tm_1)
		# propn = '((tm > {0}) & (tm < {1}) & ! ('.format(tm_0, tm_1)
		# prop = '((tm = {0}) & ('.format(tm)
		# propn = '((tm = {0}) & ! ('.format(tm)

		i = 1
		for key in outputEqns.keys():
			if i < len(data) - ffl:
				if data[i] >= 0:
					k1, k2 = (data[i]*(1-data_noise), data[i]*(1+data_noise))
				else:               
					k1, k2 = (data[i]*(1+data_noise), data[i]*(1-data_noise))
				#eqn = key #
				eqn = outputEqns[key]
				pr = '(({0}) > {1}) & (({0}) < {2})'.format(eqn, k1, k2)

				prop += pr if i == 1 else ' & '+ pr 
				propn += pr if i == 1 else ' & '+ pr 
			i+= 1
			#break
		prop += '));' 
		propn += '));' 
		print('getPropertyFromData', prop, propn)
		return prop, propn

	
	def getEstimatedParameters(model, params, d, klen, all_props): #, satfile):
		sbox_all = getBox(params)
		#print(box, type(box))
		sat_box = {}
		unsat_box = {}
		undet_box = {}
		# global noise
		# noise.update({'radius':0.001})
		# # noise.update({'g': 0.001})	
		# noise.update({'K': 0.001})
		# for par in params:
		# 	noise.update({par: d})

		par_names = sorted((sbox_all.get_map()).keys())
		pnames = ','.join(sorted(par_names))
		print('par_names', par_names)
		# pnames = ','.join(sorted(par_names))
		True_Boxes = []
		for j in range(len(default_param_boxes)):
			b_edges = {}
			for it in par_names:
				#b_edges.update({it:PyInterval(default_params[i]*(1-MIN_DELTA), default_params[i]*(1+MIN_DELTA))})
				b_edges.update({it:PyInterval(default_param_boxes[j][it])}) #, default_param_box[it])})
			default_box = Box(b_edges) 
			print('default_box', default_box, pnames)
			if sbox_all.contains(default_box):
				True_Boxes.append(default_box)
			if j >= 1:
				break

		#True_Boxes = [default_box]
		print('starting with default points', True_Boxes, 'bounding box', sbox_all)
		default_box = True_Boxes[0]
		t_bs = [default_box]#, True_Boxes[1], True_Boxes[2]]

		sbox_map = sbox_all.get_map()
		# par_low = []
		# par_up = []
		# for it in sorted(par_names):
		# 	par_low.append(sbox_map[it].leftBound())
		# 	par_up.append(sbox_map[it].rightBound())
		# engine = qmc.LatinHypercube(len(par_names))#,)
		# lhs_sample1 = engine.random(n=max(20,2**len(par_names)))
		# lhs_sample = qmc.scale(lhs_sample1,  par_low, par_up)
		# print('LHS discrepancy', qmc.discrepancy(lhs_sample1))
		# print(lhs_sample)


		# x_test = {}
		# for i in range(len(par_names)):
		# 	pn = par_names[i]
		# 	il = bfact.min_left_coordinate_value(t_bs, pn) * 10.0/15
		# 	ul = bfact.max_right_coordinate_value(t_bs, pn) * 15.0/10 
		# 	x_test.update({pn:PyInterval(il, ul)})
		# b2 = Box(x_test)
		# print('starting box', str(b2))
		# delta_to_use = getBoxDelta(b2) #b2.max_side_width()*N_GRIDS
		# boxes_to_check = [(b2, delta_to_use)]

		boxes_to_check = []
		if not OPT:
			delta_to_use = getBoxDelta(sbox_all) #b2.max_side_width()*N_GRIDS
			boxes_to_check.append((sbox_all, delta_to_use))
		else:
			for tb in True_Boxes:
				x_test = {}
				for i in range(len(par_names)):
					pn = par_names[i]
					# il = bfact.min_left_coordinate_value([tb], pn) * 10.0/12.0
					# ul = bfact.max_right_coordinate_value([tb], pn) * 12.0/10 
					il = bfact.min_left_coordinate_value([tb], pn) - EB_N
					ul = bfact.max_right_coordinate_value([tb], pn) + EB_N
					x_test.update({pn:PyInterval(il, ul)})
				b2 = Box(x_test)
				print('starting box', str(b2))
				delta_to_use = getBoxDelta(b2) #b2.max_side_width()*N_GRIDS
				boxes_to_check.append((b2, delta_to_use))


		pnames = sorted(par_names)
		# for lhs in lhs_sample:
		# 	x_test = {}
		# 	for i in range(len(par_names)):
		# 		pn = pnames[i]	
		# 		il = lhs[i]* 10.0/10.5
		# 		ul = lhs[i]* 10.5/10
		# 		x_test.update({pn:PyInterval(il, ul)})
		# 	b2 = Box(x_test)
		# 	delta_to_use = getBoxDelta(b2) #b2.max_side_width()*N_GRIDS
		# 	boxes_to_check.append((b2, delta_to_use))
		# boxes_to_check = new_boxes_to_check_after_regres
	
		# print('GP_xtest box size {0}'.format(len(extra_boxes)), len(partitions), len(boxes_to_check))     
		
		(bt, bf, bu) = getBoxes(model, params, all_props, sbox_all, boxes_to_check, klen, True_Boxes)
		print('sat boxes', len(bt), len(all_props)) #, bt)

		# sat_box.update({pnames:bt})
		# unsat_box.update({pnames:bf})
		# undet_box.update({pnames:bu})

		sat_box = bt
		unsat_box = bf
		undet_box = bu
	
		return (sat_box, unsat_box, undet_box)
	

	def boxtopoint(b, param_names = []):
		bmap = b.get_map()
		if len(param_names) == 0:
			param_names = sorted(bmap.keys())
		pts = []
		for it in sorted(param_names):
			pts.append((bmap[it].leftBound() + bmap[it].rightBound())/2)
		# print(p, bmap)
		return pts

	def getBoxDelta(b):
		#return b.min_side_width()*N_GRIDS 
		return max(b.min_side_width()*N_GRIDS, MIN_DELTA)
	
	def ifBoxEPS(b, tp):
		return (b.max_side_width() < max(EPS*(E_GRIDS**tp), MIN_EPS))

	def getBoxes(model, params, all_props, sbox_all, sboxes, klen, True_boxes = [], act_pt = None): 
		stime = time.time()
		#EPS = 0.01 #max(sbox.min_side_width() * 0.1, d)
		#mgr = multiprocessing.Manager()
		GP_X = []
		GP_y = []
		# pre_GP_x = len(GP_X)
		par_names = sorted((sbox_all.get_map()).keys())

		mcpu = multiprocessing.cpu_count()
		np1 = int(mcpu)-2 if mcpu > 30 else int(mcpu)-2
		# vol_covered = 0.0
		ratio = 0.0
		ratio2 = 0.0
		ratio1 = 0.0
		
		sat_box = []
		unsat_box = []
		undet_box = []
		# next_appr_sat_box = []
		# decided_boxes = []
		checked_boxes = []

		initPriority = 10
		stack = HeapQ() 

		total_volume = sbox_all.volume()
		# total_volume = 0.0
		env_volume = 0.0
		# if rank == 0:
		datatp = 0
		starting_box = []
		for i in range(len(sboxes)): 
			sbox, delta_to_use= sboxes[i]
			starting_box.append(sbox)
			# print('getBoxes',  len(sboxes), type(sbox))
			# total_volume += sbox.volume()
			# delta_to_use = #DELTA*(0.1**(datatp))
			stack.push(BoxInfo(initPriority, sbox, delta_to_use, datatp, BoxTyp.EXT))
			
		sat_box += [] #default_box]
		print('getBoxes', len(sboxes), total_volume)#, 'nodes', nParts)

			
		# print('------------------- Run for row ', datatp, '-------------------------')
		# print('----- data ----', prop)
		print('Starting with --', stack.size(), len(sat_box), 'num props', len(all_props), 'k_len', klen)
		
		c1 = 0
		c2 = 0
		gen = 0

		c2 = len(True_boxes)
		# kdt_npg_sat = kdivt.IntervalTree()
		i = 0   
		pre_count = 0
		pre_gen = 0
		while(not stack.isEmpty()):
			min_pri = initPriority
			max_pri = -1
			# while(len(List) > 0):
			#b = stack.pop()
			# np = 1 #
			# new_boxes = []
			count = 0
			count2 = 0
			count1 = 0
			min_delta = DELTA

			num_proc = min(np1, stack.size())
			POOL = True if num_proc > 1 else False
			#num_proc = 1 # min(np1, stack.size())
			# POOL = False #True if num_proc > 1 else False
			allboxes = []   
			for j in range(num_proc):
				bi = stack.pop()
				pr, bx, dt, tp, bttyp = bi.getInfo()
				allboxes.append(bi)
			
			num_proc = len(allboxes)  #min(num_proc, )
			sk = 'Using multiprocessing : {0}, prces {1},  Boxes left -- {2}, time -- {3:2f} hrs-- pri: {4}, {5} '.format(POOL, num_proc, stack.size(), ((time.time()-stime)/3600), max_pri, min_pri)

			if POOL:
				pool = ProcessingPool(num_proc)
				inputs = [[model, all_props, allboxes[j], klen, j] for j in range(num_proc)]
				#print(inputs)
				results = pool.map(evaluate, inputs)
			else:
				results = [evaluate([model, all_props, allboxes[j], klen, 0]) for j in range(num_proc)]

			if POOL:
				pool.close()
				pool.join()
				pool.clear()

			# print('results', results, num_proc)
			# fflag = False
			for j in range(num_proc):       
				bi = allboxes[j]    
				pr, b, dt, tp, bttyp = bi.getInfo()
				# delta_to_use = delta_to_use1*(EPS**tp)
				be = b.get_map()
				if min_delta > dt:
					min_delta = dt
				
				
				sys.stdout.flush()
				(r, instance) = results[j]      
				# print(j, r)
				
				if DEBUG:
					print('Box min: '+ str(b.min_side_width())+ ' max :' + str(b.max_side_width()), 'delta: ', dt, delta_to_use)
					print('Result: ', r, instance)

				if(r == FALSE):
					'''if given range subset of b then return false '''
					#return (False, b)
					unsat_box.append(b)
					# vol_covered += b.volume()
					if DEBUG:
						print('@@@@ False box : --decided', tp, '<', len(all_props), str(b))
					count2 += 1
					#break

				elif(r == TRUE):    
					''' if given range subset of b then return true '''
					if tp+1 < len(all_props):
						if DEBUG:
							print('## 1-sat box : ', tp, '<', len(all_props), str(b))	
						''' check for next property '''
						dt1 = getBoxDelta(b) #max(b.max_side_width()*N_GRIDS, MIN_DELTA) #max(b.minDimension()*EPS, MIN_DELTA)
						# if tp < 2:
						stack.push(BoxInfo(pr-1, b, dt1, tp+1, bttyp)) # adding to stack
						# else:
						# 	stack.push(BoxInfo(pr-1, b, dt1, tp+1, BoxTyp.PART)) # adding to stack
						# new_boxes.append((b, dt1))
						if DEBUG:
							print('---- check next property {0} with delta {1}'.format(tp+1, dt1))
						count += 1
						
					else:
						''' all properties are covered --- stop '''
						sat_box.append(b)
						# vol_covered += b.volume()
						print('@@@@ True box -- decided : ', tp, '<', len(all_props), str(b))              
						count2 += 1

						c2 += 1
						True_boxes.append(b)
						# decided_boxes.append(b)     
						checked_boxes.append(b)

				else: 
					''' UNDET '''
					# dts = max(delta_to_use, MIN_DELTA)
					eps = ifBoxEPS(b, tp) #MIN_EPS #max(EPS*(N_GRIDS**(tp+1)), MIN_EPS)
					if DEBUG:
						print('## undet box : ', tp, ' ', str(b), ' Box min: '+ str(b.min_side_width())+ ' max :' + str(b.max_side_width()), 'delta: ', dt, 'delta to use', delta_to_use, 'min delta', MIN_DELTA, 'eps', eps, MIN_EPS)                   
					if eps: #(b.max_side_width() < eps):
						if DEBUG:
							print('## 2-sat box : ', tp, '<', len(all_props), str(b))
						if tp+1 < len(all_props):
							''' check for next property '''
							dt1 = getBoxDelta(b) #max(b.max_side_width()*N_GRIDS, MIN_DELTA)
							# dt1 = max(b.minDimension()*EPS, MIN_DELTA)
							#dt1 = max(b.minDimension()*EPS, dt*EPS)
							#if tp < DATA_tp-1:
							stack.push(BoxInfo(pr-1, b, dt1, tp+1, bttyp)) # adding to stack
							# else:
							count += 1
							if DEBUG:
								print('---- check next property {0} with delta {1}'.format(tp+1, dt1))
							
						else:
							''' all properties are covered --- stop '''
							# sys.stdout.flush()
							inst1 = instance.p1
							inst2 = instance.p2
							
							satb = inst1.getSATBox()
							# negsatb = inst2.getSATBox()
							# point = inst1.getSATPoint()
							# negpoint = inst2.getSATPoint()

							sat_box.append(satb)
							# vol_covered += b.volume()     
							count2 += 1

							# extra_boxes = bfact.remove(b, satb)
							# for b2 in extra_boxes:
							#   vol_covered += b2.volume()  
							print('@@@@ < EPS {0:0.4f} SAT box -- decided'.format(max(EPS*(E_GRIDS**tp), MIN_EPS)), tp, 'boxtype', bttyp)#, 'removed {0} smaller boxes'.format(len(extra_boxes)))

							c2 += 1
							True_boxes.append(satb)
							# decided_boxes.append(satb)     
							checked_boxes.append(b)

					else:
						# delta1 = dt #min(sbox.minDimension() * 0.1, MIN_DELTA)
						boxes = heuristicPartition(b, instance, eps) #MIN_DELTA) #MIN_EPS) # dt*EPS)
						# print('--- box partition', b, len(boxes), MIN_DELTA)
						# print('getBoxes', np.array(GP_X).shape, np.array(GP_t).shape, c1, c2)
						# gp_prob_dict = {}
						nbx = []
						#print('Checking partitioned boxes...')
						for b2 in boxes:
							# print(b2)
							f1 = b2[0] # box type
							b1 = b2[1] # box
							pr1 = pr - 1
							dt1 = getBoxDelta(b1) #max(b1.max_side_width()*N_GRIDS, MIN_DELTA)
							# if dt1 < dt:
							#stack.push(BoxInfo(pr1, b1, dt1, tp)) # adding to stack
							
							stack.push(BoxInfo(pr1, b1, dt1, tp, bttyp))
								#else:
							if DEBUG:
								print('Added box after only partitioning', str(b1)) 
							#new_boxes.append((b1, dt1))
							count += 1
							# else:
					
						if DEBUG:
							print('checking partitioned boxes...', len(boxes), len(nbx))#, str(nbx))
		
			sk+= '\n\tNew boxes added: {0}/{1}, previous {2}, decided {3}, min delta {4}'.format(count, c2, pre_count, count2, min_delta) #+ ' box decided: '+ str(count2) + ' min_delta: '+ str(min_delta)
			# if pre_count == count and count2 == 0:
			#   print('New boxes', new_boxes)
			#   break
			print(sk)
			# print(i, queue_size)

			# if i%2 == 0 : #and i > 0:
				
			# sys.stdout.flush()

			if (c2 - pre_count) > FREQUENT :#or (i > 100 and (c2 - pre_count) < 5.0):
				st_clone = stack.clone()
				writeToFile(sat_box, st_clone, par_names, unsat_box) 
				print('---- Number of trainig points', len(sat_box), len(unsat_box), c2, pre_count)
				print('@@@ --- temp plot ---', (time.time() - stime)/3600, 'hr')
				pre_count = c2
				pre_gen = i
				print('--- plotting for gen ', i, 'NPG', str(i%2==0))
				figs = plot_sat_boxes(sbox_all, sat_box, par_names, ratio1, [], [], i, unsat_box) 
				if PLOT:
					pp = PdfPages(plotName)
					for fig  in figs:
						pp.savefig(fig)
					pp.close()
				sys.stdout.flush()

			i += 1
			
			if TEST and i > 5 and len(sat_box) > 500 :
				st_clone = stack.clone()
				writeToFile(sat_box, st_clone, par_names, unsat_box) 
				break

		print('{0}% of the Box covered .....'.format(ratio1))
		figs = plot_sat_boxes(sbox_all, sat_box, par_names, ratio1, unsat_box=unsat_box) 
		if PLOT:
			pp = PdfPages(plotName)
			for fig  in figs:
				pp.savefig(fig)
			pp.close()
		sys.stdout.flush()
		return (sat_box, unsat_box, undet_box)      

	def plot_sat_boxes(sbox_all, sat_boxes, par_names, ratio=100, npg = [], cluster_centers = [], gen = 0, unsat_box = []):
		figs = []
		n_par = len(par_names)  
		contract = {}
		for i in range(n_par):
			contract.update({par_names[i]:i})
		b_map = sbox_all.get_map()
		param_bounds_all = []
		for it in sorted(par_names):
			pb = [b_map[it].leftBound(), b_map[it].rightBound()]
			param_bounds_all.append(pb)

		if len(par_names) >= 3:
			subsets_3 = findsubsets(par_names, 3)
			for sub in subsets_3: 
				par = [contract[i] for i in sub]
				axisname = list(sub)
				fig1 = plt.figure()
				plt.title('3D project {0}%'.format(ratio))
				ax = fig1.add_subplot(111, projection="3d")
				ax.set_xlabel(axisname[0])
				ax.set_ylabel(axisname[1])
				ax.set_zlabel(axisname[2])
				xlim, ylim, zlim = param_bounds_all[par[0]], param_bounds_all[par[1]], param_bounds_all[par[2]]
				x1 = []
				x2 = []
				x3 = []
				for satb in sat_boxes:
					x = []
					w = []
					p1, p2, p3 = sub
					sat = satb.get_map()
					x11 = (sat[p1].mid().leftBound())
					x12 = (sat[p2].mid().leftBound())
					x13 = (sat[p3].mid().leftBound())
					x1.append(x11)
					x2.append(x12)
					x3.append(x13)
				
				ax.plot3D(x1, x2, x3, 'b.')

				x1 = []
				x2 = []
				x3 = []
				for satb in npg:
					x = []
					w = []
					p1, p2, p3 = sub
					sat = satb.get_map()
					x11 = (sat[p1].mid().leftBound())
					x12 = (sat[p2].mid().leftBound())
					x13 = (sat[p3].mid().leftBound())
					x1.append(x11)
					x2.append(x12)
					x3.append(x13)

				if gen%2 ==0:
					ax.plot3D(x1, x2, x3, 'r.')
				else:
					ax.plot3D(x1, x2, x3, 'm.')
				x1 = []
				x2 = []
				x3 = []
				for satb in cluster_centers:
					p1, p2, p3 = sub
					sat = satb.get_map()
					x11 = (sat[p1].mid().leftBound())
					x12 = (sat[p2].mid().leftBound())
					x13 = (sat[p3].mid().leftBound())
					x1.append(x11)
					x2.append(x12)
					x3.append(x13)

				ax.plot3D(x1, x2, x3, 'g.')
				# ax.set_xlim(xlim)     
				# ax.set_ylim(ylim)     
				# ax.set_zlim(zlim)
				figs.append(fig1)

		if len(param_names) >= 2:
			param_len = 2 
			# for all_params in dependent:
			#   # param_len = len(all_params.keys())
			subsets_2 = findsubsets(param_names, param_len)
			for sub in subsets_2: #for each subset of size 2 
				# print('For param set', sub, type(sub))
				par = [contract[i] for i in sub]

				xlim, ylim = param_bounds_all[par[0]], param_bounds_all[par[1]]
				x = []
				w = []
				axisname = list(sub)
				for p in par:
					x.append(param_bounds_all[p][0])
					w.append(param_bounds_all[p][1] - param_bounds_all[p][0])
				
				fig_t = plt.figure()
				plt.title('2D project '+str(axisname)) #+ '{0}%'.format(ratio))
				# xlim = [1.0*x[0], 1.0*(x[0] + w[0])]
				# ylim = [1.0*x[1], 1.0*(x[1] + w[1])]
				# print(xlim, ylim)
				plt.xlabel(axisname[0])
				plt.ylabel(axisname[1])
				# plt.ylim()
				currentAxis = plt.gca()
				currentAxis.set_xlim(xlim)
				currentAxis.set_ylim(ylim)
				
				x1 = []
				x10 = []
				x2 = []
				for satb in sat_boxes:
					p1, p2 = sub
					
					sat = satb.get_map()
					x11 = (sat[p1].mid().leftBound())
					x12 = (sat[p2].mid().leftBound())
					# x13 = (sat[p3].mid().leftBound())
					x10.append(x11)
					x1.append(x11)
					x2.append(x12)
					#currentAxis.add_patch(Rectangle((x[0], x[1]), w[0], w[1], facecolor='grey', alpha=1))
				
				currentAxis.plot(x10, x2, 'b.')

				if len(param_names) == 2:
					x1 = []
					x10 = []
					x2 = []
					for satb in npg:
						p1, p2 = sub
						
						sat = satb.get_map()
						x11 = (sat[p1].mid().leftBound())
						x12 = (sat[p2].mid().leftBound())
						# x13 = (sat[p3].mid().leftBound())
						x10.append(x11)
						x1.append(x11)
						x2.append(x12)
						#currentAxis.add_patch(Rectangle((x[0], x[1]), w[0], w[1], facecolor='grey', alpha=1))
					
					if gen%2 ==0:
						currentAxis.plot(x10, x2, 'r.')
					else:
						currentAxis.plot(x10, x2, 'm.')

					x1 = []
					x10 = []
					x2 = []
					for satb in cluster_centers:
						p1, p2 = sub
						
						sat = satb.get_map()
						x11 = (sat[p1].mid().leftBound())
						x12 = (sat[p2].mid().leftBound())
						# x13 = (sat[p3].mid().leftBound())
						x10.append(x11)
						x1.append(x11)
						x2.append(x12)

					currentAxis.plot(x10, x2,  'g.')
				x1 = []
				x10 = []
				x2 = []
				for satb in unsat_box:
					p1, p2 = sub
					sat = satb.get_map()
					x = (sat[p1].leftBound(), sat[p2].leftBound())
					w = (sat[p1].width(), sat[p2].width())
					# x13 = (sat[p3].mid().leftBound())
					# x10.append(x11)
					# x1.append(x11)
					# x2.append(x12)
					currentAxis.add_patch(Rectangle((x[0], x[1]), w[0], w[1], facecolor='grey', alpha=1))
				figs.append(fig_t)

		if len(param_names) > 2:
			# param_len = 2 
			subsets_n1 = findsubsets(param_names, len(param_names)-1)
			for sub in subsets_n1: 
				extra_par = ''
				for p in param_names:
					if p not in sub:
						extra_par = p
						break
				extra_par_id = contract[extra_par]
				extra_par_bounds = param_bounds_all[extra_par_id] #param_bounds_all[extra_par_id]
				lb , ub = extra_par_bounds[0], extra_par_bounds[1]
				extra_par_slices = np.linspace(lb, ub, int((ub - lb)/0.01))
				# print(extra_par, 'extra_par_slices', extra_par_slices)

				par = [contract[i] for i in sub]
				xlims = [param_bounds_all[par[i]] for i in range(len(sub))]
				axisname = list(sub)
				max_slice = (0, [], 0)
				for j in range(1, len(extra_par_slices)):
					extra_l, extra_u = extra_par_slices[j-1], extra_par_slices[j]
					# xlim, ylim, zlim = param_bounds_all[par[0]], param_bounds_all[par[1]], param_bounds_all[par[2]]
					xs = [[] for i in sub]
					# flag = False
					k = 0
					xs_1 = []
					for satb in sat_boxes:
						sat = satb.get_map()
						# x12 = (sat[p2].mid().leftBound())
						# x13 = (sat[p3].mid().leftBound())
						if extra_l <= sat[extra_par].leftBound() < extra_u and extra_l < sat[extra_par].rightBound() <= extra_u:
							for i in range(len(sub)):
								p = sub[i]
								x11 = (sat[p].mid().leftBound())
								xs[i].append(x11)
							# flag = True
							k += 1
							xs_1.append(satb)
					k = bfact.get_cover(xs_1, check = False).volume() if len(xs_1) > 0 else 0
					if k > max_slice[0]:
						max_slice = (k, xs, j)
				
				k, xs, j = max_slice
				#print(k, j, len(xs))
				fig = plt.figure()
				plt.title('2 slice '+extra_par+' -- max slice') # {0}, {1} -- {2}%'.format(j, k, ratio))
				if len(sub) == 3 and len(xs) > 2:
					# print('plot 3 dim')
					ax = fig.add_subplot(111, projection="3d")
					ax.set_xlabel(axisname[0])
					ax.set_ylabel(axisname[1])
					ax.set_zlabel(axisname[2])
					ax.plot3D(xs[0], xs[1], xs[2], 'b.')
					# print('[')
					# for i in range(k):
					# 	print('[{0}, {1}, {2}]'.format(xs[0][i], xs[1][i], xs[2][i]))
					# print(']')

				elif len(sub) == 2 and len(xs) > 1:
					# print('plot 2 dim')
					ax = plt.gca() #fig.add_subplot(111) #, projection="3d")
					ax.set_xlabel(axisname[0])
					ax.set_ylabel(axisname[1])
					# ax.set_zlabel(axisname[2])
					ax.plot(xs[0], xs[1], 'b.')
					ax.set_xlim(xlims[0])       
					ax.set_ylim(xlims[1])
				figs.append(fig)

			subsets_n1 = findsubsets(param_names, 2)
			for sub in subsets_n1: 
				extra_pars = []
				for p in param_names:
					if p not in sub:
						extra_pars.append(p)
						# break
				# print('extra_pars', extra_pars)
				# extra_par_id = [contract[ep] for ep in extra_pars]
				# extra_par_bounds = [param_bounds[epd] for epd in extra_par_id] #param_bounds_all[extra_par_id]
				extra_par_slices = {}
				for ep in extra_pars:
					ep_bound = param_bounds_all[contract[ep]]
					lb , ub = ep_bound[0], ep_bound[1]
					extra_par_slices[ep] = np.linspace(lb, ub, int((ub - lb)/0.01))
					# print(ep, 'extra_par_slices', len(extra_par_slices[ep]))

				par = [contract[i] for i in sub]
				xlims = [param_bounds_all[par[i]] for i in range(len(sub))]
				axisname = list(sub)
				slices_list = []
				for ep in extra_pars: 
					ep_slice = list(range(1, len(extra_par_slices[ep])))
					# print(ep, ep_slice)
					slices_list.append(ep_slice)
				
				max_slice = (0, [], 0)
				sl = 0
				for element in itertools.product(*slices_list):
					# print(element)
					xs = [[] for i in sub]
					# flag = False
					k = 0
					for satb in sat_boxes:
						sat = satb.get_map()
						ep_fl = False
						fp = 0
						for p in range(len(extra_pars)): 
							ep = extra_pars[p]
							# if sl < 1 and k < 3:
							# print(sl, element, ep, element[p], len(extra_par_slices[ep]))
							el, eu = extra_par_slices[ep][element[p]-1], extra_par_slices[ep][element[p]]
							
							if el <= sat[ep].leftBound() < eu and el < sat[ep].rightBound() <= eu:
								fp += 1
						if fp == len(extra_pars):
							for i in range(len(sub)):
								p = sub[i]
								x11 = (sat[p].mid().leftBound())
								# x11 = (sat[p][0]+sat[p][1])/2
								xs[i].append(x11)
							k += 1
					# print(k, xs, sl)
					
					if k > max_slice[0]:
						max_slice = (k, xs, sl)
					sl += 1
				
				k, xs, j = max_slice
				fig = plt.figure()
				plt.title('3 project '+str(extra_pars)+' -- max slice {0} -- {1}'.format(j, ratio))
				# print(len(xs), xs)
				# print(xs[0], xs[1])
				if len(xs) > 1:
				# print('plot 2 dim')
					ax = plt.gca()#fig.add_subplot(111) #, projection="3d")
					ax.set_xlabel(axisname[0])
					ax.set_ylabel(axisname[1])
					# ax.set_zlabel(axisname[2])
					ax.plot(xs[0], xs[1], 'b.')
					ax.set_xlim(xlims[0])       
					ax.set_ylim(xlims[1])
				figs.append(fig)
		return figs

	def readFromFile():
		stack = []
		sat_box = []
		pnames = sorted(param_names)
		with open(all_sat_file, 'r') as fp_sat:
			csv_reader = csv.reader(fp_sat, delimiter=',')
			for row in csv_reader:
				b_edges = {}
				j = 0
				# p1 = row[0]
				# p2 = row[1]
				tf = int(row[-1])
				if tf == 1:
					for l in range(0, 2*len(pnames), 2):
						l1 = float(row[l]) #.split(';')[0])
						l2 = float(row[l+1]) #.split(';')[1])
						it = pnames[int(l/2)]  
						b_edges.update({it:PyInterval(l1, l2)})
						j += 1
					dbox = Box(b_edges) 
					sat_box.append(dbox)

		with open(all_q_file, 'r') as fp_q:
			csv_reader = csv.reader(fp_q, delimiter=',')
			for row in csv_reader:
				b_edges = {}
				j = 0
				lst = 1
				# p1 = row[0]
				# p2 = row[1]
				for l in range(lst, lst+len(pnames)):
					l1 = float(row[l].split(';')[0])
					l2 = float(row[l].split(';')[1])
					it = pnames[int(l-lst)]   
					b_edges.update({it:PyInterval(l1, l2)})
					j +=1
				
				# print(row[0], l, row[l], j)
				j = lst+len(pnames)
				dbox = Box(b_edges) 
				b_pri = int(row[j]) 
				b_del = float(row[j+1])
				b_tp = int(row[j+2])
				b_ttyp = int(row[j+3])
				# sat_box.append(dbox)
				stack.append(BoxInfo(b_pri, dbox, b_del, b_tp, b_ttyp))
		return (sat_box, stack, pnames)

	def writeToFile(sat_box, st_q, par_names, unsat_box = []):
		print('--- writeToFile -- ', len(sat_box), st_q.size(), len(unsat_box), par_names)
		pnames = ','.join(par_names)
		with open(all_sat_file, 'w+') as fp_sat:
			csv_writer = csv.writer(fp_sat, delimiter=',')
			for sb in sat_box:
				row = [] #pnames]
				sb_edegs = sb.get_map()
				for it in sorted(sb_edegs.keys()):
					l1 = sb_edegs[it].leftBound()
					l2 = sb_edegs[it].rightBound()
					# s = '{0},{1}'.format(l1, l2)
					row.append(l1)
					row.append(l2)
				row.append(1)
				csv_writer.writerow(row)
			for sb in unsat_box:
				row = [] #pnames]
				sb_edegs = sb.get_map()
				for it in sorted(sb_edegs.keys()):
					l1 = sb_edegs[it].leftBound()
					l2 = sb_edegs[it].rightBound()
					# s = '{0},{1}'.format(l1, l2)
					# row.append(s)
					row.append(l1)
					row.append(l2)
				row.append(0)
				csv_writer.writerow(row)
		with open(all_q_file, 'w+') as fp_q:
			csv_writer = csv.writer(fp_q, delimiter=',')
			# for binfo in st_q:
			while(not st_q.isEmpty()):
				binfo = st_q.pop()
				b_pri, dbox, b_del, b_tp, b_ttyp = binfo.getInfo()
				row = [pnames] #'{0}'.format(datatp)]
				dbox_edges = dbox.get_map()
				for it in sorted(dbox_edges.keys()):
					l1 = dbox_edges[it].leftBound()
					l2 = dbox_edges[it].rightBound()
					s = '{0};{1}'.format(l1, l2)
					row.append(s)
				row.append('{0}'.format(b_pri))
				row.append('{0}'.format(b_del))
				row.append('{0}'.format(b_tp))
				row.append('{0}'.format(b_ttyp))
				
				csv_writer.writerow(row)

	ha1 = getModel(inputfile)

	outputEqns = getEquationsFile(outputfile)
	for var in outputEqns:
		print(var + ' : '+ str(outputEqns[var]))
		ha1.macros.update({var:outputEqns[var]})
	#print(str(ha1))
	ha = ha1.simplify(ONLY=False)
	#print(str(ha))
	'''detect parameters'''
	print('model parsed')
	
	all_data = []
	all_props = []
	tp = 0
	with open(datafile) as fp:
		fr = csv.reader(fp, delimiter=',')
		for row in fr:
			tp += 1
			# if tp == PropRow:
			data = [float(row[i]) for i in range(len(row))]
			(prop, propneg) = getPropertyFromData(data, outputEqns)
			 #convert2Prop(data, dtype) # convert a row to property
			if DEBUG:
				print('Property: ', prop) #, ' State: ', dtype)
				print('PropertyNeg: ', propneg) #, ' State: ', dtype)
			(pn, negpn) = getProperties((prop, propneg)) #, dtype)
			all_data.append(data)
			all_props.append((pn, negpn))

			if tp >= DATA_tp:
				break

	all_params = {}
	if not paramfile == '':
		all_params = getParam(paramfile)
	else:
		inits = {}
		for c in ha.init.condition:
			# print(str(c.literal1), str(c.literal2))
			inits.update({str(c.literal1):str(c.literal2)})

		for var in ha.variables.keys():
			if var == 'time':
				continue
			rng = ha.variables[var]
			if var in all_params.keys() or var not in inits.keys():
				all_params.update({var:rng})

	'''detect parameters'''
	# for par in all_params:
	#   print(par + ' : '+ str(all_params[par]))
		
	#dataSet = Data(datafile)

	#print(str(pha))
	# with open(outfile, 'w') as csvfile:
	#   print('outfile', outfile)

	# param_len = 2 
	param_len = len(all_params.keys())
	subsets_2 = findsubsets(list(all_params.keys()), param_len)
	
	figs = []
	
	for sub in subsets_2: #for each subset of size 2 
		print('For param set', sub, type(sub), default_params, default_param_box)
		params = {}
		other_params = {}
		for key in all_params:
			# print(key)
			if key in sub:
				params.update({key:all_params[key]})
				# d_iv = '(0.5*({0}+{1}))'.format(all_params[key].getleft(), all_params[key].getright())
			else:
				val = all_params[key]
				if key in default_param_box:
					val = default_param_box[key]
				other_params.update({key:val})
				#pha.addInit(key, all_params[key])
				print('params fixed: ', key, str(val))

		pha = convertHA2PHA(ha, params)
		for key in other_params:
			pha.addInit(key, other_params[key])
		print('params', params)
		print('model parameters', str(pha.parameters))

		#(satparam, unsatparam, undetparam) = getEstimatedParameters(pha, params, dataSet, d, k)
		(satparam, unsatparam, undetparam) = getEstimatedParameters(pha, params, d, k_length, all_props)#, satfile)

		sbox_all = getBox(params)
		par_names = sorted((sbox_all.get_map()).keys())
		pnames = ','.join(sorted(par_names))

		print('##################################')
		print('SAT boxes ---', pnames)

		with open(outfile, 'a+') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',')
			i = 0
			names = []
			for b in satparam:
				b_map = b.get_map()
				row = ['data']
				for key in sorted(b_map.keys()):
					row.append(key+'_low')
					row.append(key+'_up')
					names.append(key)
				i += 1          
				spamwriter.writerow(row)
				if i > 0:
					break

			xdata = []
			for b in satparam:
				b_map = b.get_map()
				row = [pnames]
				xr = []
				for key in sorted(b_map.keys()):
					#row.append((float(b_map[key].leftBound()), float(b_map[key].rightBound())))
					row.append(float(b_map[key].leftBound()))
					row.append(float(b_map[key].rightBound()))
					xr.append((b_map[key].leftBound()+b_map[key].rightBound())/2)
				row.append(TRUE)
				xdata.append(xr)
				spamwriter.writerow(row)
			# for b in unsatparam:
			#   b_map = b.get_map()
			#   row = [pnames]
			#   for key in sorted(b_map.keys()):
			#       #row.append((float(b_map[key].leftBound()), float(b_map[key].rightBound())))
			#       row.append(float(b_map[key].leftBound()))
			#       row.append(float(b_map[key].rightBound()))
			#   row.append(FALSE)
			#   spamwriter.writerow(row)

			i += 1
		print('##################################')
		# break
		figs += plot_sat_boxes(sbox_all, satparam, par_names)
		
	if PLOT:
		pp = PdfPages(plotName)
		for fig  in figs:
			pp.savefig(fig)
		pp.close()


	
if __name__ == "__main__":
	start_time = time.time()
	main(sys.argv[1:])
	end_time = time.time()
	print('Code Running duration: ', (end_time - start_time)/3600, ' hrs')

