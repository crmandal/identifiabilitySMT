
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
from scipy.stats import qmc

from sklearn.metrics import davies_bouldin_score

from scipy.integrate import odeint
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from scipy.stats import qmc

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
from util.parseOutput import *
from util.smtOutput import *
#from paramUtil.readDataBB import *
from model.node_factory import *
from ha2smt.smtEncoder import *
import ha2ode.HA2ode as hode


from paramUtil.interval import *
from paramUtil.box import *
import paramUtil.box_factory as bfact
import paramUtil.kd_interval as kdi
import paramUtil.kd_intervaltree as kdivt

import approximation as approx

import numpy
import time

PLOT = hode.PLOT
if PLOT:
	# import matplotlib
	# # matplotlib.use('Agg')
	# import matplotlib.pyplot as plt
	from matplotlib.backends.backend_pdf import PdfPages
	# from matplotlib.patches import Rectangle
	import plotUtil as plt

import itertools

# import gp_regres_grad_test as gp_grad

# fig = plt.figure()
import random
# random.seed(10)

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
# read these setup and filenames from config file ---json file

DEBUG = True
DEBUG = False
dRealCmd = "dReal3"
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
IVT = approx.IVT
OPT = False
OPT = True


# INSTANCE = False
INSTANCE = True
FREQ = 2 #0
CLEAN_GEN = 300
RF = False

reachedBoundaryFlag = False

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
		opts, args = getopt.getopt(argv,"hc:", ["ifile="])
	except getopt.GetoptError:
			print("check_box_test_approx.py -c config_file.json > logs/log.txt ")
			sys.exit(2)
			
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print("check_box_test_approx.py -c config_file.json > logs/log.txt ")
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
	NN = configuration["NN"]  #default 2*MIN_EPS
	RES = configuration["RES"] #default 2*MIN_EPS
	LHS = configuration["LHS"] if "LHS" in configuration else 1

	DATA_tp = data_tp

	TB_N = 0.04 #8*MIN_EPS #if FNAME == 'pc' else 8*MIN_EPS
	MAX_EPS = 50*MIN_EPS
	#GRID_SIZE = 3*MIN_EPS

	MIN_EPS_2 = MIN_EPS
	MIN_EPS_3 = 2*MIN_EPS #0.5*MIN_EPS
	#MIN_EPS_3 = 4*MIN_EPS #0.5*MIN_EPS

	# RES = 2*MIN_EPS
	MIN_EPS_4 = NN #10*MIN_EPS
	
	k_length = PATH_LEN
	d = DELTA   

	pName = 'plotBox_'+FNAME+'.pdf'
	satName = 'satbox_'+FNAME+'.csv'

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
			
			if len(f_row) > 0:
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
	max_degree = 1
	connectedComponents = []
	if len(connectedfile) > 0:
		with open(connectedfile, 'r') as csvfile:
			# creating a csv reader object
			csvreader = csv.reader(csvfile)
			for row in csvreader:
				rm = ['{0}'.format(row[r]) for r in range(len(row) - 1)]
				if ifSubset(param_names, rm):
					connectedComponents.append((rm, int(row[-1])))
					if max_degree < int(row[-1]):
						max_degree = int(row[-1])
	else:
		connectedComponents = [(param_names, 1)]
	
	print('connectedComponents', connectedComponents, 'max_degree', max_degree)

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

		# etm = time.time()
		# if tp == 3 and pid == 0 and EV_time = :

		sys.stdout.flush()
		
		if flag:
			exit()
		return ret
				
	def getSAT(model, i, tp, pid = 0, neg = False):

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
		partition of the box through the middle point
		in case the box cannot be partitioned anymore mark it with high priority

	'''
	def partition(sbox, eps):
		boxes = []
		
		if DEBUG:
			print(sbox, eps)
			sys.stdout.flush()
		
		# global noise
		emap = {}
		b_map = sbox.get_map()
		for it in b_map.keys():     
			# emap.update({it: PyInterval(0.01 * noise[it])})
			emap.update({it: PyInterval(eps, eps)})

		#emap = {}
		#b_map = sbox.get_map()
		#for it in b_map.keys():        
		#   emap.update({it: PyInterval(EPS)})
		
		if DEBUG:
			print('Partition: Box = ', str(sbox)) 
		
		boxes = bfact.bisect(sbox, emap, NOMARK) #, mPoint) #negpoint)
		return boxes
			
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

	def len_of_curve(current_box, true_boxes, flag = False):
		smap = current_box.get_map()
		cover = bfact.get_cover(true_boxes, check = False)
		cmap = cover.get_map()
		loc  = 0
		bound = 0
		max_loc, max_p = 0, 0
		lenp = len(cmap.keys())
		lp = (len(true_boxes) >= int(0.8*(2**(lenp-1))*int(current_box.max_side_width()/NN)))
		lp1 = len(true_boxes) >= (int(current_box.max_side_width()/NN))
		lp2 = len(true_boxes) >= int(0.8*(current_box.max_side_width()/NN))
		sc = 0.4 #(1+(lenp-1)*(EB_N-1))
		print('len_of_curve','current_box', str(current_box), 'cover', str(cover), 'loc', loc)
		for it in cmap.keys():
			lpd = cmap[it].width()
			spd = smap[it].width()
			if (lpd/spd >= 0.50): 
			#if (lpd/spd >= sc and lp) or (lpd/spd >= 0.45 and not lp and lp1) or (lpd/spd > 0.5 and lp2): #(spd-lpd) <= 2*sc*lpd): # or (not flag and lpd >= 0.4*spd):
				loc += 1
			if (cmap[it].leftBound() - RES <= smap[it].leftBound()) or (cmap[it].rightBound() + RES >= smap[it].rightBound()):
				bound += 1
			max_loc = max(max_loc, lpd)
			max_p = max(max_p, lpd/spd)
			print('len_of_curve at param {0}'.format(it), lpd, 'max loc', max_loc, 'spd', spd, lpd, 'ratio', lpd/spd, 'lp2', 0.8*(int(current_box.max_side_width()/NN)), 'max_ratio', max_p)
		if (bound > 0 or loc >= lenp - 1 or current_box.atBoundary(cover.addDelta(RES))) and len(true_boxes) > 2*lenp:
			print('True', 'len_of_curve', 'current_box', str(current_box), 'cover', str(cover), 'loc', loc)
			return True, max_p, lp
		else:
			return False, max_p, lp

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

	
	def getEstimatedParameters(model, all_param_len, params, d, klen, all_props, connected): #, satfile):
		sbox_all = getBox(params)
		print('sbox_all', sbox_all, type(sbox_all))
		sat_box = {}
		unsat_box = {}
		undet_box = {}

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
			if j >= len(par_names)-1:
				break

			if len(params) < all_param_len: # only the first given value is correct
				break


		#True_Boxes = [default_box]
		print('Starting with default points', True_Boxes, 'bounding box', sbox_all)
		default_box = True_Boxes[0]
		t_bs = [default_box]#, True_Boxes[1], True_Boxes[2]]

		sbox_map = sbox_all.get_map()
		par_low = []
		par_up = []
		for it in sorted(par_names):
			par_low.append(sbox_map[it].leftBound())
			par_up.append(sbox_map[it].rightBound())
		engine = qmc.LatinHypercube(len(par_names))#,)
		lhs_sample1 = engine.random(n=min(100,5**len(par_names)))
		lhs_sample = qmc.scale(lhs_sample1,  par_low, par_up)
		print('LHS discrepancy', qmc.discrepancy(lhs_sample1))
		print(lhs_sample)

		boxes_to_check = []
		if not OPT:
			delta_to_use = getBoxDelta(sbox_all) #b2.max_side_width()*N_GRIDS
			boxes_to_check.append((sbox_all, delta_to_use))
		else:
			for tb in True_Boxes:
				x_test = {}
				for i in range(len(par_names)):
					pn = par_names[i]
					#il = bfact.min_left_coordinate_value([tb], pn) * 10/12.0
					#ul = bfact.max_right_coordinate_value([tb], pn) *12.0/10
					il = bfact.min_left_coordinate_value([tb], pn)  / EB_N #*(1 - EB_N)
					ul = bfact.max_right_coordinate_value([tb], pn) * EB_N #*(1 + EB_N)
					x_test.update({pn:PyInterval(il, ul)})
				b3 = Box(x_test)
				# b2 = getGRID(b3, sbox_all) #.addDelta(2*RES)
				print('starting box', str(b3))
				#for b22 in [b3]: 
				for b22 in approx.getGRID_boxes(b3, sbox_all, RES):
					b2 = approx.getGRID(b22, sbox_all, RES)
					delta_to_use = getBoxDelta(b2) #b2.max_side_width()*N_GRIDS
					boxes_to_check.append((b2, delta_to_use))
			if LHS or len(True_Boxes) == 1:
				for bmap in lhs_sample: #approx.getGRID_boxes(b3, sbox_all, RES):
					x_test = {}
					for i in range(len(par_names)):
						pn = par_names[i]				
						ll = max(math.floor(bmap[i]/RES)*RES,  sbox_map[pn].leftBound())
						ul = min(math.ceil(bmap[i]/RES)*RES,  sbox_map[pn].rightBound())
						x_test.update({pn:PyInterval(ll, ul)})
					b2 = Box(x_test)
					delta_to_use = getBoxDelta(b2) #b2.max_side_width()*N_GRIDS
					boxes_to_check.append((b2, delta_to_use))

		print('starting box', len(boxes_to_check))
		print('starting box', (boxes_to_check))
		pnames = sorted(par_names)
		
		(bt, bf, bu), figs = getBoxes(model, params, all_props, sbox_all, boxes_to_check, klen, connected, True_Boxes)
		#(bt, bf, bu), figs = getBoxes(model, params, all_props, sbox_all, boxes_to_check, klen, [])
		print('sat boxes', len(bt), len(all_props)) #, bt)

		sat_box = bt
		unsat_box = bf
		undet_box = bu
	
		return (sat_box, unsat_box, undet_box), figs

		
	def getBoxDelta(b):
		#return b.min_side_width()*N_GRIDS 
		return max(b.min_side_width()*N_GRIDS, MIN_DELTA)
	
	def ifBoxEPS(b, tp):
		return (b.max_side_width() < max(EPS*(E_GRIDS**tp), MIN_EPS))
	
	def inStatingBox(sboxes, b):
		for bx1 in sboxes:
			sb, delta_to_use = bx1
			#bi, sb, d, datatp, bttyp = bx.getInfo()
			if sb.contains(b):
				return True
		return False

	def reduceQueue(que, n):
		que1 = que.clone()
		que2 = HeapQ()
		ij = 0
		while(not que1.isEmpty()):
			bi = que1.pop()
			pr, bx, dt, tp, bttyp = bi.getInfo()
			max_side = bx.max_side_width()
			if max_side < RES/2:
				que2.push(bi)
				ij += 1
			if ij >= n:
				break
		return que2


	def getBoxes(model, params, all_props, sbox_all, sboxes, klen, connected, True_boxes = [], act_pt = None): 
		print('########## starting to find the tautology boxes ##############')
		stime = time.time()
		#EPS = 0.01 #max(sbox.min_side_width() * 0.1, d)
		#mgr = multiprocessing.Manager()
		GP_X = []
		GP_y = []
		# pre_GP_x = len(GP_X)
		par_names = sorted((sbox_all.get_map()).keys())

		FREQUENT =  FREQ**(len(par_names)-1)

		mcpu = multiprocessing.cpu_count()
		np1 = int(mcpu)-2 if mcpu > 30 else int(mcpu)-2
		# vol_covered = 0.0
		ratio = 0.0
		ratio2 = 0.0
		ratio1 = 0.0

		fp_count = 0
		
		sat_box = []
		unsat_box = []
		undet_box = []
		# next_appr_sat_box = []
		# decided_boxes = []
		checked_boxes = []
		
		figs = []

		GPC = True
		extra_count = 0
		true_count = 0

		reachedBoundaryFlag = False
		eqns = []

		gpc_count = 0
		# if IVT:
		kdt_false = kdivt.IntervalTree()
		kdt_true = kdivt.IntervalTree()
		print('kdtree', kdt_true)

		False_boxes = []

		initPriority = 10 #-1*sbox_all.volume()

		queue = HeapQ() 

		total_volume = sbox_all.volume()
		# total_volume = 0.0
		env_volume = 0.0
		# if rank == 0:

		if RESUME:
			(sat_bs, qs, par_names)  = plt.readFromFile(all_sat_file)
			sat_box += sat_bs
			True_boxes += sat_bs
			checked_boxes += sat_bs
			for b in sat_bs:
				#bg = approx.getGRID(b, sbox_all, RES).addDelta(MIN_EPS_4)
				# --- oct 10, 2023
				#bg = approx.getGRID(b, sbox_all, RES)
				bg = approx.getGRID(b, sbox_all, NN)
				kdt_true = kdt_true.insert(kdi.KDInterval(bg))
			covered_region = bfact.get_cover(True_boxes, check = False)
			current_sbox = approx.extendEnvelope(covered_region, sbox_all, 10*MIN_EPS)	
			cq =0
			if len(qs)> 0:
				for qb in qs:
					#queue.push(qb)
					# bi, sb, d, datatp, bttyp = qb.getInfo()
					queue.push(qb)
					cq += 1
			else:
				next_boxes_to_check, c33 =  approx.getAPPROXBoxes(sbox_all, pre_envelope, current_sbox, kdt_true, kdt_false, True_boxes, par_names, \
												0, MIN_EPS, MIN_EPS_3, RES, connected, extra = False) #[1]

				#getAPPROXBoxes(sbox_all, pre_envelope, current_sbox, kdt_true, kdt_false, True_boxes, par_names, 0, extra = False)
				for bb in next_boxes_to_check:
					delta_to_use = getBoxDelta(bb)
					pri = initPriority
					# ''' prioritize boxes based on their size '''
					pri = bb.volume()
					queue.push(BoxInfo(pri, bb, delta_to_use, 0, BoxTyp.APPROX))	
					cq += 1
				# total_volume += sb.volume()
			# prop = all_props[datatp-1]
			# delta_to_use = delta_to_use1 #*(N_GRIDS**(datatp))
			print('Resuming from last checkpoint')
			print('getBoxes', cq, total_volume)

		else:
			datatp = 0
			starting_box = []
			for i in range(len(sboxes)): 
				sbox, delta_to_use= sboxes[i]
				starting_box.append(sbox)
				print('getBoxes - iterate',  str(sbox), type(sbox))
				# total_volume += sbox.volume()
				# delta_to_use = #DELTA*(0.1**(datatp))
				pri = initPriority
				# ''' prioritize boxes based on their size '''
				pri = sbox.volume()
				queue.push(BoxInfo(pri, sbox, delta_to_use, datatp, BoxTyp.EXT))

			covered_region = bfact.get_cover(starting_box, check = False)

			sat_box += [] #default_box]
			print('getBoxes', len(sboxes), total_volume)#, 'nodes', nParts)

			current_sbox = bfact.get_cover(starting_box, check = False)	

			'''for b in True_boxes:
				#bg = approx.getGRID(b, sbox_all, RES)
				bg = approx.getGRID(b, sbox_all, NN)
				print('Box', str(b), 'kd_interval', str(bg))
				kdt_true = kdt_true.insert(kdi.KDInterval(bg))'''
		
		true_covered = bfact.get_cover(True_boxes, check = False)
		# print('------------------- Run for row ', datatp, '-------------------------')
		# print('----- data ----', prop)
		print('Starting with --', current_sbox, sbox_all, covered_region, queue.size(), len(sat_box), 'num props', len(all_props), 'k_len', klen)
		pre_envelope = current_sbox #True_boxes[0] #sbox_all


		c1 = 0
		c2 = 0
		gen = 0
		class_flag=0
		NO_HEU_COUNT = 0
		next_boxes_to_check, c33 = [], []
		npg, centres = [] , []
		approx_box_added = 0
		c2 = len(True_boxes)
		i = 0   
		pre_count = 0
		pre_gen = 0
		decided_count = 0
		while(not queue.isEmpty()):
			clean_flag = 0
			#c2 = len(True_boxes)
			min_pri = sbox_all.min_side_width()
			max_pri = 0
			# while(len(List) > 0):
			#b = queue.pop()
			# np = 1 #
			# new_boxes = []
			count = 0
			count2 = 0
			count1 = 0
			min_delta = DELTA

			num_proc = min(np1, queue.size())
			POOL = True if num_proc > 1 else False
			#num_proc = 1 # min(np1, queue.size())
			# POOL = False #True if num_proc > 1 else False
			allboxes = []   
			# for j in range(num_proc):
			# fraction = num_proc
			fraction = int(num_proc/2.0) #
			if OPT and c2 > 10:
				if (i%3 == 0 or GPC_conditions):
					fraction = int(num_proc/4.0)  
				else:
					fraction = int(3*num_proc/4.0)
			j = 0
			# if len(True_boxes) < 10 :
			# 	hull = None
			# elif i%10 == 0:
			# 	hull = getHull(True_boxes) 
			djk = 0
			if not reachedBoundaryFlag:
				while j < fraction and not queue.isEmpty():
					bi = queue.pop()
					pr, bx, dt, tp, bttyp = bi.getInfo()
					min_side = bx.min_side_width()
					if OPT: # and i%3 == 0:
						# pr, bx, dt, tp = bi.getInfo()
						# added on 25/03/2022 -- not include closest points to true regions
						# added on 12/04/2022 -- not includong closest points to true boxes
						#if reachedBoundaryFlag:
						#	bx1 = kdi.KDInterval(bx.addDelta(MIN_EPS_4)) #bx.max_side_width()*1.5))
						#else:
						#bx1 = kdi.KDInterval(bx)    #.addDelta(0)) #bx.max_side_width()*1.5))
						#bx1 = kdi.KDInterval(approx.getGRID(approx.getGRID(bx, sbox_all, RES), sbox_all, NN))
						if len(True_boxes) > 10*(len(params)-1):
							bx1 = kdi.KDInterval(approx.getGRID(bx, sbox_all, RES)) #NN))
						else:
							bx1 = kdi.KDInterval(bx) #approx.getGRID(bx, sbox_all, RES)) #kdi.KDInterval(bx) 
						rt = True
						rf = False
						if IVT:
							rt = True if (bttyp == BoxTyp.EXT and bx.min_side_width() > RES) else (not kdt_true.search(bx1))   #kdt_true.search(bx1)
							# if (bttyp == BoxTyp.EXT and bx.min_side_width() > 1.3*MIN_EPS_2):
							# 	rt = True
							# else:
							# 	#if not inStatingBox(sboxes, bx): #c2 > 500  and bx.min_side_width() > MIN_EPS_2:
							# 	rt = not kdt_true.search(bx1) 
							#rf = kdt_false.searchContains(kdi.KDInterval(bx))
							rf = kdt_false.searchContains(bx1)
						#rr = np.random.uniform(0, 1, 1)[0] < 0.1
						if DEBUG:
							print('Queue -- ', str(approx.getGRID(bx, sbox_all, RES)), kdt_true.search(bx1), 'rt', rt, 'rf', rf)
						if (rt and not rf): # or rr:
							allboxes.append(bi)
							j +=1

							if max_pri < min_side:
								max_pri = min_side
							if min_pri > min_side:
								min_pri = min_side
						else:
							djk += 1
					else:
						allboxes.append(bi)
						if max_pri < min_side:
							max_pri = min_side
						if min_pri > min_side:
							min_pri = min_side
						j +=1

			while j < num_proc and not queue.isEmpty():
				bi1 = queue.popFromEnd() #queue.randomPop()
				pr, bx, dt, tp, bttyp = bi1.getInfo()
				min_side = bx.min_side_width()
				# prioritise this box and its partitions
				bi = BoxInfo(pr, bx, dt, tp, bttyp)
				if OPT: #i%3 == 0:
					# pr, bx, dt, tp = bi.getInfo()
					# added on 25/03/2022 -- not include closest points to true regions
					# added on 12/04/2022 -- not includong closest points to true boxes
					# if reachedBoundaryFlag:
					# 	bx1 = kdi.KDInterval(bx.addDelta(MIN_EPS_4)) #bx.max_side_width()*1.5))
					# else:
					# 	bx1 = kdi.KDInterval(bx)
					#bx1 = kdi.KDInterval(approx.getGRID(approx.getGRID(bx, sbox_all, RES), sbox_all, NN))
					if len(True_boxes) > 10*(len(params)-1):
						bx1 = kdi.KDInterval(approx.getGRID(bx, sbox_all, RES)) #NN))
					else:
						bx1 = kdi.KDInterval(bx) #approx.getGRID(bx, sbox_all, RES)) #kdi.KDInterval(bx) 
					rt = True
					rf = False
					if IVT:
						rt = True if (bttyp == BoxTyp.EXT and bx.min_side_width() > RES) else (not kdt_true.search(bx1)) # and not isInHull(bx, hull)  #kdt_true.search(bx1)
						# rf = kdt_false.search(kdi.KDInterval(bx))
						# rt = True #if (bttyp == BoxTyp.EXT and bx.min_side_width() > 2*MIN_EPS_2) else not kdt_true.search(bx1)   #kdt_true.search(bx1)
						# if (bttyp == BoxTyp.EXT and bx.min_side_width() > 2*MIN_EPS_2):
						# 	rt = True
						# else:
						# 	#if not inStatingBox(sboxes, bx): #c2 > 100 and bx.min_side_width() > MIN_EPS_2:
						# 	rt = not kdt_true.search(bx1) 

						#rf = kdt_false.searchContains(kdi.KDInterval(bx))
						rf = kdt_false.searchContains(bx1)
					#rr = np.random.uniform(0, 1, 1)[0] < 0.1
					if DEBUG:
						print('Queue -- ', str(approx.getGRID(bx, sbox_all, RES)), kdt_true.search(bx1),'rt', rt, kdt_false.searchContains(bx1), 'rf', rf)
					if (rt and not rf) : #or rr:
						allboxes.append(bi)
						j +=1

						if max_pri < min_side:
							max_pri = min_side
						if min_pri > min_side:
							min_pri = min_side
					else:
						djk += 1
				else:
					allboxes.append(bi)

					if max_pri < min_side:
						max_pri = min_side
					if min_pri > min_side:
						min_pri = min_side
					j +=1

			print('discarded box', djk, len(allboxes), kdt_true.len())
			num_proc = len(allboxes)  #min(num_proc, )
			print('Using multiprocessing : {0}, prces {1},  Boxes left -- {2}, time -- {3:2f} hrs-- pri: {4}, {5} '.format(POOL, num_proc, queue.size(), ((time.time()-stime)/3600), max_pri, min_pri))

			s_ptime = time.time()
			if POOL and num_proc > 1:
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
			e_ptime = time.time()
			print('------- TIME ---------- Time taken for {0} boxes {1} hr'.format(num_proc, (e_ptime - s_ptime)/3600.0))
			print('------- TIME ---------- Time taken for checking {0} boxes {1} hr'.format(decided_count, (e_ptime - stime)/3600.0))
			for j in range(num_proc):       
				bi = allboxes[j]    
				pr, b, dt, tp, bttyp = bi.getInfo()
				# delta_to_use = delta_to_use1*(EPS**tp)
				be = b.get_map()
				if min_delta > dt:
					min_delta = dt
				
				# covered_1 = bfact.get_cover([covered_region]+[b])
				# covered_region = covered_1
				
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
					decided_count += 1
					#break
					# decided_boxes.append(b)
					False_boxes.append(b)


					if b.min_side_width() >= 0.5*MIN_EPS: # RES/2:
						# bg = b #.addDelta()#approx.getGRID(b, sbox_all, MIN_EPS)
						# bg = kdi.KDInterval(b)
						bg = kdi.KDInterval(approx.getGRID(b, sbox_all, RES))
						print('False in ivt', str(b), 'NN', str(bg))
						kdt_false = kdt_false.insert(bg)

				elif(r == TRUE):    
					''' if given range subset of b then return true '''
					if tp+1 < len(all_props):
						if DEBUG:
							print('## 1-sat box : ', tp, '<', len(all_props), str(b))	
						''' check for next property '''
						dt1 = getBoxDelta(b) #max(b.max_side_width()*N_GRIDS, MIN_DELTA) #max(b.minDimension()*EPS, MIN_DELTA)
						# if tp < 2:
						pri = pr-1
						queue.push(BoxInfo(pri, b, dt1, tp+1, bttyp)) # adding to queue
						if DEBUG:
							print('---- check next property {0} with delta {1}'.format(tp+1, dt1))
						count += 1
						
					else:
						''' all properties are covered --- stop '''
						sat_box.append(b)
						# vol_covered += b.volume()           
						count2 += 1
						decided_count += 1

						c2 += 1
						True_boxes.append(b)
						# decided_boxes.append(b)     
						checked_boxes.append(b)
						
						#bg = approx.getGRID(b, sbox_all, RES)#.addDelta(MIN_EPS_4)
						# -- oct 10, 2023
						bg = approx.getGRID(b, sbox_all, NN)
						#approx.getGRID(b, sbox_all, RES), sbox_all, NN)
						kdt_true = kdt_true.insert(kdi.KDInterval(bg))
						print('@@@@ True box -- decided : ', tp, '<', len(all_props), str(b), 'NN for ivt', str(bg))   

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
							#pri = pr -1
							# ''' prioritize boxes based on their size '''
							pri = b.volume() #min_side_width()
							queue.push(BoxInfo(pri, b, dt1, tp+1, bttyp)) # adding to queue
						
							count += 1
							if DEBUG:
								print('---- check next property {0} with delta {1}'.format(tp+1, dt1))
							
						else:
							''' all properties are covered --- stop '''
							# sys.stdout.flush()
							inst1 = instance.p1
							inst2 = instance.p2
							
							satb = inst1.getSATBox()

							sat_box.append(satb)
							# vol_covered += b.volume()     
							count2 += 1
							decided_count += 1

							# extra_boxes = bfact.remove(b, satb)
							# for b2 in extra_boxes:
							#   vol_covered += b2.volume()  
					
							c2 += 1
							True_boxes.append(satb)
							# decided_boxes.append(satb)     
							checked_boxes.append(b)
							# if IVT:
							
							#bg = approx.getGRID(b, sbox_all, RES) #.addDelta(MIN_EPS_4)
							# -- oct 10, 2023
							bg = approx.getGRID(b, sbox_all, NN)
							#approx.getGRID(approx.getGRID(satb, sbox_all, RES), sbox_all, NN)

							kdt_true = kdt_true.insert(kdi.KDInterval(bg)) #.addDelta(MIN_EPS_4)))		
							print('@@@@ < EPS {0:0.4f} SAT box -- decided'.format(max(EPS*(E_GRIDS**tp), MIN_EPS)), tp, 'boxtype', bttyp,'box',str(satb), 'NN for ivt', str(bg))  #, 'removed {0} smaller boxes'.format(len(extra_boxes)))
							extra_count = 0
					else:
						# delta1 = dt #min(sbox.minDimension() * 0.1, MIN_DELTA)
						boxes = partition(b, eps) #MIN_DELTA) #MIN_EPS) # dt*EPS)
						# print('--- box partition', b, len(boxes), MIN_DELTA)
				
						nbx = []
						#print('Checking partitioned boxes...')
						for b1 in boxes:
							# print(b2)
							#pri = pr #- 1
							dt1 = getBoxDelta(b1) #max(b1.max_side_width()*N_GRIDS, MIN_DELTA)
							# if dt1 < dt:
							pri = b1.volume()
							queue.push(BoxInfo(pri, b1, dt1, tp, bttyp))
								#else:
							if DEBUG:
								print('Added box after only partitioning', str(b1)) 
							#new_boxes.append((b1, dt1))
							count += 1
							# else:
							#   vol_covered += b1.volume()  
							#   print('-- discarding box {0} after multiple attempts'.format(b1))
						ts1 = kdt_true.len() if kdt_true else 0.0
						ts2 = kdt_false.len() if kdt_false else 0.0
					
						#if DEBUG:
						print(ts1, ts2, 'checking partitioned boxes...', len(boxes), len(nbx))#, str(nbx))
			
			# if fflag:
			# 	break
			queue_size = queue.size()
			decided_box = True_boxes+unsat_box
			next_boxes_to_check = []
			vol_covered = 0.0
			for bb in  sat_box:
				vol_covered += bb.volume()
			for bb in  unsat_box:
				vol_covered += bb.volume()
			# envelope = bfact.get_cover(sat_box)
			vol_checkd = 0.0
			for cb in checked_boxes:
				vol_checkd += cb.volume()

			pre_true_covered = true_covered
			true_covered =  bfact.get_cover(True_boxes, check=False)
			all_covered = bfact.get_cover(decided_box, check=False) #if decided_box > 0 else 


			c2 = len(True_boxes)
			loc, max_p, lp = len_of_curve(current_sbox, True_boxes, flag = True)
			if loc: #(c2 - pre_count) > FREQUENT and:
				pre_envelope = current_sbox #true_covered
				current_sbox = approx.extendEnvelope(current_sbox, sbox_all, 2*RES)
				print('##### Increased current envelope', str(current_sbox), 'pre envelope', str(pre_envelope), '--- queue_size', queue_size, 'lp', lp)
				if queue_size > 500:
					#queue.clean(10)
					queue = reduceQueue(queue, min(0.1*queue_size, 50*(len(params)-1)))
					queue_size = queue.size()
					clean_flag = 1
				print('--- current queue_size', queue_size)
				if c2:
					print('True boxes', str(true_covered)) #[str(tb) for tb in True_boxes])
				if pre_envelope == current_sbox:
					pre_envelope = true_covered
					current_sbox = approx.extendEnvelope(true_covered, sbox_all, 2*RES)
			'''elif (c2 - pre_count) > FREQUENT:
				if c2 % 50 == 0:
					pre_envelope = true_covered
					current_sbox = approx.extendEnvelope(true_covered, sbox_all, 2*RES)
				elif gen % 50 == 25:
					pre_envelope = true_covered
					current_sbox = approx.extendEnvelope(true_covered, sbox_all, 2*RES)'''

			if OPT : # max_pri < 2*MIN_EPS_2:
				ifCovered = (current_sbox.volume() <= pre_envelope.volume() and current_sbox.volume() >= sbox_all.volume())
				print('In OPT', OPT, 'ifCovered', ifCovered)
				if (all_covered.volume() >= sbox_all.volume() and current_sbox.volume() >= sbox_all.volume()) or  vol_covered >= sbox_all.volume():
					print('may be stopped -- gen', (i, gen), queue_size, all_covered.volume(), sbox_all.volume(),\
						'No exploration -- all covered ({0}),  true_covered ({1}), current env {2}, sbox ({3})'.format(all_covered, true_covered, current_sbox, sbox_all))

					#print('No more extension -- gen', (i, gen), queue_size, vol_covered, sbox_all.volume(),\
					#	'-- all covered ({0}), true_covered ({1}), current env {2}, sbox ({3})'.format(all_covered, true_covered, current_sbox, sbox_all))

				elif ifCovered: 
					print('Covered Extension --- calculating once -- gen', (i, gen), queue_size, vol_covered, sbox_all.volume(),\
						'-- all covered ({0}), true_covered ({1}), current env {2}, sbox ({3})'.format(all_covered, true_covered, current_sbox, sbox_all))

					if GPC and c2 > len(par_names):
						next_boxes_to_check, c33 =  approx.getAPPROXBoxes(sbox_all, pre_envelope, current_sbox, kdt_true, kdt_false, True_boxes, False_boxes, par_names, \
												gen, MIN_EPS, MIN_EPS_3, RES, connected, extra = False, class_flag=class_flag)
						#approx.getAPPROXBoxes(sbox_all, pre_envelope, current_sbox, kdt_true, kdt_false, True_boxes, par_names, gen, extra = False)
						cc1 = 0
						nbx = []
						for bb in next_boxes_to_check:
							pri = pr #np.random.uniform((min_pri+max_pri)/2, max_pri, 1)[0] #initPriority-2
							delta_to_use = getBoxDelta(bb)
							pri = bb.volume()
							queue.push(BoxInfo(pri, bb, delta_to_use, 0, BoxTyp.APPROX))	
							approx_box_added += 1
							cc1 += 1
					GPC = False
				#((i-pre_gen) > 20 and i%5 == 0) or or queue_size%200 == 0

				elif GPC:
					GPC_conditions = True #False
					class_flag = 0
					''' only search for new nodes if current nodes are checked'''
					'''if c2 >= 2*len(par_names):
						other_cond = (max_pri < RES or (NO_HEU_COUNT > 10 )) #and max_pri < 0.25*RES))
						if (FREQUENT/2 < (c2 - pre_count) < FREQUENT and i%5 == 0 and other_cond) or (queue_size < 5*np1 and i%2 == 0): # queue_size < 5*np1 and ## last case when queue is depleting
							GPC_conditions = True
						if 5*np1 <= queue_size < 200 and ((c2 - pre_count) > FREQUENT or i%3 == 0) and other_cond:
							GPC_conditions = True
						if 200 <= queue_size < 500 and ((c2 < 500 or (c2 - pre_count) > FREQUENT) and i%7 < 2) and other_cond:
							GPC_conditions = True
						if 500 <= queue_size < 1000 and ((c2 < 1000 or (c2 - pre_count) > FREQUENT) and i%20 < 2) and other_cond:
							GPC_conditions = True
						if 1000 <= queue_size and (((c2 - pre_count) > FREQUENT) and i%50 < 2) and other_cond:
							GPC_conditions = True
					'''

					if not GPC_conditions:
						NO_HEU_COUNT += 1
					if queue_size > 500*(len(par_names)-1):
						GPC_conditions = False

					print('gen', (i, gen), 'GPC_conditions', GPC_conditions, 'queue_size', queue_size, 'reachedBoundaryFlag', reachedBoundaryFlag)
					if GPC_conditions and not reachedBoundaryFlag:
						NO_HEU_COUNT = 0

						print('GPC_conditions', (i, gen), queue_size, 'not reachedBoundaryFlag')
						gpc_count += 1
						next_boxes_to_check, c33 =  approx.getAPPROXBoxes(sbox_all, pre_envelope, current_sbox, kdt_true, kdt_false, True_boxes, False_boxes, par_names, \
												gen, MIN_EPS, MIN_EPS_3, RES, connected, extra = False, class_flag=class_flag)
						#approx.getAPPROXBoxes(sbox_all, pre_envelope, current_sbox, kdt_true, kdt_false, True_boxes, par_names, gen, extra = False)
						cc1 = 0
						nbx = []
						# for nb in kdt_true.items():
						# 	bb = nb.toBox(param_names)
						for bb in next_boxes_to_check:
							#pri = pr #initPriority #np.random.uniform((min_pri+max_pri)/2, max_pri, 1)[0] #initPriority-2
							delta_to_use = getBoxDelta(bb)
							# ''' prioritize boxes based on their size '''
							pri = bb.volume()
							queue.push(BoxInfo(pri, bb, delta_to_use, 0, BoxTyp.APPROX))	
							cc1 += 1
							approx_box_added += 1
							# nbx.append(bb)
						# true_covered =  bfact.get_cover(kdt_npg_sat)
						print('After GPC added Approx -- gen', (i, gen), queue_size, 'New boxes added', cc1, 'out of',\
							len(next_boxes_to_check), vol_covered, sbox_all.volume(),'---- Extra---',\
							'Extended all covered ({0}),  true_covered ({1}), current env {2}, sbox ({3})'.format(all_covered, true_covered, \
								current_sbox, sbox_all))
						npg, centres = next_boxes_to_check, c33
						gen += 1


						# if reachedBoundaryFlag:
						# 	reachedBoundaryFlag = False

				diff_box = true_covered.addDelta(MIN_EPS_4) #2*RES)
				print('Current -- gen', (i, gen), queue_size, current_sbox.fullyContains(diff_box), 'true_covered', true_covered, \
					'diff', diff_box, 'current', current_sbox, 'pre_env', pre_envelope)	
				if  ((clean_flag < 1 and len(params) > 2 and (true_covered.volume() > 0.0 and vol_covered/true_covered.volume() > 0.75)  or current_sbox.atBoundary(true_covered) or \
					not current_sbox.fullyContains(diff_box) or (queue_size < np1 and max_pri < MIN_EPS and i > 20))):	#or current_sbox.atBoundary(all_covered)	
				#if  ((true_covered.volume() > 0.0 and vol_covered/true_covered.volume() > 0.75)  or current_sbox.atBoundary(true_covered) or \
				#	not current_sbox.fullyContains(diff_box) or (queue_size < np1 and max_pri < MIN_EPS and i > 20)):	#or current_sbox.atBoundary(all_covered)	
		
					print('----- Checking to extend bounding box -- gen', (i, gen), queue_size, 'Extending --', 'sbox', sbox_all, 'covered', true_covered, \
					'all covered ({0}),  true_covered ({1}), current env {2}, sbox ({3}), diff {4}'.format(all_covered, true_covered, current_sbox, sbox_all, diff_box))
					ifCovered1 = (true_covered.volume() >= sbox_all.volume())
					#and max_pri < MIN_EPS_2
					print('Check conditions: ', i> 20, queue_size < np1, not ifCovered1, not GPC)
					#if  OPT  and (gpc_count  >= 50 or (i> 20 and queue_size < np1 and not ifCovered1 and not GPC)):
					if  OPT  and ((i> 20 and queue_size < 2*np1 and not ifCovered1 and not GPC) or (queue_size < np1 and len(next_boxes_to_check) < 1)):

						#extra_boxes = [extra_box] 
						extended_box1 = approx.extendEnvelope(true_covered, sbox_all, 2*RES) #10*MIN_EPS) #EPS)

						# current_sbox = extended_box1
						extra_boxes = bfact.remove(extended_box1, true_covered)
						print(extra_boxes)
						for bb in extra_boxes:
							delta_to_use = getBoxDelta(bb) #bb.max_side_width()*N_GRIDS
							# pri = np.random.uniform(min_pri, max_pri, 1)[0] #
							gpc_count  = 0
							# pri = initPriority+1 #+1
							pri = -2 #min_pri-2 #+1

							# ''' prioritize boxes based on their size '''
							pri = bb.volume()
							queue.push(BoxInfo(pri, bb, delta_to_use, 0, BoxTyp.EXT))
							approx_box_added += 1

							#kdt_true = kdt_true.insert(kdi.KDInterval(bb.addDelta(0)))
							print('Extend extra -- gen', (i, gen), queue_size, 'Extending --', bb)

						extra_count += 1
						print('In extend box', extra_count)

					if (diff_box.volume() >= sbox_all.volume()):
						reachedBoundaryFlag = True
						print('Extended -- gen', (i, gen), queue_size, 'reachedBoundaryFlag')
			
				else:
					print('---- Not extension  --gen', (i, gen), queue_size, vol_covered, sbox_all.volume(), \
						'all covered ({0}),  true_covered ({1}), current env {2}, sbox ({3})'.format(all_covered, true_covered, current_sbox, sbox_all))
			pre_ratio = ratio2

			ratio = (100.0*vol_covered/total_volume) if total_volume > 0.0 else 100.0
			ratio1 = (100.0*env_volume/total_volume) if total_volume > 0.0 else 100.0
			ratio2 = (100.0*vol_checkd/total_volume) if total_volume > 0.0 else 100.0
			if abs(pre_ratio - ratio2) > 0.5 : #not int(pre_ratio) == int(ratio):
				print('___________________________________')
				print('Envelope {0:0.2f}% {1:0.2f}% of the Box checked, {2:0.2f}% decided .....'.format(ratio, ratio2, ratio1))
				print('___________________________________')
			print('\n----- New boxes added: {0}/{1}, previous {2}, decided {3}, min delta {4}, decided {5:0.2f}%, checked {6:0.2f}-{7:0.2f}%, envelope {8:0.2f}%'.format(count, c2, pre_count, count2, min_delta, ratio1, pre_ratio, ratio2, ratio)) #+ ' box decided: '+ str(count2) + ' min_delta: '+ str(min_delta)
			# if pre_count == count and count2 == 0:
			#   print('New boxes', new_boxes)
			#   break
			#print(sk)
			# print(i, queue_size)
			if (c2 - pre_count) == 0 and gen%4 == 0:
				true_count += 1
			else:
				true_count = 0

			if (true_covered.volume() > pre_true_covered.volume()):
				true_count = 0

			if (c2 - pre_count) > FREQUENT :#or (i > 100 and (c2 - pre_count) < 5.0):
				print(' --- Plotting started ----')
				st_clone = queue.clone()
				next_boxes_to_check, c33 =  approx.getAPPROXBoxes(sbox_all, pre_envelope, current_sbox, kdt_true, kdt_false, True_boxes, False_boxes, par_names, \
												gen, MIN_EPS, MIN_EPS_3, RES, connected, extra = False,class_flag=1)
				#getAPPROXBoxes(sbox_all, pre_envelope, current_sbox, kdt_true, kdt_false, True_boxes, par_names, gen, extra = False)
				
				plt.writeToFile(all_sat_file, all_q_file, sat_box, st_clone, par_names, unsat_box) 
				print('---- Number of valid points', len(sat_box), 'total decided', len(checked_boxes), len(unsat_box), c2, pre_count)
				print('@@@ --- temp plot ---', (time.time() - stime)/3600, 'hr')
				pre_count = c2
				pre_gen = i
				print('--- plotting for gen ', i, 'NPG', str(i%2==0))
				pre_eqn = eqns
				figs, eqns = plt.plot_sat_boxes(sbox_all, sat_box, par_names, ratio1, npg, centres, i, unsat_box, df=connected[1]) 
				print('------------- regression equation --------', fp_count, '-----------')
				print(pre_eqn)				
				print(eqns)
				print('------------------------------------------', queue_size)
				if PLOT:
					pp = PdfPages(plotName)
					for fig  in figs:
						pp.savefig(fig)
					pp.close()
				#sys.stdout.flush()

				if eqns == pre_eqn:
					fp_count += 1
				else:				
					fp_count = 0

				
				# else:
				# 	print('--------------- Condition 1.1: completing iterations -------------', len(par_names), max_degree, fp_count)

				# 	if len(par_names) <= 2 and max_degree < 2  and (fp_count > 50 or (fp_count > 5 and lp)): # 2D, curve
				# 		print('1.1 --------------- completing iterations -------------', len(par_names), max_degree, fp_count)
				# 		break
				# 	elif len(par_names) > 2 and max_degree < 2 and (fp_count > 50 or (fp_count > 5 and lp)): # >2D, curve
				# 		print('2.1 --------------- completing iterations -------------', len(par_names), max_degree,  fp_count)
				# 		break
				# 	elif len(par_names) > 2 and  max_degree >= 2 and (fp_count > 50 or (fp_count > 5 and lp)): # >2D, surface
				# 		print('3.1 --------------- completing iterations -------------', len(par_names), max_degree, fp_count)
				# 		break

				print('--- Plotting completed ----')
			#diff_box = true_covered.addDelta(MIN_EPS_4)
			if len(sat_box) > 0:
				loc, max_p, lp  = len_of_curve(sbox_all, True_boxes)
			else:
				loc, max_p, lp = False, 0.0, False
			df = connected[1]
			print('--- len_of_curve (full_box) ---', loc, str(sbox_all), 'valid ', len(sat_box), 'box_lenth', sbox_all.max_side_width(), 'lp', lp, 'threshold', (2**(len(par_names)-1))*int(sbox_all.max_side_width()/NN), 'max_p', max_p)
			#if loc:
			print('--------------- Condition 1: completing iterations -------------', len(par_names), df, max_degree, fp_count, loc, reachedBoundaryFlag)

			if len(par_names) <= 2 and df < 2  and ((fp_count > 100 and max_p > 0.5) or (fp_count > 5 and (loc or reachedBoundaryFlag))): # 2D, curve
				print('1 --------------- completing iterations -------------', len(par_names), df, max_degree, fp_count)
				break
			elif len(par_names) > 2 and df < 2 and ((fp_count > 50 and max_p > 0.5) or (fp_count > 5 and (loc or reachedBoundaryFlag))): # >2D, curve
				print('2 --------------- completing iterations -------------', len(par_names), df, max_degree,  fp_count)
				break
			elif len(par_names) > 2 and  df >= 2 and ((fp_count > 50 and max_p > 0.5) or (fp_count > 5 and (loc or reachedBoundaryFlag))): # >2D, surface
				print('3 --------------- completing iterations -------------', len(par_names), df, max_degree, fp_count)
				break
			
			if TEST and i > 5 and len(sat_box) > 500 :
				print('---- Condition 2 ----')
				st_clone = queue.clone()
				plt.writeToFile(all_sat_file, all_q_file, sat_box, st_clone, par_names, unsat_box) 
				print('4 ---- Number of valid points', len(sat_box), 'total decided', len(checked_boxes), len(unsat_box), c2)
				break


			if true_count > 200 and gen > 500:
				print('---- Condition 2.1 ----', extra_count, true_count, gen)
				st_clone = queue.clone()
				plt.writeToFile(all_sat_file, all_q_file, sat_box, st_clone, par_names, unsat_box) 
				print('4.1 ---- Number of valid points', len(sat_box), 'total decided', len(checked_boxes), len(unsat_box), c2)
				break


			if len(sat_box) > len(par_names)*100: 
				print('---- Condition 3 ----')
				diff_box = true_covered.addDelta(MIN_EPS_4)
				if not sbox_all.fullyContains(diff_box):
					print('5 --------------- completing iterations -------------', sbox_all, diff_box, true_covered)
					print('---- Number of valid points', len(sat_box), 'total decided', len(checked_boxes), len(unsat_box), c2)
					st_clone = queue.clone()
					plt.writeToFile(all_sat_file, all_q_file, sat_box, st_clone, par_names, unsat_box) 
					break

			#	if queue_size > 0:
			i += 1
			print('iteration', i, 'gen', gen, '--------------- queue size -------------', queue_size)
			#sys.stdout.flush()


		print('--- Beyond the while loop ----')
		checked_boxes = sat_box+unsat_box
		if len(checked_boxes) > 0:
			covered_box = bfact.get_cover(checked_boxes)
			print('volumes', covered_box.volume(), sbox_all.volume(), 'total decided', len(checked_boxes))
		print('{0}% of the Box covered .....'.format(ratio1))
		if len(sat_box) > len(par_names): 
			st_clone = queue.clone()
			plt.writeToFile(all_sat_file, all_q_file, sat_box, st_clone, par_names, unsat_box) 
			figs, eqns = plt.plot_sat_boxes(sbox_all, sat_box, par_names, ratio1, unsat_box=unsat_box, df = max_degree) 
			print('------------- regression equation --------')		
			print(eqns)
			print('------------------------------------------')
			print('---- Number of valid points', len(sat_box), 'total decided', len(checked_boxes), len(unsat_box), c2)
			if len(sat_box) < 10:
				print('Sat boxes', [str(b.get_mean())+'\n' for b in sat_box])
		if PLOT and len(figs)>0:
			pp = PdfPages(plotName)
			for fig  in figs:
				pp.savefig(fig)
			pp.close()

		sys.stdout.flush()

		return (sat_box, unsat_box, undet_box), figs      


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

	# param_len = 2 
	param_len = len(all_params.keys())
	#subsets_2 = findsubsets(list(all_params.keys()), param_len)
	
	figs = []
	ic = 0
	for connected in connectedComponents:
		ic += 1
		cc1, df  = connected
		print('######## Running for Connected component {0} with dof {1} ######'.format(cc1, df))
		# = len(cc1)
		# subsets_2 = findsubsets(list(all_params.keys()), param_len)
		# for sub in subsets_2: #for each subset of size 2 
		# 	if list(sub) == cc1:
		sub = cc1		
		
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
			intrvl = other_params[key]
			pha.addInit(key, intrvl)
			#edges = sbox.get_map()
		'''
		#for it in edges:
			#if DEBUG:
			#	print('InUpdate Model: ', intrvl.leftBound(), str(intrvl.rightBound()))
			param = Range(Node(intrvl), Node(intrvl))
			if key in pha.parameters:
				pha.parameters.update({key: param})
			if key in pha.variables:
				pha.variables.update({key: param}) '''
		s = ''
		for key in pha.parameters:
			s += key + ':' + str(pha.parameters[key])+ ' , '
		print('updatedModel', 'model.params: ', s)
		print('params', params, 'all_params', param_len)
		print('model parameters', str(pha.parameters))
		print('------------')
		#(satparam, unsatparam, undetparam) = getEstimatedParameters(pha, params, dataSet, d, k)
		(satparam, unsatparam, undetparam), fig1 = getEstimatedParameters(pha, param_len, params, d, k_length, all_props, connected)#, satfile)

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
		figs += fig1 #plot_sat_boxes(sbox_all, satparam, par_names)

		temp_plot_name = prefix + 'plotBox_'+FNAME+'connected_{0}.pdf'.format(ic)
		pp = PdfPages(temp_plot_name)
		for fig  in figs:
			pp.savefig(fig)
		pp.close()
		
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

