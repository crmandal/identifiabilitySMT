import os
import subprocess
import re
import sys, getopt
import time
from multiprocessing import Event
import threading

from collections import OrderedDict
from time import gmtime, strftime

from model.haModel import *
from model.property import *
from parser.parseModel import *
from parser.parseProperty import *
from util.graph import *

INT_MAX = 1000
INT_MIN = 1/100000
cmd = "dReach";
SAT = 51
UNSAT = 52
ERROR = -1
tempfolder = os.path.join('outputs','temp')

def timing(event):
	print ("timer starts")
	ts = time.time()
	event.wait()
	te = time.time()
	elapsed_time = te - ts
	print ("Elapsed Time " + str(elapsed_time))

def getReachabilityOld(model, goal, k, d, tempfile, prefixTable, tcount, filter = False) :
	out = ERROR
	#print(goal)
	if(goal == SAT or goal == UNSAT or goal == ERROR):
		out = goal
		print('getReachability: goal: ', goal, ' out: ', out)
		#print(goal, out)
		return (out,tcount)
	else:
		if(filter):
			#out = UNSAT
			print('Path based dReach check')
			g = model.getGraph()
			g.printG()
			st = model.init.mode
			tgt = goal.mode
			cnd = goal.condition
			#print('path(', st, tgt, ') = ')
			#paths = 
			#print('path(', st, tgt, ') = ')
			#if(len(paths) > 0):
			print('checking paths...')
			count1 = 0
			count2 = 0
			for path in g.getPathsofLength(st, tgt, k):
			#	print(string(path))
				count1 = count1+1
				if(count1 > 10):
					break;
				flag = True
				prefixes = getPrefixes(path)
				#print(prefixes)
				# if(len(prefixTable) > 0):
				# 	print('checking prefixes...')
				for pre in prefixes:
					prestr = string(pre)
					#print('in prefix check', pre, prestr, '####')
					if(len(prefixTable) == 0):
						break;
					elif(prestr in prefixTable):
						#print('#############',prestr)
						if(prefixTable[prestr] == UNSAT): # filter out the paths where prefixes are not reachable
						#	print(prestr, ' prefix match')
							flag = False
							break;				
				if(flag):
					#print(path)
					count2 = count2 + 1
					#print(count1, string(path), count2)
					pathstr = string(path)
					out = reach(model, goal, k, d, tempfile, pathstr)
					if(out == UNSAT):
						if(len(cnd) == 1 and ('true' in cnd)):
							prefixTable.update({pathstr : out})
						#print(prefixTable)
					elif(out == SAT):
						print('getReachability: goal: ', goal.mode, goal.condition, ' out: ', out, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
						return out
				else:
					out = UNSAT;
					#print(count1, string(path))
					continue;

			print('getReachability: goal: ', goal.mode, goal.condition, ' out: ', out, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
			print('Number of paths checked with Filter = '+ str(filter)+ ' for goal <'+goal.mode+', '+str(out)+'> : '+str(count2))
			tcount += count2
			print('Number of paths to be checked without Filter = '+ str(filter)+ ' for goal <'+goal.mode+', '+str(out)+'> : '+str(count1))
			return (out, tcount)
		else:
			print('Direct dReach check')
			out = reach(model, goal, k, d, tempfile)
			print('getReachability: goal: ', goal.mode, goal.condition, ' out: ', out, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
			return (out, 0)


def getReachabilityV1(model, goal, k, d, tempfile, prefixTable, tcount, filter = False) :
	out = ERROR
	#print(goal)
	if(goal == SAT or goal == UNSAT or goal == ERROR):
		out = goal
		print('getReachability: goal: ', goal, ' out: ', out)
		#print(goal, out)
		return (out, tcount)
	else:
		if(filter):
			#out = UNSAT
			print('Path based dReach check')
			g = model.getGraph()
			#g.print()
			st = model.init.mode
			tgt = goal.mode
			cnd = goal.condition
			#print('path(', st, tgt, ') = ')
			#paths = g.getPathsofLength(st, tgt, k)
			#print('path(', st, tgt, ') = ')
			#if(len(paths) > 0):
			print('checking paths...')
			count1 = 0
			count2 = 0
			for path in g.getPathsofLength(st, tgt, k):
			#	print(string(path))
				count1 = count1+1
				flag = True
				prefixes = getPrefixes(path)
				#print(prefixes)
				# if(len(prefixTable) > 0):
				# 	print('checking prefixes...')
				for pre in prefixes:
					prestr = string(pre)
				#	print('in prefix check', pre, prestr, '####')
					if(len(prefixTable) == 0):
						break;
					elif(prestr in prefixTable):
						#print('#############',prestr)
						if(prefixTable[prestr] == UNSAT): # filter out the paths where prefixes are not reachable
						#	print(prestr, ' prefix match')
							prefixTable.update({pathstr : UNSAT})
							flag = False
							break;	
					else:
						count2 = count2 + 1
						pathstr = string(path)
						g = Goal(pre[len(pre)-1], 'true')
						out1 = reach(model, g, k, d, tempfile, prestr)
						print('#############',prestr, out1)
						if(out1 == UNSAT):
							prefixTable.update({prestr : out1})
							#print(prefixTable)
							flag = False
							break;
						elif(out1 == SAT):
							prefixTable.update({prestr : out1})
				if(flag):
					
					count2 = count2 + 1
					#print(count1, string(path), count2)
					pathstr = string(path)
					print('check path ', pathstr)
					out = reach(model, goal, k, d, tempfile, pathstr)
					if(out == UNSAT):
						if(len(cnd) == 1 and ('true' in cnd)):
							prefixTable.update({pathstr : out})
						#print(prefixTable)
					elif(out == SAT):
						print('getReachability: goal: ', goal.mode, goal.condition, ' out: ', out, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
						return out
				else:
					out = UNSAT;
					#print(count1, string(path))
					continue;

			print('getReachability: goal: ', goal.mode, goal.condition, ' out: ', out, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
			print('Number of paths checked with Filter = '+ str(filter)+ ' for goal <'+goal.mode+', '+str(out)+'> : '+str(count2))
			tcount += count2
			print('Number of paths to be checked without Filter = '+ str(filter)+ ' for goal <'+goal.mode+', '+str(out)+'> : '+str(count1))
			return (out, tcount)
		else:
			print('Direct dReach check')
			out = reach(model, goal, k, d, tempfile)
			print('getReachability: goal: ', goal.mode, goal.condition, ' out: ', out, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
			return (out, 0)

def getReachability(model, goal, k, d, tempfile, prefixTable, tcount, filter = False) :
	print('getReachability: ',tempfile)
	out = ERROR
	#print(goal)
	tcount1, tcount2 = tcount
	tc = tcount2
	tc1 = tcount1
	if(goal == SAT or goal == UNSAT or goal == ERROR):
		out = goal
		print('getReachability: goal: ', goal, ' out: ', out)
		#print(goal, out)
		return (out, tcount1, tcount2)
	else:
		if(filter):
			#out = UNSAT
			print('Path based dReach check', k)
			g = model.getGraph()
			#g.printG()
			st = model.init.mode
			tgt = goal.mode
			cnd = goal.condition
			#print('path(', st, tgt, ') = ')
			#paths = g.getPathsofLength(st, tgt, k)
			#print('path(', st, tgt, ') = ')
			#if(len(paths) > 0):
			print('checking paths...')
			count1 = 0
			count2 = 0
			for path in g.getPathsofLength(st, tgt, k):
				pathstr = string(path)	
				#print(pathstr)			
				count1 = count1+1
				#if(count1 > 10):
				#	break
				flag = True
				prefixes = getPrefixes(path)
				#revPrefixes = getPrefixes(path, True)
				#print(prefixes)
				# if(len(prefixTable) > 0):
				# 	print('checking prefixes...')
				for pre in prefixes:
					prestr = string(pre)
				#	print('in prefix check', pre, prestr, '####')
					if(len(prefixTable) == 0):
						break;
					elif(prestr in prefixTable):
						#print('#############',prestr)
						if(prefixTable[prestr] == UNSAT): # filter out the paths where prefixes are not reachable
						#	print(prestr, ' prefix match')
							prefixTable.update({pathstr : UNSAT})
							out = UNSAT
							flag = False
							break;	
				if(flag):					
					count2 = count2 + 1
					#print(count1, string(path), count2)
					print('check path ', pathstr)
					out = reach(model, goal, k, d, tempfile, pathstr)
					if(out == UNSAT):
						if(len(cnd) == 1 and ('true' in cnd)):
							prefixTable.update({pathstr : out})
					elif(out == SAT):
						for pre in prefixes:
							prestr = string(pre)
							prefixTable.update({prestr : SAT})
						print('getReachability: goal: ', goal.mode, goal.condition, ' out: SAT', strftime("%Y-%m-%d %H:%M:%S", gmtime()))
						break;
						#return out
				else:
					out = UNSAT;
					#print(count1, string(path))
					#continue;
			# end for loop
			print('getReachability: goal: ', goal.mode, goal.condition, ' out: ', out, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
			tc += count2
			tc1 += count1
			print('Number of paths checked with Filter = '+ str(filter)+ ' for goal <'+goal.mode+', '+str(out)+'> : '+str(count2))
			print('Number of paths to be checked without Filter = '+ str(filter)+ ' for goal <'+goal.mode+', '+str(out)+'> : '+str(count1))
			return (out, tc1, tc)
		else:
			print('Direct dReach check')
			out = reach(model, goal, k, d, tempfile)
			print('getReachability: goal: ', goal.mode, goal.condition, ' out: ', out, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
			return (out,0,0)


def getReachabilityV3(model, goal, k, d, tempfile, prefixTable, tcount, filter = False) :
	out = ERROR
	#print(goal)
	tc = tcount
	if(goal == SAT or goal == UNSAT or goal == ERROR):
		out = goal
		print('getReachability: goal: ', goal, ' out: ', out)
		#print(goal, out)
		return (out, tcount)
	else:
		if(filter):
			#out = UNSAT
			print('Path based dReach check')
			g = model.getGraph()
			#g.print()
			st = model.init.mode
			tgt = goal.mode
			cnd = goal.condition
			#print('path(', st, tgt, ') = ')
			#paths = g.getPathsofLength(st, tgt, k)
			#print('path(', st, tgt, ') = ')
			#if(len(paths) > 0):
			print('checking paths...')
			count1 = 0
			count2 = 0
			for path in g.getPathsofLength(st, tgt, k):
			#	print(string(path))
				pathstr = string(path)				
				count1 = count1+1
				
				flag = True
				prefixes = getPrefixes(path)
				revPrefixes = getPrefixes(path, True)
				#print(prefixes)
				# if(len(prefixTable) > 0):
				# 	print('checking prefixes...')
				for pre in prefixes:
					prestr = string(pre)
				#	print('in prefix check', pre, prestr, '####')
					if(len(prefixTable) == 0):
						break;
					elif(prefixTable.has_word(prestr)):
						data = prefixTable.getValue(prestr)
						#print('#############',prestr)
						if(data == UNSAT): # filter out the paths where prefixes are not reachable
						#	print(prestr, ' prefix match')
							prefixTable.add(pathstr, UNSAT)
							flag = False
							break;	
				if(flag):					
					count2 = count2 + 1
					#print(count1, string(path), count2)
					print('check path ', pathstr)
					out = reach(model, goal, k, d, tempfile, pathstr)
					if(out == UNSAT):
						if(len(cnd) == 1 and ('true' in cnd)):
							prefixTable.add(pathstr, out)
							#prefixTable.update({pathstr : out})
							for rpre in prefixes:
								rprestr = string(rpre)
								if(rprestr not in prefixTable):
									g = Goal(rpre[len(rpre)-1], 'true')
									#if(goal.mode == g.mode):
									print('check prefix path ', rprestr)
									out1 = reach(model, g, k, d, tempfile, rprestr)
									count2 = count2 + 1
									if(out1 == UNSAT):
										prefixTable.update({rprestr : UNSAT})
										break;
							# for rpre in rprefixes:
							# 	rprestr = string(rpre)
							# 	if(rprestr not in prefixTable):
							# 		g = Goal(rpre[len(rpre)-1], 'true')
							# 		#if(goal.mode == g.mode):
							# 		print('check prefix path ', rprestr)
							# 		out1 = reach(model, g, k, d, tempfile, rprestr)
							# 		count2 = count2 + 1
							# 		if(out1 == UNSAT):
							# 			prefixTable.update({rprestr : UNSAT})
							# 		elif(out1 == SAT):
							# 			prefixTable.update({rprestr : SAT})
							# 			subprefixes = getPrefixes(rpre)
							# 			for spre in subprefixes:
							# 				sprestr = string(spre)
							# 				prefixTable.update({sprestr : SAT})
							# 			break;

					elif(out == SAT):
						for pre in prefixes:
							prestr = string(pre)
							prefixTable.update({prestr : SAT})
						print('getReachability: goal: ', goal.mode, goal.condition, ' out: SAT', strftime("%Y-%m-%d %H:%M:%S", gmtime()))
						break;
						#return out
				else:
					out = UNSAT;
					#print(count1, string(path))
					continue;

			print('getReachability: goal: ', goal.mode, goal.condition, ' out: ', out, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
			tc += count2
			print('Number of paths checked with Filter = '+ str(filter)+ ' for goal <'+goal.mode+', '+str(out)+'> : '+str(count2))
			print('Number of paths to be checked without Filter = '+ str(filter)+ ' for goal <'+goal.mode+', '+str(out)+'> : '+str(count1))
			return (out, tc)
		else:
			print('Direct dReach check')
			out = reach(model, goal, k, d, tempfile)
			print('getReachability: goal: ', goal.mode, goal.condition, ' out: ', out, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
			return (out,0)



def reach(model, goal, k, d, tempfile, path = "") :
	fname = os.path.join(tempfolder, tempfile)
	
	print("reach: temp name ", tempfile, fname)
	sys.stdout.flush()
	addGoalToModel(model, goal, fname) #tempfile);	
#	print(path)
	#fname = os.path.join(tempfolder, tempfile)
	#fname = tempfile
	if(path == ""):
		st = [cmd, "-z", "-k", str(k), fname, "--precision", str(d)]
	else:
		st = [cmd, "-z", "-k", str(k), "-p", path, fname, "--precision", str(d)]
		#st = cmd + ' -p ' + path + ' ' + tempfile + ' --precision '+ str(d)
		#shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)]

	print(st)
	p = subprocess.Popen(st)
	(output, err) = p.communicate()  
	
	#This makes the wait possible
	p_status = p.wait()	
	#This will give you the output of the command being executed
	print ("Command output: " + str(p_status))
	out = p_status
	#print('goal : ', goal, 'out : ', out)
	sys.stdout.flush()
	return out
	
def addGoalToModel(model, p, temp):
	#print("temp name ", temp)
	model.deleteGoal()
	model.addGoal(p.mode, p.condition)
	fname = temp
	#print(str(goal))
	model.saveModel(fname)
	#model1 = model

if __name__ == "__main__": 
   main(sys.argv[1:])
