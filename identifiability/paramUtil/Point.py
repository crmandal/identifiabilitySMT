from __future__ import division
from paramUtil.interval import *
import random
from paramUtil.box import *
import math

class Point(Box):
	def __init__(self, values):
		edges = {}
		for it in values:
			intrvl = PyInterval(values[it], values[it])
			edges.update({it:intrvl})
		Box.__init__(self, edges)
		self._dim = self.size()
	
	def dimension(self):
		return self._dim

	def left(self):
		lower = []
		edges = self.get_map()
		for key in self.edges:
			lower.append(edges[key].left)
		return lower
	
	def right(self):
		upper = []
		edges = self.get_map()
		for key in self.edges:
			upper.append(edges[key].right)
		return upper

	def addValue(self, it, value):
		intrvl = PyInterval(value, value)
		self.addInterval(it, intrvl)
		
	def __repr__(self):
		edges = self.get_map()
		s = '('
		for key in sorted(edges.keys()):
			s += ''+key+': '+ str(edges[key]) + ','
		s += ')'
		return s

	def __str__(self):
		return self.__repr__()
	
	def __add__(self, rhs):
		if isinstance(rhs, self.__class__):
			box = Box.__add__(self, rhs)
			edges = box.get_map()
			res = {}
			for it in edges:
				res.update({it:edges[it].leftBound()})
			return Point(res)
			
		elif isinstance(rhs, int) or isinstance(rhs, float):
			edges = self.get_map()
			res = {}
			for it in edges:
				val = edges[it].leftBound()+rhs
				res.update({it:val})
			return Point(res)
		else:
			raise TypeError("unsupported operand type(s) for +: '{}' and '{}'").format(self.__class__, type(other))
	
	def __sub__(self, rhs):
		if isinstance(rhs, self.__class__):
			box = Box.__sub__(self, rhs)
			edges = box.get_map()
			res = {}
			for it in edges:
				res.update({it:edges[it].leftBound()})
			return Point(res)
			
		elif isinstance(rhs, int) or isinstance(rhs, float):
			edges = self.get_map()
			res = {}
			for it in edges:
				val = edges[it].leftBound()-rhs
				res.update({it:val})
			return Point(res)
		else:
			raise TypeError("unsupported operand type(s) for -: '{}' and '{}'").format(self.__class__, type(other))
	
	def __mul__(self, rhs):
		if isinstance(rhs, self.__class__):
			box = Box.__mul__(self, rhs)
			edges = box.get_map()
			res = {}
			for it in edges:
				res.update({it:edges[it].leftBound()})
			return Point(res)
			
		elif isinstance(rhs, int) or isinstance(rhs, float):
			edges = self.get_map()
			res = {}
			for it in edges:
				val = edges[it].leftBound()*rhs
				res.update({it:val})
			return Point(res)
		else:
			raise TypeError("unsupported operand type(s) for *: '{}' and '{}'").format(self.__class__, type(other))
			
						
	def __truediv__(self, rhs):
		if isinstance(rhs, self.__class__):
			box = Box.__truediv__(self, rhs)
			edges = box.get_map()
			res = {}
			for it in edges:
				res.update({it:edges[it].leftBound()})
			return Point(res)
			
		elif isinstance(rhs, int) or isinstance(rhs, float):
			edges = self.get_map()
			res = {}
			for it in edges:
				val = edges[it].leftBound()/rhs
				res.update({it:val})
			return Point(res)
		else:
			raise TypeError("unsupported operand type(s) for /: '{}' and '{}'").format(self.__class__, type(other))
	
	def __div__(self, rhs):
		if isinstance(rhs, self.__class__):
			box = Box.__truediv__(self, rhs)
			edges = box.get_map()
			res = {}
			for it in edges:
				res.update({it:edges[it].leftBound()})
			return Point(res)
			
		elif isinstance(rhs, int) or isinstance(rhs, float):
			edges = self.get_map()
			res = {}
			for it in edges:
				val = edges[it].leftBound()/rhs
				res.update({it:val})
			return Point(res)
		else:
			raise TypeError("unsupported operand type(s) for /: '{}' and '{}'").format(self.__class__, type(other))
		
		
def distance(point1, point2):
	res = point2 - point1
	rmap = res.get_map()
	disSum = 0
	for it in rmap:
		r = rmap[it].leftBound() * rmap[it].leftBound()
		disSum += r
	return math.sqrt(disSum)
		
def middlePoint(point1, point2):
	# print ('middlePoint', point1, point2)
	# edges = self.get_map()
	# res = {}
	# for it in edges:
	# 	val = edges[it].mid()
	# 	res.update({it:val})
	# p = Point(res)
	p = (point1 + point2)/2.0
	# if DEBUG:
	#print ('Point -- Middle Point: ', str(p))
	#print (point1, point2, res)
	return p

def randomDisplacement(point, EPS):
	dispValues = {}	
	edges = point.get_map()
	for it in edges:
		disp = random.uniform(EPS, 2*EPS)
		sign = (-1)**random.randrange(2)		
		intrvl = edges[it].leftBound() + disp*sign
		#print(str(edges[it]), str(disp*sign), str(intrvl))
		dispValues.update({it:intrvl})
	return Point(dispValues)	

def randomDisplacementwrtPoint(point1, point2, EPS, sign): 
	''' A small random displacement of point1 in with respect to point2 depending on the direction Sign = -1/1'''
	dist = distance(point1, point2)	 
	dispValues = {}
	p1_edges = point1.get_map()
	p2_edges = point2.get_map()
	disp = random.uniform(10*EPS, 20*EPS)
	t = disp*sign
	print('randomDisplacementwrtPoint, t=', str(t))
	for it in p1_edges:		
		intrvl = p1_edges[it].leftBound() * (1 - t) + p2_edges[it].leftBound() * t
		print(str(it), str(intrvl))		
		dispValues.update({it:intrvl})
	return Point(dispValues)
	
def randomDisplacementTowards(point1, point2, EPS): 
	''' A small random displacement of point1 in the same direction of point2'''
	pt = randomDisplacementwrtPoint(point1, point2, EPS, 1)
	print('randomDisplacementTowards: ', str(pt))
	return pt

def randomDisplacementOpposite(point1, point2, EPS): 
	''' A small random displacement of point1 in the opposite direction of point2'''
	pt =  randomDisplacementwrtPoint(point1, point2, EPS, -1)
	print('randomDisplacementOpposite: ', str(pt), ' EPS: ', EPS)
	return pt
