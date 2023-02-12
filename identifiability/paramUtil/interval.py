
 # Created by devleena on 07/07/20.

# #include <iostream>
# //#include <capd/capdlib.h>
# #include <capd/intervals/lib.h>
# #include "interval.h"
# #include <boost/python.hpp>
# #include <boost/python/raw_function.hpp>

# #include <iostream>
# #include <typeinfo>
# #include <Python.h>
# using namespace std;
# using namespace boost::python;
# using namespace capd;
import portion as DInterval
# from portion.const import Bound, inf

PInf = 9999999999.0
NInf = -PInf

def adjacent(lhs, rhs):
	if lhs.left == DInterval.CLOSED and rhs.right == DInterval.OPEN or \
		lhs.left == DInterval.OPEN and rhs.right == DInterval.CLOSED or \
		lhs.right == DInterval.CLOSED and rhs.left == DInterval.OPEN or \
		lhs.right == DInterval.OPEN and rhs.left == DInterval.CLOSED:
		return lhs.adjacent(rhs)
	else : #if lhs.left == DInterval.CLOSED and rhs.right == DInterval.CLOSED and lhs.right == DInterval.CLOSED and rhs.left == DInterval.CLOSED :
		a = lhs.lower
		b = lhs.upper
		c = rhs.lower
		d = rhs.upper
		if a == d or b == c:  # The intervals share a boundary x.lower = y.upper or x.upper = y.lower
			return True
	return False


class PyInterval(object):
	"""docstring for ClassName"""
	def __init__(self, left=None, right=None, closed = 0, box=True):
		# by default closed interval
		'''
		0 = both closed
		1 = left closed
		2 = right closed
		3 = both open
		4 = empty
		5 = singleton
		'''
		if left is None:
			self.component = DInterval.empty()
		elif right is None:
			# self.component = DInterval.closed(left, PInf)
			# self.closed = 0
			self.component = DInterval.singleton(left)
		elif closed == 0:
			if right is None or left is None:
				print('Incorrect interval: bounds should be specified')
				return
			self.component = DInterval.closed(left, right)
			self.closed = 0
		elif closed == 1:
			if right is None or left is None:
				print('Incorrect interval: bounds should be specified')
				return
			self.component = DInterval.closedopen(left, right)	
			self.closed = 1	
		elif closed == 2:
			if right is None or left is None:
				print('Incorrect interval: bounds should be specified')
				return
			self.component = DInterval.openclosed(left, right)
			self.closed = 2
		elif closed == 3:
			if right is None or left is None:
				print('Incorrect interval: bounds should be specified')
				return
			self.component = DInterval.open(left, right)
			self.closed = 3
		else:
			print('Incorrect interval: bounds should be specified')
			return
		self.marker = False
		self.__data = []
		self.box = box
		# print('base', self.component, left, right, self.closed)

	def setData(self,data):
		self.__data = data
	def getData(self):
		return self.__data

	def getComponent(self):
		return self.component

	def setComponent(self, component):
		self.component = component

	def mark(self):
		self.marker = True

	def unmark(self):
		self.marker = False

	def isMarked(self):
		return self.marker

	def leftBound(self):
		return (self.component).lower

	def rightBound(self):
		return (self.component).upper
	
	def __str__(self):
		s = DInterval.to_string(self.component)
		# print('box ', self.box)
		if not self.box:
			s += ', data: ['
			i = 0
			for b in self.__data:
				if i == 0:
					s += 'box ('+ str(b)+')'
				else:
					s += ', box ('+ str(b)+')'
				i += 1
			s += ']'
		# s += ')'
		return s

	def __repr__(self):
		return self.__str__()

	def clone(self):
		left = self.leftBound()
		right = self.rightBound()
		closed = self.closed
		box = self.box
		# interval.component = DInterval.closed(self.leftBound(), self.rightBound())
		interval = PyInterval(left, right, closed, box)				
		interval.marker = self.marker
		interval.setData(self.__data)
		return interval
		
	def empty(self):
		it = self.component
		# print('Empty :', it)
		return it.empty #()

	def mid(self):
		it = self.component
		mid = (it.lower + it.upper)/2
		it1 = PyInterval(mid)
		return it1

	def width(self):
		it = self.component
		if self.empty():
			width = 0.0
		else:
			# print('width :', it)
			width = (it.upper - it.lower)
		# print('width :', it,width)
		return width

	def diam(self):
		it1 = PyInterval(self.width())
		return it1

	def __lt__(self, other):
		if isinstance(other, PyInterval):
			return self.component < other.component
		elif isinstance(other, int) or isinstance(other, float) :
			return self.component < other
		else:
			print('Incorrect data type for < ', type(self), type(other))
			return False

	def __le__(self, other):
		if isinstance(other, PyInterval):
			return self.component <= other.component
		elif isinstance(other, int) or isinstance(other, float) :
			return self.component <= other
		else:
			print('Incorrect data type for <= ', type(self), type(other))
			return False

	def __ge__(self, other):
		if isinstance(other, PyInterval):
			return self.component > other.component
		elif isinstance(other, int) or isinstance(other, float) :
			return self.component > other
		else:
			print('Incorrect data type for >= ', type(self), type(other))
			return False

	def __gt__(self, other):
		if isinstance(other, PyInterval):
			return self.component >= other.component
		elif isinstance(other, int) or isinstance(other, float) :
			return self.component >= other
		else:
			print('Incorrect data type for > ', type(self), type(other))
			return False

	def __eq__(self, other):
		if isinstance(other, PyInterval):
			return self.component == other.component
		elif isinstance(other, int) or isinstance(other, float) :
			return self.component == other
		elif other is None:
			return False
		else:
			print('Incorrect data type for == ', type(self), type(other))
			return False


	def __ne__(self, other):
		return not self.__eq__(other)

	def __add__(self, other):
		a = self.leftBound()
		b = self.rightBound()
		rhs = None
		if isinstance(other, PyInterval):
			rhs = other
		elif isinstance(other, int) or isinstance(other, float) :
			rhs = PyInterval(other)			
		else:
			print('Incorrect data type for + ', type(self), type(other))
			return None
		c = rhs.leftBound()
		d = rhs.rightBound()
		# print('interval add', self, other)
		# print(a, b, c, d)
		p = PyInterval(a + c,  b + d)
		# print(a + c,  b + d, p)
		return p

	def __sub__(self, other):
		a = self.leftBound()
		b = self.rightBound()
		rhs = None
		if isinstance(other, PyInterval):
			rhs = other
		elif isinstance(other, int) or isinstance(other, float) :
			rhs = PyInterval(other)			
		else:
			print('Incorrect data type for - ', type(self), type(other))
			return None
		c = rhs.leftBound()
		d = rhs.rightBound()
		return PyInterval(a - d,  b - c)

	def __mul__(self, other):
		a = self.leftBound()
		b = self.rightBound()
		rhs = None
		if isinstance(other, PyInterval):
			rhs = other
		elif isinstance(other, int) or isinstance(other, float) :
			rhs = PyInterval(other)			
		else:
			print('Incorrect data type for * ', type(self), type(other))
			return None
		c = rhs.leftBound()
		d = rhs.rightBound()
		return PyInterval(np.min([a*c, a*d, b*c, b*d]), np.max([a*c, a*d, b*c, b*d]))


	def __truediv__(self, other):
		a = self.leftBound()
		b = self.rightBound()
		rhs = None
		if isinstance(other, PyInterval):
			rhs = other
		elif isinstance(other, int) or isinstance(other, float) :
			rhs = PyInterval(other)			
		else:
			print('Incorrect data type for / ', type(self), type(other))
			return None
		c = rhs.leftBound()
		d = rhs.rightBound()
		return PyInterval(np.min([a/c, a/d, b/c, b/d]), np.max([a/c, a/d, b/c, b/d]))

	def contains(self, other):
		if isinstance(other, PyInterval):
			return self.containsInterval(other)
		elif isinstance(other, int) or isinstance(other, float) :
			return self.containsPoint(other)
		else:
			print('Incorrect data type for \\in ', type(self), type(other))
			return None

	def fullyContains(self, other, d = 0.0001):
		if isinstance(other, PyInterval):
			r1 = self.leftBound() - other.leftBound()
			r2 = self.rightBound() - other.rightBound()
			# print('FullyContains['+it+'] ', r1, r2, d)
			if (r1 > d and r2 > d):
				return True
			else:
				return False
		elif isinstance(other, int) or isinstance(other, float) :
			return self.containsPoint(other)
		else:
			print('Incorrect data type for \\in ', type(self), type(other))
			return None

	def containsInterval(self, other):
		lhs = self.component
		rhs = other.component
		return lhs.contains(rhs)


	def containsPoint(self, e):
		lhs = self.component
		return lhs.contains(e)


	def intersects(self, other):		
		lhs = self.component
		rhs = other.component
		res = lhs.intersection(rhs)
		return ~(res.is_empty())

	def union(self, other):		
		lhs = self.component
		rhs = other.component
		res = lhs.union(rhs)
		data = self.getData()
		other_data = other.getData()
		box = other.box			
		res_data = []
		if len(data) == 0:
			res_data = other_data
		else:
			for d in data + other_data:
				res_data.append(d)	
		
		#[todo:] in case of multiple disjoint intervals
		pyint = PyInterval(box = box) #res.lower, res.upper, box = box)
		pyint.setComponent(res)
		pyint.setData(res_data)
		return pyint
		# if res.atomic:
		# 	pyint = PyInterval(box = box) #res.lower, res.upper, box = box)
		# 	pyint.setComponent(res)
		# 	pyint.setData(res_data)
		# 	return pyint
		# else:
		# 	result = []
		# 	for r in res:
		# 		# print('iteration :' , str(r))
		# 		pyint = PyInterval(box = box) #r.lower, r.upper, box = box)
		# 		pyint.setComponent(res)
		# 		pyint.setData(res_data)
		# 		result.append(pyint)
		# 	return result

	def intersection(self, other):
		lhs = self.component
		rhs = other.component
		res = lhs.intersection(rhs)
		data = self.getData()
		other_data = other.getData()
		box = other.box		
		res_data = []
		if len(data) == 0:
			res_data = other_data
		else:
			for d in data:
				if d in other_data:
					res_data.append(d)	
		pyint = PyInterval(res.lower, res.upper, box = box)
		pyint.setData(res_data)
		return pyint


	def is_atomic(self):
		#True if and only if the interval is a disjunction of a single (possibly empty) interval.
		return self.component.atomic

	def get_enclosure(self):
		# smallest atomic interval that includes the current one.
		res = self.component.enclosure
		return PyInterval(res.lower, res.upper)

	def complement(self, other):
		res = self.component - other.component	
		data = self.getData()
		other_data = other.getData()
		box = True #other.box	
		res_data = []
		if len(data) == 0:
			res_data = other_data
		else:
			for d in data:
				if d not in other_data:
					res_data.append(d)	
		# print('complement', res_data)
		pyint = PyInterval(box = box) #res.lower, res.upper, box = box)
		pyint.setComponent(res)
		pyint.setData(res_data)
		return pyint
		# if res.atomic:
		# 	pyint = PyInterval(res.lower, res.upper, box = box)
		# 	pyint.setData(res_data)
		# 	return [pyint]
		# else:
		# 	result = []
		# 	for r in res:
		# 		# print('iteration :' , str(r))
		# 		pyint = PyInterval(r.lower, r.upper, box = box)
		# 		pyint.setData(res_data)
		# 		result.append(pyint)
		# 	return result

	def difference(self, other):
		lhs = self.component
		rhs = other.component
		res = lhs.difference(rhs)		
		return PyInterval(res.lower, res.upper)

	def isAdjacent(self, other):
		lhs = self.component
		rhs = other.component
		return lhs.overlaps(rhs) and (lhs.difference(rhs) == lhs)
		# print('isAdjacent', str(lhs), str(rhs), lhs.adjacent(rhs))
		
		
	def overlaps(self, other):
		lhs = self.component
		rhs = other.component
		return lhs.overlaps(rhs)

	def merge(self, interval2):
		interval1 = self
		# if interval1.isAdjacent(interval2):
		it = interval1.union(interval2)
		# print('union', str(it))
		return it.get_enclosure()
		# else:
		# 	print('Intervals cannot be merged')


# def cover(interval):
# 	i = 0
# 	union_int = None
# 	print('cover', interval)
# 	for inode in interval.component:
# 		for int1 in inode:
# 			if i == 0:
# 				print('1. union_int', i, union_int, ',', int1) #.component)
# 				union_int = int1#.component
# 			else:

# 				print('1. union_int', i, union_int, ',',  int1) #.component)
# 				#if adjacent(union_int, int1.component):
# 				union_int = union_int.union(int1) #.component)
# 				# if union_int.isAdjacent(int1):
# 				# ui = union_int.component.union(int1.component)
# 				# union_int = PyInterval(ui.lower, ui.upper)
# 				# elif union_int.contains(int1):					
# 					# ui = union_int.component - int1.component
# 					# union_int = PyInterval(ui.lower, ui.upper)
# 			print('2. union_int', i, union_int) #, ',', int1.component)
# 			i += 1
# 	if union_int is None:
# 		return []
# 	else:
# 		return union_int
# 	# elif union_int.atomic:
# 	# 	return [PyInterval(union_int.lower, union_int.upper)]
# 	# else:
# 	# 	result = []
# 	# 	for r in union_int:
# 	# 		# print('iteration :' , str(r))
# 	# 		result.append(PyInterval(r.lower, r.upper))
# 	# 	return result


