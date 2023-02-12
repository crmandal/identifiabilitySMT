
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from paramUtil.interval import *
from paramUtil.box import *
from paramUtil.kd_interval import *
import numpy as np
from paramUtil.const import *
import random as rnd
#from paramUtil.cartesian import *
#from itertools import product
import itertools
DEBUG = False

def cartesianProduct(set_a, set_b): 
	result =[] 
	for i in range(0, len(set_a)): 
		for j in range(0, len(set_b)): 
			# for handling case having cartesian 
			# prodct first time of two sets 
			if type(set_a[i]) != list:
				set_a[i] = [set_a[i]] 
			# coping all the members 
			# of set_a to temp 
			temp = [num for num in set_a[i]] 
			# add member of set_b to  
			# temp to have cartesian product      
			temp.append(set_b[j])              
			result.append(temp)   
	if DEBUG:
		print('cartesianProduct', result)
	return result 

# Function to do a cartesian  
# product of N sets  
def cartesian(list_a):
	n = len(list_a) 
	# result of cartesian product 
	# of all the sets taken two at a time 
	result = list_a[0] 
	# do product of N sets  
	for i in range(1, n): 
		result = cartesianProduct(result, list_a[i]) 
	return result
  


class IntervalType(object):
	"""docstring for ClassName"""
	def __init__(self):
		self._leftPoint = None
		self._righPoints = []
		self._numInterval = 0
	@property
	def lP(self):
		return self._leftPoint
	@property
	def rP(self):
		return self._righPoints
	@property
	def nI(self):
		return self._numInterval

	def __repr__(self):
		# s = 'lp: '+ str(self.lP) + ', rp: ' + str(self.rP) + ', nI: '+ str(self.nI)
		s = str(self.lP) + ', ' + str(self.rP) + ', '+ str(self.nI)
		return s

	def __str__(self):
		return self.__repr__()

	def add(self, interval):
		if not isinstance(interval, KDInterval):
			return NotImplemented
		if interval.atomic:
		# for interval in kdinterval:
			if self._leftPoint is None:
				self._leftPoint = interval.lower
				self._righPoints.append(interval.upper)
				self._numInterval += 1
				self._righPoints = sorted(self._righPoints)

			elif self._leftPoint == interval.lower:
				# self._leftPoint = interval.lower
				self._righPoints.append(interval.upper)
				self._numInterval += 1
				self._righPoints = sorted(self._righPoints)
			else:
				print('Interval not added', self.lP, interval.lower)
				return
			if DEBUG:
				print('Interval added', self, interval.lower)

	def remove(self, interval): 
		if not isinstance(interval, KDInterval):
			return NotImplemented      
		if kdinterval.atomic:
			if self._leftPoint == interval.lower and interval.upper in self.rP:
				if self._numInterval - 1 == 0 :
					self._leftPoint = []
				self._righPoints.remove(interval.upper)
				self._numInterval -= 1

	def overlap(self, interval):
		if not isinstance(interval, IntervalType):
			return NotImplemented
		if kdinterval.atomic:
			if self.rP[self.nI - 1] <= interval.lP or interval.rP[interval.nI - 1] <= self.lP:
				return False
			return True
		return NotImplemented


class IntervalTree(object):
	"""docstring for ClassName"""
	def __init__(self):
		self._intervalNode = IntervalType()
		self._rmax = KDPoint()
		self._height = 0
		self._leftchild = None
		self._rightchild = None
		self._empty = True
	
	@property
	def dimension(self):
		return self._rmax.dimension
	
	@property
	def I(self):
		return self._intervalNode

	@property
	def rMax(self):
		return self._rmax
  
	@property
	def height(self):
		return self._height 

	@property
	def lC(self):
		return self._leftchild 

	@property
	def rC(self):
		return self._rightchild

	@property
	def isEmpty(self):
		return self._empty
	
	def len(self):
		if self is None:
			return 0
		l = self.lC.len() if self.lC else 0
		r = self.rC.len() if self.rC else 0
		return self.I.nI+l+r

	def copy(self, itree):
		self._intervalNode = itree.I
		self._rmax = itree.rMax
		self._height = itree.height
		self._leftchild = itree.lClC
		self._rightchild = itree.rC
		self._empty = itree.isEmpty

	def inorder(self):
		if self is None:
			return ''
		s = ''
		if self.lC is not None:
			s += ' '+ self.lC.inorder()
		# s += ' ['+str(self._intervalNode) +  ', rMax: '+ str(self.rMax) + ']'
		s += ' ('+str(self._intervalNode) +  ', '+ str(self.rMax) + ')'
		if self.dimension > 1:
			s += '\n'
		if self.rC is not None:
			s += ' '+self.rC.inorder()
		return s

	def preorder(self):
		if self is None:
			return ''
		# s = ' ['+str(self._intervalNode) +  ', rMax: '+ str(self.rMax) + ']'
		s = ' ('+str(self._intervalNode) +  ', '+ str(self.rMax) + ')'
		if self.dimension > 1:
			s += '\n'
		if self.lC is not None:
			s += ' '+self.lC.preorder()
		if self.rC is not None:
			s += ' '+self.rC.preorder()
		return s

	def __repr__(self):
		s = '\nIO:' + self.inorder()+ '\nPO:'+ self.preorder()
		return s

	def __str__(self):
		return self.__repr__()

	def items(self):
		if self is None:
			return []
		s = []
		if self.lC is not None:
			s += self.lC.items()
		# s += ' ['+str(self._intervalNode) +  ', rMax: '+ str(self.rMax) + ']'
		l = []
		for u in self.I.rP:
			l.append(KDInterval(Atomic(self.I.lP, u)))
		s += l
		if self.rC is not None:
			s += self.rC.items()
		return s
	
	def leftRotate(self):
		if DEBUG:
			print('1 left rotate', self.I)
		root = self
		y = root.rC
		t2 = y.lC

		# Perform rotation 
		y._leftchild = root
		root._rightchild = t2

		# Update heights
		root._height = 1 + max(root.lC.height if root.lC  else 0, root.rC.height if root.rC  else 0)
		y._height = 1 + max(y.lC.height if y.lC  else 0, y.rC.height if y.rC  else 0)

		# Return the new root

		if DEBUG:
			print('leftRotate-after', y) 
		return y
		# print('after', self.I)

	def rightRotate(self):
		if DEBUG:
			print('1 right rotate', self.I)
		root = self
		y = root.lC 
		T3 = y.rC 
  
		# Perform rotation 
		y._rightchild = root 
		root._leftchild = T3 
  
		# Update heights       
		root._height = 1 + max(root.lC.height if root.lC  else 0, root.rC.height if root.rC  else 0)
		y._height = 1 + max(y.lC.height if y.lC else 0, y.rC.height if y.rC  else 0)
  
		# Return the new root
		if DEBUG:
			print('rightRotate-after', y) 
		return y 
		

	def getBalance(self):
		if self is None or self.isEmpty: 
			return 0
		lch = self.lC.height if self.lC else 0
		rch = self.rC.height if self.rC  else 0
		balance = lch-rch
		if DEBUG:
			print('balance at (', self.I, ')', lch, rch, balance)
		return balance

	def avlbalance(self, interval):
		# get the balance factor

		balance = self.getBalance()
		if abs(balance) > 1:
			if DEBUG:
				print('avlbalance : ', balance, self.I) #
				print('before ', self)
				print('----------------- AVL balancing ------------------------------')
		else:
			return self
		#, interval.upper < self.lC.rMax, interval.upper > self.lC.rMax, interval.upper < self.rC.rMax, interval.upper > self.rC.rMax)
		# Step 4 - If the node is unbalanced,  
		# then try out the 4 cases 
		# Case 1 - Left Left 
		if balance > 1 and self.lC and interval.lower < self.lC.I.lP: 
			if DEBUG:
				print('case-1', interval.lower, self.lC.I.lP, interval.lower < self.lC.I.lP)
			root = self.rightRotate() 
			# self = root
			if DEBUG:
				print('@@ avlbalance ', root)
			return root
  
		# Case 2 - Right Right 
		if balance < -1 and self.rC and interval.lower > self.rC.I.lP: 
			if DEBUG:
				print('case-2', interval.lower, self.rC.I.lP, interval.lower > self.rC.I.lP)
			root = self.leftRotate() 
			# self = root
			if DEBUG:
				print('@@ avlbalance', root)
			return root
  
		# Case 3 - Left Right 
		if balance > 1 and self.lC and interval.lower > self.lC.I.lP: 
			if DEBUG:
				print('case-3', interval.lower, self.lC.I.lP, interval.lower > self.lC.I.lP)
			root_lc = self._leftchild.leftRotate() 
			self._leftchild = root_lc
			root = self.rightRotate() 
			if DEBUG:
				print('@@ avlbalance', root)
			return root
  
		# Case 4 - Right Left 
		if balance < -1 and self.rC and interval.lower < self.rC.I.lP: 
			if DEBUG:
				print('case-4', interval.lower, self.rC.I.lP, interval.lower < self.rC.I.lP)
			root_rc = self._rightchild.rightRotate() 
			self._rightchild = root_rc
			root = self.leftRotate()
			if DEBUG:
				print('@@ avlbalance', root)
			return root

		# if abs(balance) > 1:
		#     print('after', self)

	def update_rMax(self, dim):
		rmax1 = self.lC.rMax if self.lC is not None else [-Inf for i in range(dim)]
		rmax2 = self.rC.rMax if self.rC is not None else [-Inf for i in range(dim)]
		if DEBUG:
			print('rmax calculation:', self.I.rP, self.I.nI-1, rmax1, rmax2)
		self._rmax = np.max([self.I.rP[self.I.nI-1], rmax1, rmax2])

	def update_height(self):
		self._height = 1 + max(self.lC.height if self.lC is not None else 0 , self.rC.height if self.rC is not None else 0) 

	def insert(self, interval):
		if DEBUG:
			print('In tree insert: ', interval)
		if not isinstance(interval, KDInterval):
			if isinstance(interval, list) or isinstance(interval, tuple):
				p1 = [i*(100.0 - (10.0 if i>0 else -10.0))/100.0 for  i in interval]
				p2 = [i*(100.0 + (10.0 if i>0 else -10.0))/100.0  for  i in interval]
				interval1 = KDInterval(Atomic(p1, p2))
				return self.insert(interval1)
			elif isinstance(interval, KDPoint):
				p1 = [i*(100.0 - (10.0 if i>0 else -10.0))/100.0 for  i in interval.V]
				p2 = [i*(100.0 + (10.0 if i>0 else -10.0))/100.0  for  i in interval.V]
				interval1 = KDInterval(Atomic(p1, p2))
				return self.insert(interval1)
			else:
				return NotImplemented
		if interval.atomic:
			if self.isEmpty:
				if DEBUG:
					print('[1] ## ',' interval: ', interval, 'adding node to empty node: ', self, 'interval: ', interval)
				self._intervalNode.add(interval)
				self._empty = False
			else:
				if self.I.lP == interval.lower:
					if DEBUG:
						print('[2] ## ', 'interval: ', interval, ' adding node to already existed node: ', self.I)
					self._intervalNode.add(interval)                
				else:    
					if DEBUG:
						print('@@ ', 'interval: ', interval,' deciding on children at ', self.I, ',rMax', self.rMax,  'right' if self.I.lP < interval.lower else 'left')
					if self.I.lP < interval.lower:
						if not self.rC:  
							self._rightchild = IntervalTree()

						# print('Adding to right child: ', self._rightchild, '\nroot', self)
						rc = self._rightchild.insert(interval)
						self._rightchild  =  rc # recursive call to right  
						# print('After adding to right child: ', self._rightchild , '\nroot', self)
					else:
						if not self.lC:                    
							self._leftchild = IntervalTree()
						# print('Adding to left child: ', self._leftchild, '\nroot', self)
						lc = self._leftchild.insert(interval)
						self._leftchild = lc # recursive call to left 
						# print('After adding to left child: ', self._leftchild, '\nroot', self)

			# update height of the root node     
			self.update_height()
			# print('height', self._height)
			# print('@@@@ before balancing current:', self.I, '\n'+str(self))
			self = self.avlbalance(interval)
			# print('@@@@ after balancing current:', self.I, '\n'+str(self))
			self.update_rMax(interval.dimension)
			self.update_height()
			if DEBUG:
				print('[3] ## after adding ', self.I, ', rMax:', self.rMax) #, self.lC.I.nI if self.lC else 0, self.rC.I.nI if self.rC else 0)   
					
			return self  
		else:
			for i_int in interval:
				self = self.insert(i_int)
			return self

	def search(self, interval):
		if DEBUG:
			print('search', self.I)
		if not self or self.isEmpty:
			return False
		if isinstance(interval, KDInterval):
			if interval.atomic:
				for int_upper in self.I.rP:
					interval_1 = KDInterval(Atomic(self.I.lP, int_upper))
					if interval_1.intersects(interval) and not (interval_1.adjacent(interval) and interval.adjacent(interval_1)):
						if DEBUG:
							print('search', self.I.lP, int_upper, interval_1, interval)
						return True
				# if self.I.lP.le(interval.lower):
				#     for int_upper in self.I.rP:
				#         print('search', self.I.lP, int_upper, interval) #, type(self.I.lP), type(int_upper))
				#         interval_1 = KDInterval(self.I.lP.V, int_upper.V)
				#         # print('search', interval_1)
				#         if interval in interval_1:
				#             return True
				if interval.lower < self.I.lP and self.lC: #self.lC.rMax < interval.lower:
					return self.lC.search(interval)
				else:
					if self.lC and interval.lower <= self.lC.rMax:
						return self.lC.search(interval)
					elif self.rC: #self.lC and interval.lower > self.lC.rMax and
						return self.rC.search(interval)
				# else:
				return False
			else:
				res = False
				for i_int in interval:
					res |= self.search(i_int)
				return res

		elif isinstance(interval, KDPoint):
			if DEBUG:
				print('search - point', interval)
			for int_upper in self.I.rP:
				interval_1 = KDInterval(Atomic(self.I.lP, int_upper))
				if DEBUG:
					print('search- current', self.I.lP, int_upper, interval_1, interval)
				if interval_1.contains(interval):                    
					return True
			if interval.V < self.I.lP and self.lC: #self.lC.rMax < interval.lower:
				if DEBUG:
					print('search left')
				return self.lC.search(interval)
			else:
				if self.lC and interval.V <= self.lC.rMax:
					if DEBUG:
						print('search left', interval.V,  self.lC.rMax)
						return self.lC.search(interval)
				elif self.rC: #self.lC and interval.V > self.lC.rMax
					if DEBUG:
						print('search right', interval.V)
					return self.rC.search(interval)                    
				return False
		elif isinstance(interval, list) or isinstance(interval, tuple):
			if DEBUG:
				print('search - list/tuple', interval)
			point = KDPoint(interval)
			if DEBUG:
				print('search - list/tuple to point', point)
			return self.search(point)
		else:
			return False

	def rightMost(self):
		if self.rC:
			return self.rC.rightMost()
		else:
			return self.rC

	def leftMost(self):
		if self.lC:
			return self.lC.leftMost()
		else:
			return self.lC

	def delete(self, interval):
		if not self or self.isEmpty:
				return False
		if isinstance(interval, KDInterval):
			if self.I.lP == interval.lower:
				if interval in self.I.rP:
					self.I.delete(interval)
					if self.I.nI == 0:
						sc = self
						if self.lC:
							sc = self.lC.rightMost()
						elif self.rC:                            
							sc = self.rC.leftMost()
						sc1 = IntervalTree()
						sc1.copy(sc)
						self.I.lP.copy(sc)
						# sc1.copy(self.I.lP)
						# suc._leftchild = sc
						sc1 = None
						# update height of the root node     
						 # update height of the root node     
						self.update_height()
						# print('height', self._height)
						# print('@@@@ before balancing current:', self.I, '\n'+str(self))
						self = self.avlbalance(interval)
						# print('@@@@ after balancing current:', self.I, '\n'+str(self))
						self.update_rMax(interval.dimension)
						self.update_height()
						return self

			elif  self.I.lP < interval.lower:
				return self.rC.delete(interval)
			else:
				return self.lC.delete(interval)
			# return False
		else:
			return NotImplemented

	def interval_cover_complement(self, U):
		a = U[0]
		b = U[1]
		U_interval = KDInterval(Atomic(a, b))
		#E_interval = KDInterval(Atomic())
		if DEBUG:
			print('interval_cover_complement', U_interval, 'kdt', self.items())
		# intervals = []
		#c_intervals = []
		#self.shrink()
		#combined_intervals = []
		
		c_interval = None
		combined = None

		i = 0
		for interval in self.items():
			#if DEBUG:
			#print(i, 'interval_cover_complement') #, interval, type(interval))
			# if U_interval.contains(interval):
			#if DEBUG:
			#print(i, 'contains') #, interval, U_interval)
			if not c_interval:
				c_interval = U_interval.difference(interval)
			else:
				c_interval &= U_interval.difference(interval)


			# if not combined:
			# 	combined = interval
			# else:
			# 	combined = combined.union(interval)

			#print(type(combined), type(interval), type(KDInterval(interval)))
			#if DEBUG:
			#	print(i, '1. c_interval -- complement') #, interval, 'complement', c_interval)
			# else:
			#     if DEBUG:
			#         print('2. c_interval -- no complement')
			#     else:
			#         pass
			# intervals = cover(c_intervals)
			sys.stdout.flush()
			i += 1
		#intervals = KDInterval(c_intervals).enclosure()
		if DEBUG:
			print('Interval_cover complement --', i, len(c_interval.items())) #, len(intervals.items()))
		# if combined:
		# 	j = len(combined.items())
		# 	if DEBUG:
		# 		print('Interval_cover -- ', i, j)
		# 	if j < i/2: 
		# 		kdt = IntervalTree()
		# 		for j_int in combined.items():
		# 			kdt = kdt.insert(j_int)
		# 		self = kdt
		
		return c_interval

	def get_uncovered_regions(self, U_interval):
		# keys = self.keys()
		return self.interval_cover_complement(U_interval)
		#print('1. get_uncovered_regions', self.len(), len(regions))
		# if DEBUG:
		#if DEBUG:
		#	print('1. get_uncovered_regions', regions)
		#return regions

	# function to find cartesian product of two sets  
	def generate_point_in_uncovered_region(self, U_interval, k=-1):
		if DEBUG:
			print('generate_point_in_uncovered_region', k)
		sys.stdout.flush()
		regions = self.get_uncovered_regions(U_interval)
		if not regions :
			return []
		if k == -1:
			k = min(100, len(regions.items()))
		#if DEBUG:
		#print('KDInterval', len(regions))
		#sys.stdout.flush()
		# region = KDInterval(*regions_sorted)
		
		#regions_list = list(regions.items())
		##print('in intervaltree-- ', len(regions_list))
		#regions_sorted = sorted(regions_list, key=lambda x: x.size, reverse=True)
		#print('@@@@@@@@@@@@@@@@@@@', len(regions_sorted), len(regions_list))
		dimension = self.dimension
		sys.stdout.flush()
		'''points = []
		for d in range(dimension):
			point_dim = []
			for reg in regions_sorted: 
				# for r in reg:# all the atomic intervals of inetrvals
					# print(key, r)
				if DEBUG:
					print(reg)
				if not reg.isEmpty:
					# select middle point 
					p = (reg.lower[d] + reg.upper[d])/2 
					#rnd.uniform(reg.lower[d], reg.upper[d])
					point_dim.append(p)
			points.append(point_dim)
		print('before cartesian --', len(points))
		#sample_points = cartesian(points)   
		sample_points = itertools.product(*points) '''
		k1 = 0
		#all_points = []
		sampleKpoints = []
		for reg in regions.items(): #regions_sorted: 	
			if not reg.isEmpty:		
				point_dim = []
				for d in range(dimension):
					if DEBUG:
						print(reg)
					# select middle point
					p = (reg.lower[d] + reg.upper[d])/2 
					#rnd.uniform(reg.lower[d], reg.upper[d])
					#point_dim.append(p)
					point_dim.append(rnd.uniform(0.97*p, 1.03*p))
				#print('point', point_dim)
				yield point_dim
				k1 += 1
				sampleKpoints.append(point_dim)
				if k1 == k:
					break
			#all_points.append(point_dim)
		#print('no cartesian --', len(sampleKpoints))
		#if len(sampleKpoints) > 0:
		#	print('sample_point', k, len(sampleKpoints), sampleKpoints[0])
		#sample_points = cartesian(points)   
		#sample_points = itertools.product(*points) 
		
		while k1 < k: # repeat to get more mid points
			for ch_point in sampleKpoints:
				for reg in regions.items(): 
					if not reg.isEmpty:
						new_points = []
						for d in range(dimension):
							new_point_dim = []
							mid_r_d = ch_point[d]
							p1 = (reg.lower[d] + mid_r_d)/2
							p2 = (reg.upper[d] + mid_r_d)/2
							#new_point_dim.append(p1)						
							#new_point_dim.append(p2)
							new_point_dim.append(rnd.uniform(0.97*p1, 1.03*p1))
							new_point_dim.append(rnd.uniform(0.97*p2, 1.03*p2))
							new_points.append(new_point_dim)
						
						for sp in itertools.product(*new_points):
							#print('sample ', k1, sp, len(sp), dimension)
							sampleKpoints.append(sp)
							#print('point', sp)
							yield sp
							k1 += 1
							if k1 == k:
								break
					if k1 == k:
						break
				if k1 == k:
					break
		#if DEBUG:
		#	print('generate_point_in_uncovered_region', len(sampleKpoints))   
		#return sampleKpoints

