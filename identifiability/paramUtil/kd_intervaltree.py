
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
# DEBUG = True

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
	def __init__(self, l = None, r = None):
		self._leftPoint = l
		self._righPoint = r
		self._numInterval = 0

	def clone(self):
		lc = self._leftPoint.clone()
		rc = self._righPoint.clone()
		return IntervalType(lc, rc)
	@property
	def lP(self):
		return self._leftPoint
	@property
	def rP(self):
		return self._righPoint
	@property
	def nI(self):
		return self._numInterval

	def __repr__(self):
		# s = 'lp: '+ str(self.lP) + ', rp: ' + str(self.rP) + ', nI: '+ str(self.nI)
		s = 'L: '+str(self.lP) + ', H:' + str(self.rP) #+ ', '+ str(self.nI)
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
				self._righPoint = interval.upper
				self._numInterval += 1
				# self._righPoints = sorted(self._righPoints)

			elif self._leftPoint == interval.lower:
				# self._leftPoint = interval.lower
				# print('iVt -- add', 'before', self._righPoints, 'add', interval.upper)
				self_r = None #interval.upper
				flag = False
				if self._righPoint is not None:
					r = self._righPoint 
					# print(r, r.gt(interval.upper), r.ge(interval.upper), r.lt(interval.upper), r.le(interval.upper))
					if r.ge(interval.upper):
						self_r = r
						flag=True
					elif r.le(interval.upper):# or r.lt(interval.upper):
						self_r = interval.upper
						flag = True
					else:
						self_r = r
				if not flag:
					self_r = interval.upper
				self._righPoint = self_r #.append(interval.upper)
				self._numInterval = 1 #len(self_r)
				# self._righPoints = sorted(self._righPoints)
				# print('iVt -- add', self._righPoints)#, interval)
			else:
				print('Interval not added', self.lP, interval.lower)
				return
			if DEBUG:
				print('Interval added', self, interval.lower)

	def remove(self, interval): 
		if isinstance(interval, IntervalType):
			kdinterval = KDInterval(interval.lP, interval.rP)  
		if interval.atomic:
			kdinterval = interval

		if self._leftPoint == kdinterval.lower and kdinterval.upper <= self.rP:
			#if self._numInterval - 1 == 0 :
			self._leftPoint = None
			self._righPoint = None #self.rP - kdinterval.upper #.remove(interval.upper)
			self._numInterval -= 1
		return self

	def overlap(self, interval):
		# if not isinstance(interval, IntervalType):
		# 	return NotImplemented
		# if kdinterval.atomic:
		# 	if self.rP[self.nI - 1] <= interval.lP or interval.rP[interval.nI - 1] <= self.lP:
		# 		return False
		# 	return True
		# return NotImplemented

		if isinstance(interval, IntervalType):
			kdinterval = KDInterval(interval.lP, interval.rP)
			selfInterval = KDInterval(self.lP, self.rP)
			if kdinterval.overlaps(selfInterval):
			# if self.rP <= interval.lP or interval.rP <= self.lP:
				return True
		if isinstance(interval, KDInterval):
			selfInterval = KDInterval(self.lP, self.rP)
			if interval.overlaps(selfInterval):
			# if self.rP <= interval.lP or interval.rP <= self.lP:
				return True
			return False
		return False

	def mergeable(self, interval):
		if isinstance(interval, IntervalType):
			kdinterval = KDInterval(interval.lP, interval.rP)
			selfInterval = KDInterval(self.lP, self.rP)
			if kdinterval.mergable(selfInterval):
			# if self.rP <= interval.lP or interval.rP <= self.lP:
				return True
		if isinstance(interval, KDInterval):
			#print('Types:', self.lP, self.rP, type(self.lP), type(self.rP))
			selfInterval = KDInterval(self.lP, self.rP)
			if interval.mergable(selfInterval):
			# if self.rP <= interval.lP or interval.rP <= self.lP:
				return True
			return False
		return False
	
	def merge(self, interval):
		if isinstance(interval, IntervalType):
			kdinterval = KDInterval(interval.lP, interval.rP)
			selfInterval = KDInterval(self.lP, self.rP)
			# l = interval.lP
			# r = interval.rP
		if isinstance(interval, KDInterval):
			# l = interval.lower
			# r = interval.upper
			kdinterval = interval
			selfInterval = KDInterval(self.lP, self.rP)
		# if l > self.lP:
		# 	l = self.lP
		# if r < self.rP:
		# 	r = self.rP
		m_int = selfInterval.merge(kdinterval)
		newInterval = IntervalType(m_int.lower, m_int.upper)
		return newInterval



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
		self._leftchild = itree.lC
		self._rightchild = itree.rC
		self._empty = itree.isEmpty

	def inorder(self, level = 0):
		if self is None:
			return ''
		s = ''
		if self.lC is not None:
			s += ' L{0}: '.format(level+1)+ self.lC.inorder(level+1)
		# s += ' ['+str(self._intervalNode) +  ', rMax: '+ str(self.rMax) + ']'
		s += ' {0} '.format(level) +' ('+str(self._intervalNode) #+  ', '+ str(self.rMax) + ')'
		if self.dimension > 1:
			s += '\n'
		if self.rC is not None:
			s += ' R{0}: '.format(level+1)+self.rC.inorder(level+1)
		return s

	def preorder(self):
		if self is None:
			return ''
		# s = ' ['+str(self._intervalNode) +  ', rMax: '+ str(self.rMax) + ']'
		s = ' ('+str(self._intervalNode) #+  ', '+ str(self.rMax) + ')'
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
		if self.I.rP is not None:
			l.append(KDInterval(Atomic(self.I.lP, self.I.rP)))
		s += l
		if self.rC is not None:
			s += self.rC.items()
		return s
	
	def leftRotate(self, root):
		if root and root.rC:
			if DEBUG:
				print('1 left rotate', root.I)
			# root = self
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
		else:
			return root

	def rightRotate(self, root):
		if root and root.lC:
			if DEBUG:
				print('1 right rotate', root.I)
			# root = self
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
		else:
			return root
		

	def getBalance(self, root):
		if root is None or root.isEmpty: 
			return 0
		lch = root.lC.height if (root.lC is not None and not root.lC.isEmpty) else 0
		rch = root.rC.height if (root.rC is not None and not root.rC.isEmpty)  else 0
		balance = lch-rch
		if DEBUG:
			print('balance at (', root.I, ')', lch, rch, balance)
		return balance

	def avlbalance(self, root, interval):
		# get the balance factor

		balance = self.getBalance(root)
		if abs(balance) > 1:
			if DEBUG:
				print('avlbalance : ', balance, root.I) #
				print('before ', root)
				print('----------------- AVL balancing ------------------------------')
		else:
			return root
		#, interval.upper < self.lC.rMax, interval.upper > self.lC.rMax, interval.upper < self.rC.rMax, interval.upper > self.rC.rMax)
		# Step 4 - If the node is unbalanced,  
		# then try out the 4 cases 
		# Case 1 - Left Left 
		if balance > 1 and root.lC and not (root.lC is None or root.lC.isEmpty) and interval.lower < root.lC.I.lP: 
			if DEBUG:
				print('case-1', str(root), 'cond', interval.lower, root.lC.I.lP, interval.lower < root.lC.I.lP)
			root1 = self.rightRotate(root) 
			# self = root
			if DEBUG:
				print('@@ avlbalance ', root1)
			return root1
  
		# Case 2 - Right Right 
		if balance < -1 and root.rC and not (root.rC is None or root.rC.isEmpty) and interval.lower > root.rC.I.lP: 
			if DEBUG:
				print('case-2', str(root), 'cond', interval.lower, root.rC.I.lP, interval.lower > root.rC.I.lP)
			root1 = self.leftRotate(root) 
			# self = root
			if DEBUG:
				print('@@ avlbalance', root1)
			return root1
  
		# Case 3 - Left Right 
		if balance > 1 and root.lC and not (root.lC is None or root.lC.isEmpty) and interval.lower > root.lC.I.lP: 
			if DEBUG:
				print('case-3', str(root), 'cond', interval.lower, root.lC.I.lP, interval.lower > root.lC.I.lP)
			root_lc = self.leftRotate(root._leftchild) 
			root._leftchild = root_lc
			root1 = self.rightRotate(root) 
			if DEBUG:
				print('@@ avlbalance', root1)
			return root1
  
		# Case 4 - Right Left 
		if balance < -1 and root.rC and not (root.rC is None or root.rC.isEmpty) and interval.lower < root.rC.I.lP: 
			if DEBUG:
				print('case-4', str(root), 'cond', interval.lower, root.rC.I.lP, interval.lower < root.rC.I.lP)
			root_rc = self.rightRotate(root._rightchild) 
			root._rightchild = root_rc
			root1 = self.leftRotate(root)
			if DEBUG:
				print('@@ avlbalance', root1)
			return root1

		# if abs(balance) > 1:
		#     print('after', self)

	def update_rMax(self, root, dim):
		if root:
			rmax1 = root.lC.rMax if (root.lC and root.lC is not None) else [-Inf for i in range(dim)]
			rmax2 = root.rC.rMax if (root.rC and root.rC is not None) else [-Inf for i in range(dim)]
			if DEBUG:
				print('rmax calculation:', root.I.rP, root.I.nI-1, rmax1, rmax2)
			root._rmax = np.max([root.I.rP, rmax1, rmax2])

	def update_height(self, root):
		if root:
			root._height = 1 + max(root.lC.height if root.lC is not None else 0, \
				root.rC.height if root.rC is not None else 0) 

	def check_mergeable(self, root, interval):
		# print('check_mergeable -- root', root._intervalNode)
		if not isinstance(interval, KDInterval):
			if isinstance(interval, list) or isinstance(interval, tuple):
				p1 = [i*(100.0 - (10.0 if i>0 else -10.0))/100.0 for  i in interval]
				p2 = [i*(100.0 + (10.0 if i>0 else -10.0))/100.0  for  i in interval]
				interval1 = KDInterval(Atomic(p1, p2))
				#return self.insert(root, interval1)
			elif isinstance(interval, KDPoint):
				p1 = [i*(100.0 - (10.0 if i>0 else -10.0))/100.0 for  i in interval.V]
				p2 = [i*(100.0 + (10.0 if i>0 else -10.0))/100.0  for  i in interval.V]
				interval1 = KDInterval(Atomic(p1, p2))
				#return self.insert(root, interval1)
			elif isinstance(interval, IntervalType):
				interval1 = KDInterval(interval.lP, interval.rP)
				#return self.insert(root, interval1)
			# else:
			# 	return NotImplemented
		else:
			interval1 = interval
		

		interval_to_add = interval1
		# if (not self.isEmpty) and self._intervalNode.mergeable(interval):
		# 	mergedNode = interval.merge(self._intervalNode)
		# 	self.delete()
		# 	return self.insert(mergedNode)
		# else:
		# 	return self.insert(interval)
		if root is None or root.isEmpty:
			# root = self.insert(root, interval_to_add)
			return interval_to_add, root
		else:
		# if not (root.isEmpty or root is None):
			isMergeable = root._intervalNode.mergeable(interval_to_add)
			# print('Mergeable interval : ', str(root._intervalNode), str(interval_to_add), isMergeable)
			if isMergeable:
			# 	self._intervalNode.add(interval)  
			# else:
				mergedNode = root._intervalNode.merge(interval_to_add)
				interval_to_add = KDInterval(mergedNode.lP, mergedNode.rP)
				d_node = root._intervalNode.clone()
				root = self.delete(root, d_node, False)
				# print('mergedNode', interval_to_add) #type(interval_to_add)
				# print('deleted_node', d_node, root) #type(d_node),
				# interval1 = KDInterval(mergedNode.lP, mergedNode.rP)
				# return self.insert(root, interval1)
				
				#return self.insert(root, interval_to_add)  
				# root = rr 
				#return interval_to_add, root
			# else:
			if root and root.I.lP >= interval_to_add.lower:
				if root.lC:
					# print('check_mergeable - left')
					interval_to_add, rl = self.check_mergeable(root._leftchild, interval_to_add)
					root._leftchild = rl
			else:
				if root and root.rC:
					# print('check_mergeable - right')
					interval_to_add, rr = self.check_mergeable(root._rightchild, interval_to_add)
					root._rightchild = rr
			#return interval_to_add, root 
			#else:
			#return interval_to_add, root
			# root = self.insert(root, interval_to_add)
			return interval_to_add, root
	
	#def insert_AVL(self, root, interval):

	def insert(self, interval):
		root = self
		# print('insrte_extend -- before check_mergeable', interval, root)
		# interval_to_add, root  = self.check_mergeable(root, interval)
		# print('insert_extend -- after check_mergeable', interval_to_add, root)
		interval_to_add = interval
		root = self.insert_AVL(root, interval_to_add)
		return root



	def insert_AVL(self, root, interval):
		if DEBUG:
			print('[0] ## In tree insert: ', root.I if root else root, interval)
		
		if interval.atomic:
			interval_to_add = interval
			if root is None:
				if DEBUG:
					print('[0.5] ## interval: ', interval, 'adding node to None node: ', root, 'interval: ', interval_to_add)
				root = IntervalTree()
				root._intervalNode.add(interval_to_add)
				root._empty = False
				#print('added node', root, self)
			elif root.isEmpty:
				if DEBUG:
					print('[1] ## interval: ', interval_to_add, 'adding node to empty node: ', root, 'interval: ', interval_to_add)
				root._intervalNode.add(interval_to_add)
				root._empty = False
			else:
				# isMergeable = root._intervalNode.mergeable(interval_to_add)
				# print('Mergeable interval : ', str(root._intervalNode), str(interval_to_add), isMergeable)
				# if isMergeable:
				# # 	self._intervalNode.add(interval)  
				# # else:
				# 	mergedNode = root._intervalNode.merge(interval_to_add)
				# 	interval_to_add = KDInterval(mergedNode.lP, mergedNode.rP)
				# 	d_node = root._intervalNode.clone()
				# 	root = self.delete(root, d_node)
				# 	print('mergedNode', interval_to_add, type(interval_to_add))
				# 	print('deleted_node', d_node, type(d_node), root)
				# 	# interval1 = KDInterval(mergedNode.lP, mergedNode.rP)
				# 	# return self.insert(root, interval1)
				# 	# rr = self.insert(root, interval1)  
				# 	# root = rr 
				# else:	
				if root.I.lP == interval_to_add.lower:
					if DEBUG:
						print('[2] ## ', 'interval: ', interval_to_add, ' adding node to already existed node: ', root.I)

					root._intervalNode.add(interval_to_add)  
							
				else:    
					if DEBUG:
						print('@@ ', 'interval: ', interval_to_add,' deciding on children at ', root.I, \
							',rMax', root.rMax,  'right' if root.I.lP < interval_to_add.lower else 'left')
					if root.I.lP < interval_to_add.lower:
						if not root.rC:  
							root._rightchild = IntervalTree()

						# print('Adding to right child: ', self._rightchild, '\nroot', self)
						rc = self.insert_AVL(root._rightchild, interval_to_add)
						root._rightchild  =  rc # recursive call to right  
						# print('After adding to right child: ', self._rightchild , '\nroot', self)
					else:
						if not self.lC:                    
							root._leftchild = IntervalTree()
						# print('Adding to left child: ', self._leftchild, '\nroot', self)
						lc = self.insert_AVL(root._leftchild, interval_to_add)
						root._leftchild = lc # recursive call to left 
						# print('After adding to left child: ', self._leftchild, '\nroot', self)

			# update height of the root node     
			self.update_height(root)
			# print('height', self._height)
			# print('@@@@ before balancing current:', self.I, '\n'+str(self))
			root = self.avlbalance(root, interval_to_add)
			# print('@@@@ after balancing current:', self.I, '\n'+str(self))
			self.update_rMax(root, interval_to_add.lower.dimension)
			self.update_height(root)
			if DEBUG:
				print('[3] ## after adding ', self.I, ', rMax:', self.rMax) #, self.lC.I.nI if self.lC else 0, self.rC.I.nI if self.rC else 0)   
					
			return root  
		else:
			for i_int in interval:
				root = self.insert(root, i_int)
			return root

	def searchContains(self, interval):
		if DEBUG:
			print('searchContains', self.I)
		if not self or self.isEmpty:
			return False
		if isinstance(interval, KDInterval):
			if interval.atomic:
				int_upper = self.I.rP ;
				interval_1 = KDInterval(Atomic(self.I.lP, int_upper))
				if interval_1.fullyContains(interval):
					if DEBUG:
						print('searchContains', self.I.lP, int_upper, interval_1, interval)
					return True
				# if self.I.lP.le(interval.lower):
				#     for int_upper in self.I.rP:
				#         print('search', self.I.lP, int_upper, interval) #, type(self.I.lP), type(int_upper))
				#         interval_1 = KDInterval(self.I.lP.V, int_upper.V)
				#         # print('search', interval_1)
				#         if interval in interval_1:
				#             return True
				if interval.lower < self.I.lP and self.lC: #self.lC.rMax < interval.lower:
					return self.lC.searchContains(interval)
				else:
					if self.lC and interval.lower <= self.lC.rMax:
						return self.lC.searchContains(interval)
					elif self.rC: #self.lC and interval.lower > self.lC.rMax and
						return self.rC.searchContains(interval)
				# else:
				return False
			else:
				res = False
				for i_int in interval:
					res |= self.searchContains(i_int)
				return res

		elif isinstance(interval, KDPoint):
			if DEBUG:
				print('searchContains - point', interval)
			for int_upper in self.I.rP:
				interval_1 = KDInterval(Atomic(self.I.lP, int_upper))
				if DEBUG:
					print('searchContains - current', self.I.lP, int_upper, interval_1, interval)
				if interval_1.fullyContains(interval):                    
					return True
			if interval.V < self.I.lP and self.lC: #self.lC.rMax < interval.lower:
				if DEBUG:
					print('searchContains left')
				return self.lC.searchContains(interval)
			else:
				if self.lC and interval.V <= self.lC.rMax:
					if DEBUG:
						print('searchContains left', interval.V,  self.lC.rMax)
						return self.lC.searchContains(interval)
				elif self.rC: #self.lC and interval.V > self.lC.rMax
					if DEBUG:
						print('searchContains right', interval.V)
					return self.rC.searchContains(interval)                    
				return False
		elif isinstance(interval, list) or isinstance(interval, tuple):
			if DEBUG:
				print('searchContains - list/tuple', interval)
			point = KDPoint(interval)
			if DEBUG:
				print('searchContains - list/tuple to point', point)
			return self.searchContains(point)
		else:
			return False

	def search(self, interval):
		if DEBUG:
			print('search', self.I)
		if not self or self.isEmpty:
			return False
		if isinstance(interval, KDInterval):
			if interval.atomic:
				int_upper = self.I.rP ;
				interval_1 = KDInterval(Atomic(self.I.lP, int_upper))
				if interval_1.intersects(interval):
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
			return self		

	def leftMost(self):
		if self.lC:
			return self.lC.leftMost()
		else:
			return self

	def delete(self,root, interval, flagBal = True):
		if not root or root.isEmpty:
			return root
		if DEBUG:
			print('--------- In  before delete', 'int', interval, 'at root', root.I, str(root))
		# if isinstance(interval, KDInterval):
		if  root.I.lP < interval.lP:
			return self.delete(root.rC, interval)
		elif root.I.lP > interval.rP:
			return self.delete(root.lC, interval)
		else:
			if isinstance(interval, IntervalType):
				kdinterval = KDInterval(interval.lP, interval.rP)
			else:
				kdinterval = interval
			#print('KD_intervalTree-delete1', kdinterval, 'root', root.I)
			# if self.I.lP == kdinterval.lower:
			# if kdinterval.upper <= root.I.rP:
				# self.I.delete(kdinterval)
				# if self.I.nI == 0:
			if root.lC is None and not(root.rC is None):
				#print('KD_intervalTree-delete2 - root.lC', root.lC, root.rC)
				temp = root.rC #.clone()
				root = None
				#print('temp', temp)
				return temp
			
			elif root.rC is None and not(root.lC is None):
				#print('KD_intervalTree-delete2 - root.rC', root.lC, root.rC)
				temp = root.lC #.clone()
				root = None
				#print('temp', temp)
				return temp
				
			# check recursively 
			# sc = self
			if root.lC is not None:
				sc = root.lC.rightMost()
				root._intervalNode = sc.I.clone()
				#print('kd_intervalTree-delete3-lC', sc.I, root.I, kdinterval)
				#print('root', root)
				root._leftchild = self.delete(root.lC, sc.I)
				#print('after -deletion-lC', root) #root.lC)
			elif root.rC is not None:                            
				sc = root.rC.leftMost()
				#print('kd_intervalTree-delete3-rC', sc.I, root.I, kdinterval)
				#print('root', root)
				root._intervalNode = sc.I.clone()
				root._rightchild = self.delete(root.rC, sc.I)
				#print('after -deletion-rC', root)
			
			if root is None:
				return root
			# update height of the root node     
			# update height of the root node     
			self.update_height(root)
			# print('height', self._height)
			# print('@@@@ before balancing current:', self.I, '\n'+str(self))
			if flagBal:
				root = self.avlbalance(root, kdinterval)
				# print('@@@@ after balancing current:', self.I, '\n'+str(self))
				self.update_rMax(root, kdinterval.dimension)
				self.update_height(root)

			if DEBUG:
				print('---- In after delete','int', interval, 'at root', root.I, str(root))
			return root

		return root
		# return False
		# else:
		# 	return NotImplemented

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



def test():
    k = PyInterval(0.94, 0.99) 
    r = PyInterval(0.1, 0.35)
    # g = PyInterval(9.79, 9.87)
    edges = {}
    edges.update({'k': k})
    edges.update({'r': r})
    # edges.update({'g': g})
    box = Box(edges)

    print('Box 1 ', box)

    k1 = PyInterval(0.95, 0.96) 
    r1 = PyInterval(0.45, 0.5)
    # g1 = PyInterval(9.8, 9.81)
    edges1 = {}
    edges1.update({'k': k1})
    edges1.update({'r': r1})
    # edges1.update({'g': g1})
    box1 = Box(edges1)

    print('Box 2 ', box1)

    k1 = PyInterval(0.1, 0.3) 
    r1 = PyInterval(0.4, 0.5)
    # g1 = PyInterval(9.82, 9.85)
    edges3 = {}
    edges3.update({'k': k1})
    edges3.update({'r': r1})
    # edges3.update({'g': g1})
    box3 = Box(edges3)
    print('Box 3 ', box3)

    k1 = PyInterval(1.0, 1.2) 
    r1 = PyInterval(0.05, 0.1)
    # g1 = PyInterval(9.7, 9.81)
    edges4 = {}
    edges4.update({'k': k1})
    edges4.update({'r': r1})
    # edges4.update({'g': g1})
    box4 = Box(edges4)
    print('Box 4 ', box4)
	
    k = PyInterval(0.94, 0.95) 
    r = PyInterval(0.35, 0.5)
    # g = PyInterval(9.79, 9.87)
    edges = {}
    edges.update({'k': k})
    edges.update({'r': r})
    # edges.update({'g': g})
    box5 = Box(edges)
    print('Box 5 ', box5)

    k = PyInterval(0.96, 0.99) 
    r = PyInterval(0.35, 0.5)
    # g = PyInterval(9.79, 9.87)
    edges = {}
    edges.update({'k': k})
    edges.update({'r': r})
    # edges.update({'g': g})
    box6 = Box(edges)
    print('Box 6 ', box6)

    k = PyInterval(0.95, 0.96) 
    r = PyInterval(0.35, 0.45)
    # g = PyInterval(9.79, 9.87)
    edges = {}
    edges.update({'k': k})
    edges.update({'r': r})
    # edges.update({'g': g})
    box7 = Box(edges)
    print('Box 7 ', box7)

    k = PyInterval(0.94, 0.96) 
    r = PyInterval(0.05, 0.1)
    # g = PyInterval(9.79, 9.87)
    edges = {}
    edges.update({'k': k})
    edges.update({'r': r})
    # edges.update({'g': g})
    box8 = Box(edges)
    print('Box 8 ', box8)

    k = PyInterval(0.92, 0.94) 
    r = PyInterval(0.05, 0.5)
    # g = PyInterval(9.79, 9.87)
    edges = {}
    edges.update({'k': k})
    edges.update({'r': r})
    # edges.update({'g': g})
    box9 = Box(edges)
    print('Box 9 ', box9)

    boxes = [box, box1, box3, box7, box9, box4, box5, box6, box8]
    intervals = []
    for b in boxes:
        intervals.append(KDInterval(b))
    print('intervals:', intervals)

    tree = IntervalTree()
    # tree = IntervalTree()
    print('empty tree:', tree)

    ic = 0
    for i in intervals:
        print(ic, '--------- inserting  ---------- ',i)
        tree = tree.insert( i)
        print(ic, ' tree after insertion --------- ',i, tree)
        ic += 1

if __name__ == "__main__":
	test()
