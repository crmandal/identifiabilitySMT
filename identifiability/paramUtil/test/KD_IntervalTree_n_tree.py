from intervaltree import Interval as INode
from intervaltree import IntervalTree as ITree
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from intervalUtil.interval import *
from intervalUtil.box import *
from intervalUtil.box_factory import *
import random as rnd


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
	# print('cartesianProduct', result)
	return result 

# Function to do a cartesian  
# product of N sets  
def cart(list_a):
	n = len(list_a) 
	# result of cartesian product 
	# of all the sets taken two at a time 
	result = list_a[0] 
	# do product of N sets  
	for i in range(1, n): 
		result = cartesianProduct(result, list_a[i]) 
	return result


# def updateData(interval, islower):
# 	print('before updateData', interval, islower)
# 	if islower:
# 		oldlimit = interval[islower]
# 	else:
# 		oldlimit = interval[1]
# 	print('before updateData', interval, islower)
# 	return oldlimit

# def mergeData(interval, )

class IntervalTree(ITree):
	"""docstring for ClassName"""
	def __init__(self, key, intervals = []):		
		self.key = key
		iNodeSet = []
		for interval in intervals:
			if isinstance(interval, PyInterval):
				inode = self.convertToINode(interval)
			elif isinstance(interval, INode):
				inode = interval
			elif isinstance(interval, tuple) or isinstance(interval, list):
				if len(interval) > 2:
					inode = validateData(INode(interval[0], interval[1], interval[2]))
				else:
					inode = INode(interval[0], interval[1], [])
			iNodeSet.append(inode)

		if len(iNodeSet) >  0:
			self.tree = ITree(iNodeSet)
		else:
			self.tree = ITree()
		# super.__init__(iNodeSet)


	def items(self):
		s = []
		for item in self.tree.items():
			s.append(self.convertFromINode(item))
		return s

	def __str__(self):
		s = '['
		i = 0
		for item in self.items():
			if i == len(self.items())-1:
				s += '\tInterval ('+ str(item) + ')'
			else:
				s += '\tInterval ('+ str(item)+ ')\n'
			i += 1
		s += ']'
		return s

	def __repr__(self):
		self.__str__()

	def validateData(self, iNode):
		a, b, data = iNode
		data_updated = []
		for box in data:
			edges = box.get_map()
			edge_interval = edges[self.key]
			if edge_interval.leftBound() >= a and edge_interval.rightBound() <= b:
				if box not in data_updated:
					data_updated.append(box)
		inode1 = INode(a, b, data_updated)
		return inode1

	def convertToINode(self, interval):
		a = interval.leftBound()
		b = interval.rightBound()
		data = interval.getData()
		inode = self.validateData(INode(a, b, data))
		return inode

	def convertFromINode(self, iNode):
		a, b, data = self.validateData(iNode)
		interval = PyInterval(a, b, box = False)
		interval.setData(data)
		return interval

	def setTree(self, tree):
		# print('in setTree', str(tree))
		if tree is None:
			return
		tree_updated = []
		for iNode in tree:
			inode1 = self.validateData(iNode)
			tree_updated.append(inode1)

		# print('in setTree- updated', str(tree_updated))
		self.tree = ITree(tree_updated)


	def remove(self, interval):
		inode = self.convertToINode(interval)
		self.tree.remove(inode)

	def add(self, interval):
		inode = self.convertToINode(interval)
		self.tree.add(inode)

	# def discard(self, interval):
	# 	inode = convertToINode(interval)
	# 	self.tree.discard(inode)

	def remove_overlap(self, range):
		# removes all overlapping the range
		a = range[0]
		b = range[1]
		self.tree.remove_overlap(a, b)

	def remove_envelop(self, range):
		# removes all enveloped in the range
		a = range[0]
		b = range[1]
		self.tree.remove_envelop(a, b)

	def searchPoint(self, point):
		res = self.tree.at(point)
		print('searchPoint', res)
		return res


	def searchRange(self, range):
		a = range[0]
		b = range[1]
		res = self.tree.overlap(a, b)
		res_tree = IntervalTree(self.key, res)
		# print('searchRange {0} - {1}'.format(a, b), str(res_tree))
		return res_tree

	def containsInterval(self, interval):
		return interval in self.tree

	def containsPoint(self, point):
		return point in self.tree

	def is_empty(self):
		return self.tree.is_empty()

	def len(self):
		return len(self.tree.items())

	def begin(self):
		return self.tree.begin()

	def end(self):
		return self.tree.end()

	def union(self, other_trees):
		t1 = self.tree
		t3 = t1
		for ot in other_trees:
			t2 = ot.tree
			t3 |= t2
		t = IntervalTree(self.key)	
		t.setTree(t3)
		return t 

	def difference(self, other_trees):
		t1 = self.tree
		t3 = t1
		for ot in other_trees:
			t2 = ot.tree
			t3 -= t2
		t = IntervalTree(self.key)	
		t.setTree(t3)
		return t  

	def intersection(self, other_trees):
		t1 = self.tree
		t3 = t1
		for ot in other_trees:
			t2 = ot.tree
			t3 &= t2
		t = IntervalTree(self.key)	
		t.setTree(t3)
		return t  

	def symmetric_difference(self, other_trees):
		t1 = self.tree
		t3 = t1
		for ot in other_trees:
			t2 = ot.tree
			t3 ^= t2
		t = IntervalTree(self.key)	
		t.setTree(t3)
		return t

	def __le__(self, other):
		t1 = self.tree
		t2 = other.tree		
		return t1 <= t2
	def __ge__(self, other):
		t1 = self.tree
		t2 = other.tree		
		return t1 >= t2
	def __lt__(self, other):
		t1 = self.tree
		t2 = other.tree		
		return t1 < t2
	def __gt__(self, other):
		t1 = self.tree
		t2 = other.tree		
		return t1 > t2
	def __eq__(self, other):
		t1 = self.tree
		t2 = other.tree		
		return t1 == t2
	def __ne__(self, other):
		t1 = self.tree
		t2 = other.tree		
		return not t1 == t2

	def slice(self, point):
		# slice intervals at point
		t1 = self.tree
		t2 = t1.slice(point)
		t = IntervalTree(self.key)
		t.setTree(t2)
		return t

	def merge_overlaps(self):
		def reducer(old, new):
			# print(old, new)
			return old+new
		# joins overlapping intervals into a single interval, optionally merging the data fields
		# print('In merge_overlaps')
		t1 = self.tree
		# print('before merge ', str(t1))
		t1.merge_overlaps(data_reducer = reducer, strict= False)
		# print('after merge ', str(t1))
		t = IntervalTree(self.key)		
		t.setTree(t1)
		return t

	def merge_equals(self):
		# joins intervals with matching ranges into a single interval
		# print('In merge_equals')
		t1 = self.tree
		t1.merge_equals()
		# print(t1)
		t = IntervalTree(self.key)		
		t.setTree(t1)
		return t

	def shrink(self):
		t1 = self.merge_overlaps()
		# t2 = t1.merge_equals()
		return t1

	def interval_cover(self):
		intervals = []
		t1 = self.shrink()
		for interval in t1.items():
			intervals.append((interval.leftBound(), rightBound()))
		return intervals

	def interval_cover_complement(self, U):
		a = U[0]
		b = U[1]
		U_interval = PyInterval(a, b)
		# intervals = []
		c_intervals = []
		t1 = self.shrink()
		for interval in t1.items():
			# print('interval_cover_complement', U_interval, 'interval', interval)
			# if U_interval.contains(interval):
			c_interval = U_interval.complement(interval)
			# print('1. c_interval', interval, 'complement', c_interval)
			c_intervals.append(c_interval)
			# else:
				# print('2. c_interval -- no complement')
		# intervals = cover(c_intervals)
		return c_intervals



class KD_IntervalTree(object):
	"""docstring for KD_IntervalTree"""
	def __init__(self, k, boxes = []):
		# super(KD_IntervalTree, self).__init__()
		# self.arg = arg
		self.dimension = k
		self.trees = {}

		keys = []
		data = boxes
		intervals = []
		if len(boxes) > 0:
			box = boxes[0]
			edges = box.get_map()
			dim = len(edges)
			self.dimension = dim
			for key in edges.keys():
				keys.append(key)

		for key in keys:
			key_interval = []
			for box in boxes:			
				edges = box.get_map()
				interval = edges[key].clone()
				interval.box = False
				interval.setData(data)
				key_interval.append(interval)
			tree = IntervalTree(key, key_interval)
			self.optimizedTree(key, tree)

	def len(self):
		keys = self.trees.keys()
		l = [len(keys)]
		for key in keys:
			t = self.trees[key]
			l.append(t.len())
		return l
		
	def __repr__(self):
		s = 'KD_IntervalTree \n['
		keys = self.trees.keys()
		for key in keys:
			t = self.trees[key]
			s += ' Tree '+ str(key) + ' -- ' + str(t) + '\n'
		s += ']'
		return s

	def addBox(self, box):
		# print('addBox--before', self)
		data = [box]		
		edges = box.get_map()
		keys = self.trees.keys()
		if len(self.trees) == 0:
			keys = edges.keys()
		for key in keys:
			interval = edges[key].clone()
			interval.box = False
			interval.setData(data)
			if key in self.trees:
				self.trees[key].add(interval)
				self.optimizedTree(key, self.trees[key])
			else:
				t = IntervalTree(key, [interval])
				# print('addBox--key',key, t)
				self.optimizedTree(key, t)
		# print('addBox--after', self)
		# return self

	def optimizedTree(self, key, tree):
		t = tree.shrink()
		self.trees.update({key:t})

	# def removeBox(self, box):
	def contains(self, box):
		keys = self.trees.keys()
		data_check = {}
		# i = 0
		for key in keys:
			self.optimizedTree(key, self.trees[key])
			t = self.trees[key]
			edges = box.get_map()
			interval = edges[key]
			interval_tree = t.searchRange((interval.leftBound(), interval.rightBound()))
			# print('Intervals : ', interval_tree, type(interval_tree))
			for interval1 in interval_tree.items():
				if interval1 is not None: 
					data = interval1.getData()
					if len(data_check) == 0 or key not in data_check:
						data_check.update({key:data})
					else:
						d = data_check[key]
						data_check.update({key:d+data})
			# i += 1
				else:
					data_check.update({key:[]})
		# print('contains', data_check)

		if len(data_check) < len(keys):
			return False
		else:
			data1 = []
			i = 0
			for key in keys:
				d = data_check[key]				
				# print('1. contains ', key, 'd', d, 'data1', data1)
				if i == 0:
					data1 = d
				else:
					data2 = []
					for item in data1:
						# print('item', item)
						if item in d:
							data2.append(item)
					# print('2.1. contains ', key, 'd', d, 'data1', data2)
					data1 = data2
				# print('2. contains ', key, 'd', d, 'data1', data1)
				i += 1
			# print('intersection', data1)
			if len(data1) == 0:
				return False
			else:
				return True

	def get_uncovered_regions(self, box):
		keys = self.trees.keys()
		regions = {}
		# i = 0
		for key in keys:
			self.optimizedTree(key, self.trees[key])
			t = self.trees[key]
			U_interval = (box.get_map())[key]
			key_boxes = t.interval_cover_complement((U_interval.leftBound(), U_interval.rightBound()))
			regions.update({key:key_boxes})
			# print('1. get_uncovered_regions', key, '--', key_boxes)
		# print('2. get_uncovered_regions', regions)
		return regions

	# function to find cartesian product of two sets  
	def generate_point_in_uncovered_region(self, box):
		# print('box',box, self)
		regions = self.get_uncovered_regions(box)
		keys = list(self.trees.keys())
		# print('@@@@@@@@@@@@@@@@@@@', keys, regions)
		len_keys = len(keys)
		points = []
		for key in keys:
			point_dim = []
			for reg in regions[key]:
				int_key = reg.component
				# print('##', int_key)
				for r in int_key:
					# print(key, r)
					p = rnd.uniform(r.lower, r.upper)
					point_dim.append(p)
			points.append(point_dim)
		# print(points)
		sample_points = cart(points)	
		# print('generate_point_in_uncovered_region', len(sample_points))	
		return sample_points

	# def generate_point_in_uncovered_region(self, box):
	# 	regions = self.get_uncovered_regions(box)
	# 	keys = list(self.trees.keys())
	# 	len_keys = len(keys)
	# 	points = []
	# 	i = 0
	# 	count = 0
	# 	# print('1. keys', keys, len_keys)
	# 	# print('region', i, regions[keys[i]], type(regions[keys[i]][0]), len(regions[keys[i]]))
	# 	for region_0 in regions[keys[i]]:
	# 		data_key1 = region_0.getData()
	# 		int_key1 = region_0.component
	# 		for d in data_key1:				
	# 			count = 0
	# 			int_key2s = {}
	# 			int_key2s.update({keys[i]:int_key1})
	# 			for j in (1, len_keys-1):
	# 				# print(j, len_keys)
	# 				# print('region', j, regions[keys[j]], type(regions[keys[j]][0]), len(regions[keys[j]]))
	# 				for region_2 in regions[keys[j]]:
	# 					data_key2 = region_2.getData()
	# 					int_key2 = region_2.component
	# 					if d in data_key2:
	# 						count += 1
	# 						int_key2s.update({keys[j]:int_key2})
	# 						break
	# 			# print('count == len_keys', count, len_keys)
	# 			if count == len_keys:
	# 				edges = {}
	# 				for k in int_key2s:
	# 					comps = int_key2s[k]
	# 					values = []
	# 					for comp in comps:
	# 						value = rnd.uniform(comp.lower, comp.upper)
	# 						values.append(value)
	# 					value = rnd.sample(values, 1)[0]
	# 					edges.update({k: PyInterval(value)})
	# 				box = Box(edges)
	# 				print('added', box)
	# 				points.append(box)
	# 	# print(points)
	# 	return rnd.sample(points, 1)[0]




