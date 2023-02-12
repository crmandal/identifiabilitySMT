#from interval import interval, inf, imath
import sys, os
from collections import OrderedDict
#from pyicl import Interval
# from interval import *
ONLYMARKED = 1
NOMARK =0
DEBUG = False

print('Using this one...............')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paramUtil.interval import *
from paramUtil.box import *
from util.queue import *
from util.stack import *
from util.heap import *

# definition of __getinitargs__
#def interval_getinitargs(self):
#	return (self.leftBound(), self.rightBound(),)

# now inject __getinitargs__ (Python is a dynamic language!)
#PyInterval.__getinitargs__ = interval_getinitargs

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
	
'''
# Cartesian product
'''
def cartesian_product(edges): #std::map<std::string, std::vector<capd::interval>> m)
	''' checking if the map is empty'''
	#print('cartesian_product')
	if(len(edges) == 0):
		return []
	
	size = 1
	for it in edges:
		intrvl = edges[it]
		size *= len(intrvl)
		
	product = []
	for i in range(size):
		index = i
		tmp_m = {}
		for it1 in sorted(edges.keys()):
			intrvl1 = edges[it1]
			mult = 1
			for it2 in sorted(edges.keys(), reverse=True): #edges.keys():
				#print('it1-it2', it1, it2)
				if(it1 != it2):
					#print('it1 != it2', it1, it2)
					intrvl2 = edges[it2]
					mult *= len(intrvl2)
				else:
					break
			tmp_index = int(index / mult)
			# print(index , mult, tmp_index, type(index), type(mult))
			tmp_m.update({it1: intrvl1[tmp_index]})
			index -= tmp_index * mult
		b = Box(tmp_m)
		if(b not in product):
			product.append(b)
	return product

''' partitioning a box'''
def partition(b, e): # Box b, double precision e
	# setting up a precision map
	e_map = {}
	edges = b.get_map()
	for it in edges.keys():
		e_map.update({it: PyInterval(e)})
	# main algorithm
	
	q = Queue()
	q.enqueue(b) # Queue
	
	res = []
	while(not q.isEmpty()):
		tmp_b = q.dequeue()
		tmp_v = bisect(tmp_b, e_map)
		if(len(tmp_v) == 1):
			#res.append(tmp_b)
			yield tmp_b
		else:
			for tmp in tmp_v:
				q.enqueue(tmp)
	#return res

def partition_int(b, amount): # Box b, int amount
	res = Queue()
	res.enqueue(b) # Queue
	
	while(len(res) < amount):
		b_tmp = res.dequeue()
		b_bisect = bisect(b_tmp)
		for tmp in tmp_v:
			res.enqueue(b_bisect)
	return res

def partition_map(b, e_map): # Box b, e_map
	''' checking if precision map is empty'''
	if(e_map.empty()) :
		return [b]
	
	'''checking whether partition map contains does not contain undefined variables'''
	b1 = Box(e_map)
	if(not get_keys_diff(b1, b).empty()):
		s = 'partition map \'' + b1 + '\' contains variables not defined in the box \'' + b + '\''
		raise ValueError()
		
	''' main algorithm'''
	q = Queue()
	q.enqueue(b) # Queue
	
	res = []
	while(not q.empty()):
		tmp_b = q.dequeue()
		tmp_v = bisect(tmp_b, e_map)
		
		if(len(tmp_v) == 1):
			res.append(tmp_b)
		else:
			for tmp in tmp_v:
				q.enqueue(tmp)
	return res

'''
 Dividing the box in a single dimension producing 2 boxes of
 different/same size according to the precision vector e and
 Partition plane point
 Some dimensions may be skipped for partitioning if flag in ONLYMARKED
'''
def bisect1D(box, emap = None, flag = NOMARK, point = None): 
	b_map = box.get_map()
	if(emap == None):
		emap = {}
		for it in b_map.keys():		
			emap.update({it: PyInterval(0.0)})
	elif(len(emap) == 0):
		return [b]
	
	# print('In bisect1D. Point = ',str(point)) 
	
	if(point is not None):
		p_map = point.get_map()	
	(skit, skintrvl) = findIntervalWithMaximumSkewness(box, point, emap, flag)
	# else: # the middle point is not passed

	tmp_m = {}
	fg = False
	for it in b_map.keys():
		intrvl = b_map[it]
		#e = emap[it]
		if(not(skintrvl == None) and it == skit):# and intrvl == skintrvl): 
			''' The interval with maximum skewness will be partitioned only; others will remain as it is '''
			#print('bisected interval \'{0}\''.format(it))
			tmp_v = []
			if((point is not None) and (it in p_map)):
				b1 = PyInterval(intrvl.leftBound(), p_map[it].leftBound())
				b2 = PyInterval(p_map[it].leftBound(), intrvl.rightBound())
			else:
				b1 = PyInterval(intrvl.leftBound(), intrvl.mid().rightBound())
				b2 = PyInterval(intrvl.mid().leftBound(), intrvl.rightBound())
					
			# if(flag == ONLYMARKED and intrvl.isMarked()):	
			# 	b1.mark()
			# 	b2.mark()
				
			tmp_v.append(b1)
			tmp_v.append(b2)
			tmp_m.update({it: tmp_v})
			fg = True
		else:
			#print('not bisected interval \'{0}\''.format(it))
			tmp_m.update({it: [intrvl]})

	if(fg):
		cart = cartesian_product(tmp_m)
		if DEBUG:# or flag == ONLYMARKED:
			print('Bisected box {0} along point {1} along dimension \' {2} \''.format(str(box), str(point), skit))
			print('before cartesian')
			s = ''
			for it in tmp_m:
				s += str(it) +':'
				for  v in tmp_m[it]:
					s += str(v)
				s += '\n'
			print(s)
			print('after cartesian')
			s = ''
			for it in cart:
				s += str(it) + '\n'
			print(s)
		return cart
	else:
		#print('No bisection possible')
		return [box]

def findIntervalWithMaximumSkewness(box, point, emap, flag = NOMARK):
	b_map = box.get_map()
	pmap = {}
	if point is not None:
		pmap = point.get_map()
	skInterval = None
	skit = ''
	for it in b_map:
		intrvl = b_map[it]
		e = emap[it]
		r = True 
		if point is not None:
			r = pmap[it].leftBound() > intrvl.leftBound() and  pmap[it].leftBound() < intrvl.rightBound()
		#print('findIntervalWithMaximumSkewness', 'interval', str(intrvl), str(r))
		if(intrvl.width() > e.leftBound() and r): 
			#print('findIntervalWithMaximumSkewness', 'feasible interval', str(intrvl.width()), '>',  str(e.leftBound()), str(r))		
			''' Considering only feasible interval'''
			if(flag == ONLYMARKED):
				if(intrvl.isMarked()):	
					''' In case the flag is MARKED, consider only marked interval to search for skewed one'''
					if skInterval == None:
						skInterval = intrvl
						skit = it
					else:
						if skInterval.width() < intrvl.width():
							skInterval = intrvl
							skit = it
					skInterval.mark()
			else:  
				''' Take the interval with the largest width as the skewed interval and the point is fully contained in that interval'''
				if skInterval == None:
					skInterval = intrvl
					skit = it
				else:
					if skInterval.width() < intrvl.width():
						skInterval = intrvl
						skit = it
	#print('findIntervalWithMaximumSkewness', skit, str(skInterval))
	return (skit, skInterval)		
'''
 Dividing the box in all n dimensions producing 2^n boxes of the same size according to the precision vector e
'''
'''box b, map<std::string, capd::interval> emap)'''
def bisect(b, emap = None, flag = NOMARK, point = None): 
	b_map = b.get_map()
	if(emap == None):
		emap = {}
		for it in b_map.keys():		
			emap.update({it: PyInterval(0.0)})
	elif(len(emap) ==0):
		return [b]
	if(point is not None):
		p_map = point.get_map()	
	tmp_m = {}
	#b_map = b.get_map()
	for it in b_map.keys():
		intrvl = b_map[it]
		e = emap[it]
		if(flag == ONLYMARKED):
			if(intrvl.isMarked()):				
				if(intrvl.width() > e.leftBound()):
					tmp_v = []
					if((point is not None) and (it in p_map)):
						b1 = PyInterval(intrvl.leftBound(), p_map[it].leftBound())
						b2 = PyInterval(p_map[it].leftBound(), intrvl.rightBound())
					else:
						b1 = PyInterval(intrvl.leftBound(), intrvl.mid().rightBound())
						b2 = PyInterval(intrvl.mid().leftBound(), intrvl.rightBound())
					
					b1.mark()
					b2.mark()
					tmp_v.append(b1)
					tmp_v.append(b2)
					tmp_m.update({it: tmp_v})
				else:
					tmp_m.update({it: [intrvl]})
			else:
					tmp_m.update({it: [intrvl]})
		else:
			if(intrvl.width() > e.leftBound()):
				tmp_v = []
				if(not (point == None) and (it in p_map)):
					b1 = PyInterval(intrvl.leftBound(), p_map[it].leftBound())
					b2 = PyInterval(p_map[it].leftBound(), intrvl.rightBound())
				else:
					b1 = PyInterval(intrvl.leftBound(), intrvl.mid().rightBound())
					b2 = PyInterval(intrvl.mid().leftBound(), intrvl.rightBound())
				tmp_v.append(b1)
				tmp_v.append(b2)
				tmp_m.update({it: tmp_v})
			else:
				tmp_m.update({it: [intrvl]})
	#print()

	cart = cartesian_product(tmp_m)
	if DEBUG:
		print('Bisected box along point '+str(point))
		print('before cartesian')
		s = ''
		for it in tmp_m:
			s += str(it) +':'
			for  v in tmp_m[it]:
				s += str(v)
			s += '\n'
		print(s)
		print('after cartesian')
		s = ''
		for it in cart:
			s += str(it) + '\n'
		print(s)
	return cart


def mergeBoxList(boxes, check = True): #std::vector<box> boxes):
	i = 0;
	boxes_cp = []
	for it in boxes:
		boxes_cp.append(it)
	#blen = len(boxes)
	while(i < len(boxes_cp)):
		previous_size = len(boxes_cp)
		for j in range(i+1, len(boxes_cp)):
			b = merge(boxes_cp[i], boxes_cp[j], check)
			if( not b.empty()):
				boxes_cp.insert(i, b)
				for k in (0, j+1):
					item = boxes[j-k]
					boxes.remove(item)
				i = 0
				break
		if(len(boxes) == previous_size):
			i = i+1	
	return boxes
	
'''
merge 2 boxes with shared boundary
'''
def merge(lhs, rhs, check=True): #box lhs, box rhs)
	l_map = lhs.get_map()
	r_map = rhs.get_map()
	if check:
		for it in l_map:
			if(not it in r_map):
				s = 'Variables of the compared boxes are not the same'
				raise ValueError()
	neq_counter = 0
	neq_dim = ''
	for it in l_map:
		intrvl = l_map[it]
		#print(it + ' ' + intrvl + '\n')
		if(not intrvl == r_map[it]):
			neq_counter = neq_counter+1
			neq_dim = it
		if(neq_counter > 1):
			return Box()
			
	if(l_map[neq_dim].rightBound() == r_map[neq_dim].leftBound()):
		l_map.update({neq_dim: PyInterval(l_map[neq_dim].leftBound(), r_map()[neq_dim].rightBound())})
		return Box(l_map)
	else:
		if(l_map[neq_dim].leftBound() == r_map[neq_dim].rightBound()):
			l_map.update({neq_dim: PyInterval(r_map[neq_dim].leftBound(), l_map[neq_dim].rightBound())})
			return Box(l_map)
		else:
			return Box()

def get_mean(q): #vector<box> q
	f_box = q.get[0]
	f_map = f_box.get_map()
	
	# setting the initial box
	init_map = {}
	div_map = {}
	
	for it in f_map:
		intrvl = f_map[it]
		init_map.update({it: PyInterval(0.0)})
		div_map.update({it: PyInterval(len(q))})
		
	res = Box(init_map)
	div = Box(div_map)
	for b in q:
		res = res + b.get_mean()
	return res / div

def get_stddev(q): # vector<box> q
	mean = get_mean(q)
	f_map = mean.get_map()
	# setting the initial box
	init_map = {}
	div_map = {}
	for it in f_map:
		init_map.update({it: PyInterval(0.0)})
		div_map.update({it: PyInterval(len(q))})
		
	resSum = Box(init_map)
	div = Box(div_map)
	for b in q :
		resSum = resSum + (b.get_mean() - mean) * (b.get_mean() - mean)
	return sqrt(resSum/div)


#def sqrt(b): #box b
	#b_map = b.get_map()
	#res = {}
	#for it in f_map:
		#intrvl = f_map[it]
		#res.update(it, intrvl.sqrt())
	#return Box(res)

#def log(b): #box b
	#b_map = b.get_map()
	#res = {}
	#for it in b_map:
		#intrvl = b_map[it]		
		#res.update(it, intrvl.log())
	#return Box(res)


def get_keys_diff(lhs, rhs): # box lhs, box rhs
	res = {}
	lhs_map = lhs.get_map()
	rhs_map = rhs.get_map()
	for it in lhs_map:
		intrvl = lhs_map[it]		
		if(not it in rhs_map):
			res.update({it: intrvl})
	return Box(res)

def bubbleSort(alist):
	blist = []
	for it in alist:
		blist.append(it)
	for passnum in range(len(blist)-1,0,-1):
		for i in range(passnum):
			if blist[i] > blist[i+1]:
				temp = blist[i]
				blist[i] = blist[i+1]
				blist[i+1] = temp
	return blist		
		
def min_left_coordinate_value(boxes, it):
	# if(compatible(boxes)):
	i = 0
	m = 0
	for box in boxes:
		edges = box.get_map()
		intrvl = edges[it] 
		if i==0:
			m = intrvl.leftBound()
		else:
			if intrvl.leftBound() < m:
				m = intrvl.leftBound()
		i= i+1
	return m
	# else:
		# print('Could not min_left_coordinate_value boxes. The boxes have different sets of variables')
		# raise ValueError()

def max_right_coordinate_value(boxes, it):
	# if(compatible(boxes)):
	i = 0
	m = 0
	for box in boxes:
		edges = box.get_map()
		intrvl = edges[it] 
		if i==0:
			m = intrvl.rightBound()
		else:
			if intrvl.rightBound() > m:
				m = intrvl.rightBound()
		i= i+1
	return m
	# else:
	# 	print('Could not max_right_coordinate_value boxes. The boxes have different sets of variables')
	# 	raise ValueError()
		
def sort(q): #vector<pair<box, capd::interval>> q : dictionary<box, interval>
	#sq = OrderedDict(sorted(q.items(), key=lambda x: x[1].mid()))
	# print('before sort:')
	# for it in q:
		# print(str(it))
		
	sq = bubbleSort(q) # key=lambda x: x.count, reverse=False)
	
	# print('after sort:')
	# for it in sq:
		# print(str(it))
	return sq

def get_cover(q, check = True): #vector<box> q1
	#q = sort(q1)
	#qf = q[0]
	#qb = q[-1]
	#if(not get_keys_diff(qf, qb).empty() or not get_keys_diff(qb, qf).empty()):
		#print('could not get_cover for ' + qf + ' and ' + qb + '. The boxes have different sets of variables')
		#raise ValueError()
	
	#qmin = qf.get_map()
	#qmax = qb.get_map()
	#res = {}
	#for it in qmin.keys():
		#intrvl = qmin[it]
		#res.update({it: PyInterval(intrvl.leftBound(), qmax[it].rightBound())})
	res = {}
	qf = q[0]
	q_map = qf.get_map()
	if check and not compatible(q):
		print('Could not get_cover boxes. The boxes have different sets of variables')
		raise ValueError()
	for it in q_map:
		intmin = min_left_coordinate_value(q, it)
		intmax = max_right_coordinate_value(q, it)
		res.update({it: PyInterval(intmin, intmax)})
	box = Box(res)
	# print('get_cover', 'boxes', str(q[0]), str(q[1]), 'Cover = ', str(box))
	# sys.stdout.flush()
	return box

def compatible(q): #vector<box> q
	for b in q:
		for c in q:
			if(not b.compatible(c)):
				return False
	return True


def map_box(ratio, b): 
	b_map = b.get_map()
	res_map = {}
	for it in b_map.keys():
		intrvl = b_map[it]
		res_map.update({it: intrvl.leftBound() + ratio.get_map()[it] * intrvl.width()})
	return Box(res_map)

def get_intersection_conflicts(original, compared): #map<box, capd::interval> original, map<box, capd::interval> compared)
	original_conflict = {}
	compared_conflict = {}
	for it in compared.keys():
		intersect_flag = False
		b = it
		intrvl1 = compared[it]
		for it2 in original.keys():
			intrvl2 = original[it2]
			if(b.intersects(it2)):
				intersect_flag = True
				if(not intersects(intrvl1,intrvl2)):
					original_conflict.update({it: intrvl1})
					compared_conflict.update({it2: intrvl2})
				break
		if (not intersect_flag):
			original_conflict.update({it: intrvl1})
	return ({original_conflict: compared_conflict})

def boxIntersection(box1, box2):
	b1edges = box1.get_map();
	b2edges = box2.get_map();

	# print('boxIntersection: box1 edges ', str(box1))
	# print('boxIntersection: box2 edges ', str(box2))
	edges = {}
	for it in b1edges.keys():
		intrvl1 = b1edges[it]
		if(it in b2edges.keys()):
			intrvl2 = b2edges[it]
			intrvl = intrvl1.intersection(intrvl2)
			edges.update({it: intrvl})		
			# print('update edge: ', it, str(intrvl))		
		else:
			print('The target box does not contain variable: \'' + it + '\'')
			raise ValueError()
	# print('boxIntersection: new edges ')
	# for i in edges.keys():
	# 	print(i, str(edges[i]))
	return Box(edges)

def intersectionBoxMap(boxlst1, boxlst2):
	resLst = []
	for b1 in boxlst1:
		for b2 in boxlst2:
			#if(b1 != b2 and b1.intersects(b2)):
			if(b1.intersects(b2)):
				res = boxIntersection(b1, b2)
				resLst.append(res)
	return resLst

'''
# remove b = intersection(box1, box2) from box1 and returns a set of convex non overlapping boxes
'''
def remove(box1, box2, emap= None, flag = NOMARK): 
	if(not box1.compatible(box2)):
		s = 'cannot perform remove operation for ' + str(box1) + ' and ' + str(box2) + '. The boxes have different sets of variables'
		raise ValueError()
	# print('box_factory: remove', str(box1), str(box2))
	boxes = []
	b1 = box1
	b2 = box2
	old_pt = None
	i = 0
	intersect_b = boxIntersection(b1, b2)
	# print('box_factory: remove', 'after boxIntersection', str(intersect_b))
	corners = getcorners(intersect_b)
	# print('box_factory: remove', 'intersection corners', corners)
	st_b = Stack()
	st_b.push(box1)
	for pt in corners:
		box = []
		while(not st_b.isEmpty()): #True):
			b1 = st_b.pop()
			if b1 and b1.fullyContains(pt):
				b1parts = bisect(b1, emap, flag, pt)
				if len(b1parts) > 1:
					for b2 in b1parts:
						st_b.push(b2)
						# print('box_factory: remove','current stack', str(b2))
				else:
					if DEBUG:
						print('\n box cant be partitioned '+ str(b1))
					if not b1.intersects(intersect_b):
						boxes.append(b1)
				break
			elif b1:
				box.append(b1)
		for b1 in box:
			st_b.push(b1)

	# print('box_factory: remove','boxes -1')
	# for b1 in boxes:
	# 	print('\tbox_factory: remove','boxes -1', str(b1))
	while(not st_b.isEmpty()):
		b1 = st_b.pop() 
		if b1 in boxes or b1.empty() or b1.minDimension() == 0 or (b1 == intersect_b):
			continue
		else:
			boxes.append(b1)
	# print('box_factory: remove','boxes -2')
	# for b1 in boxes:
	# 	print('\tbox_factory: remove','boxes -2', str(b1))
	return boxes

def remove_old(box1, box2, emap= None, flag = NOMARK): 
	if(not box1.compatible(box2)):
		s = 'cannot perform remove operation for ' + str(box1) + ' and ' + str(box2) + '. The boxes have different sets of variables'
		raise ValueError()
	# print('remove', str(box1), str(box2))
	boxes = []
	b1 = box1
	b2 = box2
	old_pt = None
	i = 0
	while(True):
		b = boxIntersection(b1, b2)
		# print('###### In loop Iteration {0}   ######'.format(i))
		# print('intersection of  b1 and b2 is b :[{0}]'.format(str(b)))
		if(old_pt == None):
			pt = cornerInside(b, b1, np = i)
			# print('1 remove: pt', str(pt))
		else:
			r = b1.fullyContains(old_pt)
			# print('2 remove: old_pt', str(old_pt),'Fully contains: ', str(r))
			if (r):
				pt = old_pt
			else:
				pt = cornerInside(b, b1, old_pt, np = i)
				# print('3 remove: pt', str(pt))
		
		if(pt == None):
			break	
		old_pt = pt
		# print('remove: by pt', str(pt))
		b1parts = bisect1D(b1, emap, flag, pt)
		#print('intersection b1 and b2', str(b))
		if(len(b1parts) > 1):
			b11 = b1parts[0]
			b12 = b1parts[1]
			# print(' b11: '+str(b11)+ '\n b12: '+str(b12)+ '\n b: '+str(b))
			if (b11 == b or b12 == b):
				# print('Intersection matches')
				if(b11 == b and not b12.contains(b)):
					boxes.append(b12)
					# print('1. Added box b12', str(b12))
				elif(b12 == b and not b11.contains(b)):
					boxes.append(b11)
					# print('1. Added box b11', str(b11))				
			#elif(b12 == b):
				break
			else:
				# print('Intersection does not match')
				if b11.contains(b):
					b1 = b11
					b2 = b
				else:
					boxes.append(b11)
					# print('2. Added box b11', str(b11))
				if b12.contains(b):
					b1 = b12
					b2 = b
				else:
					boxes.append(b12)
					# print('2. Added box b12', str(b12))	
		else:
			# print('old pt :', str(old_pt))
			old_pt = cornerInside(b, b1, old_pt)
			if(old_pt == None):
				break
		i += 1
		#if i == 4:
		#	break
	return boxes

def getcorners(box1):
	b1_edges = box1.get_map()	
	point_map = {}

	for it in sorted(b1_edges.keys()):
		intrvl = b1_edges[it]
		tmp_v = [PyInterval(intrvl.leftBound()), PyInterval(intrvl.rightBound())]
		point_map.update({it:tmp_v})

	corners = cartesian_product(point_map)
	return corners

LEFT = 1
RIGHT = 2
''' returns a corner of box1 which is inside box2
	Starting from the leftmost point in all dimension and 
	searching through closest points
'''
def cornerInside(box1, box2, old_pt = None, np = 0):
	b1_edges = box1.get_map()	
	max_num = 2**len(b1_edges)
	d = box2.getPrecision()
	if DEBUG:
		print('cornerInside','old_pt', str(old_pt), str(d))
	if (old_pt == None or np == 0):
		if DEBUG:
			print('Here old_pt None')
		point_map = {}	
		for it in sorted(b1_edges.keys()):
			intrvl = b1_edges[it]
			point_map.update({it:PyInterval(intrvl.leftBound(), intrvl.leftBound())})
		old_pt = Box(point_map)
		r = box2.fullyContains(old_pt)
		if r:
			return old_pt
	#else:
	old_edges = old_pt.get_map()
	pmap = {}
	for it in sorted(old_edges.keys()):
		intrvl = old_edges[it]
		b_intrvl = b1_edges[it]
		if(intrvl.leftBound() == b_intrvl.leftBound()):
			pmap.update({it:0})
		elif(intrvl.leftBound() == b_intrvl.rightBound()):
			pmap.update({it:1})
	if DEBUG:
		print('pmap', pmap)

	number = 0		
	i = 0
	#print('binary from lsb: ')
	for it in sorted(pmap.keys()): # binary of the number
		#print(pmap[it])
		number += (pmap[it])* 2**(max_num-i-1)
		i += 1	
	if DEBUG:
		print('number', number)		
	n1 = grayToBinary(number)
	if DEBUG:
		print(n1)
	if( np >= max_num-1): #n1 >= max_num-1):
		return None
			
	# print('Finding Next point')
	flag = False
	j = 0	
	while(not flag):
		nextPt = nextPoint(box1, old_pt)		
		#pt = Box(point_map)
		if DEBUG:
			print('loop iteration {0} cornerInside oldpt {1} ,next point = {2}, precision = {3}'.format(j, str(old_pt), str(nextPt), d))
		r = box2.fullyContains(nextPt)
		if r:
			flag = True
			return nextPt
		else:
			old_pt = nextPt
		j += 1
		if(j == max_num):
			break;
	return None
				
''' Next pt of box1'''
def nextPoint(box1, old_pt):
	if DEBUG:
		print('nextPoint: old_pt', str(old_pt))
	b1_edges = box1.get_map()
	old_edges = old_pt.get_map()
	pmap = {}
	dimensions = {}
	i = 0
	for it in sorted(old_edges.keys()):
		intrvl = old_edges[it]
		b_intrvl = b1_edges[it]
		if(intrvl.leftBound() == b_intrvl.leftBound()):
			pmap.update({it:0})
		elif(intrvl.leftBound() == b_intrvl.rightBound()):
			pmap.update({it:1})
		dimensions.update({i:it})
		i += 1
	k = i
	
	i = 0
	number = 0	
	#print('binary from lsb: ')
	for it in pmap: # binary of the number
		#print(pmap[it])
		number += (pmap[it])* 2**(k-i-1)
		i += 1		
	# print('current {0} bit  number {1}'.format(str(k), number))
	
	''' Get one of closest point which is not visited yet, may be using next gray code'''
	next_number = nextGray(number, k)
	next_pmap = {}
	# number to binary
	q = next_number
	i = 0
	while(not int(q) == 0 and i < k):
		r = q%2
		it = dimensions[k-i-1]
		next_pmap.update({it:r})
		q = q//2
		i += 1
	for it in sorted(b1_edges.keys()):
		if it not in next_pmap:
			next_pmap.update({it:0})
			
	# print(string(next_pmap))
			
	point_map = {}	
	for it in sorted(next_pmap.keys()):
		intrvl = b1_edges[it]
		if(next_pmap[it] == 1):
			intrvl2 = PyInterval(intrvl.rightBound(), intrvl.rightBound())
		else:
			intrvl2 = PyInterval(intrvl.leftBound(), intrvl.leftBound())
		point_map.update({it:intrvl2})		
		
	
	pt = Box(point_map)
	if DEBUG:
		print('nextPoint: new_pt', str(pt))
	return pt

def string(pmap):
	s = ''
	for it in pmap:
		s += str(pmap[it])
	return s	
	
def gray(n):
	#n = n-1
	n1 = n ^ (n >> 1)
	# print('gray', 'number {0}, next numver {1}'.format(str(n), str(n1)))
	return n1
  
def nextGray(num, k):
	n1 = grayToBinary(num)
	n2 = n1 + 1
	if n2 > 2**k - 1:
		n2 = 0
	num1 = binaryToGray(n2)
	# print('nextGray', 'number: {0}, binary: {1} next number: {2}'.format(str(num), str(n1), str(num1)))
	return num1
	
'''The purpose of this function is to convert an unsigned
        binary number to reflected binary Gray code.

        The operator >> is shift right. The operator ^ is exclusive or.
'''
def binaryToGray(num):
	#num -= 1
	num1 =  (num >> 1) ^ num
	#print(str(num), 'binaryToGray:',  str(num1))
	return num1
	
'''The purpose of this function is to convert a reflected binary
        Gray code number to a binary number. '''
def grayToBinary(num):
	mask = num >> 1
	num1 = num
	while(not mask == 0):
		num1 = num1 ^ mask
		mask = mask >> 1
	#(str(num), 'grayToBinary:',  str(num1))
	return num1

def intersectsAny(b, boxes):
	for b1 in boxes:
		if b.intersects(b1):
			return True
	else:
		return False
		
# def intersect(lhs, rhs): #capd::interval lhs, capd::interval rhs)
# 	return lhs.contains(rhs.leftBound()) or  lhs.contains(rhs.rightBound()) or rhs.contains(lhs.leftBound()) or rhs.contains(lhs.rightBound())

def toRange(it):
		return Range(Node(it.leftBound()), Node(it.rightBound()))

def diag_to_Box(edges, p1, p2):
	b_edges = {}
	i = 0
	for it in sorted(edges):
		b_edges.update({it:PyInterval(p1[i], p2[i])})
		i+= 1
	box = Box(b_edges)
	# print('diag_to_Box', p1, p2, box)
	return box
