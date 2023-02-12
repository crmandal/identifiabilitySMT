# from __future__ import division

import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paramUtil.interval import *
DEBUG = False
'''
# definition of __getinitargs__
def interval_getinitargs(self):
	return (self.leftBound(), self.rightBound(),)

# now inject __getinitargs__ (Python is a dynamic language!)
PyInterval.__getinitargs__ = interval_getinitargs
'''
class Box:
	def __init__(self, edges = {}, d = 0.001):
		flag = 1
		for key in sorted(edges.keys()):
			if edges[key].width() < 0:
				flag = 0
				print('invalid interval '+ key+ ' : '+ str(edges[key])+' for a box' )
				break
				# throw invalid argument
		if(flag == 1):
			self.edges = edges
		else:
			raise ValueError
		self.precision = self.minDimension() * d

	#box::box(string line)
	#{
		#// removing whitespaces
		#line.erase(remove_if(line.begin(), line.end(), ::isspace), line.end());
		#size_t pos = 0;
		#map<string, capd::interval> b_map;
		#while (line.length() > 0)
		#{
			#pos = line.find(';');
			#if(pos == string::npos)
			#{
				#ostringstream s;
				#s << 'Every variable definition must finish with \';\'';
				#throw invalid_argument(s.str());
			#}

			#string edge_string = line.substr(0, pos);
			#line.erase(0, pos + 1);

			#size_t pos2 = edge_string.find(':');
			#string var = edge_string.substr(0, pos2);

			#string interval_string = edge_string.substr(pos2 + 1, edge_string.length() - pos2);

			#size_t pos3 = interval_string.find(',');
			#b_map.insert(make_pair(var, capd::interval(interval_string.substr(1, pos3 - 1), interval_string.substr(pos3 + 1, interval_string.length() - pos3 - 2))));
		#}
		#this->edges = b_map;
	#}
	
	def size(self):
		return len(self.edges)
		
	def get_map(self):
		return self.edges
	
	def set_map(self, edges):
		self.edges = edges
	
	def minDimension(self):
		edges = self.get_map()
		i = 0
		minD = 0
		for it in edges:
			w = edges[it].width()
			if i == 0:
				minD = w
			else:
				if minD > w:
					minD = w
			i = i+1
		return minD

	def empty(self):
		return self.maxDimension() == 0

	def getInterval(self, it):
		edges = self.get_map()
		intrvl = edges[it]
		return intrvl
	
	def addInterval(self, it, interval):
		edges = self.get_map()
		edges.update({it: interval})
		self.set_map(edges)
	
	def addPrecision(self, delta):
		self.precision = delta_perturb
	
	def getPrecision(self):
		return self.precision
				
	def printEdges(self):
		edges = self.get_map()
		os = ''
		j = 0
		for it in sorted(edges.keys()):
			intrvl = edges[it]
			if j == 0:
				os +=  str(it)+ ':' + str(intrvl)
			else:
				os += ', '+ str(it)+ ':' + str(intrvl)
			j += 1
		return os
	
	def volume(self):
		edges = self.get_map()
		vol = 1.0
		for it in sorted(edges.keys()):
			vol *= edges[it].width()
		return vol
		
	def __str__(self):		
		os = self.printEdges() #+';'
		return os

	def __repr__(self):		
		os = self.printEdges() #+';'
		return os

	'''returns true if box is empty'''
	def empty(self):
		return (len(self.get_map()) == 0)

	def contains(self, b):
		edges = self.get_map();
		b_edges = b.get_map();
		for it in sorted(edges.keys()):
			intrvl = edges[it]
			if(it in b_edges):
				#print('Box['+it+'] : '+ str(intrvl)+', SubBox['+it+'] : '+ str(b_edges[it]))
				if(not intrvl.contains(b_edges[it])):
					return False;
			else:
				print('The target box does not contain variable: \'' + it + '\'')
				raise ValueError()
		return True;
	
		
	''' checks if a point is fully contained in a box 
		the point position in any of the dimension should be within the left and right value of that dimension of Box
	'''
	def fullyContains(self, point):
		d = self.precision
		edges = self.get_map();
		p_edges = point.get_map();
		res = self.contains(point)
		if res:
			flag = True
			for it in sorted(edges.keys()):
				intrvl = edges[it]
				if(it in p_edges.keys()):
					p_intrvl = p_edges[it]
					# print('Box['+it+'] : '+ str(intrvl)+', SubBox['+it+'] : '+ str(p_intrvl))
					#if(intrvl.contains(p_intrvl)):
					r1 = p_intrvl.leftBound() - intrvl.leftBound()
					r2 = intrvl.rightBound() - p_intrvl.rightBound()
					# print('FullyContains['+it+'] ', r1, r2, d)
					if(r1 > d and r2 > d):
						#if(p_intrvl.leftBound() < intrvl.leftBound() and  p_intrvl.leftBound() > intrvl.rightBound()):
						#if(p_intrvl.leftBound() > intrvl.leftBound() and  p_intrvl.rightBound() < intrvl.rightBound()):
						# print('Fullycontains:', str(it), str(intrvl), 'point: ' ,str(p_intrvl))
						return True
				else:
					print('The target box does not contain variable: \'' + it + '\'')
					raise ValueError()
		# print('Point not FullyContained')
		return False;
		
	def intersects(self, b):
		edges = self.get_map();
		b_edges = b.get_map();
		for it in sorted(edges.keys()):
			intrvl = edges[it]
			if(it in b_edges.keys()):
				if(not (intrvl.contains(b_edges[it].leftBound()) or intrvl.contains(b_edges[it].rightBound()) or b_edges[it].contains(intrvl.leftBound()) or b_edges[it].contains(intrvl.rightBound()))):
					return False
			else:
				print('The target box does not contain variable: \'' + it + '\'')
				raise ValueError()
		return True

	def adjacent11(self, b):
		if DEBUG:
			print('Box-adjacent', self, b, b.size())
		edges = self.get_map();
		b_edges = b.get_map();
		lc = 0
		for it in sorted(edges.keys()):
			intrvl = edges[it]
			if(it in b_edges.keys()):
				if (intrvl.leftBound() == b_edges[it].rightBound()) or (intrvl.rightBound() == b_edges[it].leftBound()):
					lc += 1
					if DEBUG:
						print('Box-adjacent', lc, 'intervals', intrvl.leftBound(), '==', b_edges[it].rightBound(), \
						intrvl.rightBound(), '==', b_edges[it].leftBound())
					#return True
			else:
				print('The target box does not contain variable: \'' + it + '\'')
				raise ValueError()
		if lc > 0 and lc == b.size():
			return True
		return False
	
	def overlaps(self, b):
		#print('Box-overlaps', self, b, b.size())
		s_edges = self.get_map();
		b_edges = b.get_map();
		isOverlaps = True
		for it in sorted(s_edges.keys()):
			s_intrvl = s_edges[it]
			b_intrvl = b_edges[it]
			isOverlaps  = isOverlaps and (s_intrvl.overlaps(b_intrvl))

		if DEBUG:
			print('Box-overlaps', self, b, isOverlaps)
		return isOverlaps
	
	def merge(self, b):
		#print('Box-overlaps', self, b, b.size())
		s_edges = self.get_map();
		b_edges = b.get_map();
		m_edges = {}
		for it in sorted(s_edges.keys()):
			s_intrvl = s_edges[it]
			b_intrvl = b_edges[it]
			m_int  = s_intrvl.merge(b_intrvl)
			m_edges.update({it:m_int})
		m = Box(m_edges)
		if DEBUG:
			print('Box- merge', self, b, m)
		return m
	
	def adjacent(self, b):
		if DEBUG:
			print('Box-adjacent', self, b, b.size())
		return self.overlaps(b)
		
	def atBoundary(self, b):
		edges = self.get_map();
		b_edges = b.get_map();
		for it in sorted(edges.keys()):
			intrvl = edges[it]
			if(it in b_edges.keys()):
				if intrvl.leftBound() == b_edges[it].leftBound() or intrvl.rightBound() == b_edges[it].rightBound() or intrvl.leftBound() == b_edges[it].rightBound() or intrvl.rightBound() == b_edges[it].leftBound():
					return True
			else:
				print('The target box does not contain variable: \'' + it + '\'')
				raise ValueError()
		return False
		
	def adjacentEdges(self, b):
		edges = self.get_map();
		b_edges = b.get_map();
		adj_Edges = []
		for it in sorted(edges.keys()):
			intrvl = edges[it]
			if(it in b_edges.keys()):
				if intrvl.leftBound() == b_edges[it].rightBound() or intrvl.rightBound() == b_edges[it].leftBound():
					#return True
					adj_Edges.append(it)
			else:
				print('The target box does not contain variable: \'' + it + '\'')
				raise ValueError()
		return adj_Edges

	def compatible(self, b):
		return self.get_keys_diff(b).empty() and b.get_keys_diff(self).empty()


	def __le__(self, rhs): #checking if dimensions of the boxes are the same
		if( not self.get_vars() == rhs.get_vars()):
			s = ''
			s+= 'Variables of the compared boxes are not the same';
			raise ValueError()
			
		d = self.precision
		
		lhs_map = self.get_map()
		rhs_map = rhs.get_map()
		for it in sorted(lhs_map.keys()):
			intrvl = lhs_map[it]
			#if(intrvl.leftBound() < rhs_map[it].leftBound()):
			#	return True
			res = intrvl.leftBound() - rhs_map[it].leftBound()
			if(res > d):
				return False
		return True
	
	def __ge__(self, rhs): #checking if dimensions of the boxes are the same
		if( not self.get_vars() == rhs.get_vars()):
			s = ''
			s+= 'Variables of the compared boxes are not the same';
			raise ValueError()
				
		d = self.precision
		
		lhs_map = self.get_map()
		rhs_map = rhs.get_map()
		for it in sorted(lhs_map.keys()):
			intrvl = lhs_map[it]
			res = rhs_map[it].rightBound() - intrvl.rightBound()
			if(res > d):
				return False
		return True
	
	''' checking if dimensions of the boxes are the same'''
	def __eq__(self, rhs): #
		if(self is None or rhs is None):
			return False
		if( not self.get_vars() == rhs.get_vars()):
			s = 'Variables of the compared boxes are not the same: {0}, {1}'.format(self.get_vars(),rhs.get_vars());
			print(s)
			raise ValueError()
			
		d = self.precision
		
		lhs_map = self.get_map()
		rhs_map = rhs.get_map()
		for it in sorted(lhs_map.keys()):
			intrvl = lhs_map[it]
			r1 = abs(intrvl.leftBound() - rhs_map[it].leftBound()) 
			r2 = abs(intrvl.rightBound() - rhs_map[it].rightBound())
			#print(it, r1, r2, d)
			if( r1 >= d or r2 >= d ):
				return False
		return True

	def __add__(self, rhs):
		if(not self.get_keys_diff(rhs).empty() or not rhs.get_keys_diff(self).empty()):
			s = 'cannot perform \'+\' operation for ' + str(self) + ' and ' + str(rhs) + '. The boxes have different sets of variables'
			print(s)
			raise ValueError()
		
		lhs_map = self.get_map()
		rhs_map = rhs.get_map()
		res = {}
		for it in sorted(lhs_map.keys()):
			intrvl1 = lhs_map[it]
			intrvl2 = rhs_map[it]
			# print('box add', it, intrvl1, intrvl2, type(intrvl1))
			res.update({it: intrvl1 + intrvl2})
		# print(res)
		return Box(res)

	def __sub__(self, rhs):
		if(not self.get_keys_diff(rhs).empty() or not rhs.get_keys_diff(self).empty()):
			s = 'cannot perform \'-\' operation for ' + str(self) + ' and ' + str(rhs) + '. The boxes have different sets of variables'
			raise ValueError()
		
		lhs_map = self.get_map()
		rhs_map = rhs.get_map()
		res = {}
		for it in sorted(lhs_map.keys()):
			intrvl1 = lhs_map[it]
			intrvl2 = rhs_map[it]
			res.update({it: (intrvl1 - intrvl2)})
		return Box(res)
		
	def __mul__(self, rhs):
		if(not self.get_keys_diff(rhs).empty() or not rhs.get_keys_diff(self).empty()):
			s = 'cannot perform \'*\' operation for ' + str(self) + ' and ' + str(rhs) + '. The boxes have different sets of variables'
			raise ValueError()
		
		lhs_map = self.get_map()
		rhs_map = rhs.get_map()
		res = {}
		for it in sorted(lhs_map.keys()):
			intrvl = lhs_map[it]
			res.update({it: (intrvl * rhs_map[it])})
		return Box(res)

	def __truediv__(self, rhs):
		if(not self.get_keys_diff(rhs).empty() or not rhs.get_keys_diff(self).empty()):
			s = 'cannot perform \'/\' operation for ' + str(self) + ' and ' + str(rhs) + '. The boxes have different sets of variables'
			raise ValueError()
		
		rhs_map = rhs.get_map()
		lhs_map = self.get_map()
	
		res = {}
		for it in sorted(lhs_map.keys()):
			intrvl1 = lhs_map[it]
			intrvl2 = rhs_map[it]
			res.update({it: (intrvl1 / intrvl2)})
		return Box(res)		
	
	def __div__(self, rhs):
		if(not self.get_keys_diff(rhs).empty() or not rhs.get_keys_diff(self).empty()):
			s = 'cannot perform \'/\' operation for ' + str(self) + ' and ' + str(rhs) + '. The boxes have different sets of variables'
			raise ValueError()
		
		rhs_map = rhs.get_map()
		lhs_map = self.get_map()
	
		res = {}
		for it in sorted(lhs_map.keys()):
			intrvl1 = lhs_map[it]
			intrvl2 = rhs_map[it]
			res.update({it: (intrvl1 / intrvl2)})
		return Box(res)	
	
	'''
	# Returns the edges of the box
	'''
	def get_intervals(self):
		i = []
		edges = self.get_map()
		for it in sorted(edges.keys()):
			intrvl = edges[it]  
			i.append(intrvl)
		return i

	'''
	#Returns the variables of the box
	'''
	def get_vars(self):
		v = []
		edges = self.get_map()
		for it in sorted(edges.keys()):
			v.append(it)
		return v

	def get_mean(self):
		edges = self.get_map()
		mu_map = {}
		for it in sorted(edges.keys()):
			intrvl = edges[it]  
			mu_map.update({it: intrvl.mid()})
		return Box(mu_map)

	def get_stddev(self):
		edges = self.get_map()
		sigma_map = {}
		for it in sorted(edges.keys()):
			intrvl = edges[it] 
			lb = intrvl.leftBound()
			mb = intrvl.mid().leftBound()
			i = PyInterval(lb, mb)
			sigma_map.update({it: i.width()});
		return Box(sigma_map)

	# def max_left_coordinate_value(self):
	# 	edges = self.get_map()
	# 	i = 0
	# 	m = 0
	# 	for it in sorted(edges.keys()):
	# 		intrvl = edges[it] 
	# 		if i==0:
	# 			m = intrvl.leftBound()
	# 		else:
	# 			if intrvl.leftBound() > m:
	# 				m = intrvl.leftBound()
	# 		i= i+1
	# 	return m
			
	def min_left_coordinate_value(self):
		edges = self.get_map()
		i = 0
		m = {}
		for it in sorted(edges.keys()):
			intrvl = edges[it] 
			m.update({it:intrvl.leftBound()})
		return m
	
	
	def max_right_coordinate_value(self):
		edges = self.get_map()
		i = 0
		m = {}
		for it in sorted(edges.keys()):
			intrvl = edges[it] 
			m.update({it:intrvl.rightBound()})
		return m

	def max_side_width(self):
		edges = self.get_map()
		i = 0
		m = 0
		for it in edges.keys():
			intrvl = edges[it] 
			if i==0:
				m = intrvl.width()
			else:
				if intrvl.width() > m:
					m = intrvl.width()
			i= i+1
		return m

	def min_side_width(self):
		edges = self.get_map()
		i = 0
		m = 0
		for it in edges.keys():
			intrvl = edges[it] 
			if i==0:
				m = intrvl.width()
			else:
				if intrvl.width() < m:
					m = intrvl.width()
			i= i+1
		return m
		
	def get_keys_diff(self, b):
		res = {}
		lhs_map = self.get_map()
		rhs_map = b.get_map()
		for it in lhs_map.keys():
			intrvl = lhs_map[it]
			if(it not in rhs_map):
				res.update({it: intrvl})
		return Box(res);

	def fmod(self, mod):
		res = {}
		edges = self.get_map()
		for it in edges.keys():
			intrvl = edges[it] 
			res.update({it: PyInterval(intrvl.leftBound()-(long)(intrvl.leftBound()))})
		return Box(res)

	def addDelta(self, delta):
		b_edges =  self.get_map()
		edges = {}
		for it in b_edges:
			intrvl = b_edges[it]
			lb = intrvl.leftBound() #- delta/2
			ub = intrvl.rightBound() #+ delta/2
			if (ub - lb) > 5*delta:
				lb = intrvl.leftBound() - delta/2
				ub = intrvl.rightBound() + delta/2
				if lb < 0:
					lb = intrvl.leftBound()
			else:
				dd = (ub-lb)*0.01
				lb = intrvl.leftBound() - dd/2
				ub = intrvl.rightBound() + dd/2
				if lb < 0:
					lb = intrvl.leftBound()
			if lb > ub :
				lb1 = intrvl.leftBound()*(1-0.01) #- delta/2
				ub1 = intrvl.rightBound()*(1-0.001)
				lb = min(lb, ub)
				ub = max(lb, ub)

			edges.update({it:PyInterval(lb,  ub)})

		return Box(edges)

	def diag_bound(self):
		b_edges =  self.get_map()
		p1 = []
		p2 = []

		for it in sorted(b_edges.keys()):
			intrvl = b_edges[it]

			lb = intrvl.leftBound()
			ub = intrvl.rightBound()
			p1.append(lb)
			p2.append(ub)
		return p1, p2




