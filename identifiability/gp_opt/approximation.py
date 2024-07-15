
import math
import numpy as np
import random
from scipy import interpolate
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn_extra.cluster import KMedoids
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
from scipy.spatial.qhull import QhullError

import collections
from collections import OrderedDict

from paramUtil.interval import *
from paramUtil.box import *
import paramUtil.box_factory as bfact
import paramUtil.kd_interval as kdi
import paramUtil.kd_intervaltree as kdivt

import GPR as gp_grad
import MLP as mlp

import numpy as np
import random as rnd
import time
DEBUG = False

import itertools

CLUSTER = True
# CLUSTER = False
MAX_clusters = 30
EPS = 0.001
samples_per_dim = 3
IVT = False
IVT = True
NN = False
#NN = True
prob_thres = 0.499

prev_ldu, prev_udu = 0.0, 0.0


def extendEnvelope(covered_region, sbox, du =  0.05):
	cr = covered_region #getGRID(covered_region, sbox, du)
	sbox_edges = sbox.get_map()
	en_edges = cr.get_map()
	en_ex_edges = {}
	for it in en_edges.keys():
		# it_lb = max(sbox_edges[it].leftBound()/du, sbox_edges[it].leftBound()) #max(en_edges[it].leftBound()*10/25.0, sbox_edges[it].leftBound())
		# it_ub = min(sbox_edges[it].rightBound()*du, sbox_edges[it].rightBound())	 #min(en_edges[it].rightBound()*25/10.0, sbox_edges[it].rightBound())	
		it_lb = max(en_edges[it].leftBound() - du, sbox_edges[it].leftBound())
		it_ub = min(en_edges[it].rightBound() + du, sbox_edges[it].rightBound())
		en_ex_edges.update({it:PyInterval(it_lb, it_ub)})
	extended_box = Box(en_ex_edges)

	return extended_box

def getGRID(b, sbox, RES):
	big_s_map = sbox.get_map()
	b_map = b.get_map()

	x_map = {}
	for i in b_map.keys():
		ll = max(math.floor(b_map[i].leftBound()/RES)*RES,  big_s_map[i].leftBound())
		ul = min(math.ceil(b_map[i].rightBound()/RES)*RES,  big_s_map[i].rightBound())
		x_map.update({i:PyInterval(ll, ul)})
	g = Box(x_map)
	return g

def getGRID_boxes(b, sbox, RES):
	GS = 4*RES
	g = getGRID(b, sbox, RES)
	max_dim = 1 #math.ceil(g.max_side_width()/(GS))
	b_map = g.get_map()
	# global noise
	emap = {}
	for it in b_map.keys():     
		# emap.update({it: PyInterval(0.01 * noise[it])})
		emap.update({it: PyInterval(GS, GS)})
	# grid_boxes  = []
	boxes = [g]
	for i in range(max_dim):
		g_boxes = []
		for b1 in boxes:
			bbs = bfact.bisect(b1, emap)  # if b.max_side_width() > 2*RES]
			for bb in bbs:
				g_boxes.append(bb)
		boxes = g_boxes
		if len(g_boxes) >= 64:
			break
		# print('getGRID_boxes', i, ['({0})'.format(b1) for b1 in boxes])
	return boxes #bfact.partition(g, 2*RES)



def boxtopoint(b, param_names = []):
	bmap = b.get_map()
	if len(param_names) == 0:
		param_names = sorted(bmap.keys())
	pts = []
	for it in sorted(param_names):
		pts.append((bmap[it].leftBound() + bmap[it].rightBound())/2)
	# print(p, bmap)
	return pts

def getHull(boxes):
	true_points = [boxtopoint(tb) for tb in boxes]
	lpn = len(boxes[0].get_map().keys()) if len(boxes) > 0 else 4
	if len(boxes) > max(5, lpn):
		try:
			hull = ConvexHull(true_points)
		except QhullError:
			hull = None 
	else:
		hull =  None #true_points
	return hull

def getHullBoundary(boxes, hull = None):
	if len(boxes) > 5:
		if not hull:
			hull = getHull(boxes)
		if hull:
			select_c3 = [boxes[i] for i in hull.vertices]
		else:
			select_c3 = boxes
	else:
		select_c3 = boxes
	return select_c3


def isInHull(box, hull, tolerance=1e-12):
	if hull:
		try:
			point = boxtopoint(box)
		except TypeError:
			print('isInHull', str(box))
			raise TypeError('Incorect box', str(box))
		return all( (np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)
	else:
		return True

def isValid(x):
	if  math.isnan(x) or math.inf == x or -math.inf == x:
		return False
	else:
		# if isinstance(x, list) and len(x) > 1:
		# 	return False
		return True

def find_k_closest(centroids, data, k=1, distance_norm=2):
	"""
	Arguments:
	----------
		centroids: (M, d) ndarray
			M - number of clusters
			d - number of data dimensions
		data: (N, d) ndarray
			N - number of data points
		k: int (default 1)
			nearest neighbour to get
		distance_norm: int (default 2)
			1: Hamming distance (x+y)
			2: Euclidean distance (sqrt(x^2 + y^2))
			np.inf: maximum distance in any dimension (max((x,y)))

	Returns:
	-------
		indices: (M,) ndarray
		values: (M, d) ndarray
	"""

	kdtree = cKDTree(data) #, leafsize=leafsize)
	distances, indices = kdtree.query(centroids, k, p=distance_norm)
	if k > 1:
		indices = indices[:,-1]
	values = data[indices]
	return indices, values

def getAPPROXBoxes(sbox, pre_envelope, current_sbox, kdt_true, kdt_false, true_boxes, false_boxes, par_names, gen, MIN_EPS, MIN_EPS_3, RES, connected,  extra = False, class_flag = 0):
	
	# random.seed(gen)
	t_boxes = true_boxes
	f_boxes = false_boxes

	stm1 = time.time()

	# changed on 20/01/2023 
	# MIN_EPS_3 = 0.01
	# #gen%2 == 1 or	
	# MIN_EPS_3 = 2*0.005 = 0.01
	# NN = 0.05
	# RES = 0.01
	if  gen%4 < 2:
		ldu = MIN_EPS_3 * 2
		udu = MIN_EPS_3 * 5		
		# 0.02, 0.05	
	else:
		ldu = MIN_EPS_3 * 5
		udu = MIN_EPS_3 * 10
	rn =  np.random.uniform(0, 1, 1)[0]
	if rn < 0.25:
		ldu = MIN_EPS_3 * 10
		udu = MIN_EPS_3 * 20
		# 0.05, 0.08
	# du = ldu 
	''' changed on 20/10/2023
	global prev_ldu, prev_udu
	if gen%10 == 0:
		if len(par_names) > 2:
			ldu = min(prev_udu, 0.1) if gen > 0 else MIN_EPS_3 * 4
			udu = min(ldu*1.2, 0.2) if ldu > 0 else MIN_EPS_3 * 10
		else:
			ldu = min(prev_udu, 0.1) if gen > 0 else MIN_EPS_3 * 4
			udu = min(ldu*2.0, 0.2) if ldu > 0 else MIN_EPS_3 * 10
		prev_ldu, prev_udu = ldu, udu
	else:
		ldu, udu = prev_ldu, prev_udu
	'''
	du = np.random.uniform(ldu, udu, 1)[0] #0.05
	print('du', du, 'ldu', ldu, 'udu', udu)

	npg = []
	# print('t_boxes', len(t_boxes), 'gen', gen)
	#NPG = False # For gradient only
	#NPG = True # For GPR only
	NPG = True if ((gen%2==0) and len(t_boxes) > len(par_names)) else False
	print('In Approx boxes', du, gen, NPG, gen%2==0, len(t_boxes), len(par_names))
	
	rn1 =  np.random.uniform(0, 1, 1)[0]
	if len(t_boxes) <  10*(len(par_names)-1) or rn1 < 0.10: #*len(par_names): # and len(c33) == 0:
		npg_pi, c33 = getNextBoxesFromNN(sbox, pre_envelope, current_sbox, kdt_true, kdt_false, t_boxes,f_boxes, du, gen, MIN_EPS, MIN_EPS_3,  RES, connected,extra = extra)
	else:
		if NPG :
			npg_pi, c33 = getNextBoxesFromGradient(sbox, pre_envelope, current_sbox, kdt_true, kdt_false, t_boxes,f_boxes, du, gen, MIN_EPS, MIN_EPS_3,  RES, connected,extra = extra, class_flag = class_flag)
		else:
			npg_pi, c33 = getNextBoxesFromGPR(sbox, pre_envelope, current_sbox, kdt_true, kdt_false, t_boxes,f_boxes, du, gen, MIN_EPS, MIN_EPS_3, RES, connected,extra = extra, class_flag = class_flag)
		
	npg = npg_pi

	etm1 = time.time()
	print('---------- TIME ------------- time taken for approximation using {0} boxes {1} hr'.format(len(t_boxes), (etm1 - stm1)/3600.0))
	return npg, c33 #kdt_npg_sat, npg

def getNN(current_sbox, extra_boxes, params, pn, du, RES):
	GS = RES
	s_map = current_sbox.get_map()
	# en_edges = envelope.get_map()

	big_s_map = current_sbox.get_map()

	# all_params = s_map.keys()
	# all_param_len = len(all_params)
	# param_names_sorted = sorted(all_params)
	# param_index_dict = {}
	# i = 0
	# for par in param_names_sorted:
	# 	param_index_dict.update({par:i})
	# 	i +=1

	ulist = np.linspace(-du, du, samples_per_dim)
	#[-du, du/2, 0, du/2, du] 
	#[-du, du/2, 0, du/2, du]
	selected = []

	for b in extra_boxes:
		#print('--- from box ---- ', str(b), 'du', du)
		# b = c3[ci]
		#for b in select_c3:
		blist = []
		bmap = b.get_map()
		i = 0
		npg_cc = []
		for key in params:
			itb = (bmap[key].leftBound()+bmap[key].rightBound())/2
			i += 1
			#npg_cc.append(blist)
			blist = [(itb + element) for element in ulist]
			npg_cc.append(blist)
			#print('blist1', blist)

		for elem in itertools.product(*npg_cc):
			#print('elem', elem)
			flag = 1 
			bmap = {}
			i = 0
			for key in params:
				ll = max(math.floor(elem[i]/GS)*GS,  big_s_map[pn].leftBound())
				ul = min(math.ceil(elem[i]/GS)*GS,  big_s_map[pn].rightBound())
				if ll > ul:
					flag = 0
					break
				else:
					bmap.update({key:PyInterval(ll, ul)})
				i += 1

			# changed on 22/10/2023
			#blist1 = [blist[i] *(1 + element[i]) for i in range(len(params)-1)]
			
			if flag:
				bb = Box(bmap)
				if (bb not in extra_boxes) and (bb not in selected):
					selected.append(bb)
				#print('NN box', str(bb))
	return selected


def getNextBoxesFromGPR(sbox, pre_envelope, current_sbox, kdt_true, kdt_false, true_boxes, false_boxes, du, gen, MIN_EPS, MIN_EPS_3, RES, connected, extra = False, class_flag = 0):
	GS = RES
	stm1 = time.time()

	ParValue = collections.namedtuple('ParValue', ['id', 'val', 'std'])
	#big_s_map = sbox.get_map()
	s_map = current_sbox.get_map()
	big_s_map = current_sbox.get_map()
	# en_edges = envelope.get_map()
	all_params = s_map.keys()
	all_param_len = len(all_params)
	param_names_sorted = sorted(all_params)
	param_index_dict = {}
	i = 0
	for par in param_names_sorted:
		param_index_dict.update({par:i})
		i +=1

	
	def pointtobox(p): #, param_names = param_names_sorted):
		bmap = {}
		ij = 0
		for it in sorted(param_names_sorted):
			# changed on 11/05/2022
			bmap.update({it:PyInterval(p[ij]*(1-MIN_DELTA/2), p[ij]*(1+MIN_DELTA/2))})
			ij += 1
		# print(p, bmap)
		return Box(bmap)

	envelope = bfact.get_cover(true_boxes, check=False)
	# true_covered =  bfact.get_cover(true_boxes, check=False)
	#extended_box1 = extendEnvelope(envelope, current_sbox, du) #10*MIN_EPS) #EPS)
	#extra_boxes = bfact.remove(extended_box1, envelope)

	c31 = [b for b in true_boxes] #rnd.sample(true_boxes, min(len(true_boxes), 7**all_param_len))]

	# stm1 = time.time()

	hull = getHull(true_boxes)

	c32 = c31 #+ select_c3
	if len(true_boxes) > 20*(all_param_len-1):
		if CLUSTER:
			# MAXC = min(MAX_clusters, len(select_c3))
			# changed on 11/05/2022 
			MAXC = min(5**(all_param_len-1), len(c32))
			#min(all_param_len*MAX_clusters, 5**(all_param_len-1), len(c32)) ##MAX(MAX_clusters, len(c32))
			
			# changed on 20/01/2023 
			# MAXC = min(max(int(envelope.volume()/((2*MIN_EPS)**all_param_len)), 5**all_param_len), len(c32)) ##MAX(MAX_clusters, len(c32))

			X_1 = np.array([boxtopoint(b) for b in c32])
			chosen_k = MAXC
		
			'''pre_savg = -1
			pre_score = -1
			for n_clusters in range(2*all_param_len, MAXC):
			# Initialize the clusterer with n_clusters value and a random generator
			# seed of 10 for reproducibility.
				#clusterer = KMeans(n_clusters=n_clusters, random_state=50)
				clusterer = KMedoids(n_clusters=n_clusters, random_state=gen) #, init='heuristic'
				cluster_labels = clusterer.fit_predict(X_1)

				# The silhouette_score gives the average value for all the samples.
				# This gives a perspective into the density and separation of the formed
				# clusters
				# if int(gen/2)%3 == 0 and len(cluster_labels) > 1 and len(X_1) > 1:
				# 	# silhouette_avg = -1*davies_bouldin_score(X_1, cluster_labels)
				# 	silhouette_avg = silhouette_score(X_1, cluster_labels)
				# 	#print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
				# 	if pre_score < silhouette_avg:
				# 		pre_score = silhouette_avg
				# 		chosen_k = n_clusters
				# else:		
				silhouette_avg = (pre_savg - clusterer.inertia_) # elbow method
				if silhouette_avg < EPS:
					chosen_k = n_clusters
					break				
				pre_savg =  clusterer.inertia_
			'''

			clusterer =  KMedoids(n_clusters=chosen_k,  random_state=gen).fit(X_1) #init='heuristic',
			centroids = clusterer.cluster_centers_
			#centroids_1 = centroids
			# clusterer = KMeans(n_clusters=chosen_k, random_state=gen).fit(X_1)
			# centroids = clusterer.cluster_centers_
			center_indices, centroids_1 = find_k_closest(centroids, X_1)
			c33 = [] 
			# c33 = select_c3
			# X = list(X_1)
			for cc in center_indices: #centroids_1:
				b = c32[cc]
				c33.append(b) #pointtobox(cc))

			# changed on 11/05/2022 	
			#c3 += c33

		else:
			# # added on 15/01/2023
			# MAXC = min(all_param_len*MAX_clusters, 5**all_param_len, len(c32)) ##MAX(MAX_clusters, len(c32))

			# changed on 20/01/2023 
			MAXC = min(max(int(envelope.volume()/((2*MIN_EPS)**all_param_len)), 5**all_param_len), len(c32)) ##MAX(MAX_clusters, len(c32))
			
			X_1 = np.array([boxtopoint(b) for b in c32])

			per_dim = int(MAXC/all_param_len)
			s_indices = np.linspace(0, len(X_1)-1, per_dim)
			s_points = []
			for i in range(all_param_len):
				X_1_sorted_i = sorted(X_1, key = lambda x: x[i])
				for j in s_indices:
					s_points.append(X_1_sorted_i[int(j)])

			center_indices, centroids_1 = find_k_closest(s_points, X_1)
			
			c33 = []
			for cc in center_indices: #centroids_1:
				b = c32[cc]
				c33.append(b) #pointtobox(cc))
	else:
		c33 = c31
	if len(c33) < all_param_len:
		c33 = c31
	# c3 = c33
	# c33 = c33 + select_c3
	st2 = time.time()
	print('gpr -- ', st2 -stm1, 's')
	# end changes on 26/04/2022

	# select some boundary points ??
	#extra_points = sorted(rnd.sample(range(0, len(c3)), min(len(c3), rate)))
	select1_c3 = getHullBoundary(true_boxes, hull)
	# rate += 10
	jk = 0

	#npg_nn, _ = getNextBoxesFromNN(sbox, pre_envelope, current_sbox, kdt_true, kdt_false, true_boxes, false_boxes, du, gen, MIN_EPS, MIN_EPS_3, RES, connected, extra = extra)	
	new_boxes_to_check_after_regres = []
	selected = []
	npg_cc = []
	# = connectedComponents
	cc1, df  = connected
	jk += 1
	cc = sorted(cc1)
	pi = int((gen/2)%len(cc)) #0 #cc.index(pn)
	pn = cc[pi]
	#for pn in cc:
	# 				
	# pn = random.choice(cc)
	# pi = cc.index(pn)
	out_param = param_names_sorted.index(pn) #rnd.sample(range(param_len), 1)[0]
	print('gpr', jk, 'cc', cc, out_param, pi, pn)
	params = cc
	param_len = len(params)

	# moved on 06/02/2023
	if df == 1:
		select_c3 = []
		ss = [(boxtopoint(select1_c3[i]), i) for i in range(len(select1_c3))]
		for ii in range(all_param_len):
			sss = sorted(ss, key = lambda k: k[0][ii])
			# select_c3.append(select1_c3[sss[0][1]])
			# select_c3.append(select1_c3[sss[-1][1]])
			if select1_c3[sss[0][1]] not in select_c3:
				select_c3.append(select1_c3[sss[0][1]])
			
			if select1_c3[sss[-1][1]] not in select_c3:
				select_c3.append(select1_c3[sss[-1][1]])
	else:
		select_c3 = select1_c3
		print('DF {0} -- no GPR'.format(df))
		#return [], []
	
	c3 = select_c3 #+ c33

	extra_boxes = []
	rate = 0
	# extra_points = []
	# extra_boxes = []
	for i in range(len(c3)):
		tb = c3[i]
		extra_boxes.append(tb)
		# extra_points.append(i)
		rate += 1 

	predict_boxes = getNN(current_sbox, extra_boxes, params, pn, du, RES)
	extra_boxes += predict_boxes

	# extra_points += sorted(rnd.sample(range(0, len(c3)), min(len(c3), 10))) 
	if df == 1:
		other_out_param = []
		for i in range(param_len):
			key = params[i]
			# print(i, param_index_dict[key],out_param)
			if (param_index_dict[key] == out_param):
				continue
			else:
				other_out_param.append(i)

		# extra_points += sorted(rnd.sample(range(0, len(c3)), min(len(c3), 10))) 
		subsets_2 = [(params[pi], params[i]) for i in other_out_param] #range(1, param_len)]
		for subs in subsets_2:
			#print('subs', subs, 'out_param', out_param)
			def getGPRTrainingPoint(sat_boxes, out_par, in_par):
				xlist = [] #[dt]
				ylist = []
				others1 = []
				for b in sat_boxes:
					smap = b.get_map()
					i = 0
					blist = []
					for key in params:
						if (param_index_dict[key] == out_par):
							ylist.append((smap[key].leftBound()+smap[key].rightBound())/2)
						elif (param_index_dict[key] == in_par):
							xlist.append([(smap[key].leftBound()+smap[key].rightBound())/2])
						else:
							blist.append((smap[key].leftBound()+smap[key].rightBound())/2)

						i += 1
					others1.append(blist)
				return xlist, ylist, others1

			GP_x, GP_y, others1 = getGPRTrainingPoint(c33, param_index_dict[subs[0]], param_index_dict[subs[1]])

			GP_xtest = []

			ulist = [-du, du] #np.linspace(-du, du, samples_per_dim)
			# print('----', du, ulist)
			others = {}
			sys.stdout.flush()
			for b in extra_boxes:
				# b = c3[ci]
				#for b in select_c3:
				blist = []
				smap = b.get_map()
				i = 0
				for key in params:
					if (param_index_dict[key] == param_index_dict[subs[1]]):
						blist.append((smap[key].leftBound()+smap[key].rightBound())/2)
					elif not (param_index_dict[key] == out_param):
						others.update({key:(smap[key].leftBound()+smap[key].rightBound())/2})
					i += 1

				for element in ulist:
					blist1 = [(blist[i] + element) for i in range(len(blist))]

				# changed on 22/04/2022
				#blist1 = [blist[i] *(1 + element[i]) for i in range(len(params)-1)]
				GP_xtest.append(blist1)
			#print('----', others)
			x_edges = []
			#print('X', GP_x, np.array(GP_x).shape, 'y', GP_y, np.array(GP_y).shape, \
		    #      'xtest', GP_xtest, np.array(GP_xtest).shape)
			output_extended, out_mean, mse = gp_grad.regres(np.array(GP_x), np.array(GP_y), np.array(GP_xtest))

			for i in range(len(GP_xtest)):
				xlist = GP_xtest[i]
				y_pred = output_extended[i] #+ mse[i]
				if isValid(y_pred):
					row = []
					# row = [ParValue(id = params[k], val = 0) for k in range(param_len)]
					row.append(ParValue(id = params[pi], val = y_pred, std= 1.96*mse[i]))
					k = 0
					for key in params:
						if (param_index_dict[key] == param_index_dict[subs[1]]):
							#print('-- ', key, param_index_dict[key],  'out_par', out_param, 'pi', pi, 'out', params[pi], params[k])
							row.append(ParValue(id = key, val = xlist[k], std = RES))
							k += 1
						elif not (param_index_dict[key] == out_param):
							#print('-- ', key, param_index_dict[key],  'out_par', out_param, 'pi', pi, 'out', params[pi], params[k])
							row.append(ParValue(id = key, val = others[key], std = RES))
							#k += 1

					#print('----', [params[k] for k in range(param_len)], xlist, 'out', params[pi], y_pred, row)
					x_edges.append(row)

			print(row)
		if DEBUG:
			print('x_edges', len(x_edges), x_edges[0])

		npg_cc.append(x_edges)

	else:
		def getGPRTrainingPoint(sat_boxes, out_par):
			xlist = [] #[dt]
			ylist = []
			for b in sat_boxes:
				smap = b.get_map()
				i = 0
				blist = []
				for key in params:
					if (param_index_dict[key] == out_par):
						ylist.append((smap[key].leftBound()+smap[key].rightBound())/2)
					else:
						blist.append((smap[key].leftBound()+smap[key].rightBound())/2)
					i += 1
				xlist.append(blist)
			return xlist, ylist

		GP_x, GP_y = getGPRTrainingPoint(c33, out_param)
		# print('out_param', out_param)

		GP_xtest = []

		ulist = [np.linspace(-du, du, samples_per_dim) for j in range(len(params)-1)]
		# print('----', du, ulist)
		sys.stdout.flush()
		for b in extra_boxes: #predict_boxes:
			# b = c3[ci]
			#for b in select_c3:
			blist = []
			smap = b.get_map()
			i = 0
			for key in params:
				if not (param_index_dict[key] == out_param):
					blist.append((smap[key].leftBound()+smap[key].rightBound())/2)
				i += 1

			for element in itertools.product(*ulist):
				blist1 = [blist[i] + element[i] for i in range(len(params)-1)]

				# changed on 22/04/2022
				#blist1 = [blist[i] *(1 + element[i]) for i in range(len(params)-1)]
			GP_xtest.append(blist1)

		x_edges = []
		# print('X', np.array(GP_x).shape, 'y', np.array(GP_y).shape, 'xtest', np.array(GP_xtest).shape)
		output_extended, out_mean, mse = gp_grad.regres(np.array(GP_x), np.array(GP_y), np.array(GP_xtest))

		for i in range(len(GP_xtest)):
			xlist = GP_xtest[i]
			y_pred = output_extended[i] #+ mse[i]
			if isValid(y_pred):
				row = []
				# row = [ParValue(id = params[k], val = 0) for k in range(param_len)]
				row.append(ParValue(id = params[pi], val = y_pred, std= 1.96*mse[i]))
				k = 0
				for key in params:
					if not (param_index_dict[key] == out_param):
						# print('-- ', key, param_index_dict[key],  'out_par', out_param, 'pi', pi, 'out', params[pi], params[k])
						row.append(ParValue(id = key, val = xlist[k], std = RES))
						k += 1

				# print('----', [params[k] for k in range(param_len)], xlist, 'out', params[pi], y_pred, row)
				x_edges.append(row)

			# print(row)
		if DEBUG:
			print('x_edges', len(x_edges), x_edges[0])

		npg_cc.append(x_edges)

	# print(len(npg_cc), len(npg_cc[0]))
	#if len(connectedComponents)> 1:
	npg_pi = []
	for elem in itertools.product(*npg_cc):
		elem1 = []
		for item in elem:
			elem1 += item
		npg_pi.append(elem1)
	# else:
	# 	npg_pi = x_edges

	for x11 in npg_pi:
		# print()
		x1 = [0 for k in range(all_param_len)]
		x1_std = [0 for k in range(all_param_len)]
		for pk in x11:
			# print('pk', pk)
			pid, pval, mse = param_index_dict[pk.id], pk.val, pk.std
			x1[pid] = pval
			x1_std[pid] = mse
		# print('x11', x11, x1)
		sys.stdout.flush()
		flag = True
		x_map = {}
		for i in range(len(param_names_sorted)):
			pn = param_names_sorted[i]
			if x1[i] < 0 or not (big_s_map[pn].leftBound() < x1[i] < big_s_map[pn].rightBound()):
				flag = False
				break

			# changed on 8/02/2023
			if (i == out_param):
				y = x1[i] - max(min(x1_std[i], RES), RES/2),  x1[i] + max(min(x1_std[i], RES), RES/2)
				#y = x1[i] - (min(x1_std[i], 2*RES)),  x1[i] + (min(x1_std[i], 2*RES))
				ll = max(math.floor(y[0]/GS)*GS,  big_s_map[pn].leftBound())
				ul = min(math.ceil(y[1]/GS)*GS,  big_s_map[pn].rightBound())
				#print(pn, ll, ul, y)
			else:
				# changed on 27/01/2023
				ll = max(math.floor(x1[i]/GS)*GS,  big_s_map[pn].leftBound())
				ul = min(math.ceil(x1[i]/GS)*GS,  big_s_map[pn].rightBound())
			#print(pn, out_param, ll, ul)
			if ll > ul:
				# ll = s_map[pn].rightBound()*11/12.0
				# ul = s_map[pn].rightBound()
				flag = False
				break
			# print(i, pn, ll, ul)
			x_map.update({pn:PyInterval(ll, ul)})
			#x_map.update({pn:PyInterval(x1[i])})
		# print(x1, x_map)
		if flag:
			b2 = Box(x_map)
			# fb2 = 0
			# for bb in npg_nn:
			# 	if b2.overlaps(bb):
			# 		fb2 = 1
			# 		break
			# if fb2 == 1:
			selected.append(b2)
	
	y_prob = getProbNextBoxesFromMLP(sbox, true_boxes, false_boxes, selected, class_flag)
	jk = -1
	for b2 in selected:
		# print('Selected', str(b2))
		jk += 1
		# added on 25/05/2022
		rt = False
		rf = False
		if IVT:
			bg = kdi.KDInterval(getGRID(b2, sbox, RES))
			rt = kdt_true.search(bg) #.addDelta(0)))
			rf = kdt_false.searchContains(bg)

			if DEBUG:
				print('IVT check in GPR', str(b2), str(getGRID(b2, sbox, RES)), 'rt', rt, 'rf', rf )

		if current_sbox.contains(b2) and not pre_envelope.contains(b2):		
			if not (rt or rf):
				if b2 not in new_boxes_to_check_after_regres:
					new_boxes_to_check_after_regres.append(b2)
					if DEBUG:
						print('Added GPR box {0} with prob {1} outside envelope {2}'.format(str(b2), y_prob[jk][0], str(pre_envelope)))

		elif current_sbox.contains(b2) and pre_envelope.contains(b2) and y_prob[jk][0] > prob_thres:
			
			# added on 21/05/2022
			# For GPR, only search inside the envelope
			if not isInHull(b2, hull) and envelope.contains(b2):
			#if not isInHull(b2, hull) or not envelope.contains(b2):
				# rt = kdt_true.search(kdi.KDInterval(b2.addDelta(MIN_EPS)))

				# not closer to a true region
				if not (rt or rf):
					# not closer to a false region
					#if len(true_boxes) < 100: #5**all_param_len:
					if b2 not in new_boxes_to_check_after_regres:
						new_boxes_to_check_after_regres.append(b2)
					if DEBUG:
						print('Added GPR box {0} with prob {1} in envelope {2}'.format(str(b2), y_prob[jk][0], str(envelope)))
				
					# not closer to a true region may be closer to a false region
					# elif rf and (np.random.uniform(0, 1, 1)[0] < 0.1):
					# 	new_boxes_to_check_after_regres.append(b2)	
			# else:
			# 	if not rt and (np.random.uniform(0, 1, 1)[0]  < 0.1): # closer to a false region but not closer true region
			# 		new_boxes_to_check_after_regres.append(b2)
			#if len(true_boxes) < 10: #5**all_param_len:
			#	new_boxes_to_check_after_regres.append(b2)
				
		elif y_prob[jk][0] > prob_thres:
			rt = False
			rf = False
			if IVT:
				bg = kdi.KDInterval(getGRID(b2, sbox, RES))
				rt = kdt_true.search(bg) #.addDelta(0)))
				rf = kdt_false.search(bg)
				#rt = kdt_true.search(kdi.KDInterval(b2)) #.addDelta(0)))
				#rf = kdt_false.searchContains(kdi.KDInterval(b2))
				# rf = kdt_false.search(kdi.KDInterval(b2))
				# rt = kdt_true.search(kdi.KDInterval(b2)) #.addDelta(0)))
			# if not rt:
			# npg.append(bb)
			if not (rt or rf) :
				if b2 not in new_boxes_to_check_after_regres:
					new_boxes_to_check_after_regres.append(b2)
				if DEBUG:
					print('Added GPR box {0} with prob {1}'.format(str(b2), y_prob[jk][0]))

		# changed on 11/05/2022 	
		#c3 += c33
	print('gen', gen, 'gpr', len(new_boxes_to_check_after_regres), len(extra_boxes), 'time taken -- ', time.time() - st2, 's') 

	return new_boxes_to_check_after_regres, c33+select_c3


def getNextBoxesFromGradient(sbox, pre_envelope, current_sbox, kdt_true, kdt_false, true_boxes, false_boxes, du, gen, MIN_EPS, MIN_EPS_3,  RES, connected, extra = False, class_flag = 0):
	GS = RES
	stm1 = time.time()
	
	#npg_nn, _ = getNextBoxesFromNN(sbox, pre_envelope, current_sbox, kdt_true, kdt_false, true_boxes, false_boxes, du, gen, MIN_EPS, MIN_EPS_3, RES, connected, extra = extra)
	envelope = bfact.get_cover(true_boxes, check=False)
	# true_covered =  bfact.get_cover(true_boxes, check=False)
	# extended_box1 = extendEnvelope(envelope, current_sbox, du) #10*MIN_EPS) #EPS)
	# extra_boxes = bfact.remove(extended_box1, envelope)

	#big_s_map = sbox.get_map()

	big_s_map = current_sbox.get_map()

	ParValue = collections.namedtuple('ParValue', ['id', 'val'])
	s_map = current_sbox.get_map()
	# en_edges = envelope.get_map()
	all_params = s_map.keys()
	all_param_len = len(all_params)
	param_names_sorted = sorted(all_params)
	param_index_dict = {}
	i = 0
	for par in param_names_sorted:
		param_index_dict.update({par:i})
		i +=1

	def pointtobox(p):
		bmap = {}
		ij = 0
		for it in sorted(param_names_sorted):
			bmap.update({it:PyInterval(p[ij]*(1-MIN_EPS/2), p[ij]*(1+MIN_EPS/2))})
			ij += 1
		# print(p, bmap)
		return Box(bmap)


	hull = getHull(true_boxes)
	select1_c3 = getHullBoundary(true_boxes, hull)

	# c3 += select_c3
	
	#c23 = [b for b in rnd.sample(true_boxes, min(len(true_boxes), 10**all_param_len))]
	c31 = [b for b in true_boxes] #[b for b in rnd.sample(true_boxes, min(len(true_boxes), 10**all_param_len))]
	# c3 = select_c3
	c32 = c31 #+ select_c3

	# c3 = select_c3 #c32
	
	if len(true_boxes) > 20*(all_param_len-1):
		if CLUSTER:
			# changed on 26/04/2022
			# MAXC = min(all_param_len*MAX_clusters, 5**all_param_len, len(c32)) ##MAX(MAX_clusters, len(c32))
			#MAXC = len(select_c3) ##MAX(MAX_clusters, len(c32))

			# changed on 20/01/2023 
			MAXC = min( 5**(all_param_len-1), len(c32)) 
			#MAXC = min(max(int(envelope.volume()/((MIN_EPS)**all_param_len)), 5**all_param_len), len(c32)) ##MAX(MAX_clusters, len(c32))
			
			X_1 = np.array([boxtopoint(b) for b in c32])
			chosen_k = MAXC
			'''
			pre_savg = -1
			pre_score = -1
			for n_clusters in range(2*all_param_len, MAXC):
			# Initialize the clusterer with n_clusters value and a random generator
			# seed of 10 for reproducibility.
				#clusterer = KMeans(n_clusters=n_clusters, random_state=50)
				clusterer = KMedoids(n_clusters=n_clusters, random_state=gen) #, init='heuristic'
				cluster_labels = clusterer.fit_predict(X_1)

				# The silhouette_score gives the average value for all the samples.
				# This gives a perspective into the density and separation of the formed
				# clusters
				# if int(gen/2)%3 == 0 and len(cluster_labels) > 1 and len(X_1) > 1:
				# 	# silhouette_avg = -1*davies_bouldin_score(X_1, cluster_labels)
				# 	silhouette_avg = silhouette_score(X_1, cluster_labels)
				# 	#print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
				# 	if pre_score < silhouette_avg:
				# 		pre_score = silhouette_avg
				# 		chosen_k = n_clusters
				# else:		
				silhouette_avg = (pre_savg - clusterer.inertia_) # elbow method
				if silhouette_avg < EPS:
					chosen_k = n_clusters
					break				
				pre_savg =  clusterer.inertia_

			'''
				

			clusterer =  KMedoids(n_clusters=chosen_k,  random_state=gen).fit(X_1) #init='heuristic',
			centroids = clusterer.cluster_centers_
			#centroids_1 = centroids
			# clusterer = KMeans(n_clusters=chosen_k, random_state=gen).fit(X_1)
			# centroids = clusterer.cluster_centers_
			# center_indices, centroids_1 = find_k_closest(centroids, X_1)
			center_indices, centroids_1 = find_k_closest(centroids, X_1)
			c33 = [] 
			# c33 = select_c3
			# X = list(X_1)
			for cc in center_indices: #centroids_1:
				b = c32[cc]
				c33.append(b) #pointtobox(cc))
			# c33 = select_c3
			# # X = list(X_1)
			# for cc in centroids_1:
			# 	# b = c32[X.index()]
			# 	c33.append(pointtobox(cc))
			# changed on 11/05/2022 	
			#c3 += c33

		else:	
			# added on 15/01/2023
			# MAXC = min(all_param_len*MAX_clusters, 5**all_param_len, len(c32)) ##MAX(MAX_clusters, len(c32))

			# changed on 20/01/2023 
			MAXC = min(max(int(envelope.volume()/((MIN_EPS)**all_param_len)), 5**all_param_len), len(c32)) ##MAX(MAX_clusters, len(c32))
			
			X_1 = np.array([boxtopoint(b) for b in c32])
			per_dim = int(MAXC/all_param_len)
			s_indices = np.linspace(0, len(X_1)-1, per_dim)
			s_points = []
			for i in range(all_param_len):
				X_1_sorted_i = sorted(X_1, key = lambda x: x[i])
				#print('X_1', len(X_1), len(X_1[0]), len(X_1_sorted_i), s_indices)
				for j in s_indices:
					s_points.append(X_1_sorted_i[int(j)])

			center_indices, centroids_1 = find_k_closest(s_points, X_1)
			
			c33 = []
			for cc in center_indices: #centroids_1:
				b = c32[cc]
				c33.append(b) #pointtobox(cc))
	else:
		c33 = c31
	if len(c33) < all_param_len:
		c33 = c31
	
	# c33 = c33 + select_c3

	st2 = time.time()
	# c3 = c33
	print('npg -- ', time.time()-stm1, 's')
	# end changes on 26/04/2022

	#envelope = bfact.get_cover(c3)
	#rate = 50
	# select some boundary points ??
	#extra_points = sorted(rnd.sample(range(0, len(c3)), min(len(c3), rate)))
	#true_points = [boxtopoint(tb) for tb in true_boxes]
	

	jk = 0
	new_boxes_to_check_after_regres = []
	selected = []
	npg_cc = []
	#for connected in connectedComponents:
	cc1, df = connected
	jk += 1
	cc = sorted(cc1)
	pi = int((gen/2)%len(cc)) #0 #cc.index(pn)
	pn = cc[pi]
	#for pn in cc:
	# 				
	# pn = random.choice(cc)
	# pi = cc.index(pn)
	# # print(cc1, pn)
	#pi = cc.index(pn)
	out_param = param_names_sorted.index(pn) #rnd.sample(range(param_len), 1)[0]
	#print('npg -- ',jk, 'cc', cc, out_param, 'index', pi, pn, len(c33))
	params = cc
	param_len = len(params)

	
	# moved on 06/02/2023
	if df ==1:
		select_c3 = []
		ss = [(boxtopoint(select1_c3[i]), i) for i in range(len(select1_c3))]
		for ii in range(all_param_len):
			sss = sorted(ss, key = lambda k: k[0][ii])
			# select_c3.append(select1_c3[sss[0][1]])
			# select_c3.append(select1_c3[sss[-1][1]])
			if select1_c3[sss[0][1]] not in select_c3:
				select_c3.append(select1_c3[sss[0][1]])
			
			if select1_c3[sss[-1][1]] not in select_c3:
				select_c3.append(select1_c3[sss[-1][1]])
	else:
		select_c3 = select1_c3

	c3 = select_c3 # + c33

	rate = 0
	extra_points = []
	extra_boxes = []
	for i in range(len(c3)):
		tb = c3[i]
		extra_boxes.append(tb)
		extra_points.append(i)
		rate += 1 

	other_out_param = []
	for i in range(param_len):
		key = params[i]
		# print(i, param_index_dict[key],out_param)
		if (param_index_dict[key] == out_param):
			continue
		else:
			other_out_param.append(i)
	subsets_2 = [(params[pi], params[i]) for i in other_out_param] #range(1, param_len)]
	#print('subsets_2', subsets_2)

	X_ = {}
	Y_ = {}
	j = 0
	for b in c33:
		b_map = b.get_map()
		for it in params:
			i = param_index_dict[it]
			if it == params[out_param]:
				x = (b_map[it].leftBound() + b_map[it].rightBound())/2
				X_.update({j:x})
			else:
				y = (b_map[it].leftBound() + b_map[it].rightBound())/2
				Y_.update({(j,i):y})
		j+= 1
	# print('X_', X_)
	# print('Y_', Y_)
	X_all = {}
	Y_all = {}
	j = 0
	for b in c3:
		b_map = b.get_map()
		for it in params:
			i = param_index_dict[it]
			if it == params[out_param]:
				x = (b_map[it].leftBound() + b_map[it].rightBound())/2
				X_all.update({j:x})
			else:
				y = (b_map[it].leftBound() + b_map[it].rightBound())/2
				Y_all.update({(j,i):y})
		j+= 1

	xv = list(X_.values())
	xvn, ijk = np.unique(xv, return_index = True)
	k_degree = 3 #min(3, len(xvn))
	# print('sub', k_degree, len(xvn))
	if len(xvn) < all_param_len:
		X_ = {}
		Y_ = {}
		j = 0
		for b in c33 +select_c3:
			b_map = b.get_map()
			for it in params:
				i = param_index_dict[it]
				if it == params[out_param]:
					x = (b_map[it].leftBound() + b_map[it].rightBound())/2
					X_.update({j:x})
				else:
					y = (b_map[it].leftBound() + b_map[it].rightBound())/2
					Y_.update({(j,i):y})
			j+= 1
		
		xv = list(X_.values())
		xvn, ijk = np.unique(xv, return_index = True)
		k_degree = 3 #min(3, len(xvn))

	xv_all = list(X_all.values())
	keys = []
	YV_grad = {}
	YV_ = {}
	for sub in subsets_2:
		s1, s2 = sub[0], sub[1]
		j1, j2 = param_index_dict[s1], param_index_dict[s2]
		yv = [Y_[(j,j2)] for j in X_.keys()]
		# print('np.gradient -- ', s1, len(xv), s2, len(yv), xv, yv) 
		
		# print('np.gradient -- ', s1, len(xv), s2, len(yv), xvn, len(ijk), ijk) 
		yvn = [yv[ijj] for ijj in ijk]
		# print('np.gradient -- ', s1, len(xv), s2, len(yv), xvn, yvn)
		if len(xvn) >= 4:
			tck = interpolate.splrep(xvn, yvn, s = 0, k = k_degree)
		else:
			tck = -1
		# grad = interpolate.splev(xv)
		grad = np.gradient(xv, yv) #, axis = 0)
		# print('------------', len(grad), grad)
		YV_grad.update({j2:(grad, yv, tck)})
		YV_.update({j2:[Y_all[(j,j2)] for j in X_all.keys()]})
		keys.append(j2)

	grad_interpolate = {}
	for j in keys:
		grad_all = YV_grad[j][0]
		x_all = xv
		xvn, ijk = np.unique(x_all, return_index = True)
		# print('np.gradient -- ', s1, len(xv), s2, len(yv), xvn, ijk) 
		yvn = [grad_all[ijj] for ijj in ijk]
		f = interpolate.interp1d(xvn, yvn,  fill_value="extrapolate")
		# f = interpolate.interp1d(xvn, yvn, kind = 'cubic', fill_value="extrapolate")
		grad_interpolate.update({j:f})

	# print(YV_grad)
	def del_z(u, i):
		#grad = [YV_grad[j][0][i] for j in YV_grad.keys()]
		unorm = np.linalg.norm(list(u))
		#grad = [YV_grad[j][0][i] for j in YV_grad.keys()]
		if du > 5*MIN_EPS_3 and len(xvn) >= 4:
			# changed on 10/05/2022
			grad = [interpolate.splev(xv_all[i], YV_grad[j][2], der=1) for j in keys]
		else:
			# # changed on 26/04/2022
			grad = [grad_interpolate[j](xv_all[i]) for j in keys]
		dz = np.dot(grad, u) #/unorm # divided by norm of u??
		# if DEBUG:
		# dxz = xv[i]+dz 
		dxz = xv_all[i]+dz #[0]
		# print('del_z', 'grad', grad, 'u', u, 'dz', dz, 'index', i, 'x', xv_all[i], dxz)
		return dxz, dz

	ulist = [np.linspace(-du, du, samples_per_dim) for j in YV_grad.keys()] #YV_grad.keys()]
	# print('ulist', ulist)
	# if DEBUG:
	# print('np.gradient', len(YV_grad), YV_grad.keys(), 'c3', len(c3), 'extra_points', rate, len(extra_points), extra_boxes)
	x_edges = []
	for element in itertools.product(*ulist):
		if all([x == 0 for x in element]):
			continue
		# print('eelement', element) #, YV_grad.keys(), [YV_grad[j][1] for j in YV_grad.keys()])
		inputs_extended = {}
		k = 0
		for j in keys: #YV_grad.keys():
			u = element[k]
			# yv = YV_grad[j][1]
			# # changed on 26/04/2022
			yv = YV_[j]
			yv1 = []
			for i in extra_points:
				yv1.append(yv[i] + u)
				# changed on 22/04/2022
				# yv1.append(yv[i] *(1 + u))
				
				#yv1.append(extra_boxes[i] *(1 + u))
			inputs_extended.update({j:yv1})
			k += 1
		#[[(YV_grad[j][1][i] + element[j]) for i in extra_points] for j in YV_grad.keys()]
		output_extended_dz = [del_z(element, i)[1] for i in extra_points]
		output_extended = [del_z(element, i)[0] for i in extra_points]
		#print('elem' ,element, output_extended[0], params[keys[0]], inputs_extended[keys[0]][0])
		for i in range(len(extra_points)):
			row = []
			# changed on 11/05/2022 -- to limit the distant points in approximation
			if isValid(output_extended[i]): # and  abs(output_extended_dz[i]) < 50*du: #not math.isnan(output_extended[i]):
				# row.append(output_extended[i])
				row.append(ParValue(id = params[pi], val = output_extended[i]))
				for j in keys: #range(len(keys)): #YV_grad.keys())):
					# row.append(inputs_extended[j][i])
					if j == pi:
						continue
					row.append(ParValue(id = params[j], val = inputs_extended[j][i]))
				x_edges.append(row)

				#print('--', row, out_param, pi)

	if DEBUG:
		print('x_edges', x_edges)
	npg_cc.append(x_edges)

	# print(len(npg_cc), len(npg_cc[0]))
	#if len(connectedComponents)> 1:
	npg_pi = []
	for elem in itertools.product(*npg_cc):
		# print(elem)
		elem1 = []
		for item in elem:
			elem1 += item
		npg_pi.append(elem1)
		# print(elem1)
	# else:
	# 	npg_pi = x_edges

	for x11 in npg_pi:
		# print(x11)
		x1 = [0 for k in range(all_param_len)]
		for pk in x11:
			# print('pk', pk)
			pid, pval = param_index_dict[pk.id], pk.val
			x1[pid] = pval
		# print('x11', x11, x1)
		sys.stdout.flush()
		flag = True
		x_map = {}
		for i in range(len(param_names_sorted)):
			pn = param_names_sorted[i]
			if x1[i] < 0 or not (big_s_map[pn].leftBound() <= x1[i] <= big_s_map[pn].rightBound()):
				#print(pn, x1[i])
				flag = False
				break

			# changed on 27/01/2023
			ll = max(math.floor(x1[i]/GS)*GS,  big_s_map[pn].leftBound())
			ul = min(math.ceil(x1[i]/GS)*GS,  big_s_map[pn].rightBound())

			# ll = max(x1[i] - 2*du, s_map[pn].leftBound())
			# ul = min(x1[i] + 2*du, s_map[pn].rightBound())
			if ll > ul:
				# ll = s_map[pn].rightBound()*11/12.0
				# ul = s_map[pn].rightBound()
				flag = False
				break
			# print(i, pn, ll, ul)
			x_map.update({pn:PyInterval(ll, ul)})
			#x_map.update({pn:PyInterval(x1[i])})
		#print('detected', x1, x_map)
		if flag:
			b2 = Box(x_map)
			# print('b2', str(b2))
			# fb2 = 0
			# for bb in npg_nn:
			# 	if b2.overlaps(bb):
			# 		fb2 = 1
			# 		break
			# if fb2 == 1:
			selected.append(b2)

	y_prob = getProbNextBoxesFromMLP(sbox, true_boxes, false_boxes, selected, class_flag)
	jk = -1
	for b2 in selected:
		#print('Selected', str(b2))
		jk += 1
	# for b2 in selected:
		# print('Selected', str(b2))
		# added on 25/05/2022
		rt = False
		rf = False
		if IVT:
			bg = kdi.KDInterval(getGRID(b2, sbox, RES))
			rt = kdt_true.search(bg) #.addDelta(0)))
			rf = kdt_false.searchContains(bg)
			# rt = kdt_true.search(kdi.KDInterval(b2)) #.addDelta(0)))
			# rf = kdt_false.searchContains(kdi.KDInterval(b2))
			if DEBUG:
				print('IVT check in NPG', str(b2), str(getGRID(b2, sbox, RES)), 'rt', rt, 'rf', rf )

		if current_sbox.contains(b2) and not pre_envelope.contains(b2):
			# rt = kdt_true.search(kdi.KDInterval(b2)) #.addDelta(0)))
			# rf = kdt_false.searchContains(kdi.KDInterval(b2))
			if not (rt or rf):
				if b2 not in new_boxes_to_check_after_regres:
					new_boxes_to_check_after_regres.append(b2)
					if DEBUG:
						print('NPG Added box {0} with prob {1} outside box {2}'.format(str(b2), y_prob[jk][0], str(envelope)))
			
		elif current_sbox.contains(b2) and pre_envelope.contains(b2) and y_prob[jk][0] > prob_thres:
			# new_boxes_to_check_after_regres.append(b2)
			# rf = kdt_false.searchContains(kdi.KDInterval(b2))
			# rt = kdt_true.search(kdi.KDInterval(b2)) #.addDelta(0)))

			# updated on 20/01/2023
			# if not rt  and np.random.uniform(0, 1, 1)[0] < 0.2 : #or len(true_boxes) < 20 :
			# 	new_boxes_to_check_after_regres.append(b2)


			# updated on 21/05/2022
			# For NPG, only search outside the envelope

			if not isInHull(b2, hull) or not envelope.contains(b2) :# and extended_box.contains(b2) : #or len(selected) < 20:
				if not (rt or rf):
					if b2 not in new_boxes_to_check_after_regres:
						new_boxes_to_check_after_regres.append(b2)#, delta_to_use))
						if DEBUG:
							print('NPG Added box {0} with prob {1} inside box {2}'.format(str(b2), y_prob[jk][0], str(envelope)))
				# elif rf and not rt and (np.random.uniform(0, 1, 1)[0] < 0.1):
				# 	new_boxes_to_check_after_regres.append(b2)#, delta_to_use))

			#elif not isInHull(b2, hull) and envelope.contains(b2):
			#	#rt = kdt_true.search(kdi.KDInterval(b2.addDelta(MIN_EPS*5)))
			#	if not (rt or rf) or np.random.uniform(0, 1, 1)[0] < 0.1 or len(true_boxes) < 50 :
			#		new_boxes_to_check_after_regres.append(b2)
		elif y_prob[jk][0] > prob_thres:
			rt = False
			rf = False
			if IVT:
				# rt = kdt_true.search(kdi.KDInterval(b2)) #.addDelta(0)))
				# rf = kdt_false.searchContains(kdi.KDInterval(b2))
				# rf = kdt_false.search(kdi.KDInterval(b2))
				# rt = kdt_true.search(kdi.KDInterval(b2)) #.addDelta(0)))

				bg = kdi.KDInterval(getGRID(b2, sbox, RES))
				rt = kdt_true.search(bg) #.addDelta(0)))
				rf = kdt_false.search(bg)
			# if not rt:
			# npg.append(bb)
			if not (rt or rf) :
				if b2 not in new_boxes_to_check_after_regres:
					new_boxes_to_check_after_regres.append(b2)
					if DEBUG:
						print('NPG Added box {0} with prob {1}'.format(str(b2), y_prob[jk][0]))

	print('gen', gen, 'npg', len(new_boxes_to_check_after_regres), len(extra_points), 'time taken -- ', time.time() - st2, 's') 		
	return new_boxes_to_check_after_regres, c33+select_c3


def getNextBoxesFromNN(sbox, pre_envelope, current_sbox, kdt_true, kdt_false, true_boxes, false_boxes, du, gen, MIN_EPS, MIN_EPS_3, RES, connected, extra = False):
	
	stm1 = time.time()

	ParValue = collections.namedtuple('ParValue', ['id', 'val', 'std'])
	big_s_map = sbox.get_map()
	s_map = current_sbox.get_map()
	# en_edges = envelope.get_map()

	big_s_map = current_sbox.get_map()

	all_params = s_map.keys()
	all_param_len = len(all_params)
	param_names_sorted = sorted(all_params)
	param_index_dict = {}
	i = 0
	for par in param_names_sorted:
		param_index_dict.update({par:i})
		i +=1

	
	def pointtobox(p): #, param_names = param_names_sorted):
		bmap = {}
		ij = 0
		for it in sorted(param_names_sorted):
			# changed on 11/05/2022
			bmap.update({it:PyInterval(p[ij]*(1-MIN_DELTA/2), p[ij]*(1+MIN_DELTA/2))})
			ij += 1
		# print(p, bmap)
		return Box(bmap)

	envelope = bfact.get_cover(true_boxes, check=False)
	# true_covered =  bfact.get_cover(true_boxes, check=False)
	#extended_box1 = extendEnvelope(envelope, sbox, du) #10*MIN_EPS) #EPS)
	#extra_boxes = bfact.remove(extended_box1, envelope)

	c31 = [b for b in true_boxes] #rnd.sample(true_boxes, min(len(true_boxes), 7**all_param_len))]

	# stm1 = time.time()

	hull = getHull(true_boxes)

	c32 = c31 #+ select_c3

	# select some boundary points ??
	#extra_points = sorted(rnd.sample(range(0, len(c3)), min(len(c3), rate)))
	select1_c3 = getHullBoundary(true_boxes, hull)
	# rate += 10
	jk = 0

	new_boxes_to_check_after_regres = []
	selected = []
	npg_cc = []
	# = connectedComponents
	#for connected in connectedComponents:
	cc1, df  = connected
	jk += 1
	cc = sorted(cc1)
	pi = int((gen/2)%len(cc)) #0 #cc.index(pn)
	pn = cc[pi]
	#for pn in cc:
	# 				
	# pn = random.choice(cc)
	# pi = cc.index(pn)
	out_param = param_names_sorted.index(pn) #rnd.sample(range(param_len), 1)[0]
	#print('gpr', jk, 'cc', cc, out_param, pi, pn)
	params = cc
	param_len = len(params)

	select_c3 = []
	# moved on 06/02/2023
	if df == 1:
		select_c3 = []
		ss = [(boxtopoint(select1_c3[i]), i) for i in range(len(select1_c3))]
		for ii in range(all_param_len):
			sss = sorted(ss, key = lambda k: k[0][ii])
			if select1_c3[sss[0][1]] not in select_c3:
				select_c3.append(select1_c3[sss[0][1]])
			
			if select1_c3[sss[-1][1]] not in select_c3:
				select_c3.append(select1_c3[sss[-1][1]])
	else:
		select_c3 = select1_c3
		print('DF {0} -- no GPR'.format(df))
		#return [], []
	
	c3 = select_c3 #+ c33

	print('NN - select_c3', [str(b) for b in c3])
	rate = 0
	# extra_points = []
	extra_boxes = []
	for i in range(len(c3)):
		tb = c3[i]
		extra_boxes.append(tb)
		# extra_points.append(i)
		rate += 1 

	#print('NN - extra boxes', [str(b) for b in extra_boxes])
	selected = getNN(current_sbox, extra_boxes, params, pn, du, RES)

	#print('NN - selected boxes', [str(b) for b in selected])
	y_prob = getProbNextBoxesFromMLP(sbox, true_boxes, false_boxes, selected)
	jk = -1
	for b2 in selected:
		#print('NN Selected', str(b2))
		jk += 1
		# added on 25/05/2022
		rt = False
		rf = False
		if IVT:
			bg = kdi.KDInterval(getGRID(b2, sbox, RES))
			rt = kdt_true.search(bg) #.addDelta(0)))
			rf = kdt_false.searchContains(bg)
			if DEBUG:
				print('IVT check in NN', str(b2), str(getGRID(b2, sbox, RES)), 'rt', rt, 'rf', rf )
		#		
		# if len(true_boxes) < 10:
		# 	new_boxes_to_check_after_regres.append(b2)
		ff = current_sbox.contains(b2)
		#print('Contains --', ff)
		if len(true_boxes) < 10*(len(params)-1):
			if ff : #and (not pre_envelope.contains(b2) or not isInHull(b2, hull))):
				if b2 not in new_boxes_to_check_after_regres:
					new_boxes_to_check_after_regres.append(b2)
					if DEBUG:
						print('Added box {0} from NN in envelope {1} with prob {2}'.format(str(b2), str(current_sbox), y_prob[jk][0]), 'contains', ff)
		elif not (rt or rf) and y_prob[jk][0] > prob_thres:
			if (ff and (not pre_envelope.contains(b2) or not isInHull(b2, hull))):
				if b2 not in new_boxes_to_check_after_regres:
					new_boxes_to_check_after_regres.append(b2)
					if DEBUG:
						print('Added box {0} from NN not in envelope {1} with prob {2}'.format(str(b2), str(pre_envelope), y_prob[jk][0]))
		
		#print('--- for b2 ---', str(b2))
			# if len(true_boxes) > 10 and not isInHull(b2, hull) and envelope.contains(b2):
			# 	new_boxes_to_check_after_regres.append(b2)
			# elif y_prob[jk][0] > prob_thres:
			# 	new_boxes_to_check_after_regres.append(b2)
		# elif current_sbox.contains(b2) and y_prob[jk][0] > prob_thres:
			
		# 	# added on 21/05/2022
		# 	# For GPR, only search inside the envelope
			#if current_sbox.contains(b2) and not isInHull(b2, hull) and envelope.contains(b2):
			# if y_prob[jk][0] > prob_thres:
			# 	new_boxes_to_check_after_regres.append(b2)	
			# 	print('Added box {0} with prob {1}'.format(str(b2), y_prob[jk][0]))
				# rt = kdt_true.search(kdi.KDInterval(b2.addDelta(MIN_EPS)))

		# 		# not closer to a true region
		# 		if not (rt or rf):
		# 			# not closer to a false region
		# 			if len(true_boxes) < 100: #5**all_param_len:
		# 				new_boxes_to_check_after_regres.append(b2)
				
		# 			# not closer to a true region may be closer to a false region
		# 			# elif rf and (np.random.uniform(0, 1, 1)[0] < 0.1):
		# 			# 	new_boxes_to_check_after_regres.append(b2)	
		# 	# else:
		# 	# 	if not rt and (np.random.uniform(0, 1, 1)[0]  < 0.1): # closer to a false region but not closer true region
		# 	# 		new_boxes_to_check_after_regres.append(b2)
		# 	if len(true_boxes) < 10: #5**all_param_len:
		# 		new_boxes_to_check_after_regres.append(b2)
				

		# elif y_prob[jk][0] > prob_thres:
		# 	rt = False
		# 	rf = False
		# 	if IVT:
		# 		#rt = kdt_true.search(kdi.KDInterval(b2)) #.addDelta(0)))
		# 		#rf = kdt_false.searchContains(kdi.KDInterval(b2))
		# 		rf = kdt_false.search(kdi.KDInterval(b2))
		# 		rt = kdt_true.search(kdi.KDInterval(b2)) #.addDelta(0)))
		# 	# if not rt:
		# 	# npg.append(bb)
		# 	if not (rt or rf) :
		# 		new_boxes_to_check_after_regres.append(b2)		
	# print()
	
	print('gen', gen, 'du', du, 'NN', len(new_boxes_to_check_after_regres), len(extra_boxes), 'time taken -- ', time.time() - stm1, 's') 

	return new_boxes_to_check_after_regres, select_c3

def getProbNextBoxesFromMLP(sbox, true_boxes, false_boxes, pred_boxes, class_flag = 0):

	param_names = sorted(sbox.get_map().keys())

	def getGPRTestingPoint():
		flag = 0
		xtest = [] #[dt]
		for b in pred_boxes:
			pp = boxtopoint(b)
			xtest.append(pp)
			if flag == 0:
				flag = 1
		return xtest, flag

	GP_xtest, flag2 = getGPRTestingPoint()

	if (not NN) or len(true_boxes) == 0 or len(false_boxes) == 0:
		y_prob = [[1] for i in range(len(GP_xtest))]
		print('y_prob -- No MLP', y_prob, np.array(y_prob).shape)
		return y_prob

	'''
	t_boxes = []
	MAXC1 = min(max(2, 2**(len(param_names)-1)), len(true_boxes))
	X_1 = np.array([boxtopoint(b) for b in true_boxes])
	clusterer =  KMedoids(n_clusters=MAXC1,  random_state=1).fit(X_1) #init='heuristic',
	centroids1 = clusterer.cluster_centers_
	#centroids_1 = centroids
	# clusterer = KMeans(n_clusters=chosen_k, random_state=gen).fit(X_1)
	# centroids = clusterer.cluster_centers_
	center_indices1, centroids_1 = find_k_closest(centroids1, X_1)

	for cc in center_indices1: #centroids_1:
		b = true_boxes[cc]
		t_boxes.append(b) #pointtobox(cc))

	f_boxes = []
	MAXC2 = min(max(2, 2**(len(param_names)-1)), len(false_boxes))
	X_2 = np.array([boxtopoint(b) for b in false_boxes])
	clusterer =  KMedoids(n_clusters=MAXC2,  random_state=1).fit(X_2) #init='heuristic',
	centroids2 = clusterer.cluster_centers_
	#centroids_1 = centroids
	# clusterer = KMeans(n_clusters=chosen_k, random_state=gen).fit(X_1)
	# centroids = clusterer.cluster_centers_
	center_indices2, centroids_2 = find_k_closest(centroids2, X_2)

	for cc in center_indices2: #centroids_1:
		b = false_boxes[cc]
		f_boxes.append(b) #pointtobox(cc))
	'''
	t_boxes = true_boxes
	f_boxes = false_boxes


	def getGPRTrainingPoint():
		flag = 0
		xlist = [] #[dt]
		ylist = []
		others1 = []
		#for it in kdt_true.items():
			#pp = boxtopoint(bfact.fromInterval(param_names, it))
		#flag = 1 if (len(t_boxes) > 0.5*len(f_boxes)) else 0
		for b in t_boxes:
			pp = boxtopoint(b)
			ylist.append([1])
			xlist.append(pp)
			if flag == 0:
				flag = 1
		# for it in kdt_false.items():
		# 	pp = boxtopoint(bfact.fromInterval(param_names, it))
		for b in f_boxes:
			pp = boxtopoint(b)
			ylist.append([0])
			xlist.append(pp)
			if flag == 0:
				flag = 1
			if flag == 1:
				flag = 2
		return xlist, ylist, flag

	GP_x, GP_y, flag1 = getGPRTrainingPoint()

	print('getGPRTrainingPoint', flag1, flag2, 'X', np.array(GP_x).shape, 'y', np.array(GP_y).shape, 'xtest', np.array(GP_xtest).shape)
	print('getGPRTrainingPoint', 'X', np.array(GP_x).shape, 'y',np.array(GP_y).shape,'xtest', np.array(GP_xtest).shape)
	y_prob = [[1] for i in range(len(GP_xtest))]
	y_prob1 = [[1] for i in range(len(GP_xtest))]
	y_prob2 = [[1] for i in range(len(GP_xtest))]
	if flag1 == 2 and flag2 == 1:
		y_pred, y_prob = mlp.classify(np.array(GP_x), np.array(GP_y), np.array(GP_xtest))
		if len(y_prob) > 0:
			print('getGPRTrainingPoint', 'y_prob', np.array(y_prob).shape, GP_xtest[0], y_pred[0], y_prob[0])
		y_prob2 = [[y_prob[i][y_pred[i]]] for i in range(len(y_prob))]
		# if len(y_prob) > 0:
		# print('getGPRTrainingPoint', 'y_prob', np.array(y_prob).shape, GP_xtest, y_pred, y_prob)
	return y_prob2

