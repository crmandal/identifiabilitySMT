from math import *
import numpy as np
import csv
import random as rnd
import copy, os, sys

from scipy.optimize import minimize

# from scipy.linalg import lu

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

rnd.seed(12)

from paramUtil.kd_interval import *
from paramUtil.kd_intervaltree import *

# from scipy.linalg import lu, inv

def printMatrix(mat):
	r = len(mat)
	c = len(mat[0])
	s = '['
	for i in range(r):
		s += '['
		for j in range(c):
			s += '{0:0.2e},'.format(mat[i][j]) if j < c-1 else '{0}'.format(mat[i][j])
		s += '],\n' if i < r-1 else ']'
	s += ']'
	return s

def printArray(mat):
	r = len(mat)
	s = '['
	for i in range(r):
		s += '{0}\t'.format(mat[i])
	s+= ']'
	return s


# if __name__ == '__main__':
#     x = gausselim(np.array([[3, 2], [1, -4]]), np.array([[5], [10]]))
#     print x

def findIndex(timespace, sample, atol):
	ind = -1
	if sample in timespace:
		ind = timespace.index(sample)
	else:
		s = 0
		e = len(timespace)
		while s < e:
			m = int((s+e)/2)
			if abs(timespace[m] - sample) < atol or s == e-1:
				ind = m
				break
			else:
				if timespace[m] < sample:
					s = m
				else:
					e = m
	return ind

def getNorm(vec):
	s = 0
	for i in range(len(vec)):
		s += vec[i]**2
	return s

def unit_vector(vector):
	""" Returns the unit vector of the vector.  """
	return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
	""" Returns the angle in radians between vectors 'v1' and 'v2'::

			>>> angle_between((1, 0, 0), (0, 1, 0))
			1.5707963267948966
			>>> angle_between((1, 0, 0), (1, 0, 0))
			0.0
			>>> angle_between((1, 0, 0), (-1, 0, 0))
			3.141592653589793
	"""
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))*(180/pi)

def minifisher(params, param_names, param_range, obs_data, observable_index, obs_time, time, atol, delta = 0.001):
	#params = np.array(params)
	print('####################### minifisher #################')
	# time = time/2
	# atol = 0.1
	matrix, timespace = run(params, time)
	samples = min(int(time/atol), len(timespace), 500)
	
	# delta = 0.1
	lp = len(params)
	# print(params, len(params))
	# flag = False

	uniformSamples = np.linspace(0, time, samples) #sorted(np.random.choice(np.linspace(0, time, samples), min(samples, 20)))
	print(delta, time, samples) #, uniformSamples)

	# matrix, timespace = run(ind, time)
	big_sigma_array = []
	for k in observable_index:
		# k = observable_index[i]
		#print(k)
		tup = matrix[k]
		#print(tup)
		#for i in observable_index:
		for t1 in uniformSamples:
			tm = findIndex(timespace, t1, atol)
			#tup = obs[:, tm]
			#val = matrix_SS[observable_index[i]][tm]
			val = tup[tm]
			# print(val)
			sigma = val*np.random.normal(0, 0.05, 1)[0] #if val > 0 else np.random.normal(0.1*val, 0, 1)[0] #(val*0.1)#/T3_ss
			# print(sigma)
			big_sigma_array.append(sigma**2)

	big_sigma_array1 = np.array(big_sigma_array, dtype=np.float64)
	big_sigma = np.diagflat(big_sigma_array1)
	# params_1 = [p for p in params] #np.array(params)
	# params_2 = [p for p in params] #np.array(params)

	listX = np.zeros((lp, len(uniformSamples)*len(observable_index))) #, dtype = np.float64)

	params_1 = np.array(params)
	params_2 = np.array(params)
	for i in range(len(params)):

		u1 = params[i]

		params_1 = np.array(params)
		params_2 = np.array(params)

		params_1[i] = params[i] * (1 + delta)
		params_2[i] = params[i] * (1 - delta)

		# print(param_names[i], params[i], params_1[i], params_2[i])

		res_1, timespace1 = run(params_1, time)
		res_2, timespace2 = run(params_2, time)
		# print(len(res_1), observable_index)
		l = 0
		for j in range(len(res_1)): # number of outputs
			u = np.mean(obs_data[j])
			#print('scale factors per o/p', j, u, u1)
			# subX = []
			# print(len(res_1[j]))
			for k in uniformSamples: # number of times
				t1 = findIndex(timespace1, k, atol)
				t2 = findIndex(timespace2, k, atol)
				# print(i, j, k, timespace1[t1], timespace2[t2], res_1[j][t1], res_2[j][t2])
				del_y = (res_1[j][t1] - res_2[j][t2])/(2 * delta * params[i])
				scale = 1.0*u1/u
				sigma = 1#*np.sqrt(big_sigma[l][l])
				# print(i, k, l, sigma)
				# subX.append(del_y*scale) #*u1/u)
				#(sir_ode.yfcn(res_1, params_1) - sir_ode.yfcn(res_2, params_2)) / (2 * delta * params[i])
				listX[i][l] = del_y*scale/sigma
				l += 1

	X = listX #np.matrix(listX)
	#X1 = np.array(listX)
	#print(type(listX), type(X), type(X1))
	#print(len(listX), len(listX[0]), X.shape, X1.shape)
	X_t = X.transpose()
	FIM = np.dot(X, X_t)

	T = len(uniformSamples)
	#FIM1 = np.dot(X1, X1.transpose())
	print(param_names)
	# print(printMatrix(X))
	print('1--------')
	det = np.linalg.det(FIM)
	rank = np.linalg.matrix_rank(FIM)
	print(FIM.dtype, 'FIM det', det, 'rank', rank, T)
	# print(printMatrix(FIM))

	#int(FIM1.dtype, 'FIM det', det, 'rank', rank)

	# print(printMatrix(FIM1))

	# print('3--------')
	# A = [[1042.0914848218285,2097.7379109633894,860.512397049636,1393.685769156111,678.4501834075619],
	# 	[2097.7379109633894,4273.679861564189,1761.4333985053001,2784.454113300638,1320.6843707604792],
	# 	[860.512397049636,1761.4333985053001,732.3938360482618,1149.6063458439874,550.3931714415284],
	# 	[1393.685769156111,2784.454113300638,1149.6063458439874,1895.8301084392174,960.2630948426447],
	# 	[678.4501834075619,1320.6843707604792,550.3931714415284,960.2630948426447,592.1848275285174]]
	# A = np.array(A)
	# det = np.linalg.det(A)
	# rank = np.linalg.matrix_rank(A)
	# print(A.dtype, 'FIM det', det, 'rank', rank)
	# s, vh = np.linalg.eig(FIM)
	# print(printMatrix(vh), printArray(s))
	return FIM, X, uniformSamples, matrix, timespace

## Calculate the simplified Fisher Information Matrix (FIM) ####

# FIM = minifisher(paramests, xData, yData, delta = 0.001)
# print(np.linalg.matrix_rank(FIM), len(paramests)) #calculate rank of FIM
# print(FIM)


def copyInd(ind):
	ind1 = []
	for it in ind:
		ind1.append(it)
	return ind1

def getPerturbedInd(ind, j):
	ind1 = copyInd(ind) # to store the perturbed individual
	#l = 20
	#r = random.uniform(1/(l-1), (1- 1/(l-1)))
	#d = ind[j] + r * (KUpp[j] - KLow[j])
	#ind1[j] = ind[j] - d 
	ind1[j] = random.normal(ind[j], 0.02*ind[j]) #random.uniform(KLow[j], KUpp[j]) #ind[j], 0.5*ind[j])
	#if j == 0 and k < 2:
	#    print('sample ', ind[j], ind1[j])
	return ind1

def latin_sampling(l, h, k= 20, delta = 0.1):
	kdt = IntervalTree()
	U_lower = []
	U_upper = []
	for i in range(len(l)):
		U_lower.append((1-delta)*l[i] if l[i] > 0 else (1+delta)*l[i])
		U_upper.append((1+delta)*h[i] if h[i] > 0 else (1-delta)*h[i])
	U = (U_lower, U_upper)
	#print('U', U)
	points = kdt.generate_samples(U, k)
	return points

def getFIM_global(params, param_names, param_range, obs_data, observable_index, time, atol):
	flag = False
	lp = len(params)
	# while not flag:
	# 	FIM, S, T = getFIM_COV_global(params, param_names, param_range, obs_data, observable_index, time, atol)
	# 	det = np.linalg.det(FIM)
	# 	print('FIM det', det)
	# 	if det > 0.0:
	# 		cov = np.linalg.inv(FIM)
	# 		flag = True
	# 		for i in range(lp):
	# 			if cov[i][i] < 0.0:
	# 				flag = False
	# 				break
	# 	break

	FIM, S, T = getFIM_COV_global(params, param_names, param_range, obs_data, observable_index, time, atol)
	return FIM, S, T

def getFIM_COV_global(params, param_names, param_range, obs_data, observable_index, time, atol):
	#print('################## getFIM_COV_global #########################')
	
	#params = generate_population(50)
	# atol = 0.01
	samples = min(int(time/atol), 500)
	uniformSamples = np.linspace(0, time, samples) #sorted(np.random.choice(np.linspace(0, time, samples), min(samples, 20)))

	ind = params
	# matrix, timespace = run(ind, time)
	n_all = len(params)
	n = len(params)
	KLow = []
	KUpp = []
	default_params = params
	for i in range(n):
		KLow.append(param_range[param_names[i]][0])
		KUpp.append(param_range[param_names[i]][1])

	l = 20.0

	r_samples = 1 #20 #50 #20 # these many samples for running the Morris method
	all_params = list(latin_sampling(KLow, KUpp, r_samples - 1))
	all_params.append(ind)

	#print(KLow, KUpp, ind, len(all_params))
	# #print(len(all_params), all_params)

	#print('observable_index', observable_index)
	S_array = []
	u = 1.0	

	# timeSamples = [] # random sampling times between start and end (maxTime)
	# samplesChosen = sorted(np.random.choice(timespace, 20))
	# samples = 20
	# for sam in samplesChosen:
	# 	timeSamples.append(timespace.index(sam))  

	# #print('samples ', samples, timeSamples, time)
	# variance of i_th observation at t_th time (only diagonal elements)

	#print('parameters ', n)
	# #print('Shape of S matrix : {0}X{1}'.format(3*samples, n))
	S_array = []	
	d = []
	# ind1 = copyInd(ind)
	dk = []
	ind_01 = []
	#global KLow, KUpp
	#for i in range(n):
	#   #print(KLow[i], KUpp[i], ind[i])
		
	##print(ind)
	##print('##########')
	A = []
	J = []
	for i in range(0, n+1):
		a = []
		js = []
		for j in range(0, n):
			if j < i:
				a.append(1)
			else:
				a.append(0)
			js.append(1)
		A.append(a)
		J.append(js)

	for k in range(0, r_samples):
		dj = []
		# delta = 1/(l-1) 
		delta = l/(2*(l-1))
		# delta = rnd.uniform(1/(l-1), 1- 1/(l-1))
		hd = delta/2
		##print('delta: ', delta, hd)
		#ind1 = copyInd(ind)
		index = rnd.sample(range(0, len(all_params)), 1)[0]
		# index = 0
		ind = all_params[index]
		ind_01 = []
		for i in range(n):
			item = (ind[i] - KLow[i])/(KUpp[i] - KLow[i])
			ind_01.append(item)
		if k < 1:
			print('1','mapped to 0-1 :', ind_01)
		D = []
		P = []
		D_ul = []
		# P_star = []
		for i in range(0, n):
			d = []
			p = []
			dul = []
			for j in range(0, n):
				r = 0
				pi = 0
				rul = 0
				if i == j:
					r = rnd.sample([-1, 1], 1)[0]
					rul = KUpp[j] - KLow[j]
					##print(rul)
					pi = 1
				d.append(r)
				p.append(pi)
				dul.append(rul)
			D.append(d)
			P.append(p)
			D_ul.append(dul)
			#P_star.append(p)

		#for i in range(0, n):
		m = rnd.sample(range(0, n), 1)[0]
		##print(m)
		rr = [s for s in range(n)]
		# #print(rr)
		p1 = np.random.shuffle(rr)
		P = np.array(P).take(rr, axis=0)

		D_star = np.array(D)
		P_star = np.array(P)
		J_star = np.array(J)
		A_star = np.array(A)
		D_ul_star = np.array(D_ul)

		indices = []
		for i in range(len(P_star)):
			for j in range(len(P_star[0])):
				if P_star[i][j] == 1:
					indices.append(j)
					break

		# if k < 1:
		#     #print('D')
		#     #print(D_star)
		#     #print('##########')
		#     #print('P')
		#     #print(np.array(P))
		#     #print('##########')
		#     #print(indices)
		#     #print('##########')

		#     #print(D_star.shape, P_star.shape, J_star.shape, A_star.shape, D_ul_star.shape, np.array(ind).shape)

		#     #print(J_star.dot(ind).shape)

		#ind1 = np.transpose(ind)
		##print(np.array(ind).shape, np.array(ind1).shape)
		##print(P_star)

		#Lb = []
		#for i in range(len(KLow)):
		#    Lb.append(KLow[i])
		Lb = np.array(KLow)
		
		A_s = J_star*ind_01 + (hd*((2*A_star - J_star).dot(D_star) + J_star)).dot(P_star)
		C = J_star[0]*Lb + A_s.dot(D_ul_star)

		# if k < 1:
		#     #print('A_s', A_s.shape)
		#     #print('##########')
		#     #print('J*q')
		#     print(np.array(J_star*ind_01))
		#     print('##########')
		#     print('(2*A - J)')
		#     print((2*A_star - J_star))# + J_star)
		#     print('##########')
		#     print('(2*A - J).dot(D_star)')
		#     print((2*A_star - J_star).dot(D_star))# + J_star)
		#     print('##########')
		#     print('(2*A - J).dot(D_star) + J')
		#     print((2*A_star - J_star).dot(D_star) + J_star)
		#     print('##########')
		#     print('(delta/2) *((2*A - J).dot(D_star) + J)')
		#     print(hd*((2*A_star - J_star).dot(D_star) + J_star))
		#     print('##########')
		#     print('((delta/2) *((2*A - J).dot(D_star) + J)).dot(P)')
		#     print((hd*((2*A_star - J_star).dot(D_star) + J_star)).dot(P_star))
		#     print('##########')
		#     print('J*q +((delta/2) *((2*A - J).dot(D_star) + J)).dot(P)')
		#     print(J_star*ind_01 +(delta/2.0)*((2.0*A_star - J_star).dot(D_star) + J_star).dot(P_star))
		#     print('##########')
		#     #print('##########')
		#     print('J[0].dot(Lb)')
		#     print(J_star[0].dot(Lb))# + J_star)
		#     print('##########')
		#     print('J[0]*KLow')
		#     print(J_star[0]*KLow)# + J_star)
		#     print('##########')
		#     print('A_s')
		#     print(np.array(A_s))
		#     print('##########')
		#     print('C')
		#     print( np.array(C))
		samplesChosen = []
		for j in range(0, n):
			ind1 = copyInd(C[j][:])
			ind2 = copyInd(C[j+1][:])
			index = indices[j]
			# del_jk =  delta 
			del_jk = ind2[index] - ind1[index]

			if k < 1:
				print('2',ind1[index], ind2[index], del_jk, (ind2[index] - ind[index])/ind[index], (ind1[index] - ind[index])/ind[index])

			#if del_jk < 0.00000000001:
			#    continue
			matrix1_SS, timespace1 = run(ind1, time)
			matrix2_SS, timespace2 = run(ind2, time)

			djk = []			
			kt1 = 0 
			for tm in uniformSamples: # number of times
				t1 = findIndex(timespace1, tm, atol)
				t2 = findIndex(timespace2, tm, atol)
				# print(tm, t1, t2)
				#tm = int(900*100)
				#tup1 = matrix1_SS[:, tm] 
				#tup2 = matrix2_SS[:, tm]
				kt1 += 1
				djkt = []
				for i in observable_index:
					val1 = matrix1_SS[i][t1] #tup1[i]
					val2 = matrix2_SS[i][t2] #tup2[i]
					scale = 1.0
					'''if i == 2 or i == 7:
						scale = mol_TT4
					elif i == 0 or i == 6:
						scale = mol_TT3'''
					djkti = (val2 - val1)*scale/del_jk
					djkt.append(djkti)
				djk.append(djkt)
			dj.append(djk)
		#print(np.array(dj))
		dk.append(dj)
	#print(np.array(dk))
	if (len(dk) > 0):
		d_sum_j = np.average(dk, axis=0) #/samples # sensitivity on p_j (on all observations, without normalisation)
		print(np.array(dk).shape, np.array(d_sum_j).shape)
		d = d_sum_j
	#else:
	#   d = []
		#    for j in range0, q()
	#print(np.array(d))

	ind = default_params
	matrix, timespace = run(ind, time)
	big_sigma_array = []
	for k in observable_index:
		# k = observable_index[i]
		#print(k)
		tup = matrix[k]
		#print(tup)
		#for i in observable_index:
		for t1 in uniformSamples:
			tm = findIndex(timespace, t1, atol)
			#tup = obs[:, tm]
			#val = matrix_SS[observable_index[i]][tm]
			val = tup[tm]
			# print(val)
			sigma = val*np.random.normal(0, 0.05, 1)[0] #if val > 0 else np.random.normal(0.1*val, 0, 1)[0] #(val*0.1)#/T3_ss
			# print(sigma)
			big_sigma_array.append(sigma**2)

	big_sigma_array1 = np.array(big_sigma_array, dtype=np.float64)
	big_sigma = np.diagflat(big_sigma_array1)

	S_dash = [] # bring the matrix to proper sensitivity matrix form
	for i in observable_index:
		u = np.mean(obs_data[i])
		# print('mean : {0}, {1}'.format(i, u))
		for t1 in range(kt1): #range(len(range(timestart, maxTime, timeDiv))):
			#k = i*t1
			sigma = big_sigma[k][k]
			row = []
			for j in range(0, n): #param_indices:
				u1 = default_params[j] #ind[j]
				v =  d[j][t1][i]*u1/np.sqrt(sigma)               
				row.append(v)
			S_dash.append(row)
			# print(row)  
	#big_sigma = np.diagflat(big_sigma_array)
	#sigma_inv = linalg.inv(big_sigma)
	# timeSamples = 

	
	#sigma_inv = linalg.inv(big_sigma)
	prod = 1.0
	for item in big_sigma_array1 :
		prod *= item
	print('Sigma product ', prod)
	S = np.array(S_dash, dtype=np.float64) #np.matrix(S_array)
	print('calculated shape of S_matrix: ', np.array(S).shape)
	#print('####### Sigma #########')
	#print_matrix(sigma_inv)
	#S_t = np.transpose(S)
	#print(type(S_t))
	#S_t_sigma = S_t.dot(sigma_inv)
	#FIM = S_t_sigma.dot(S)
	#fim2 = (S_t*sigma_inv)*S
	#print('########### Sigma ########')
	#print_matrix(big_sigma_array1)
	#print(big_sigma_array1)
	#print('####### S #########')
	#print(S)
	#print_matrix(S)
	#print('####### FIM #########')
	#print_matrix(FIM)
	#print('####### FIM-1 #########')
	#print_matrix(fim2)
	
	# S = np.array(S_array, dtype=np.float64) 

	# print('calculated shape of S_matrix: ', np.array(S).shape)
	S_t = np.transpose(S)
	print( np.array(S_t).shape)
	# sigma_inv = np.linalg.inv(big_sigma)
	# print(np.array(sigma_inv).shape)
	S_t_sigma = S_t #.dot(sigma_inv)
	FIM = S_t_sigma.dot(S)
	# print_matrix(S)
	# print_matrix(S_t)
	fim = S_t.dot(S)
	# print()
	# print(np.linalg.det(sigma_inv), np.linalg.det(S))
	#print(FIM)
	det = np.linalg.det(FIM)
	print('FIM det', det, 'rank', np.linalg.matrix_rank(FIM))
	T = len(uniformSamples)
	# # print_matrix(FIM)
	# if det == 0.0:
	# 	cov = np.ones(n, n) *0.01
	# else:
	# 	cov = np.linalg.inv(FIM)
	return FIM, S, uniformSamples#, cov

# cov = getCOV_global()
# print_matrix(cov)
# print('cov_det', linalg.det(cov))

import itertools
def findsubsets(S,m):
	return set(itertools.combinations(S, m))

def intersection(set1, set2):
	flag = False
	for i in set1:
		if i in set2:
			flag = True
			break
	return flag

def getsubFIM(SM, nn):
	SM_T = SM #.transpose()
	st = []
	for i in nn:
		st.append(SM_T[i])
	st = np.array(st, dtype=np.float64)
	s = st.transpose()
	fim = st.dot(s)

	#print('getsubFIM', nn, st.shape, s.shape, fim.shape)
	# for i in n:
	# 	row = []
	# 	for j in n:
	# 		row.append(FIM[i][j])
	# 	fim.append(row)
	return fim

def checkNearlyFullRank(SM, nn):
	forall = True
	l = len(nn)
	pss = findsubsets(nn, l-1)
	for nn1 in pss:
		fim = getsubFIM(SM, nn1)
		s_fim_rank = np.linalg.matrix_rank(fim)
		if not(s_fim_rank == len(nn1)):
			forall = False
			break
	return forall, pss

def getLinearDependentRow(matrix):
	dp = []
	indp = []
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[0]):
			if i != j:
				inner_product = np.inner(
					matrix[:,i],
					matrix[:,j]
				)
				norm_i = np.linalg.norm(matrix[:,i])
				norm_j = np.linalg.norm(matrix[:,j])

				# print('I: ', matrix[:,i])
				# print('J: ', matrix[:,j])
				# print('Prod: ', inner_product)
				# print('Norm i: ', norm_i)
				# print('Norm j: ', norm_j)
				if np.abs(inner_product - norm_j * norm_i) < 1E-5:
					# print('Dependent')
					dp.append((i, j))
				else:
					# print('Independent')
					indp.append((i, j))
	return dp, indp
def getConnectedParameters(FIM, SM, T, params, param_names):
	print(params, param_names)

	# #print(#printMatrix(FIM)) 
	print('######## getConnectedParameters ###########')	
	det = np.linalg.det(FIM)
	fim_rank = np.linalg.matrix_rank(FIM)
	print('FIM det', det, 'rank', fim_rank)
	# print(FIM)

	# pl, u = lu(FIM, permute_l=True)

	# #print(pl)
	# #print(u)
	print('###################')

	independent = []
	insensitive = []
	dependent = []

	n_par = len(params)	
	contract = {}
	j = 0
	for i in range(n_par):
		contract.update({i:j})

	 # = np.linalg.matrix_rank(FIM)
	
	num_param_comb = fim_rank

	p_inds = [] #list(range(n_par))
	for i in range(n_par):
		if FIM[i][i] < 1e-20:
			#print('Insensitive parameter', i)
			insensitive.append(param_names[i])
		else:
			p_inds.append(i)


	# det = np.linalg.det(FIM)
	print('FIM det', det)
	# print(FIM)

	if not det == 0.0:
		cov = np.linalg.inv(FIM)
		n_ps = p_inds
		d_ps = []
		q = 0
		sd_params = {}
		for i in n_ps:
			# j = contract[i]
			sd = np.sqrt(cov[i][i])
			sc = sd*100/params[i]
			# print(param_names[i], sd, sc)
			sd_params.update({sc:i})
			if sc > 100.0:
				d_ps.append(i)

		# sd_params_sorted = sorted(sd_params.keys(), reverse=True)
		# pre_sc = sd_params_sorted[0]
		
		# flag = 0
		# for sc in sd_params_sorted:
		# 	i = sd_params[sc]
		# 	print(param_names[i], sc)
		# 	if sc > 100.0:
		# 		d_ps.append(i)
		# 	# ratio = pre_sc/sc
		# 	# print(param_names[i], sc)
		# 	# if i > 0 and ratio > 80.0:
		# 	# 	flag = 1
		# 	# else:
				
		# 	# if flag == 0:
		# 	# 	pre_sc = sc

		for i in n_ps:
			if i not in d_ps:
				independent.append(param_names[i])

		print('independent', independent)
		d_ps = sorted(d_ps)
		print('dependent', d_ps, [param_names[i] for i in d_ps])
		num_param_comb -= len(independent)

		print('Identifiable parameter combinations ', num_param_comb)

		all_subsets = []
		l = len(d_ps) #- 1	
		ps = list(findsubsets(d_ps, l))
		for nn in ps:
			all_subsets.append(nn)
			# l -= 1

		while l > 1:
			print('all subsets of size ', l, len(all_subsets))
			ps = []
			ps2 = []
			for nn in all_subsets:
				fullRank, pss = checkNearlyFullRank(SM, nn)
				if fullRank:
					print('subset', nn, [param_names[i] for i in nn], 'rank deficient') 
					dependent.append([param_names[i] for i in nn])
					for nn1 in pss:
						ps2.append(nn1)
				else:
					print('subset', nn, [param_names[i] for i in nn])
					for nn1 in pss:
						ps.append(nn1)
			# if len(ps) > 0:
			print(ps, ps2)
			l -= 1
			all_subsets = []
			for nn in ps:
				if nn not in all_subsets and nn not in ps2:
					all_subsets.append(nn)

	else:
		#Cauchy-Schwarz inequality
		#print('Two or more parameters are linearly depended')

		dp, indp = getLinearDependentRow(FIM)

		# lambdas, V =  np.linalg.eig(FIM)
		# print(lambdas, V)
		# # The linearly dependent row vectors 
		# print(FIM[lambdas == 0,:])

		d_ps = []
		for it in dp:
			i, j = it
			if i not in d_ps:
				d_ps.append(i)
			if j not in d_ps:
				d_ps.append(j)
		print('dependent', d_ps)
		fim_d = []
		n_ps = []
		flag = False
		l = len(p_inds) - 1
		fim_d = FIM

		l = 1
		while l < len(d_ps):
			i_dps = findsubsets(d_ps, l)
			for i_dp in i_dps:
				param_ps = [i for i in range(n_par) if i not in i_dp]
				fim = getsubFIM(SM, param_ps)
				det = np.linalg.det(fim)
				# print(det, printMatrix(fim))
				if det > 0.0:
					# print('linear depended', [i for i in range(n_par) if i not in nn])				
					fim_d = fim
					flag = True
					n_ps = param_ps
					break
			if flag:
				break

			l += 1

		# while l > 0:
		# 	param_ps = findsubsets(p_inds, l)
		# 	for nn in param_ps:
				# nn1 = range(nn)
		# 		print('subset -- ', nn)
		# 		fim = getsubFIM(SM, nn)
		# 		det = np.linalg.det(fim)
		# 		print(det, printMatrix(fim))
		# 		if det > 0.0:
		# 			# print('linear depended', [i for i in range(n_par) if i not in nn])				
		# 			fim_d = fim
		# 			flag = True
		# 			n_ps = nn
		# 			break
		# 	if flag:
		# 		break
		# 	l = l - 1

		# for i in p_inds:
		# 	if i not in n_ps:
		# 		d_ps.append(i)
		# d_ps = [i for i in range(n_par) if i not in n_ps]
		re_contract = {}
		j = 0
		for i in n_ps:
			re_contract.update({i:j})
			j+=1
		print('n_ps', n_ps)

		if len(n_ps) == 0:
			print('dependent', d_ps, [param_names[i] for i in d_ps])
		else:
			cov = np.linalg.inv(fim_d)
			# print(cov)
			# n_ps = p_inds
			# d_ps = []
			q = 0
			sd_params = {}
			for i in n_ps:
				j = re_contract[i]
				sd = np.sqrt(cov[j][j])
				sc = sd*100/params[i]
				sd_params.update({sc:i})

				print(param_names[i], sc, sd)
				if sc > 100.0:
					if i not in d_ps:
						d_ps.append(i)

			sd_params_sorted = sorted(sd_params.keys(), reverse=True)
			pre_sc = sd_params_sorted[0]
			print('-------------------')
			for sc in sd_params_sorted:
				i = sd_params[sc]
				# print(param_names[i], sc)
				ratio = pre_sc/sc
				print(param_names[i], sc)
				if i > 0 and ratio > 100.0:
					if i not in d_ps:
						d_ps.append(i)
					break
				else:
					if sc > 100.0:
						if i not in d_ps:
							d_ps.append(i)
				pre_sc = sc

		for i in p_inds:
			if i not in d_ps:
				independent.append(param_names[i])

		d_ps = sorted(d_ps)
		print('dependent', d_ps, [param_names[i] for i in d_ps])
		num_param_comb -= len(independent)

		#print('Identifiable parameter combinations ', num_param_comb)

		all_subsets = []
		l = len(d_ps) - 1	
		ps = list(findsubsets(d_ps, l))
		for nn in ps:
			all_subsets.append(nn)
			# l -= 1

		while l > 1:
			#print('all subsets of size ', l, len(all_subsets))
			ps = []
			ps2 = []
			for nn in all_subsets:
				fullRank, pss = checkNearlyFullRank(SM, nn)
				if fullRank:
					#print('subset', nn, [param_names[i] for i in nn], 'rank deficient') 
					dependent.append([param_names[i] for i in nn])
					for nn1 in pss:
						ps2.append(nn1)
				else:
					#print('subset', nn, [param_names[i] for i in nn])
					for nn1 in pss:
						ps.append(nn1)
			#print(ps, ps2)
			l -= 1
			all_subsets = []
			for nn in ps:
				if nn not in all_subsets and nn not in ps2:
					all_subsets.append(nn)

	print('#########################')
	print('independent', independent)
	print('dependent', dependent)
	print('insensitive', insensitive)
	print('#########################')
	#print('SM_T = {0}'.format(printMatrix(SM)))
	#print('T = {0}'.format(T))
	print('params = {0}'.format(params))
	print('param_names = {0}'.format(param_names))
	print('dependent = {0}'.format(dependent))
	print('#####################')
	#getEquations(SM, T, params, param_names, dependent)
	return independent, dependent, insensitive

def checkConstraints(err_func, lp, bounds = []):	
	# A = np.eye(np)
	# if len(bounds) == 0:
	# 	bounds = [(1.0, 100.0) for i in range(np)]
	# lb = [bounds[i][0] for i in range(np)]
	# ub = [bounds[i][1] for i in range(np)]
	init_params = [1.0 for i in range(lp-1)]
	# constraints = scipy.optimize.LinearConstraint(A, lb, ub, keep_feasible=False)[source]
	# 
	optimizer = minimize(err_func, init_params,  method='nelder-mead', options={'xatol': 1e-20}) #, bounds = Lbounds)
	return optimizer.x, optimizer.fun, optimizer.success, optimizer.message


def getEquations(SM_T, T, params, param_names, dependent):
	n_par = len(params)	
	contract = {}
	for i in range(n_par):
		contract.update({param_names[i]:i})

	lp = len(SM_T)
	nyt = len(SM_T[0])
	ny = list(range(0, int(nyt/T)))
	
	#print('SM_T', lp, nyt)

	def checkAdd(SM_T, p, params):
		def func_min(c1):
			t = 0.0
			for i in c1:
				if i == 0.0:
					t += 1
			if t == len(p) - 1:
				return 100.0
			
			c = list(c1)
			c.append(1.0)

			s_sum = []
			for j in ny:
				st_sum = []
				for l in range(T):
					j_ind = j*T+l
					s = 0.0
					for k in range(len(p)):
						i = p[k]
						st = 0.0
						# j_ind = j*T+l
						if SM_T[i][j_ind] == 0.0:
							continue
						else:
							st += 1.0/(SM_T[i][j_ind])
						# #print(k, c, c1)
						s += c[k]*st
					st_sum.append(s**2)
				sy_min = np.sum(st_sum) # for an observable, for all T error
				s_sum.append(sy_min)
			s_min = np.min(s_sum) # there exist an y, for all T sum_{p} ci/sji_t == 0
			return s_min**2
		c, err, suc, msg = checkConstraints(func_min, len(p))
		#print('Add', c, err, suc, msg)
		t = 0.0
		for i in c:
			if i == 0.0:
				t += 1
		if suc and t < len(p):
			return True, c
		else:
			return False, []

	def checkMult(SM_T, p, params):
		def func_min(c1):
			t = 0.0
			for i in c1:
				if i == 0.0:
					t += 1
			if t == len(p) - 1:
				return 100.0
			
			c = list(c1)
			c.append(1.0)

			s_sum = []
			for j in ny:
				st_sum = []
				for l in range(T):
					j_ind = j*T+l
					s = 0.0
					for k in range(len(p)):
						i = p[k]
						st = 0.0
						j_ind = j*T+l
						if SM_T[i][j_ind] == 0.0:
							continue
						else:
							st += 1.0/(params[i]*SM_T[i][j_ind])
						# #print(k, c, c1)
						s += c[k]*st
					st_sum.append(s**2)
				sy_min = np.sum(st_sum) # for an observable, for all T error
				s_sum.append(sy_min)
			s_min = np.min(s_sum) # there exist an y, for all T sum_{p} ci/sji_t == 0
			return s_min**2

		c, err, suc, msg = checkConstraints(func_min, len(p))
		#print('MULT', c, err, suc, msg)
		t = 0.0
		for i in c:
			if i == 0.0:
				t += 1
		if suc and t < len(p):
			return True, c
		else:
			return False, []

	# addition = []
	# multiplication = []
	# subtraction = []
	# division = []

	param_len = 2 
	for all_params in dependent:
		# param_len = len(all_params.keys())
		subsets_2 = findsubsets(all_params, param_len)
		for sub in subsets_2: #for each subset of size 2 
			# print('For param set', sub, type(sub))
			p = [contract[i] for i in sub]
			succ1, c_add = checkAdd(SM_T, p, params)

			st = str(sub)
			if succ1:
				# st += 'Add: ' + str(c_add) + '\n'
				i = 0
				for c in c_add:
					st += '{0} * {1} + '.format(c, sub[i]) #if i ==0 else ' + {0} * {1}'.format(c, sub[i])
					i+=1
				st += '{0}'.format(sub[i])
			else:
				succ2, c_mult = checkMult(SM_T, p, params)
				if succ2:
					# st += 'Mult: ' + str(c_mult) + '\n'
					i = 0
					for c in c_mult:
						st += '{0} * {1} * '.format(c, sub[i]) #if i ==0 else ' + {0} * {1}'.format(c, sub[i])
						i+=1
					st += '{0}'.format(sub[i])
				else:
					st += 'No relation'
			print('###', st)

def getDelFunc(ind):
	lp = len(ind)
	matrix, timespace, senseMatrix, sample_points, observed_data, obs_timespace = getSenseResult(ind)

	values = []
	n_all = lp
	S_array = []
	#print(observable_index)
	r_theta_arary = []
	#print('getSensitivity', n, len(t_sample), n_obv)
	for l in range(len(matrix)):
		values.append(matrix[l])
		i = 0
		for k in obs_timespace: #range(penalty_range1, penalty_range2, 15):
			t1 = findIndex(timespace, k, atol)
			t2 = findIndex(sample_points, k, atol)
			serror = ((matrix[l][t1]-observed_data[l][i])/observed_data[l][i])
			r_theta_arary.append(serror)
			#print(t1)
			row = []						
			for i1 in range(n_all):
				t = int(l*len(senseMatrix) + t2)
				tup = senseMatrix[i1][t]
				row.append(tup)
			S_array.append(row)	
			i += 1
	#print(len(S_array), len(S_array[0]))
	
	S = np.array(S_array, dtype=np.float64)	
	r_theta = np.array(r_theta_arary, dtype=np.float64)
	del_l_theta = 2 * r_theta.dot(S)
	return np.transpose(del_l_theta)

def update_param(p_k, v_k, scale = 0.005):
	#print('before line search', np.shape(s_k),np.shape(v_kp1), np.shape(np.transpose(v_kp1)))
	#scale = scipy.optimize.line_search(get_error, getDelFunc, p_k, v_k)[0] #, s_k)
	#if scale == None or scale > 0.008:
	# scale = 0.005
	#print('line seaaaaaaaaaaaaaaaaaaaaaaaaarch', scale)
	p_k1 = np.add(p_k, (scale*v_k))
	return p_k1

def quasi_newton(ind, scale = 0.005):
	#greedy_mutant = toolbox.clone(ind)
	n = len(ind)
	n_iter = 5
	#print('params', ind)	
	#if isInd(ind):
	# print('In -- quasi_newton', 'before', ind)
	#	printSol(ind)
	p_kp1 = ind
	h_kp1 = np.eye(n)
	s_kp1 = getDelFunc(p_kp1) 
	for i in range(n_iter):
		# if error:
		# 	break
		p_k = p_kp1
		s_k = s_kp1
		h_k = h_kp1
		#print('P_k', p_k)
		#print('h_k', h_k)
		#print('s_k', s_k)
		h_s = h_k.dot(s_k)
		#print('h_s', h_s)
		det = np.linalg.norm(h_s)
		v_kp1 = - h_s/det
		#print('direction', scale*v_kp1)
		p_kp1 = update_param(p_k, v_kp1, scale) #np.add(p_k, np.transpose((scale*v_kp1)))
		#print('p_k+1', p_kp1)
		s_kp1 = getDelFunc(p_kp1) #[0])
		#print('s_k+1', s_kp1)
		#print('P_k', p_k)
		y_k = np.subtract(p_kp1, p_k) #scale*v_kp1 #np.add(p_kp1,-1* p_k)
		z_k = np.subtract(s_kp1, s_k)
		#print('y, z', y_k, z_k)
		y_t = np.transpose(y_k)
		z_t = np.transpose(z_k)
		#print('transpose', y_t, z_t)
		h_z_k = h_k.dot(z_k)
		h_z_t = np.transpose(h_z_k)
		y_y_t = y_k.dot(y_t)
		#print('y_yt', y_y_t)
		y_t_z = y_t.dot(z_k)
		#print('yt_z', y_t_z)
		t1 = y_y_t/y_t_z#[0][0]
		h_z_h_z_t = h_z_k.dot(h_z_t)
		z_t_h_z = z_t.dot(h_z_k)		
		#print('z_t_h_z', z_t_h_z)
		t2 = h_z_h_z_t/z_t_h_z #[0][0]
		h_kp1 = np.subtract(np.add(h_k, t1), t2)

		if np.linalg.det(h_kp1) == 0.0:
			h_kp1 = np.eye(n)

		if np.linalg.norm(s_kp1) < 1e-50:
			break
		#print('Iteration {0} -- old {1}, new {2}'.format(i, p_k, p_kp1))
	p_kp2 = p_kp1
	# p_kp2 = interpolate(p_kp1, KLow, KUpp)
	return p_kp2 #[0]

#####################
# T = 2003
# params = [0.7, 0.4, 0.75, 0.7, 3.0]
# param_names = ['k01', 'k02', 'k12', 'k21', 'V']
# dependent = [['k01', 'k02', 'k12', 'k21']]
# #####################
# getEquations(SM_T, T, params, param_names, dependent)

'''
def getCOV_local(ind):
	timespace = np.linspace(0, stable_time_tosimulate, stable_samples)
	n_all = len(selectedIndices)
	n = len(sensitiveIndices)
	c = run_sense(timespace, ind)

	matrix = c[0]
	value_SS = c[1]
	err_msg = c[2]

	T4_ss = matrix[var_output['T4']][0]
	T3_ss = matrix[var_output['T3']][0]
	TSH_ss = matrix[var_output['TSH']][0]

	T4_o = obs[0]
	T3_o = obs[1]
	TSH_o = obs[2]


	S_array = []
	u = 1.0
	tm = -1

	maxTime = stable_time_tosimulate
	timestart = 0

	n = len(ind)

	samples = 0
	timeSamples = [] # random sampling times between start and end (maxTime)
	t1 = timestart
	while t1 < maxTime:
		timeSamples.append(t1)
		#random.seed(samples)
		timeDiv = rnd.uniform(1, maxTime/5)
		t1 += timeDiv
		samples += 1    

	# print('samples ', samples, timeSamples)
	big_sigma_array = [] # variance of i_th observation at t_th time (only diagonal elements)

	# print('parameters ', n)
	# print('Shape of S matrix : {0}X{1}'.format(3*samples, n))
	S_array = []
	for k in observable_index:
		# k = observable_index[i]
		#print(k)
		tup = matrix[k]
		#print(tup)
		#for i in observable_index:
		for t1 in timeSamples:
			tm = int(t1*stable_sampHr)
			#tm = int(900*100)
			#tup = obs[:, tm]
			#val = matrix_SS[observable_index[i]][tm]
			val = tup[tm]
			
			if k == var_output['TSH']:
				sigma = np.random.normal(0, 0.1*val, 1)[0] #(val*0.1), 1) #/TSH_ss #0.01 #(val*0.2) # actually variance
			elif k == var_output['T4']:
				sigma = np.random.normal(0, 0.1*val, 1)[0] #(val*0.1)#/T4_ss # actually variance
			elif k == var_output['T3']:
				sigma = np.random.normal(0, 0.1*val, 1)[0] #(val*0.1)#/T3_ss
			# print(sigma)
			
			sigma = np.random.normal(0, 0.1*val, 1)[0] #(val*0.1)#/T3_ss
			big_sigma_array.append(sigma**2)

	big_sigma_array1 = np.array(big_sigma_array, dtype=np.float64)
	big_sigma = np.diagflat(big_sigma_array1)
	#sigma_inv = linalg.inv(big_sigma)
	prod = 1.0
	for item in big_sigma_array1 :
		prod *= item
	# print('Sigma product ', prod)
	# print('value_SS - t4', value_SS[4][tm], value_SS[5][tm], value_SS[6][tm], value_SS[7][tm], value_SS[140][tm], value_SS[141][tm], value_SS[142][tm], value_SS[143][tm])
	# print('value_SS - t3', value_SS[17][tm],value_SS[18][tm],value_SS[19][tm],value_SS[20][tm],value_SS[144][tm], value_SS[145][tm], value_SS[146][tm], value_SS[147][tm])
	# print('value_SS - tsh_p', value_SS[29][tm], value_SS[30][tm],value_SS[31][tm], value_SS[32][tm],value_SS[148][tm], value_SS[149][tm], value_SS[150][tm], value_SS[151][tm])
	#print('getSensitivity', n, len(t_sample), n_obv)

	for i in  observable_index : #(var_output['t3p'], var_output['t4'], var_output['tsh_p']):   
		u = np.mean(observed_data[i])
		# row = []            
		
		if i == var_output['TSH']:
			u = TSH_o  
		elif i == var_output['T3']:
			u = T3_o
		elif i == var_output['T4']:
			u = T4_o  
				  
		#for k in range(n):
		#   k1 = hashmap[sensitiveIndices[k]]
		for t1 in timeSamples: #range(penalty_range1, penalty_range2, 15):
			tm = int(t1*stable_sampHr)
			row = []
			for i1 in range(n):
				k1 = hashmap[selectedIndices[i1]]
				index = (i*n_all + k1) #+ len(var_output)
				# print('@@@ 1 ', i, n, n_all, i1, k1, index, len(value_SS))
				u1 = default_params[selectedIndices[i1]]

				# if len(value_SS) == 0 :
				#   print('###########')
				#   print('@@@ 1 ', i, n, k1, index)
				#   print(len(value_SS))
				# else:
				#   print(len(value_SS), len(value_SS[0]))

				#   if len(value_SS[index])== 0:
				#       print('@@@ 2 ', i, n, k1, index)
				#       print(len(value_SS[index]))
				#       print('###########')
				#print(value_SS, value_SS[index])
				tup = value_SS[index][tm] #[tm]
				v = tup*u1/u #*ind[k1] #((tup)/u)* individual[k1]   # * ind[k]/u #frac/
				row.append(v)
			S_array.append(row) 
	# print('S', len(S_array), len(S_array[0]))
	# print('R', len(r_theta_arary), len(r_theta_arary[0]))
	S = np.array(S_array, dtype=np.float64)     
	# print('calculated shape of S_matrix: ', np.array(S).shape)
	S_t = np.transpose(S)
	# print( np.array(S_t).shape)
	sigma_inv = linalg.inv(big_sigma)
	# print(np.array(sigma_inv).shape)
	S_t_sigma = S_t.dot(sigma_inv)
	FIM = S_t_sigma.dot(S)
	# print_matrix(S)
	# print_matrix(S_t)
	det = linalg.det(FIM)
	# print('FIM det', det)
	# print_matrix(FIM)

	if det == 0.0:
		cov = np.zeros((n, n))# *0.01
	else:
		cov = linalg.inv(FIM)
	# print_matrix(cov)
	return FIM, cov

cov1 = getCOV_local(population[0])
print_matrix(cov1)
'''
