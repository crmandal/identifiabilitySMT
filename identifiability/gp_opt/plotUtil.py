
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from paramUtil.interval import *
from paramUtil.box import *
import paramUtil.box_factory as bfact
import math, itertools
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import csv

SAT = 51
UNSAT = 52
UNKNOWN = -1
TRUE = 1
FALSE = 0
UNDET = 2
ONLYMARKED = 1
NOMARK = 0
RF = False

def findsubsets(S,m):
	return set(itertools.combinations(S, m))


def plotBox(currentAxis, b, combs, col1 = 'grey', boxType= FALSE):
	#if boxType = TRUE:
	# print(boxType)
	col = col1 #'white'
	if boxType == TRUE:
		col = 'blue'
	elif boxType == UNDET:
		col = 'white'
	elif boxType == FALSE:
		col = 'black'
	# plt.figure()  
	if b.size() == 2:
		b_edges = b.get_map()
		x = []
		w = []
		for it in combs:
			intrvl = b_edges[it]
			x.append(intrvl.leftBound())
			w.append(intrvl.width())
		currentAxis.add_patch(Rectangle((x[0], x[1]), w[0], w[1], facecolor=col, alpha=1))
	# plotBox.show()
	# return plt

functions = ['y+x' , 'y-x', 'y*x', 'y/x', 'y/x^2', 'y/x^(0.5)', 'y*x^2', 'y*x^(0.5)']

#todo: regression with fixed set of functions
def fixed_fn_set(x, *pm):
	p, op = pm
	if op == 1:# ADD y = p-x
		return p-x
	elif op == 2:# SUB y = p+x
		return p+x
	elif op == 3:# MULT y = p/x
		return p/x
	elif op == 4:# SUB y = p*x
		return p*x
	elif op == 5:# FN1 y = p*x^2
		return p*x^2	
	elif op == 6:# FN1 y = p*x^(0.5)
		return p*np.sqrt(x)
	elif op == 6:# FN1 y = p/x^2
		return p/x^2	
	elif op == 6:# FN1 y = p/x^(0.5)
		return p/np.sqrt(x)
	
def rational_cf(X, Y, names = ['x', 'y'], order = 3, dec = 1):
	print('-------------- rational curve fit -----------')
	def getEqn(popt, nr = True):
		if nr:
			od = int((len(popt)+1)/2)
			p = [popt[i] for i in range(od)]
			q = [popt[od+i] for i in range(od-1)]
			print('getEqn', list(popt), od, p, q)
			num = []
			nm = '(x-1)'
			for i in range(od):
				coff1 = abs(round(p[i], dec))
				coff = '{0:0.1f}'.format(p[i])
				if (i) > 1:
					if coff1 > 0:
						var = coff+'*' + nm+'^'+ str(i) 
						num.append(var)
				elif i == 1:
					if  coff1 > 0:
						var =  coff+'*'+ nm  
						num.append(var)
				else:
					if coff1 > 0:
						var = coff
						num.append(var)
			num = '+'.join(num)
			den = ['1']
			for i in range(od-1):
				coff1 = abs(round(q[i], dec))
				coff = '{0:0.1f}'.format(q[i])
				if i > 0:
					if  coff1 > 0:
						var = coff+'* ' + nm+'^'+ str(i+1)
						den.append(var)
				else : #i == 0:
					if coff1 > 0:
						var =  coff+'*'+ nm
						den.append(var)
			den = '+'.join(den)
			s = '(('+ num + ')/(' + den + '))'
			print('y = '+ s)
			x, y = symbols('x, y')
			s1 = sympify(s).simplify().subs('x', names[0])
			#._full()
			s2 = names[1] + '= '+ str(s1).replace('**', '^')
			return s2
		else:
			p, op = popt[0], popt[1]
			s1 = ''
			s2 = ''
			if op == 1:# ADD y = p-x
				s2 = '{0} + {1} = {2:0.2f}'.format(names[0], names[1], p)  
				s1 = '{0} + {1}'.format(names[0], names[1])  
			elif op == 2:# SUB y = p+x
				s2 = '{1} - {0} = {2:0.2f}'.format(names[0], names[1], p)  
				s1 = '{1} - {0}'.format(names[0], names[1])  
			elif op == 3:# MULT y = p/x
				s2 = '{0} * {1} = {2:0.2f}'.format(names[0], names[1], p)  
				s1 = '{0} * {1}'.format(names[0], names[1])  
			elif op == 4:# SUB y = p*x
				s2 = '{1} / {0} = {2:0.2f}'.format(names[0], names[1], p)  
				s1 = '{1} / {0}'.format(names[0], names[1])  
			elif op == 5:# FN1 y = p*x^2
				s2 = '{1} = {2:0.2f}*{0}^2'.format(names[0], names[1], p)  	
				s1 = '{1}/{0}^2'.format(names[0], names[1])  	
			elif op == 6:# FN1 y = p*x^(0.5)
				s2 = '{1}^2 = {2:0.2f}*{0}'.format(names[0], names[1], p**2)  
				s1 = '{1}^2 /{0}'.format(names[0], names[1])  
			# elif op == 6:# FN1 y = p/x^2
				# s2 = '{0}^2*{1} = {2:0.2f}'.format(names[0], names[1], p)  
				# s1 = '{0}^2*{1}'.format(names[0], names[1])  
				# return p/x^2	
			# elif op == 6:# FN1 y = p/x^(0.5)
			# 	s2 = '{0}*{1}^2 = {2:0.2f}'.format(names[0], names[1], p**2)  
			# 	s1 = '{0}*{1}^2'.format(names[0], names[1])  
			# 	# return p/np.sqrt(x)
	
			return s2, s1
	def rational(x, p, q):
		"""
		The general rational function description.
		p is a list with the polynomial coefficients in the numerator
		q is a list with the polynomial coefficients (except the first one)
		in the denominator
		The zeroth order coefficient of the denominator polynomial is fixed at 1.
		Numpy stores coefficients in [x**2 + x + 1] order, so the fixed
		zeroth order denominator coefficent must comes last. (Edited.)
		"""

		num = [];
		den = [1.0];
		for i in range(len(p)):
			var = p[i]*(x**i)
			num.append(var)
		for i in range(len(q)):
			var = q[i]*x*(x**i)
			den.append(var)
		y = np.sum(num)/np.sum(den)
		#np.polyval(p, x) / np.polyval(q + [1.0], x)
		# print('rational', x, y, num, den)
		return y

	def rational3_3(x, *params): #p0, p1, p2, q1, q2):
		od = int((len(params)+1)/2)
		# print('rational3_3', params, len(params), od, order, x)
		p = [params[i] for i in range(od)]
		q = [params[od+i] for i in range(od-1)]

		return rational(x, p, q)

	def Calc_all(X, *params): #p0, p1, p2, q1, q2):
		yHat = []
		for i in range(len(X)):
			x = X[i]
			y = Y[i]
			# rational3_3(x, *params) 
			yhat = rational3_3(x, *params) if RF else fixed_fn_set(x, *params)
			# yhat = 
			yHat.append(yhat)
		return yHat

	def fixed_fn_set(x, *params):
		# print('fixed_fn_set', params)
		p, op = params[0], int(params[1])
		if op == 1:# ADD y = p-x
			return p-x
		elif op == 2:# SUB y = p+x
			return p+x
		elif op == 3:# MULT y = p/x
			return p/x
		elif op == 4:# DIV y = p*x
			return p*x
		elif op == 5:# FN1 y = p*x^2
			return p*x**2	
		elif op == 6:# FN1 y = p*x^(0.5)
			return p*np.sqrt(x)
		# elif op == 6:# FN1 y = p/x^2
		# 	return p/x**2	
		# elif op == 6:# FN1 y = p/x^(0.5)
			# return p/np.sqrt(x)
		else:
			return p

	def err_func(*params, log = False):
		err = 0
		yHat = Calc_all(X, *params)
		for i in range(len(X)):
			x = X[i]
			y = Y[i]
			# rational3_3(x, *params) 
			yhat = yHat[i]
			err += (y - yhat)**2
		# err = mean_squared_error(Y, yHat)
		if log:
			print('err_func', err, X, Y, yHat)
		return err #, bic

	# calculate bic for regression
	def calculate_bic(n, mse, num_params):
		# k*ln(n) - 2 ln(L) L : maximum likelihood func
		# L = math.exp(-mse)
		bic = 2*mse + num_params * math.log(n)
		# print('calculate_bic', mse, num_params * math.log(n), bic)
		return bic

	pre_bic = 99999999
	preopt = []
	pre_order = 0.0
	if RF:
		p = [2.0 for i in range(0, order)]
		q = [1.0 for i in range(0, order - 1)]

		for i in range(1, order):
			params = [p[j] for j in range(0,i)] + [q[j] for j in range(0,i-1)]
			# optimizer = minimize(err_func, params)#, method = 'nelder-mead') #
			popt, pcov = curve_fit(Calc_all, X, Y, p0=tuple(params))
			# popt = optimizer.x
			# mse = optimizer.fun #
			mse = err_func(*popt, True)
			# bic = mse
			bic = calculate_bic(len(Y), mse, i)
			print('------------ order', i, list(popt), mse, bic)#, getEqn(popt))
			if bic < pre_bic:
				# print('chosen order', pre_order, preopt)
				pre_bic = bic
				preopt = popt
				pre_order = i

	else:
		for i in range(1, 7):
			# i = 3
			params = [1.0, i]
			# optimizer = minimize(err_func, params)#, method = 'nelder-mead') #
			popt, pcov = curve_fit(Calc_all, X, Y, p0=tuple(params))
			# # popt = optimizer.x
			# # mse = optimizer.fun #
			mse = err_func(*popt, True)
			# # bic = mse
			bic = calculate_bic(len(Y), mse, 2)
			print('------------ order', functions[i-1], list(popt), mse, bic)#, getEqn(popt))
			if bic < pre_bic:
				# print('chosen order', pre_order, preopt)
				pre_bic = bic
				preopt = popt
				pre_order = i
	print('chosen order', pre_order, preopt)	
	popt = preopt
	#if PLOT:

	f = plt.figure()
	# plt.plot(x, ynoise, '.', label='data')
	X_sort = sorted(X)
	Y_sort = Calc_all(X_sort, *popt)
	eqn = getEqn(popt, nr = RF)
	print('regression:', eqn[0], eqn[1])
	if RF:
		plt.plot([x+1 for x in X], Y, 'b.', label='locii of boxes')
		plt.plot([x+1 for x in X_sort], Y_sort, 'r-', label='fit')
	else:
		plt.plot([x for x in X], Y, 'b.', label='locii of boxes')
		plt.plot(X_sort, Y_sort, 'g--', label='fit')
		# plt.plot(X_sort, [popt[0]/x for x in X_sort], 'k--', label='fit')
	plt.xlabel(names[0])
	plt.ylabel(names[1])
	plt.title(eqn[1])
	# print([popt[0]/x for x in X_sort], Y_sort)
	#else:
	#	f = None
	#	print(x, y, rational3_3(x, *popt))
	#
	plt.legend() 
	# plt.show()
	return popt, eqn, f


def plot_sat_boxes(sbox_all, sat_boxes, par_names, ratio=100, npg = [], cluster_centers = [], gen = 0, unsat_box = []):
	figs = []
	n_par = len(par_names)  
	contract = {}
	for i in range(n_par):
		contract.update({par_names[i]:i})
	b_map = sbox_all.get_map()
	param_bounds_all = []
	for it in sorted(par_names):
		pb = [b_map[it].leftBound(), b_map[it].rightBound()]
		param_bounds_all.append(pb)

	eqns = []

	if len(par_names) >= 3:
		subsets_3 = findsubsets(par_names, 3)
		for sub in subsets_3: 
			par = [contract[i] for i in sub]
			axisname = list(sub)
			fig1 = plt.figure()
			plt.title('3D project')# {0}%'.format(ratio))
			ax = fig1.add_subplot(111, projection="3d")
			ax.set_xlabel(axisname[0])
			ax.set_ylabel(axisname[1])
			ax.set_zlabel(axisname[2])
			xlim, ylim, zlim = param_bounds_all[par[0]], param_bounds_all[par[1]], param_bounds_all[par[2]]
			x1 = []
			x2 = []
			x3 = []
			for satb in sat_boxes:
				x = []
				w = []
				p1, p2, p3 = sub
				sat = satb.get_map()
				x11 = (sat[p1].mid().leftBound())
				x12 = (sat[p2].mid().leftBound())
				x13 = (sat[p3].mid().leftBound())
				x1.append(x11)
				x2.append(x12)
				x3.append(x13)
			
			ax.plot3D(x1, x2, x3, 'b.')

			x1 = []
			x2 = []
			x3 = []
			for satb in npg:
				x = []
				w = []
				p1, p2, p3 = sub
				sat = satb.get_map()
				x11 = (sat[p1].mid().leftBound())
				x12 = (sat[p2].mid().leftBound())
				x13 = (sat[p3].mid().leftBound())
				x1.append(x11)
				x2.append(x12)
				x3.append(x13)

			if gen%2 ==0:
				ax.plot3D(x1, x2, x3, 'r.')
			else:
				ax.plot3D(x1, x2, x3, 'm.')
			x1 = []
			x2 = []
			x3 = []
			for satb in cluster_centers:
				p1, p2, p3 = sub
				sat = satb.get_map()
				x11 = (sat[p1].mid().leftBound())
				x12 = (sat[p2].mid().leftBound())
				x13 = (sat[p3].mid().leftBound())
				x1.append(x11)
				x2.append(x12)
				x3.append(x13)

			ax.plot3D(x1, x2, x3, 'g.')
			# ax.set_xlim(xlim)     
			# ax.set_ylim(ylim)     
			# ax.set_zlim(zlim)
			figs.append(fig1)

	if len(par_names) >= 2:
		param_len = 2 
		# for all_params in dependent:
		#   # param_len = len(all_params.keys())
		subsets_2 = findsubsets(par_names, param_len)
		for sub in subsets_2: #for each subset of size 2 
			# print('For param set', sub, type(sub))
			par = [contract[i] for i in sub]

			xlim, ylim = param_bounds_all[par[0]], param_bounds_all[par[1]]
			x = []
			w = []
			axisname = list(sub)
			for p in par:
				x.append(param_bounds_all[p][0])
				w.append(param_bounds_all[p][1] - param_bounds_all[p][0])
			
			fig_t = plt.figure()
			plt.title('2D project '+str(axisname) )#+ '{0}%'.format(ratio))
			# xlim = [1.0*x[0], 1.0*(x[0] + w[0])]
			# ylim = [1.0*x[1], 1.0*(x[1] + w[1])]
			# print(xlim, ylim)
			plt.xlabel(axisname[0])
			plt.ylabel(axisname[1])
			# plt.ylim()
			currentAxis = plt.gca()
			currentAxis.set_xlim(xlim)
			currentAxis.set_ylim(ylim)
			
			x1 = []
			x10 = []
			x2 = []
			for satb in sat_boxes:
				p1, p2 = sub
				
				sat = satb.get_map()
				x11 = (sat[p1].mid().leftBound())
				x12 = (sat[p2].mid().leftBound())
				# x13 = (sat[p3].mid().leftBound())
				x10.append(x11)
				x1.append(x11)
				x2.append(x12)
				#currentAxis.add_patch(Rectangle((x[0], x[1]), w[0], w[1], facecolor='grey', alpha=1))
			
			currentAxis.plot(x10, x2, 'b.')

			xs1 = [x-1 for x in x10] if RF else  [x for x in x10] 
			xs2 = [x for x in x2]
			popt, eq, rat_f = rational_cf(xs1, xs2, [axisname[0], axisname[1]])
			eqns.append(eq[1])
			figs.append(rat_f)

			if len(par_names) == 2:
				x1 = []
				x10 = []
				x2 = []
				for satb in npg:
					p1, p2 = sub
					
					sat = satb.get_map()
					x11 = (sat[p1].mid().leftBound())
					x12 = (sat[p2].mid().leftBound())
					# x13 = (sat[p3].mid().leftBound())
					x10.append(x11)
					x1.append(x11)
					x2.append(x12)
					xx = []
					ww = []
					xx.append(sat[p1].leftBound())
					xx.append(sat[p2].leftBound())
					ww.append(sat[p1].width())
					ww.append(sat[p2].width())
					#currentAxis.add_patch(Rectangle((xx[0], xx[1]), ww[0], ww[1], facecolor='grey', alpha=1))
					plotBox(currentAxis, satb, sub, 'red') if gen%2==0 else plotBox(currentAxis, satb, sub, 'magenta') 
					#print('NPG boxes', ww)
				
				#if gen%2 ==0:
				#	currentAxis.plot(x10, x2, 'r.')
				#else:
				#	currentAxis.plot(x10, x2, 'm.')

				x1 = []
				x10 = []
				x2 = []
				for satb in cluster_centers:
					p1, p2 = sub
					
					sat = satb.get_map()
					x11 = (sat[p1].mid().leftBound())
					x12 = (sat[p2].mid().leftBound())
					# x13 = (sat[p3].mid().leftBound())
					x10.append(x11)
					x1.append(x11)
					x2.append(x12)

				currentAxis.plot(x10, x2,  'g.')
			x1 = []
			x10 = []
			x2 = []
			for satb in unsat_box:
				p1, p2 = sub
				sat = satb.get_map()
				x = (sat[p1].leftBound(), sat[p2].leftBound())
				w = (sat[p1].width(), sat[p2].width())
				# x13 = (sat[p3].mid().leftBound())
				# x10.append(x11)
				# x1.append(x11)
				# x2.append(x12)
				currentAxis.add_patch(Rectangle((x[0], x[1]), w[0], w[1], facecolor='grey', alpha=1))
			figs.append(fig_t)
			figs.append(rat_f)

	if len(par_names) > 2:
		# param_len = 2 
		subsets_n1 = findsubsets(par_names, len(par_names)-1)
		for sub in subsets_n1: 
			extra_par = ''
			for p in par_names:
				if p not in sub:
					extra_par = p
					break
			extra_par_id = contract[extra_par]
			extra_par_bounds = param_bounds_all[extra_par_id] #param_bounds_all[extra_par_id]
			lb , ub = extra_par_bounds[0], extra_par_bounds[1]
			extra_par_slices = np.linspace(lb, ub, int((ub - lb)/0.01))
			# print(extra_par, 'extra_par_slices', extra_par_slices)

			par = [contract[i] for i in sub]
			xlims = [param_bounds_all[par[i]] for i in range(len(sub))]
			axisname = list(sub)
			max_slice = (0, [], 0)
			for j in range(1, len(extra_par_slices)):
				extra_l, extra_u = extra_par_slices[j-1], extra_par_slices[j]
				# xlim, ylim, zlim = param_bounds_all[par[0]], param_bounds_all[par[1]], param_bounds_all[par[2]]
				xs = [[] for i in sub]
				# flag = False
				k = 0
				xs_1 = []
				for satb in sat_boxes:
					sat = satb.get_map()
					# x12 = (sat[p2].mid().leftBound())
					# x13 = (sat[p3].mid().leftBound())
					if extra_l <= sat[extra_par].leftBound() < extra_u and extra_l < sat[extra_par].rightBound() <= extra_u:
						for i in range(len(sub)):
							p = sub[i]
							x11 = (sat[p].mid().leftBound())
							xs[i].append(x11)
						# flag = True
						k += 1
						xs_1.append(satb)
				k = bfact.get_cover(xs_1, check = False).volume() if len(xs_1) > 0 else 0
				if k > max_slice[0]:
					max_slice = (k, xs, j)
			
			k, xs, j = max_slice
			#print(k, j, len(xs))
			fig = plt.figure()
			plt.title('slice '+extra_par+' -- max slice') # {0}, {1} -- {2}%'.format(j, k, ratio))
			if len(sub) == 3 and len(xs) > 2:
				plt.title('3D slice '+extra_par+' -- max slice') # {0}, {1} -- {2}%'.format(j, k, ratio))
				# print('plot 3 dim')
				ax = fig.add_subplot(111, projection="3d")
				ax.set_xlabel(axisname[0])
				ax.set_ylabel(axisname[1])
				ax.set_zlabel(axisname[2])
				ax.plot3D(xs[0], xs[1], xs[2], 'b.')
				# print('[')
				# for i in range(k):
				# 	print('[{0}, {1}, {2}]'.format(xs[0][i], xs[1][i], xs[2][i]))
				# print(']')

			elif len(sub) == 2 and len(xs) > 1:
				plt.title('2D slice '+extra_par+' -- max slice') # {0}, {1} -- {2}%'.format(j, k, ratio))
				# print('plot 2 dim')
				ax = plt.gca() #fig.add_subplot(111) #, projection="3d")
				ax.set_xlabel(axisname[0])
				ax.set_ylabel(axisname[1])
				# ax.set_zlabel(axisname[2])
				ax.plot(xs[0], xs[1], 'b.')
				ax.set_xlim(xlims[0])       
				ax.set_ylim(xlims[1])
				xs1 = [xp for xp in xs[0]] #if RF else  [x for x in x10] 
				xs2 = [xp for xp in xs[1]]
				popt, eq, rat_f = rational_cf(xs1, xs2, [axisname[0], axisname[1]])
				eqns.append(eq[1])
				figs.append(rat_f)
			figs.append(fig)

		subsets_n1 = findsubsets(par_names, 2)
		for sub in subsets_n1: 
			extra_pars = []
			for p in par_names:
				if p not in sub:
					extra_pars.append(p)
					# break
			# print('extra_pars', extra_pars)
			# extra_par_id = [contract[ep] for ep in extra_pars]
			# extra_par_bounds = [param_bounds[epd] for epd in extra_par_id] #param_bounds_all[extra_par_id]
			extra_par_slices = {}
			for ep in extra_pars:
				ep_bound = param_bounds_all[contract[ep]]
				lb , ub = ep_bound[0], ep_bound[1]
				extra_par_slices[ep] = np.linspace(lb, ub, int((ub - lb)/0.01))
				# print(ep, 'extra_par_slices', len(extra_par_slices[ep]))

			par = [contract[i] for i in sub]
			xlims = [param_bounds_all[par[i]] for i in range(len(sub))]
			axisname = list(sub)
			slices_list = []
			for ep in extra_pars: 
				ep_slice = list(range(1, len(extra_par_slices[ep])))
				# print(ep, ep_slice)
				slices_list.append(ep_slice)
			
			max_slice = (0, [], 0)
			sl = 0
			for element in itertools.product(*slices_list):
				# print(element)
				xs = [[] for i in sub]
				# flag = False
				k = 0
				for satb in sat_boxes:
					sat = satb.get_map()
					ep_fl = False
					fp = 0
					for p in range(len(extra_pars)): 
						ep = extra_pars[p]
						# if sl < 1 and k < 3:
						# print(sl, element, ep, element[p], len(extra_par_slices[ep]))
						el, eu = extra_par_slices[ep][element[p]-1], extra_par_slices[ep][element[p]]
						
						if el <= sat[ep].leftBound() < eu and el < sat[ep].rightBound() <= eu:
							fp += 1
					if fp == len(extra_pars):
						for i in range(len(sub)):
							p = sub[i]
							x11 = (sat[p].mid().leftBound())
							# x11 = (sat[p][0]+sat[p][1])/2
							xs[i].append(x11)
						k += 1
				# print(k, xs, sl)
				
				if k > max_slice[0]:
					max_slice = (k, xs, sl)
				sl += 1
			
			k, xs, j = max_slice
			fig = plt.figure()
			plt.title('3 project '+str(extra_pars)+' -- max slice {0} -- {1}'.format(j, ratio))
			# print(len(xs), xs)
			# print(xs[0], xs[1])
			if len(xs) > 1:
			# print('plot 2 dim')
				ax = plt.gca()#fig.add_subplot(111) #, projection="3d")
				ax.set_xlabel(axisname[0])
				ax.set_ylabel(axisname[1])
				# ax.set_zlabel(axisname[2])
				ax.plot(xs[0], xs[1], 'b.')
				ax.set_xlim(xlims[0])       
				ax.set_ylim(xlims[1])

				xs1 = [xp for xp in xs[0]] #if RF else  [x for x in x10] 
				xs2 = [xp for xp in xs[1]]
				popt, eq, rat_f = rational_cf(xs1, xs2, [axisname[0], axisname[1]])
				eqns.append(eq[1])
				figs.append(rat_f)
				
			figs.append(fig)
	return figs, eqns

def readFromFile(all_sat_file, all_q_file):
	queue = []
	sat_box = []
	pnames = sorted(par_names)
	with open(all_sat_file, 'r') as fp_sat:
		csv_reader = csv.reader(fp_sat, delimiter=',')
		for row in csv_reader:
			b_edges = {}
			j = 0
			# p1 = row[0]
			# p2 = row[1]
			tf = int(row[-1])
			if tf == 1:
				for l in range(0, 2*len(pnames), 2):
					l1 = float(row[l]) #.split(';')[0])
					l2 = float(row[l+1]) #.split(';')[1])
					it = pnames[int(l/2)]  
					b_edges.update({it:PyInterval(l1, l2)})
					j += 1
				dbox = Box(b_edges) 
				sat_box.append(dbox)

	with open(all_q_file, 'r') as fp_q:
		csv_reader = csv.reader(fp_q, delimiter=',')
		for row in csv_reader:
			b_edges = {}
			j = 0
			lst = 1
			# p1 = row[0]
			# p2 = row[1]
			for l in range(lst, lst+len(pnames)):
				l1 = float(row[l].split(';')[0])
				l2 = float(row[l].split(';')[1])
				it = pnames[int(l-lst)]   
				b_edges.update({it:PyInterval(l1, l2)})
				j +=1
			
			# print(row[0], l, row[l], j)
			j = lst+len(pnames)
			dbox = Box(b_edges) 
			b_pri = int(row[j]) 
			b_del = float(row[j+1])
			b_tp = int(row[j+2])
			b_ttyp = int(row[j+3])
			# sat_box.append(dbox)
			queue.append(BoxInfo(b_pri, dbox, b_del, b_tp, b_ttyp))
	return (sat_box, queue, pnames)

def writeToFile(all_sat_file, all_q_file, sat_box, st_q, par_names, unsat_box = []):
	print('--- writeToFile -- ', 'sat', len(sat_box), 'q', st_q.size(), 'unsat', len(unsat_box), 'par', par_names)
	pnames = ','.join(par_names)
	with open(all_sat_file, 'w+') as fp_sat:
		csv_writer = csv.writer(fp_sat, delimiter=',')
		for sb in sat_box:
			row = [] #pnames]
			sb_edegs = sb.get_map()
			for it in sorted(sb_edegs.keys()):
				l1 = sb_edegs[it].leftBound()
				l2 = sb_edegs[it].rightBound()
				# s = '{0},{1}'.format(l1, l2)
				row.append(l1)
				row.append(l2)
			row.append(1)
			csv_writer.writerow(row)
		for sb in unsat_box:
			row = [] #pnames]
			sb_edegs = sb.get_map()
			for it in sorted(sb_edegs.keys()):
				l1 = sb_edegs[it].leftBound()
				l2 = sb_edegs[it].rightBound()
				# s = '{0},{1}'.format(l1, l2)
				# row.append(s)
				row.append(l1)
				row.append(l2)
			row.append(0)
			csv_writer.writerow(row)

	with open(all_q_file, 'w+') as fp_q:
		csv_writer = csv.writer(fp_q, delimiter=',')
		# for binfo in st_q:
		while(not st_q.isEmpty()):
			binfo = st_q.pop()
			b_pri, dbox, b_del, b_tp, b_ttyp = binfo.getInfo()
			row = [pnames] #'{0}'.format(datatp)]
			dbox_edges = dbox.get_map()
			for it in sorted(dbox_edges.keys()):
				l1 = dbox_edges[it].leftBound()
				l2 = dbox_edges[it].rightBound()
				s = '{0};{1}'.format(l1, l2)
				row.append(s)
			row.append('{0}'.format(b_pri))
			row.append('{0}'.format(b_del))
			row.append('{0}'.format(b_tp))
			row.append('{0}'.format(b_ttyp))
			
			csv_writer.writerow(row)
