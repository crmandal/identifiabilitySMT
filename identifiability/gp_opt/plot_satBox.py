from math import *
import numpy as np
import csv, math
import random as rnd
import copy, os, sys

# import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from scipy.integrate import odeint
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paramUtil.interval import *
from paramUtil.box import *
import paramUtil.box_factory as bfact
import plotUtil as plt

import itertools

sat_csv = 'examples/review_data/ex_1_sat_box_npg_ex1_7hrs_80_151_6167.csv' 
pdfName = 'examples/review_data/gen/ex_1_sat_box_npg_ex1_7hrs_80_151_6167.pdf'
params = [0.7, 0.4, 1.0, 0.7]
param_names = ['k01', 'k02', 'k12', 'k21']
param_bounds_all = [[0.1,3], [0.1, 3], [0.1, 3], [0.1, 3]]
df = 1

sat_csv = 'examples/to_upload/ex2/ex22/ex_1_sat_box_npg_ex1_7hrs_80_151_6167.csv' 
pdfName = 'examples/to_upload/ex2/ex22/ex_1_sat_box_npg_ex1_7hrs_80_151_6167.pdf'
params = [0.7, 0.4, 1.0, 0.7]
param_names = ['k01', 'k02', 'k12', 'k21']
param_bounds_all = [[0.1,3], [0.1, 3], [0.1, 3], [0.1, 3]]
df = 1

# sat_csv = 'examples/review_data/pc_satbox_grad_no_slice_pc_up.csv' 
# pdfName = 'examples/review_data/gen/pc_satbox_grad_no_slice_pc_g.pdf'
# params = [1.0, 1.0]
# param_names = ['k41', 'k42']
# param_bounds = [[0.0, 2.0], [0.0, 2.0]]
# param_bounds_all = [[0.0, 2.0], [0.0, 2.0]]

# sat_csv = 'examples/review_data/th_satbox_grad_no_slice_th_ivt_13_902_1300 2.csv' 
# pdfName = 'examples/review_data/gen/th_plotBox_th_ivt_13_902_1300.pdf'

# sat_csv = 'examples/review_data/th_sat_box_gpr_th_465_1hr.csv' 
# pdfName = 'examples/review_data/gen/th_plotBox_th_gpr_1_465.pdf'

# sat_csv = 'examples/review_data/th_satbox_grad_no_slice_th_ivt_13_902_1300.csv' 
# pdfName = 'examples/review_data/gen/th_plotBox_th_ivt_13_902_1300.pdf'

# sat_csv = 'examples/review_data/th_sat_box_npg_th_ivt_8hrs_132.csv' 
# pdfName = 'examples/review_data/gen/th_sat_box_npg_th_ivt_8hrs_132.pdf'
# params = [2, 0.118]
# param_names = ['c', 'k34']
# param_bounds = [[0.1, 2.0], [0.1, 2.0]]
# param_bounds_all = [[0.1, 2.0], [0.1, 2.0]]

#sat_csv = 'examples/review_data/bbDrag_sat_box_npg_bb_12hrs_142_205.csv'
#pdfName = 'examples/review_data/gen/bbDrag_acr_205_12hr.pdf'
sat_csv = 'examples/to_upload/bb/bbDrag_satbox_acr_16_11.csv'
pdfName = 'examples/to_upload/bb/bbDrag_acr_16_11_plotSatbox.pdf'
params = [0.3, 0.47, 1.22]
param_names = ['a', 'c',  'r']
param_bounds = [[0.03, 3.0], [0.1, 3.0], [0.1,3.0]]
param_bounds_all = [[0.03, 3.0], [0.1, 3.0], [0.1,3.0]]
df = 2

# sat_csv = 'examples/bb_1/bbDrag_sat_box_npg_bb_ac 2.csv'
# pdfName = 'examples/review_data/gen/bbDrag_ac.pdf'
# params = [0.3, 0.47]
# param_names = ['a', 'c']
# param_bounds = [[0.03, 3.0], [0.1, 3.0]]
# param_bounds_all = [[0.03, 3.0], [0.1, 3.0]]

# sat_csv = 'examples/bb_2/bbDrag_sat_box_npg_bb.csv'
# pdfName = 'examples/review_data/gen/bbDrag_ar.pdf'
# params = [0.3, 1.22]
## param_names = ['a',  r'$\rho$']
# param_names = ['a',  'r']
# param_bounds = [[0.03, 3.0], [0.1,3.0]]
# param_bounds_all = [[0.03, 3.0],  [0.1,3.0]]


# sat_csv = 'examples/bb_3/bbDrag_sat_box_npg_bb.csv'
# pdfName = 'examples/review_data/gen/bbDrag_cr.pdf'
# params = [0.47, 1.22]
# param_names = ['c',  r'$\rho$']
# param_bounds = [ [0.1, 3.0], [0.1,3.0]]
# param_bounds_all = [ [0.1, 3.0], [0.1,3.0]]

n_par = len(params)	
contract = {}
for i in range(n_par):
	contract.update({param_names[i]:i})

sat_boxes = []
unsat_boxes = []
sat_count = 0
unsat_count = 0
with open(sat_csv, mode= 'r') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		#print(len(row), row)
		if len(row) > 2*len(param_names):
			UNSAT = True
		if row[0].startswith("d"):
			continue
		else:
			instances = {}
			j = 0
			while j < len(param_names):
				#print(j, 2*j, 2*j+1, len(row), len(row)/2)
				instances.update({param_names[j]:PyInterval(float(row[2*j]), float(row[2*j+1]))})
				j += 1
			bb = Box(instances)
			if UNSAT:
				if int(row[-1]) == 1:
					sat_boxes.append(bb)
					sat_count += 1
				else:
					unsat_boxes.append(bb)
					unsat_count += 1
			else:
					sat_boxes.append(bb)
					sat_count += 1
			# print('current sat',  instances)

#spamwriter.writerow(xr)
envelope = bfact.get_cover([x for x in sat_boxes], check = False)

print('envelope', envelope, unsat_count, 'sat', sat_count, 'total', sat_count+unsat_count)

en_edges = envelope.get_map()
param_bounds = []
smap = {}
j = 0
for it in param_names:
	param_bounds.append([en_edges[it].leftBound(), en_edges[it].rightBound()])
	smap.update({it:PyInterval(param_bounds_all[j][0], param_bounds_all[j][1])})
	j += 1
sbox = Box(smap)

figs, eqns = plt.plot_sat_boxes(sbox, sat_boxes, param_names, unsat_box = unsat_boxes, df = df) 	

pp = PdfPages(pdfName)
for fig  in figs:
	pp.savefig(fig)
pp.close()

exit()
def f(y0, t, p):
	# q = y0
	x1 = y0[0]
	x2 = y0[1]

	# 'k01', 'k02', 'k12', 'k21', 'V'
	k01 = p[0]
	k02 = p[1]
	k12 = p[2]
	k21 = p[3]
	v = p[4]

	# Auxillary equations
	qdot = np.zeros(2)
	
	#  ODEs
	qdot[0] = k12*x2 - (k01+k21)*x1
	qdot[1] = k21*x1 - (k02+k12)*x2
	#  ODE vector
	return qdot

def run(params, time):

	samples = 5000
	tspan = np.linspace(0, time, samples) 

	k01 = params[0]
	k02 = params[1]
	k12 = params[2]
	k21 = params[3]
	v = params[4]
	ic = [15, 0]

	soln = odeint(f, ic, tspan, args = (params, ))
	
	x1 = soln[:, 0]
	x2 = soln[:, 1]

	y = x1/v
	return [y, x2], tspan

data_csv = 'test/ex_1.csv'
data_times = []
data_vals = []
max_time = 10.0
with open(data_csv, mode= 'r') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		if row[0].startswith("k"):
			continue
		else:
			data_times.append(float(row[0]))
			data_vals.append(float(row[1]))
			if max_time < float(row[0]):
				max_time = float(row[0])

max_time = math.ceil(max_time)

cal_vals, cal_times = run(one_instance, max_time)

fig = plt.figure()
plt.plot(data_times, data_vals, 'ro')
plt.plot(cal_times, cal_vals[0], 'b-')

pp.savefig(fig)

pp.close()
