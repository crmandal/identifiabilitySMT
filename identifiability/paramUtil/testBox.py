from interval import *
from box import *
from box_factory import *

#import sys
#from ctypes import *
#lib = cdll.LoadLibrary('test.so')
#print(lib)
it = PyInterval(0.5, 1.2)
print('Interval', str(it))
mid = it.mid()
print('Interval mid', mid)
print('Interval contains mid', it.contains(mid))
print('checking plus operation', it+mid)

it1 = PyInterval(1.2, 1.4)
it2 = PyInterval(0.3, 0.5)
it3 = PyInterval(1.5, 2.0)

it_m1 = it.merge(it1)
print('merged 1', str(it_m1))

it_m2 = it.merge(it2)
print('merged 2', str(it_m2))

it_m1_2 = it_m1.merge(it2)
print('merged 1_2', str(it_m1_2))

it_m3 = it.merge(it3)
print('merged 3', str(it_m3))

itc = it_m1.complement(it)
print('complement : ', str(itc))

itc1 = it_m1_2.complement(it)
print('complement : ', str(it_m1_2), str(it))
for i in itc1:
	print(str(i))

it_4 = it1.union(it3)
print('union : ', str(it1), str(it3))
for i in it_4:
	print(str(i))

exit()

k = PyInterval(0.94, 0.99) 
r = PyInterval(0.1, 0.35)
g = PyInterval(9.79, 9.87)
edges = {}
edges.update({'k': k})
edges.update({'r': r})
edges.update({'g': g})
box = Box(edges)

print('Box 1 ', box)

k1 = PyInterval(0.95, 0.96) 
r1 = PyInterval(0.12, 0.3)
g1 = PyInterval(9.8, 9.81)
edges1 = {}
edges1.update({'k': k1})
edges1.update({'r': r1})
edges1.update({'g': g1})
box1 = Box(edges1)

print('Box 2 ', box)

b = boxIntersection(box, box1)
print('boxIntersection', str(box), str(box1), '=', str(b))

p = cornerInside(box1, box)
print('corner :'+str(p))

b1_edges = {}
b1_edges.update({'k': k})
b1_edges.update({'r': r})
b1 = Box(b1_edges)

b2_edges = {}
b2_edges.update({'k': k1})
b2_edges.update({'r': r1})
b2 = Box(b2_edges)

boxes = remove(b1, b2)
print('Remove: box1 = ', str(b1), 'box2 = ', str(b2))
for it in boxes:
	print(str(it))
	
	
b1_edges = {}
b1_edges.update({'k': k})
b1_edges.update({'r': r})
b1_edges.update({'g': g})
b1 = Box(b1_edges)

b2_edges = {}
b2_edges.update({'k': k1})
b2_edges.update({'r': r1})
b2_edges.update({'g': g1})
b2 = Box(b2_edges)

#boxes = remove(b1, b2)
#print('Remove: box1 = ', str(b1), 'box2 = ', str(b2))
#for it in boxes:
#	print(str(it))

#b = boxIntersection(b1, b2)
#print('boxIntersection', str(b))

kb = PyInterval(0.94, 0.940391)
rb = PyInterval(0.148936, 0.155131)
gb = PyInterval(9.79, 9.79031)
b1_edges = {}
b1_edges.update({'k': kb})
b1_edges.update({'r': rb})
b1_edges.update({'g': gb})
b1 = Box(b1_edges)

kr = PyInterval(0.94, 0.940391)
rr = PyInterval(0.148936, 0.150485)
gr = PyInterval(9.79, 9.79031)
b2_edges = {}
b2_edges.update({'k': kr})
b2_edges.update({'r': rr})
b2_edges.update({'g': gr})
b2 = Box(b2_edges)
EPS = 0.002
emap = {}
b_map = b1.get_map()
for it in b_map.keys():		
	emap.update({it: PyInterval(EPS)})

print('######################################')
print('Remove: \nbox1 = '+ str(b1) + '\nbox2 = ' +str(b2))
boxes = remove(b1, b2, emap)
for it in boxes:
	print(str(it))

r2 = PyInterval(0.151649, 0.151649)
g2 = PyInterval(9.79031, 9.79031)
b2_edges = {}
b2_edges.update({'radius': r2})
b2_edges.update({'g': g2})
b2 = Box(b2_edges)

# radius : [0.15114, 0.153425]; g : [9.79031, 9.79035]
r1 = PyInterval(0.15114, 0.153425)
g1 = PyInterval(9.790314, 9.79035)
b1_edges = {}
b1_edges.update({'radius': r1})
b1_edges.update({'g': g1})
b1 = Box(b1_edges)

print('b1 contains b2: ', b1.contains(b2))
# print(sys.float_info)
