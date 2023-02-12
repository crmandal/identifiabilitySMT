from interval import *
from KD_IntervalTree import *
from box import *
from box_factory import *


# k = PyInterval(0.94, 0.99) 
# r = PyInterval(0.1, 0.35)
# g = PyInterval(9.79, 9.87)
edges1 = {}
edges1.update({'k': PyInterval(0.94, 0.99) })
# edges1.update({'r': r})
# edges1.update({'g': g})
box1 = Box(edges1)
print('Box 1 ', box1)

# k1 = PyInterval(0.95, 0.96) 
# r1 = PyInterval(0.12, 0.3)
# g1 = PyInterval(9.8, 9.81)
edges2 = {}
edges2.update({'k': PyInterval(0.95, 0.96) })
# edges1.update({'r': r1})
# edges1.update({'g': g1})
box2 = Box(edges2)
print('Box 2 ', box2)

# k1 = PyInterval(0.95, 0.96) 
# r1 = PyInterval(0.12, 0.3)
# g1 = PyInterval(9.8, 9.81)
edges3 = {}
edges3.update({'k': PyInterval(0.1, 0.3) })
# edges1.update({'r': r1})
# edges1.update({'g': g1})
box3 = Box(edges3)
print('Box 3 ', box3)

# k1 = PyInterval(0.95, 0.96) 
# r1 = PyInterval(0.12, 0.3)
# g1 = PyInterval(9.8, 9.81)
edges4 = {}
edges4.update({'k': PyInterval(1.0, 1.15) })
# edges1.update({'r': r1})
# edges1.update({'g': g1})
box4 = Box(edges4)
print('Box 4 ', box4)

data = [box1, box2, box3, box4]

it = PyInterval(0.5, 1.2, box = False)
it.setData(data)
it1 = PyInterval(0.6, 1.1, box = False)
it1.setData(data)
t = IntervalTree('k', [it, it1])
print(t)
it2 = PyInterval(1.0, 2, box = False)
it2.setData(data)
it3 = PyInterval(0.3, 1.0, box = False)
it3.setData(data)
it4 = PyInterval(0.4, 1.5, box = False)
it4.setData(data)
t.add(it2)
t.add(it3)
t.add(it4)
# print(sorted(t[6]))
# print(sorted(t[2:4]))
print(t)

t.remove(it2)
print(t)
# t.remove(it2)
# print(t)

it2 = PyInterval(0.1, 0.4, box = False)
it2.setData(data)
t.add(it2)
print(t, t.len())
# t.remove_envelop((0.5, 1.2))
# print(t)
# t.remove_overlap((0.5, 1.2))
# print(t)
print('Splicing tree')
t.slice(1.0)
print(t, t.len())

print('SearchRange tree')
res=t.searchRange((0.5, 1.0))
print(res)

print('Shrink Tree')
res = t.shrink()
print(res)

print(' Tree')
res = t.shrink()
print(res)

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

boxes = [box, box1, box3]
kdt = KD_IntervalTree(3, boxes)
print(kdt)

# exit()
print('1.........')
res = kdt.contains(box1)
print('check--contains: ', box1, res)

k1 = PyInterval(0.95, 0.96) 
r1 = PyInterval(0.05, 0.1)
# g1 = PyInterval(9.8, 9.81)
edges2 = {}
edges2.update({'k': k1})
edges2.update({'r': r1})
# edges2.update({'g': g1})
box2 = Box(edges2)


print('2.........')
res = kdt.contains(box2)
print('check--contains: ', box2, res)

print('----------------------------------------------')

k1 = PyInterval(0.8, 1.5) 
r1 = PyInterval(0.01, 0.55)
# g1 = PyInterval(9.8, 9.81)
edges2 = {}
edges2.update({'k': k1})
edges2.update({'r': r1})
# edges2.update({'g': g1})
box_u = Box(edges2)

res = kdt.get_uncovered_regions(box_u)
print('----------------result----------')
print(res)

point = kdt.generate_point_in_uncovered_region(box_u)
print(point)

point = kdt.generate_point_in_uncovered_region(box_u)
print(point)


kdt = KD_IntervalTree(3, [])
print(kdt)
kdt.addBox(box4)
print(kdt)