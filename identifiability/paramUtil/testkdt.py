
from kd_interval import *
from kd_intervaltree import *
def test_1D():

    print('############################################ testing 1D ############################################')
    intervals_text = [(9,9.5), (9,10.5), (10,12), (11, 16), (12.5, 13), (14, 15), (14.5, 18), (14.5, 19.5)]
    intervals = []
    for i in intervals_text:
        intervals.append(KDInterval(i[0], i[1]))

    print('intervals:', intervals)

    tree = IntervalTree()
    print('empty tree:', tree)

    for i in intervals:
        print('------------- adding interval {0} ----------------------------------'.format(i))
        tree = tree.insert(i)
        print('-----------------------------------------------')
        
        # print('tree after insertion ', tree)
    print('-----------------------------------------------')
    print('-----------------------------------------------')
    print(tree)    
    # print('-----------------------------------------------')
    # print(tree.preorder())

    search_test = [(11.5, 13), (17, 18), (15, 17)]
    searches = []
    for i in search_test:
        searches.append(KDInterval(i[0], i[1]))

    for i in searches:
        print('------------- searching interval {0} ----------------------------------'.format(i), tree)
        res = tree.search(i)
        print('----------------', res, '-------------------------------')

    points = [KDPoint([16]), KDPoint([20]), KDPoint([11])]
    for i in points:
        print('------------- searching point {0} ----------------------------------'.format(i), tree)
        res = tree.search(i)
        print('----------------', res, '-------------------------------')

    print(tree.items())

    box_u = [8, 20]
    res = tree.get_uncovered_regions(box_u)
    print('----------------result----------')
    print(res)

    print('----------------result iterator----------')
    for i in res:
        print(i)

    print('---------------- ----------')
    point = tree.generate_point_in_uncovered_region(box_u)
    print('###', point)

    point = tree.generate_point_in_uncovered_region(box_u)
    print('###', point)

    print('############################################ testing completed ############################################')

def test_2D():

    print('############################################ testing 2D ############################################')
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

    boxes = [box, box1, box3, box4]
    intervals = []
    for b in boxes:
        # atomic = Atomic(b)
        ki = KDInterval(b)
        print(str(b), str(ki))
        intervals.append(ki)

    N = 5
    # intervals = []
    for i in range(N):
        k = [np.random.uniform(0.5, 0.95), np.random.uniform(0.5, 0.95)]
        r = [np.random.uniform(0.1, 0.3), np.random.uniform(0.1, 0.3)]
        lower = [np.min(k), np.min(r)]
        upper = [np.max(k), np.max(r)]
        interval = KDInterval(lower, upper)
        print('interval ', interval)
        intervals.append(interval)
    
    intervals = [[(0.94, 0.10),(0.99, 0.35)], [(0.95, 0.45),(0.96, 0.50)], [(0.10, 0.40),(0.30, 0.50)], [(1.00, 0.05),(1.20, 0.10)], [(0.59, 0.21),(0.77, 0.26)], [(0.74, 0.10),(0.92, 0.15)], [(0.51, 0.18),(0.65, 0.29)], [(0.51, 0.18),(0.65, 0.21)], [(0.68, 0.16),(0.74, 0.16)], [(0.70, 0.24),(0.77, 0.25)]]

    print(intervals)

    tree = IntervalTree()
    print('empty tree:', tree)

    for j in intervals:
        i_low = j[0]
        i_upp = j[1]
        interval = KDInterval(i_low, i_upp)
        print('------------- adding interval {0} ----------------------------------'.format(interval))
        tree = tree.insert(interval)
        print('-----------------------------------------------')
        
        # print('tree after insertion ', tree)
    print('-----------------------------------------------')
    print('-----------------------------------------------')
    print(tree)    

    search_test = [[(0.95, 0.47),(0.96, 0.48)], [(0.7, 0.12),(0.72, 0.16)], [(0.4, 0.05),(0.42, 0.07)]]
    searches = []
    for i in search_test:
        interval = KDInterval(i[0], i[1])
        # for i in searches:
        print('------------- searching interval {0} ----------------------------------'.format(interval), tree)
        res = tree.search(interval)
        print('----------------', res, '-------------------------------')

    points = [KDPoint([0.75, 0.22]), KDPoint([0.9, 0.1]), KDPoint([0.75, 0.2]), KDPoint([0.4, 0.17])]
    for i in points:
        print('------------- searching point {0} ----------------------------------'.format(i), tree)
        res = tree.search(i)
        print('----------------', res, '-------------------------------')

    print(tree.items())

    box_u = [(0.5, 0.01), (1.3, 0.25)]
    res = tree.get_uncovered_regions(box_u)
    print('----------------result----------')
    print(res)

    print('----------------result iterator----------')
    for i in res:
        print(i)

    print('---------------- ----------')
    points = tree.generate_point_in_uncovered_region(box_u)
    print('###', len(points), points[0])

    print('---------------- ----------')
    tree = tree.insert([0.75, 0.2])
    print('############## added point #############')
    print(tree)
    res = tree.search([0.75, 0.2])
    print(res, tree)

    # point = tree.generate_point_in_uncovered_region(box_u)
    # print('###', point)
    # print('############################### testing completed ######################')


def test():
    test_1D()
    test_2D()


test()