
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from intervalUtil.interval import *
from intervalUtil.box import *
from intervalUtil.kd_interval import *
import numpy as np

Inf = 10000000000000000

class IntervalType(object):
    """docstring for ClassName"""
    def __init__(self):
        self._leftPoint = None
        self._righPoints = []
        self._numInterval = 0
    @property
    def lP(self):
        return self._leftPoint
    @property
    def rP(self):
        return self._righPoints
    @property
    def nI(self):
        return self._numInterval

    def __repr__(self):
        s = 'lp: '+ str(self.lP) + ', rp: ' + str(self.rP) + ', nI: '+ str(self.nI)
        return s

    def __str__(self):
        return self.__repr__()

    def add(self, interval):
        if not isinstance(interval, KDInterval):
            return NotImplemented
        if self._leftPoint is None:
            self._leftPoint = interval.lower
            self._righPoints.append(interval.upper)
            self._numInterval += 1
            self._righPoints = sorted(self._righPoints)

        elif self._leftPoint == interval.lower:
            # self._leftPoint = interval.lower
            self._righPoints.append(interval.upper)
            self._numInterval += 1
            self._righPoints = sorted(self._righPoints)
        else:
            print('Interval not added', self.lP, interval.lower)
            return
        # print('Interval added', self, interval.lower)

    def remove(self, interval): 
        if not isinstance(interval, KDInterval):
            return NotImplemented      
        if self._leftPoint == interval.lower and interval.upper in self.rP:
            if self._numInterval - 1 == 0 :
                self._leftPoint = []
            self._righPoints.remove(interval.upper)
            self._numInterval -= 1

    def overlap(self, interval):
        if not isinstance(interval, IntervalType):
            return NotImplemented

        if self.rP[self.nI - 1] <= interval.lP or interval.rP[interval.nI - 1] <= self.lP:
            return False
        return True



class IntervalTree(object):
    """docstring for ClassName"""
    def __init__(self):
        self._intervalNode = IntervalType()
        self._rmax = KDPoint()
        self._height = 0
        self._leftchild = None
        self._rightchild = None
        self._empty = True
        
    @property
    def I(self):
        return self._intervalNode

    @property
    def rMax(self):
        return self._rmax
  
    @property
    def height(self):
        return self._height 

    @property
    def lC(self):
        return self._leftchild 

    @property
    def rC(self):
        return self._rightchild

    @property
    def isEmpty(self):
        return self._empty
    
    # def clone(self):
    #     copy = IntervalTree()
    #     copy._intervalNode.clone
    #     self._intervalNode = IntervalType()
    #     self._rmax = KDPoint()
    #     self._height = 0
    #     self._leftchild = None
    #     self._rightchild = None
    #     self._empty = True

    def inorder(self):
        if self is None:
            return ''
        s = ''
        if self.lC is not None:
            s += ' '+ self.lC.inorder()
        s += ' ['+str(self._intervalNode) +  ', rMax: '+ str(self.rMax) + ']'
        if self.rC is not None:
            s += ' '+self.rC.inorder()
        return s

    def preorder(self):
        if self is None:
            return ''
        s = ' ['+str(self._intervalNode) +  ', '+ str(self.rMax) + ']'
        if self.lC is not None:
            s += ' '+self.lC.inorder()
        if self.rC is not None:
            s += ' '+self.rC.inorder()
        return s

    def __repr__(self):
        s = self.inorder()
        return s

    def __str__(self):
        return self.__repr__()

    def leftRotate(self):
        y = self.rC
        t2 = y.lC

        # Perform rotation 
        y._leftchild = self
        z._rightchild = t2

        # Update heights
        self._height = 1 + max(self.lC.height if self.lC is not None else 0, self.rC.height if self.rC is not None else 0)
        y._height = 1 + max(y.lC.height if y.lC is not None else 0, y.rC.height if y.rC is not None else 0)

        # Return the new root
        self = y

    def rightRotate(self):
        y = self.lC 
        T3 = y.rC 
  
        # Perform rotation 
        y._rightchild = self 
        self._leftchild = T3 
  
        # Update heights       
        self._height = 1 + max(self.lC.height if self.lC is not None else 0, self.rC.height if self.rC is not None else 0)
        y._height = 1 + max(y.lC.height if y.lC is not None else 0, y.rC.height if y.rC is not None else 0)
  
        # Return the new root 
        self = y 

    def getBalance(self):
        if self is None or self.isEmpty: 
            return 0  
        return self.lC.height if self.lC is not None else 0 - self.rC.height if self.rC is not None else 0

    def avlbalance(self, interval):
        # get the balance factor
        balance = self.getBalance()

        # Step 4 - If the node is unbalanced,  
        # then try out the 4 cases 
        # Case 1 - Left Left 
        if balance > 1 and interval.upper < self.lC.rMax: 
            return self.rightRotate() 
  
        # Case 2 - Right Right 
        if balance < -1 and interval.upper > self.rC.rMax: 
            return self.leftRotate() 
  
        # Case 3 - Left Right 
        if balance > 1 and interval.upper > self.lC.rMax: 
            self.lC.leftRotate() 
            self.rightRotate() 
  
        # Case 4 - Right Left 
        if balance < -1 and interval.upper < self.rC.rMax: 
            self.rC.rightRotate() 
            self.leftRotate() 

    def insert(self, interval):
        # print('In tree insert: ', interval)
        if not isinstance(interval, KDInterval):
            return NotImplemented

        if self.isEmpty:
            self._intervalNode.add(interval)
            self._empty = False
        else:
            print('deciding on children ', self.rMax, interval.upper, self.rMax < interval.upper, self.rMax > interval.upper)
            if self.rMax < interval.upper: # current rMax is smaller than new interval, add to right child
                if self.rC is None:
                    self._rightchild = IntervalTree()
                self.rC.insert(interval) # recursive call to right           
            if self.rMax > interval.upper: # current rMax is larger than new interval, add to left child
                if self.lC is None:
                    self._leftchild = IntervalTree()
                self.lC.insert(interval) # recursive call to left

        # update height of the node     
        self._height = 1 + max(self.lC.height if self.lC is not None else 0 , self.rC.height if self.rC is not None else 0)

        # self.avlbalance(interval)  

        rmax1 = self.lC.rMax if self.lC is not None else [-Inf for i in range(interval.dimension)]
        rmax2 = self.rC.rMax if self.rC is not None else [-Inf for i in range(interval.dimension)]
        # print(self.I.rP[self.I.nI-1], rmax1, rmax2)
        self._rmax = np.max([self.I.rP[self.I.nI-1], rmax1, rmax2])
        # self._height = 1 + max(self.lC.height if self.lC is not None else 0 , self.rC.height if self.rC is not None else 0)



def test():
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
        intervals.append(KDInterval(b))
    print('intervals:', intervals)

    tree = IntervalTree()
    print('empty tree:', tree)

    for i in intervals:
        tree.insert(i)
        
    print('tree after insertion', tree)

test()
