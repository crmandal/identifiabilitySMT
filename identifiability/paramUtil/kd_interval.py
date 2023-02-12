# from __future__ import division

import sys, os
from collections import namedtuple
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from paramUtil.interval import *
from paramUtil.box import *
# from paramUtil.Point import *
# Inf = 10000000000000000
from paramUtil.const import *
# Inf = _PInf
DEBUG = False

class KDPoint():
    # __slots__ = ()  # Saves memory, avoiding the need to create __dict__ for each point
    """docstring for ClassName"""
    def __init__(self, values=[]) : #, infinite = 0, dim = 0):
        if DEBUG:
            print('KDPoint-init', values, len(values))
        # if infinite == 0:
        self._values = values
        self._dim = len(values)
        # for i in range(self._dim):
        #     self._values.append(values[i])
        # self._values = self.values
        # self._dim = self.dim
        # print(self._values)


        # print('KDPoint', self._values, self._dim)
    
    def clone(self):
        # if infinite == 0:
        vals = [v for v in self._values]
        return KDPoint(vals)

    @staticmethod
    def extremeLeft(dim):
        point = KDPoint()
        point._values = [-Inf for i in range(dim)]
        point._dim = dim
        return point

    @staticmethod
    def extremeRight(dim):
        point = KDPoint()
        point._values = [Inf for i in range(dim)]
        point._dim = dim
        return point

    @property
    def V(self):
        return self._values

    @property
    def dimension(self):
        return self._dim

    @property
    def isEmpty(self):
        return len(self._values)==0

    def __getitem__(self, i):
        if len(self.V) > i:
            return self.V[i]
        return None

    def __repr__(self):
        s = ''
        if self.dimension > 1:
            s += '('

        for i in range(self.dimension):
            if i == 0:
                if isinstance(self[i], float):
                    #s+= '{0:1.2e}'.format(self[i])
                    s+= '{0}'.format(self[i])
                else:                    
                    s+= '{0}'.format(self[i])
            else:
                if isinstance(self[i], float):
                    #s+= ', {0:1.2e}'.format(self[i])
                    s+= ', {0}'.format(self[i])
                else:                    
                    s+= ', {0}'.format(self[i])
        if self.dimension > 1:
            s += ')'
        return s

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other): 
        #checking if dimensions of the points are the same
        if isinstance(other, KDPoint): 
            if not(self.dimension == other.dimension):
                print('Dimension not same', self, other)
                raise ValueError()
            else:          
                left = self.V
                right = other.V
                for i in range(self.dimension):
                    if(left[i] == right[i]):
                        continue
                    else:
                        return False
                return True
        elif isinstance(other, tuple)  or isinstance(other, list):
            if not(self.dimension == len(other)):
                print('Dimension not same')
                raise ValueError()
            else:
                left = self.V
                right = other
                for i in range(self.dimension):
                    if(left[i] == right[i]):
                        continue
                    else:
                        return False
                return True   
        else:
            return NotImplemented

    def __le__(self, other): 
        # (x1, x2, ..., xn) <= (y1, y2, ..., yn) <=> [x1 <= y1 || (x1 == y1 && (x2, ..., xn) <= (y2, ..., yn))]
        #checking if dimensions of the points are the same
        if isinstance(other, KDPoint): 
            if not(self.dimension == other.dimension):
                print('Dimension not same')
                raise ValueError()
            else:          
                left = self.V
                right = other.V
                for i in range(self.dimension):
                    if(left[i] <= right[i]):
                        # continue
                        return True
                    else:
                        return False
                return False

        elif isinstance(other, tuple)  or isinstance(other, list):
            if not(self.dimension == len(other)):
                print('Dimension not same')
                raise ValueError()
            else:
                left = self.V
                right = other
                for i in range(self.dimension):
                    if(left[i] <= right[i]):
                        # continue
                        return True
                    else:
                        return False
                return False   
        else:
            return NotImplemented
    
    def __ge__(self, other): 
        # (x1, x2, ..., xn) >= (y1, y2, ..., yn) <=> [x1 >= y1 || (x1 == y1 && (x2, ..., xn) >= (y2, ..., yn))]
        #checking if dimensions of the points are the same
        if isinstance(other, KDPoint): 
            if not(self.dimension == other.dimension):
                print('Dimension not same')
                raise ValueError()
            else:          
                left = self.V
                right = other.V
                for i in range(self.dimension):
                    if(left[i] >= right[i]):
                        # continue
                        return True
                    else:
                        return False
                return False
        elif isinstance(other, tuple)  or isinstance(other, list):
            if not(self.dimension == len(other)):
                print('Dimension not same')
                raise ValueError()
            else:
                left = self.V
                right = other
                for i in range(self.dimension):
                    if(left[i] >= right[i]):
                        # continue
                        return True
                    else:
                        return False
                return False   
        else:
            return NotImplemented

    def __lt__(self, other): 
        # (x1, x2, ..., xn) < (y1, y2, ..., yn) <=> [x1 < y1 || (x1 == y1 && (x2, ..., xn) < (y2, ..., yn))]
        #checking if dimensions of the points are the same
        if isinstance(other, KDPoint): 
            if not(self.dimension == other.dimension):
                print('Dimension not same')
                raise ValueError()
            else:          
                left = self.V
                right = other.V
                for i in range(self.dimension):
                    if(left[i] < right[i]):
                        # continue
                        return True
                    elif (left[i] == right[i]):
                        continue
                    else:
                        return False
                return False
        elif isinstance(other, tuple)  or isinstance(other, list):
            if not(self.dimension == len(other)):
                print('Dimension not same')
                raise ValueError()
            else:
                left = self.V
                right = other
                for i in range(self.dimension):
                    if(left[i] < right[i]):
                        # continue
                        return True
                    elif (left[i] == right[i]):
                        continue
                    else:
                        return False
                return False   
        else:
            return NotImplemented

    def __gt__(self, other): 
        # (x1, x2, ..., xn) > (y1, y2, ..., yn) <=> [x1 > y1 || (x1 == y1 && (x2, ..., xn) > (y2, ..., yn))]
        #checking if dimensions of the points are the same
        if isinstance(other, KDPoint): 
            if not(self.dimension == other.dimension):
                print('Dimension not same')
                raise ValueError()
            else:          
                left = self.V
                right = other.V
                for i in range(self.dimension):
                    if(left[i] > right[i]):
                        # continue
                        return True
                    elif (left[i] == right[i]):
                        continue
                    else:
                        return False
                return False
        elif isinstance(other, tuple)  or isinstance(other, list):
            if not(self.dimension == len(other)):
                print('Dimension not same')
                raise ValueError()
            else:
                left = self.V
                right = other
                for i in range(self.dimension):
                    if(left[i] > right[i]):
                        # continue
                        return True
                    elif (left[i] == right[i]):
                        continue
                    else:
                        return False
                return False   
        else:
            return NotImplemented

    # strictly less than equal (comparable points)
    def le(self, other): 
        #print('le --- test')
        # (x1, x2, ..., xn) <= (y1, y2, ..., yn) <=> [x1 <= y1 && (x1 == y1 && (x2, ..., xn) <= (y2, ..., yn))]
        #checking if dimensions of the points are the same
        if isinstance(other, KDPoint): 
            if DEBUG:
                print('Dimension', self.dimension, other.dimension)
            if not(self.dimension == other.dimension):
                print('Dimension not same')
                raise ValueError()
            else:          
                left = self.V
                right = other.V

                # if DEBUG:
                    # print('le', left, right)
                for i in range(self.dimension):                    
                    if DEBUG:
                        print('le', left[i], right[i])
                    if(left[i] <= right[i]):
                        continue
                    else:
                        return False
                return True
        elif isinstance(other, tuple)  or isinstance(other, list):
            if not(self.dimension == len(other)):
                print('Dimension not same')
                raise ValueError()
            else:
                left = self.V
                right = other
                for i in range(self.dimension):
                    if(left[i] <= right[i]):
                        continue
                        # return True
                    else:
                        return False
                return True #True   
        else:
            return NotImplemented
    
    # strictly greater than equal (comparable points)
    def ge(self, other): 
        # (x1, x2, ..., xn) >= (y1, y2, ..., yn) <=> [x1 >= y1 && (x1 == y1 && (x2, ..., xn) >= (y2, ..., yn))]
        #checking if dimensions of the points are the same
        if isinstance(other, KDPoint): 
            if not(self.dimension == other.dimension):
                print('Dimension not same')
                raise ValueError()
            else:          
                left = self.V
                right = other.V
                # if DEBUG:
                #     print('ge', left, right)
                for i in range(self.dimension):
                    if DEBUG:
                        print('ge', i,  left[i], right[i])
                    if(left[i] >= right[i]):
                        continue
                        # return True
                    else:
                        return False
                return True
        elif isinstance(other, tuple)  or isinstance(other, list):
            if not(self.dimension == len(other)):
                print('Dimension not same')
                raise ValueError()
            else:
                left = self.V
                right = other
                for i in range(self.dimension):
                    if(left[i] >= right[i]):
                        continue
                        # return True
                    else:
                        return False
                return True   
        else:
            return NotImplemented


    # strictly less than (comparable points)
    def lt(self, other): 
        # (x1, x2, ..., xn) < (y1, y2, ..., yn) <=> [x1 < y1 && (x1 == y1 && (x2, ..., xn) < (y2, ..., yn))]
        #checking if dimensions of the points are the same
        if isinstance(other, KDPoint): 
            if not(self.dimension == other.dimension):
                print('Dimension not same')
                raise ValueError()
            else:          
                left = self.V
                right = other.V
                for i in range(self.dimension):
                    if(left[i] < right[i]):
                        continue
                        # return True
                        # elif (left[i] == right[i]):
                        # continue
                    else:
                        return False
                return True
        elif isinstance(other, tuple)  or isinstance(other, list):
            if not(self.dimension == len(other)):
                print('Dimension not same')
                raise ValueError()
            else:
                left = self.V
                right = other
                for i in range(self.dimension):
                    if(left[i] < right[i]):
                        continue
                        # return True
                    # elif (left[i] == right[i]):
                        # continue
                    else:
                        return False
                return True   
        else:
            return NotImplemented


    # strictly graeter than (comparable points)
    def gt(self, other): 
        # (x1, x2, ..., xn) > (y1, y2, ..., yn) <=> [x1 > y1 && (x1 == y1 && (x2, ..., xn) > (y2, ..., yn))]
        #checking if dimensions of the points are the same
        if isinstance(other, KDPoint): 
            if not(self.dimension == other.dimension):
                print('Dimension not same')
                raise ValueError()
            else:          
                left = self.V
                right = other.V
                for i in range(self.dimension):
                    if(left[i] > right[i]):
                        continue
                    #     return True
                    # elif (left[i] == right[i]):
                    #     continue
                    else:
                        return False
                return True
        elif isinstance(other, tuple)  or isinstance(other, list):
            if not(self.dimension == len(other)):
                print('Dimension not same')
                raise ValueError()
            else:
                left = self.V
                right = other
                for i in range(self.dimension):
                    if(left[i] > right[i]):
                        continue
                        # return True
                    # elif (left[i] == right[i]):
                        # continue
                    else:
                        return False
                return True   
        else:
            return NotImplemented

    def __cmp__(self, other):
        if self < other:
            return -1
        elif self > other:
            return 1
        else:
            return 0

# Atomic = namedtuple('Atomic', ['left', 'lower', 'upper', 'right'])

class Atomic():
    def __init__(self, lower = None, upper=None, data=None, dim=0):
        #self._dim = 0
        # print('params: ', lower, upper, data, dim)

        if not upper and lower:
            # input is a box
            if isinstance(lower, Box):
                # print('Box as interval')
                edges = lower.get_map()
                left = []
                right = []
                for key in sorted(edges.keys()):
                    it = edges[key]
                    left.append(it.leftBound())
                    right.append(it.rightBound())
                self._lower = KDPoint(left)
                self._upper = KDPoint(right)
                self._dim = lower.size()
                
                if dim and self._dim != dim:
                    print('1.Incorrect combination of dimensions: lower {0}, given {1}'.format(self._dim, dim))
                    raise ValueError()

                # print(edges, left, right, self._lower, self._upper)

        elif upper and lower:
            left = []
            # lower input as KDPoint
            if isinstance(lower, KDPoint):
                self._lower = lower
                self._dim = lower.dimension
            else:

                # lower input as list/tuple
                if isinstance(lower, list) or isinstance(lower, tuple):
                    left = lower
                else: 
                    # lower input as single value
                    left = [lower]
                self._lower = KDPoint(left)
                self._dim = len(left)
                if dim and self._dim != dim:
                    print('2.Incorrect combination of dimensions: lower {0}, given {1}'.format(self._dim, dim))
                    raise ValueError()

            right = []            
            # upper input as KDPoint
            if isinstance(upper, KDPoint):
                self._upper = upper
                if self._dim != upper.dimension:
                    print('3.Incorrect combination of dimension: lower {0}, upper {1}'.format(self._dim, upper.dimension))
                    raise ValueError()
            else:
                # upper input as list/tuple
                if isinstance(upper, list) or isinstance(upper, tuple):
                    if self._dim != len(upper):
                        print('4.Incorrect combination of dimension: lower {0}, upper {1}'.format(self._dim, len(upper)))
                        raise ValueError()
                    right = upper
                else:
                    # upper input as single value
                    if self._dim > 1:
                        print('5.Incorrect combination of dimension: lower {0}, upper {1}'.format(self._dim, 1))
                        raise ValueError()
                    right = [upper]
                self._upper = KDPoint(right)
            # print(left, right, self._lower, self._upper)
        # elif not upper and not lower:
        #     print('both lower and upper is none')
        #     self._lower = KDPoint()
        #     self._upper = KDPoint()
        #     self._dim = 0
        else:
            self._lower = KDPoint()
            self._upper = KDPoint()
            self._dim = 0

        if not self._lower.isEmpty and not self._upper.isEmpty and self._lower > self._upper:
            print('Error: Left boundary is greater than right boundary: lower {0}, upper {1}'.format(str(self._lower), str(self._upper)))
            self._lower = KDPoint()
            self._upper = KDPoint()
            self._dim = 0
        
    @property
    def dimension(self):
        return self._dim

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper
    
    @property
    def isEmpty(self):
        return self.lower.isEmpty
    
    def __repr__(self):
        os = '[' + str(self.lower) + ',' +  str(self.upper) +' ]'
        return os

    def __str__(self):      
        return self.__repr__()

    def adjacent(self, other):
        if not isinstance(other, Atomic):
            return NotImplemented

        if self.dimension != other.dimension:
            print('Dimension not same {0} , {1}'.format(self.dimension, other.dimension))
            raise ValueError()

        # for i in range(self.dimension):
        # if self.lower == other.upper or self.upper == other.lower:
        #     return True
        # else:
        #     return False
        lc = 0
        for i in range(self.dimension):
            if self.lower[i] == other.upper[i]:
                lc += 1
        if lc == self.dimension-1:
            return True
        rc = 0
        for i in range(self.dimension):
            if self.upper[i] == other.lower[i]:
                rc += 1
        if rc == self.dimension-1:
            return True
        # if self.lower == other.upper or self.upper == other.lower:
        #     return True
        # else:
        return False

    def overlaps(self, lower, upper=None):
        """
         Checks if interval overlaps the given k-d point, range or another interval.
        :param lower: starting point of the range, or the point, or an Interval
        :param upper: upper limit of the range. Optional if not testing ranges.
        :return: True or False
        :rtype: bool
        """
        if upper is not None:
            ''' Range given by (begin, end)
            begin and end can be tuple, list or a single value
            '''
            if len(upper) != self.dimension or len(lower) != self.dimension:
                print('6.Incorrect combination of dimensions: dimension {0}, lower {1}, upper {2}'.format(self._dim, len(lower), len(upper)))
                raise ValueError()
            else:
                # for i in range(self._dim):
                # check for full containment
                # if lower <= self.upper and upper >= self.lower:
                if lower.le(self.upper) and upper.ge(self.lower):
                    return True
                else:
                    return False                

        elif isinstance(lower, tuple)  or isinstance(lower, list) or isinstance(lower, KDPoint):
            return self.contains_point(lower)

        elif  isinstance(lower, Atomic):
            left = lower.lower
            right = lower.upper
            # for i in range(self._dim):
            # check for full containment
            # if left < self.upper and right > self.lower:
            if left.lt(self.upper) and right.gt(self.lower):
                return True
            else:
                return False
            
        else:
            if DEBUG:
                print('overlaps:', type(lower))
            print('7.Incorrect combination of dimension')
            return
            
            # return self.overlaps(begin.begin, begin.end)
    
    def mergeable(self, other):
        return self.adjacent(other) or self.overlaps(other) 

    def __and__(self, other):
        if not isinstance(other, Atomic):
            return NotImplemented
        if self.isEmpty or other.isEmpty:
            return Atomic()
        if self.dimension != other.dimension:

            print('Dimension not same {0} , {1}'.format(self.dimension, other.dimension), self, other)
            raise ValueError()
        left = []#self.lower.V
        right = [] #self.upper.V

        for i in range(self.dimension):
            if self.lower[i] < other.lower[i]:
                left.append(other.lower[i])
                #left[i] = other.lower[i]
            else:
                left.append(self.lower[i])
            if self.upper[i] > other.upper[i] and left[i] <= other.upper[i]:
                right.append(other.upper[i])
                # right[i] = other.upper[i]
            else:
                right.append(self.upper[i])
        if DEBUG:
            print('and', left, right)
        for i in range(self.dimension):
            if left[i] > right[i]:
              return Atomic()
        # if not left.le(right):
        return Atomic(left, right)

    def __or__(self, other):
        if not isinstance(other, Atomic):
            return NotImplemented

        if self.dimension != other.dimension:
            print('Dimension not same {0} , {1}'.format(self.dimension, other.dimension))
            raise ValueError()

        # if not self.mergable(other):
        #     print('Intervals cannot be merged')
        #     return None

        left = [] #self.lower
        right = [] #self.upper

        if self.mergeable(other):
            if self.lower == other.lower:
                lower = self.lower
            else:
                lower = min(self.lower, other.lower)

            if self.upper == other.upper:
                upper = self.upper
            else:
                upper = max(self.upper, other.upper)

            union = Atomic(lower, upper)
            return [union]
        else:
            return [self, other]

    def contains(self, other):
        return other in self

    def __invert__(self):
        complements = [Atomic(KDPoint.extremeLeft(self.dimension), self.lower), Atomic(self.upper, KDPoint.extremeRight(self.dimension))]
        return complements

    def __contains__(self, other):
        if DEBUG:
            print('Atomic--contains', self, other, type(other))
        if isinstance(other, Atomic):
            return self.containsInterval(other)
        elif isinstance(other, tuple) or isinstance(other, list) or isinstance(other, KDPoint):
            return self.containsPoint(other)
        else:
            if self._dim == 1 and isinstance(other, int) and isinstance(other, float) :
                oth = [other]
                return self.containsPoint(oth)
            else:
                print('The value is not of correct type: \'' + other + '\'')
                raise ValueError()

    ''' checks if an interval is fully contained in the interval 
    '''
    def containsInterval(self, other):

        if DEBUG:
            print('Atomic--containsInterval', self, other, type(other))
        left = self.lower
        right = self.upper
        other_left = other.lower
        other_right = other.upper
        # print('containsInterval', type(left), type(right), type(other_left), type(other_right))
        # for i in range(self.dimension):
        #     if other_left[i] >= left[i] and other_right[i] <= right[i]:
        #         continue
        #     else:
        #         return False
        # return True
        
        if other_left.ge(left) and other_right.le(right):
            return True
        else:
            return False
            
    ''' checks if a point (tuple/list) is fully contained in the interval 
        the point position in any of the dimension should be within the left and right value of that dimension of Box
    '''
    def containsPoint(self, point):

        if DEBUG:
            print('Atomic--containsPoint', self, point, type(point))
        left = self.lower
        right = self.upper
        if not isinstance(point, KDPoint):    
            point1 = KDPoint(point)
        else:
            point1 = point
        if DEBUG:
            print('Atomic--containsPoint-- update', point, point1, left, right)
        # check for full containment
        if left.le(point1) and point1.le(right):
            return True
        else:
            return False   

    def __le__(self, other):
        #checking if dimensions of the intervals are the same
        if isinstance(other, Atomic):  
            if not(self.dimension == other.dimension):
                print('Dimension not same')
                raise ValueError()
            else:      
                # left = self.lower
                right = self.upper
                # other_left = other.lower
                other_right = other.right

                # for i in range(self.dimension):
                # check for full containment
                # if(right <= other_right):
                if right.le(other_right):
                    return True
                else:
                    return False
            
        elif isinstance(other, tuple)  or isinstance(other, list) or isinstance(other, KDPoint):
            left = self.lower
            if isinstance(other, tuple)  or isinstance(other, list):
                other1 = KDPoint(other)
            else:
                other1 = other
            # for i in range(self.dimension):
            # check for full containment
            # if(left <= other):
            if left.le(other1):
                return True
            else:
                return False
            
        else:
            return NotImplemented

    def __ge__(self, other): 
        if isinstance(other, Atomic):  
            if not(self.dimension == other.dimension):
                print('Dimension not same')
                raise ValueError()
            else:      
                left = self.lower
                # right = self.upper
                other_left = other.lower
                # other_right = other.right

                # for i in range(self.dimension):
                # check for full containment
                # if(left >= other_left):
                if left.ge(other_left):
                    return True
                else:
                    return False
            
        elif isinstance(other, tuple)  or isinstance(other, list) or isinstance(other, KDPoint):
            right = self.upper
            if isinstance(other, tuple)  or isinstance(other, list):
                other1 = KDPoint(other)
            else:
                other1 = other
            # for i in range(self.dimension):
            # check for full containment
            # if(right >= other):
            if right.ge(other1):
                return True
            else:
                return False
            
        else:
            return NotImplemented

    def __lt__(self, other):
        #checking if dimensions of the intervals are the same
        if isinstance(other, Atomic):  
            if not(self.dimension == other.dimension):
                print('Dimension not same')
                raise ValueError()
            else:      
                # left = self.lower
                right = self.upper
                # other_left = other.lower
                other_right = other.right

                # for i in range(self.dimension):
                # check for full containment
                # if(right < other_right):
                if right.lt(other_right):
                    return True
                else:
                    return False
            
        elif isinstance(other, tuple)  or isinstance(other, list) or isinstance(other, KDPoint):
            left = self.lower
            if isinstance(other, tuple)  or isinstance(other, list):
                other1 = KDPoint(other)
            else:
                other1 = other
            # for i in range(self.dimension):
            # check for full containment
            # if(left < other):
            if left.lt(other1):
                return True
            else:
                return False
            
        else:
            return NotImplemented

    def __gt__(self, other): 
        if isinstance(other, Atomic):  
            if not(self.dimension == other.dimension):
                print('Dimension not same')
                raise ValueError()
            else:      
                left = self.lower
                # right = self.upper
                other_left = other.lower
                # other_right = other.right

                # for i in range(self.dimension):
                # check for full containment
                # if(left > other_left):
                if left.gt(other_left):
                    return True
                else:
                    return False
            
        elif isinstance(other, tuple)  or isinstance(other, list) or isinstance(other, KDPoint):
            right = self.upper
            if isinstance(other, tuple)  or isinstance(other, list):
                other1 = KDPoint(other)
            else:
                other1 = other
            # for i in range(self.dimension):
            # check for full containment
            # if(right > other):
            if right.gt(other1):
                return True
            else:
                return False
            
        else:
            return NotImplemented

    def __eq__(self, other): 
        #checking if dimensions of the intervals are the same
        if isinstance(other, Atomic): 
            if not(self.dimension == other.dimension):
                print('Dimension not same')
                raise ValueError()
            else:          
                left = self.lower
                right = self.upper
                other_left = other.lower
                other_right = other.right

                # for i in range(self.dimension):
                if(right == other_right and left == other_left):
                    return True
                else:
                    return False
            
        elif isinstance(other, tuple)  or isinstance(other, list) or isinstance(other, KDPoint):
            right = self.upper
            if isinstance(other, tuple)  or isinstance(other, list):
                other1 = KDPoint(other)
            else:
                other1 = other
            # for i in range(self.dimension):
            if(right == other):
                return True
            else:
                return False
        else:
            return NotImplemented


   
    
class KDInterval:
    # __slots__ = ()  # Saves memory, avoiding the need to create __dict__ for each interval

    '''
     lower and upper can be 1-D or K-D values or Box Object,
     it can be float values or tuple/list or Box object
    '''
    def __init__(self, *intervals):
        # if isinstance(intervals[0], Box):
        #     atomic  = Atomic(box)
        #     self._intervals.append(atomic)
        #     return 
        self._intervals = list()
        self._size = 0.0
        if len(intervals) == 2:
            if (isinstance(intervals[0], list) or isinstance(intervals[0], tuple) or isinstance(intervals[0], KDPoint) or isinstance(intervals[0], int) or isinstance(intervals[0], float))\
             and (isinstance(intervals[1], list) or isinstance(intervals[1], tuple) or isinstance(intervals[1], KDPoint) or isinstance(intervals[1], int) or isinstance(intervals[1], float)):
                self._intervals.append(Atomic(*intervals))
            else:
                for interval in intervals:
                    if isinstance(interval, KDInterval):
                        if not interval.isEmpty:
                            self._intervals.extend(interval._intervals)
                    elif isinstance(interval, Atomic):
                        self._intervals.append(interval)
                    elif isinstance(interval, Box):
                        self._intervals.append(Atomic(interval))
                    else:
                        raise TypeError('Parameters must be Interval instances')
        else:
            for interval in intervals:
                if isinstance(interval, KDInterval):
                    if not interval.isEmpty:
                        self._intervals.extend(interval._intervals)
                elif isinstance(interval, Atomic):
                    self._intervals.append(interval)
                elif isinstance(interval, Box):
                    self._intervals.append(Atomic(interval))
                else:
                    raise TypeError('Parameters must be Interval instances')

        if len(self._intervals) == 0:
            # So we have at least one (empty) interval
            self._intervals.append(Atomic())
        else:
            # Sort intervals by lower bound
            self._intervals.sort(key=lambda i: i.lower)

            i = 0
            # Try to merge consecutive intervals
            while i < len(self._intervals) - 1:
                current = self._intervals[i]
                successor = self._intervals[i + 1]

                if current.mergeable( successor):
                    if current.lower == successor.lower:
                        lower = current.lower
                    else:
                        lower = min(current.lower, successor.lower)

                    if current.upper == successor.upper:
                        upper = current.upper
                    else:
                        upper = max(current.upper, successor.upper)

                    union = Atomic(lower, upper)
                    self._intervals.pop(i)  # pop current
                    self._intervals.pop(i)  # pop successor
                    self._intervals.insert(i, union)
                else:
                    i = i + 1

    # def __init__(self, lower = None, upper=None, data=None, dim=0):
    #     #self._dim = 0
    #     # print('params: ', lower, upper, data, dim)

    #     if not upper and lower:
    #         if isinstance(lower, Box):
    #             print('Box as interval')
    #             edges = lower.get_map()
    #             left = []
    #             right = []
    #             for key in edges.keys():
    #                 it = edges[key]
    #                 left.append(it.leftBound())
    #                 right.append(it.rightBound())
    #             self._lower = KDPoint(left)
    #             self._upper = KDPoint(right)
    #             self._dim = lower.size()
                
    #             if dim and self._dim != dim:
    #                 print('Incorrect combination of dimensions: lower {0}, given {1}'.format(self._dim, dim))
    #                 raise ValueError()

    #             # print(edges, left, right, self._lower, self._upper)
    #     elif upper and lower:
    #         left = []
    #         if isinstance(lower, KDPoint):
    #             self._lower = lower
    #             self._dim = lower.dimension
    #         else:
    #             if isinstance(lower, list) or isinstance(lower, tuple):
    #                 left = lower
    #             else:
    #                 left = [lower]
    #             self._lower = KDPoint(left)
    #             self._dim = len(left)
    #             if dim and self._dim != dim:
    #                 print('Incorrect combination of dimensions: lower {0}, given {1}'.format(self._dim, dim))
    #                 raise ValueError()

    #         right = []
    #         if isinstance(upper, KDPoint):
    #             self._upper = upper
    #             if self._dim != upper.dimension:
    #                 print('Incorrect combination of dimension: lower {0}, upper {1}'.format(self._dim, upper.dimension))
    #                 raise ValueError()
    #         else:
    #             if isinstance(upper, list) or isinstance(upper, tuple):
    #                 if self._dim != len(upper):
    #                     print('Incorrect combination of dimension: lower {0}, upper {1}'.format(self._dim, len(upper)))
    #                     raise ValueError()
    #                 right = upper
    #             else:
    #                 if self._dim > 1:
    #                     print('Incorrect combination of dimension: lower {0}, upper {1}'.format(self._dim, 1))
    #                     raise ValueError()
    #                 right = [upper]
    #             self._upper = KDPoint(right)
    #         # print(left, right, self._lower, self._upper)
    #     # elif not upper and not lower:
    #     #     print('both lower and upper is none')
    #     #     self._lower = KDPoint()
    #     #     self._upper = KDPoint()
    #     #     self._dim = 0
    #     else:
    #         self._lower = KDPoint()
    #         self._upper = KDPoint()
    #         self._dim = 0

    #     if not self._lower.isEmpty and not self._upper.isEmpty and self._lower > self._upper:
    #         print('Error: Left boundary is greater than right boundary: lower {0}, upper {1}'.format(str(self._lower), str(self._upper)))
    #         self._lower = KDPoint()
    #         self._upper = KDPoint()
    #         self._dim = 0
        

    #     # if self.isEmpty():
    #     #     left = [-inf for i in range(self._dim) ]
    #     #     right = [inf for i in range(self._dim) ]
    #     #     self._lower = left
    #     #     self._upper = right
    
    @property
    def dimension(self):
        return self._intervals[0].dimension

    @property
    def lower(self):
        return self._intervals[0].lower

    @property
    def upper(self):
        return self._intervals[-1].upper
    
    def getRange_i(self, i):
        return (self.lower[i], self.upper[i])
           
    def __str__(self):      
        return self.__repr__()

    def __repr__(self):     
        intervals = []

        for interval in self._intervals:
            if interval.isEmpty:
                intervals.append('()')
            elif interval.lower == interval.upper:
                intervals.append('[{}]'.format(repr(interval.lower)))
            else:
                intervals.append('{}{},{}{}'.format('[', repr(interval.lower), repr(interval.upper),']',))
        return ' | '.join(intervals)

    '''returns true if box is empty'''
    @property
    def isEmpty(self):
        for i in self._intervals:
            if i.isEmpty:
                continue
            else:
                return False
        return True
        # return (self.lower > self.upper)
        # return self.lower.isEmpty
    @property
    def atomic(self):
        """
        True if this interval is atomic, False otherwise.
        An interval is atomic if it is composed of a single (possibly empty) atomic interval.
        """
        return len(self._intervals) == 1
    @property
    def size(self):
        if self._size == 0.0:
            s = 1.0
            for i in range(self.dimension):
                l, u = self.getRange_i(i)
                s *= u-l
            self._size = s
        return self._size
    
    @staticmethod
    def from_atomic(lower, upper):
        instance = KDInterval()
        instance._intervals = [Atomic(lower, upper)]

        if DEBUG:
            print('from_atomic', instance)
        if instance.isEmpty:
            instance = KDInterval()
            instance._intervals = []
        return instance

    @property
    def enclosure(self):
        return KDInterval.from_atomic(self.lower, self.upper)

    def adjacent(self, other):
        # print('KD_interval -- adjacent', self, other)
        sb = self.toBox()
        ob = other.toBox()
        # print('kd_interval--adjacent', self.lower,'==', other.upper, self.upper, '==', other.lower)
        # lc = 0
        return sb.adjacent(ob)
        # for i in range(self.dimension):
        #     if self.lower[i] == other.upper[i]:
        #         lc += 1
        # if lc == self.dimension-1:
        #     return True
        # rc = 0
        # for i in range(self.dimension):
        #     if self.upper[i] == other.lower[i]:
        #         rc += 1
        # if rc == self.dimension-1:
        #     return True
        # if self.lower == other.upper or self.upper == other.lower:
        #     return True
        # else:
        # return False
        #return (self & other).isEmpty and (self | other).atomic

    def overlaps(self, other):
        if not isinstance(other, KDInterval):
            raise TypeError('Unsupported type {} for {}'.format(type(other), other))
        
        for s_ai in self._intervals:
            for o_ai in other._intervals:
                if s_ai.overlaps(o_ai):
                    return True
        # else:
        return False 

    def intersection(self, other):
        return self & other

    def intersects(self, other):
       res = self.intersection(other)
       # print('intersects', res, res.isEmpty, ~res.isEmpty)
       return not res.isEmpty

    def union(self, other):
        return self | other

    def contains(self, item):
        return item in self

    def fullyContains(self, other, d = 0.0001):
        sb = self.toBox()
        ob = other.toBox()
        # print('kd_interval--adjacent', self.lower,'==', other.upper, self.upper, '==', other.lower)
        # lc = 0
        return sb.fullyContains(ob) or ob.fullyContains(sb)

    def complement(self):
        return ~self

    def difference(self, other):
        return self - other

    def mergable(self, other):
        if DEBUG:
            print('KD_interval--mergable', self.atomic, other.atomic, self.adjacent(other), self.overlaps(other))
        return self.atomic and other.atomic and (self.adjacent(other) or self.overlaps(other))

    def merge(self, other):
        if DEBUG:
            print('KD_interval--merge', self.atomic)
        if self.atomic and other.atomic:
            sb = self.toBox()
            ob = other.toBox()
            m  = sb.merge(ob)
            m_int = KDInterval(Atomic(m))
            return m_int
            



    def items(self):
        items = []
        for i in self._intervals:
            it = KDInterval.from_atomic(i.lower, i.upper)
            # print('items',i, it)
            items.append(it)
        # print('items',items)
        return items

    def __len__(self):
        return len(self._intervals)

    def __iter__(self):
        return iter(self.items())

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [KDInterval.from_atomic(i.lower, i.upper) for i in self._intervals[item]]
        else:
            it = self._intervals[item]
            return KDInterval.from_atomic(it.lower, it.upper)

    def __and__(self, other):
        if not isinstance(other, KDInterval):
            return NotImplemented
        # 
        if self.atomic and other.atomic:
            # print(self._intervals, other._intervals)
            ai = self._intervals[0] & other._intervals[0]
            lower = ai.lower
            upper = ai.upper
            return KDInterval.from_atomic(lower, upper)
        else:
            intersections = []

            for s_ai in self._intervals:
                for o_ai in other._intervals:
                    inter = s_ai & o_ai
                    if not inter.isEmpty:
                        intersections.append(inter)

            return KDInterval(*intersections)

    def __or__(self, other):
        if not isinstance(other, KDInterval):
            return NotImplemented

        if self.atomic and other.atomic:
            ai = self._intervals[0] | other._intervals[0]
            if len(ai) == 1:
                lower = ai[0].lower
                upper = ai[0].upper
                return KDInterval.from_atomic(lower, upper)
            else:
                return KDInterval(*ai)
        else:
            unions = []

            if len(self._intervals) > 0:
                for s_ai in self._intervals:
                    for o_ai in other._intervals:
                        inter = s_ai | o_ai
                        for ai in inter:                                
                            if not ai.isEmpty:
                                unions.append(ai)
                            
            else:
                for o_ai in other._intervals:
                    unions.append(o_ai)
            return KDInterval(*unions)     

    def  __contains__(self, item):
        if isinstance(item, KDInterval):
            for s_ai in self._intervals:
                for i_ai in item._intervals:
                    if s_ai.contains(i_ai):
                        return True
            return False
        else:
            for s_ai in self._intervals:
                if DEBUG:
                    print('KDInterval--contains', s_ai, item)
                if s_ai.contains(item):
                    return True

            return False

    def __invert__(self):
        # print('__invert__', self, self.dimension)
        p1 = KDPoint.extremeLeft(self.dimension)
        p2 = KDPoint.extremeRight(self.dimension)
        # print('__invert__', p1, p2) # KDPoint(infinite=-1, dim=self.dimension), self.lower)
        complements = [Atomic(p1, self.lower), Atomic(self.upper,p2)]

        for i, j in zip(self._intervals[:-1], self._intervals[1:]):
            complements.append(KDInterval.from_atomic(i.upper, j.lower))
        # print('__invert__', complements)
        return KDInterval(*complements)

    def __sub__(self, other):
        if not isinstance(other, KDInterval):
            return NotImplemented
        # print('__sub__', self, ~other)
        return self & ~other

    def __eq__(self, other):
        if not isinstance(other, KDInterval):            
            return NotImplemented

        if len(other._intervals) != len(self._intervals):
            return False
    
        for a,b in zip(self._intervals, other._intervals):
            if a==b:
                continue
            else:
                return False
        return True

    def __lt__(self, other):
        if isinstance(other, KDInterval):            
            return self.upper < other.lower
        else:
            return self.upper < other

    def __gt__(self, other):
        if isinstance(other, KDInterval):            
            return self.lower > other.upper
        else:
            return self.upper > other

    def __le__(self, other):
        if isinstance(other, KDInterval):            
            return self.upper <= other.lower
        else:
            return self.upper <= other

    def __ge__(self, other):
        if isinstance(other, KDInterval):            
            return self.lower >= other.upper
        else:
            return self.upper >= other

    def toBox(self, names=[]):
        lb = self.lower
        ub = self.upper
        if len(names) == 0:
            dim =lb.dimension
            names = ['d_{0}'.format(i) for i in range(dim)]
        b_map = {}
        for i in range(len(names)):
            nm = names[i]
            b_map.update({nm:PyInterval(lb.V[i], ub.V[i])})
        return Box(b_map)
         
        
