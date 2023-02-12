from numbers import Number
from collections import namedtuple
from Point import *
from Box import *

class Interval() : #namedtuple('IntervalBase', ['begin', 'end', 'data'])):
    __slots__ = ()  # Saves memory, avoiding the need to create __dict__ for each interval

    '''
     lower and upper can be 1-D or K-D values or Point Object,
     it can be float values or tuple/list or Point object
    '''
    def __init__(self, lower, upper, data=None, dim=1):
        self._dim = dim
        if isinstance(lower, Point):
            self._lower = lower
            self._dim = lower.dimension()

            if self._dim != dim:
                print('Incorrect combination of dimensions')
                return
        else:
            left = {}
            if isinstance(lower, list) or isinstance(lower, tuple):
                for i in range(len(lower)):
                    left.update({i:lower[i]})
            else:
                left.update({0:lower})
            point = Point(left)
            self._lower = point
            self._dim = point.dimension()
            if self._dim != dim:
                print('Incorrect combination of dimensions')
                return

        if isinstance(upper, Point):
            self._upper = upper
            if self._dim != upper.dimension():
                print('Incorrect combination of dimensions')
                return
        else:
            right = {}
            if isinstance(upper, list) or isinstance(upper, tuple):
                if self._dim != len(upper):
                    print('Incorrect combination of dimension')
                    return
                for i in range(len(upper)):
                    right.update({i:upper[i]})
            else:
                 if self._dim > 1:
                    print('Incorrect combination of dimension')
                    return
                right.update({0:upper})
            point = Point(right)
            self._upper = point
            # self._dim = point.dimension()

        if self.is_empty():
            left = {}
            right = {}
            for i in range(self._dim):
                left.update({i:-inf})
                right.update({i:inf})
            self._lower = Point(left)
            self._upper = Point(right)
    
    def overlaps(self, lower, upper=None):
        """
         Checks if interval overlaps the given k-d point, range or Box.
        :param lower: starting point of the range, or the point, or an Interval
        :param upper: upper limit of the range. Optional if not testing ranges.
        :return: True or False
        :rtype: bool
        """
        if upper is not None:
            ''' Range given by (begin, end)
            begin and end can be tuple, list or a single value
            '''
            if len(upper) != self._dim or len(lower) != self._dim:
                print('Incorrect combination of dimensions')
                return False
            else:
                for i in range(self._dim):
                    if lower[i] < self._upper[i] and upper[i] > self._lower[i]:
                        continue
                    else:
                        return False
                return True
        elif isinstance(lower, Point) :
            return self.contains_point(begin)
        elif isinstance(lower, KDInterval):
            edges = lower.get_map()
            ranges = []
            i = 0
            for key in edges.keys():
                ranges.append((edges[key].left, edges[key].right))
                i += 1
            for i in range(self._dim):
                if ranges[i][0] < self._upper[i] and ranges[i][1] > self._lower[i]:
                    continue
                else:
                    return False
            return True
        else:
            print('Incorrect combination of dimension')
            return
            
            # return self.overlaps(begin.begin, begin.end)
        

    # def overlap_size(self, begin, end=None):
    #     """
    #     Return the overlap size between two intervals or a point
    #     :param begin: beginning point of the range, or the point, or an Interval
    #     :param end: end point of the range. Optional if not testing ranges.
    #     :return: Return the overlap size, None if not overlap is found
    #     :rtype: depends on the given input (e.g., int will be returned for int interval and timedelta for
    #     datetime intervals)
    #     """
    #     overlaps = self.overlaps(begin, end)
    #     if not overlaps:
    #         return 0

    #     if end is not None:
    #         # case end is given
    #         i0 = max(self.begin, begin)
    #         i1 = min(self.end, end)
    #         return i1 - i0
    #     # assume the type is interval, in other cases, an exception will be thrown
    #     i0 = max(self.begin, begin.begin)
    #     i1 = min(self.end, begin.end)
    #     return i1 - i0

    def contains_point(self, p):
        """
        Whether the Interval contains point p.
        :param p: a point
        :return: True or False
        :rtype: bool
        """
        count = 0
        edges = p.get_map()
        ranges = []
        i = 0
        for key in edges.keys():
            ranges.append(edges[key].left) #, edges[key].right))
            i += 1
        for i in range(self._dim):
            if self._lower[i] <= ranges[i][0] < self._upper[i]:
                count += 1
        if count == self._dim:
            return True
        else:
            return False
        # return self.begin <= p < self.end
    
    def range_matches(self, other):
        """
        Whether the begins equal and the ends equal. Compare __eq__().
        :param other: Interval
        :return: True or False
        :rtype: bool
        """
        # return (
        #     self.begin == other.begin and 
        #     self.end == other.end
        # )
        count = 0
        edges1 = other.get_map()
        ranges = []
        i = 0
        for key in edges.keys():
            ranges.append((edges[key].left, edges[key].right))
            i += 1
        for i in range(self._dim):
            if ranges[i][0] == self._lower[i] and ranges[i][1] == self._upper[i]:
                count += 1
        if count == self._dim:
            return True
        else:
            return False
       
    
    def contains_interval(self, other):
        """
        Whether other is contained in this Interval.
        :param other: Interval
        :return: True or False
        :rtype: bool
        """
        count = 0
        edges = other.get_map()
        ranges = []
        i = 0
        for key in edges.keys():
            ranges.append((edges[key].left, edges[key].right))
            i += 1
        for i in range(self._dim):
            if ranges[i][0] >= self._lower[i] and ranges[i][1] <= self._upper[i]:
                count += 1
        if count == self._dim:
            return True
        else:
            return False
       
    
        # return (
        #     self.begin <= other.begin and
        #     self.end >= other.end
        # )
    
    # def distance_to(self, other):
    #     """
    #     Returns the size of the gap between intervals, or 0 
    #     if they touch or overlap.
    #     :param other: Interval or point
    #     :return: distance
    #     :rtype: Number
    #     """
    #     if self.overlaps(other):
    #         return 0
    #     try:
    #         if self.begin < other.begin:
    #             return other.begin - self.end
    #         else:
    #             return self.begin - other.end
    #     except:
    #         if self.end <= other:
    #             return other - self.end
    #         else:
    #             return self.begin - other

    def is_null(self):
        """
        Whether this equals the null interval.
        :return: True if end <= begin else False
        :rtype: bool
        """
        # count = 0
        for i in range(self._dim):
            if ranges[i][0] > ranges[i][1]:
                return True
            # else:
            #     count += 1
        # if count == self._dim:
        return False

        # return self.begin >= self.end

    def __eq__(self, other):
        """
        Whether the begins equal, the ends equal, and the data fields
        equal. Compare range_matches().
        :param other: Interval
        :return: True or False
        :rtype: bool
        """
        for i in range(self._dim):
            if not(other._lower[i] == self._lower[i] and other._lower[i] == self._upper[i]):
                return False
        # if count == self._dim:
        return True
        
    def __lt__(self, other):
        """
        Less than operator. Parrots __cmp__()
        :param other: Interval or point
        :return: True or False
        :rtype: bool
        """
        return self.__cmp__(other) < 0

    def __gt__(self, other):
        """
        Greater than operator. Parrots __cmp__()
        :param other: Interval or point
        :return: True or False
        :rtype: bool
        """
        return self.__cmp__(other) > 0

    def _raise_if_null(self, other):
        """
        :raises ValueError: if either self or other is a null Interval
        """
        if self.is_null():
            raise ValueError("Cannot compare null Intervals!")
        if hasattr(other, 'is_null') and other.is_null():
            raise ValueError("Cannot compare null Intervals!")

    def lt(self, other):
        """
        Strictly less than. Returns True if no part of this Interval
        extends higher than or into other.
        :raises ValueError: if either self or other is a null Interval
        :param other: Interval or point
        :return: True or False
        :rtype: bool
        """
        self._raise_if_null(other)
        return self.end <= getattr(other, 'begin', other)

    def le(self, other):
        """
        Less than or overlaps. Returns True if no part of this Interval
        extends higher than other.
        :raises ValueError: if either self or other is a null Interval
        :param other: Interval or point
        :return: True or False
        :rtype: bool
        """
        self._raise_if_null(other)
        return self.end <= getattr(other, 'end', other)

    def gt(self, other):
        """
        Strictly greater than. Returns True if no part of this Interval
        extends lower than or into other.
        :raises ValueError: if either self or other is a null Interval
        :param other: Interval or point
        :return: True or False
        :rtype: bool
        """
        self._raise_if_null(other)
        if hasattr(other, 'end'):
            return self.begin >= other.end
        else:
            return self.begin > other

    def ge(self, other):
        """
        Greater than or overlaps. Returns True if no part of this Interval
        extends lower than other.
        :raises ValueError: if either self or other is a null Interval
        :param other: Interval or point
        :return: True or False
        :rtype: bool
        """
        self._raise_if_null(other)
        return self.begin >= getattr(other, 'begin', other)

    def _get_fields(self):
        """
        Used by str, unicode, repr and __reduce__.

        Returns only the fields necessary to reconstruct the Interval.
        :return: reconstruction info
        :rtype: tuple
        """
        if self.data is not None:
            return self.begin, self.end, self.data
        else:
            return self.begin, self.end
    
    def __repr__(self):
        """
        Executable string representation of this Interval.
        :return: string representation
        :rtype: str
        """
        if isinstance(self.begin, Number):
            s_begin = str(self.begin)
            s_end = str(self.end)
        else:
            s_begin = repr(self.begin)
            s_end = repr(self.end)
        if self.data is None:
            return "Interval({0}, {1})".format(s_begin, s_end)
        else:
            return "Interval({0}, {1}, {2})".format(s_begin, s_end, repr(self.data))

    __str__ = __repr__

    def copy(self):
        """
        Shallow copy.
        :return: copy of self
        :rtype: Interval
        """
        return Interval(self.begin, self.end, self.data)
    
    def __reduce__(self):
        """
        For pickle-ing.
        :return: pickle data
        :rtype: tuple
        """
        return Interval, self._get_fields()
