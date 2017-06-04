import itertools
import heapq


class FixedLengthPQueue(object):

    def __init__(self, length):
        """
        :param length: max length of queue. Should be greater then 0
        """
        self.pq = []                         # list of entries arranged in a heap
        self.entry_finder = {}               # mapping of tasks to entries
        self.counter = itertools.count()     # unique sequence count
        self.length = length

    def add_task(self, task, priority=0):
        """Add a new task or update the priority of an existing task"""
        if len(self.pq) == self.length:
            self.pop_task()
        count = next(self.counter)
        entry = [- priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.pq, entry)

    def pop_task(self):
        """Remove and return the lowest priority task. Raise KeyError if empty."""
        while self.pq:
            priority, count, task = heapq.heappop(self.pq)
            del self.entry_finder[task]
            return task
        raise KeyError('pop from an empty priority queue')