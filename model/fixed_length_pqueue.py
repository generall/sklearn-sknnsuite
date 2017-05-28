import itertools
import heapq


class FixedLengthPQueue(object):

    def __init__(self):
        self.pq = []                         # list of entries arranged in a heap
        self.entry_finder = {}               # mapping of tasks to entries
        self.counter = itertools.count()     # unique sequence count

    def add_task(self, task, priority=0):
        """Add a new task or update the priority of an existing task"""
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