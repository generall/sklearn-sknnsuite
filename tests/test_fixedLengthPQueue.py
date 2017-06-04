from unittest import TestCase
from model.fixed_length_pqueue import FixedLengthPQueue


class TestFixedLengthPQueue(TestCase):
    def test_add_task(self):
        queue = FixedLengthPQueue(3)
        queue.add_task("the task1", 1)
        queue.add_task("the task2", 2)
        queue.add_task("the task3", 0)
        self.assertEqual(len(queue.pq), 3)

    def test_pop_task(self):
        queue = FixedLengthPQueue(3)
        queue.add_task("the task1", 1)
        queue.add_task("the task2", 2)
        queue.add_task("the task3", 0)
        self.assertEqual(queue.pop_task(), "the task2")
