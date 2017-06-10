from unittest import TestCase
from sknn_suite import SkNNSuite


class TestSkNNSuite(TestCase):
    def test_fit(self):
        clf = SkNNSuite(k=1, distance_function=lambda x, y: abs(x - y))
        clf.fit(X=[
            [1, 100, 10, 11],
            [1, 50, 21, 20],
            [1, 2, 3, 4, 5]
        ], y=[
            ["l1", "l2", "l3", "l3"],
            ["l1", "l4", "l5", "l5"],
            ["l1", "l1", "l1", "l1", "l1"]
        ])

        prediction = clf.predict(x_targets=[
            [1, 70, 0, 0],
            [1, 80, 23, 22],
            [1, float('inf'), 23, 22]
        ])

        self.assertEqual(["l1", "l2", "l3", "l3"], prediction[0])
        self.assertEqual(["l1", "l4", "l5", "l5"], prediction[1])
        self.assertEqual(None, prediction[2])
