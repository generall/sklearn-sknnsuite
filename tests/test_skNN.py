from unittest import TestCase
from collections import namedtuple
from model.sknn_model import Model
from sknn_tagger import SkNN

Element = namedtuple('Element', ['label', 'data'])


class TestSkNN(TestCase):

    @staticmethod
    def create_element(label, data):
        return Element(label=label, data=data)

    def test_tag(self):
        m = Model(1, lambda x, y: abs(x.data - y.data))
        
        seq1 = [
            TestSkNN.create_element("l1", 1),
            TestSkNN.create_element("l2", 100),
            TestSkNN.create_element("l3", 10),
            TestSkNN.create_element("l3", 11)
        ]

        seq2 = [
            TestSkNN.create_element("l1", 1),
            TestSkNN.create_element("l4", 50),
            TestSkNN.create_element("l5", 21),
            TestSkNN.create_element("l5", 20)
        ]

        seq3 = [
            TestSkNN.create_element("l1", 1),
            TestSkNN.create_element("l1", 2),
            TestSkNN.create_element("l1", 3),
            TestSkNN.create_element("l1", 4),
            TestSkNN.create_element("l1", 5)
        ]

        test1 = [
            TestSkNN.create_element(None, 1),
            TestSkNN.create_element(None, 70),
            TestSkNN.create_element(None, 0),
            TestSkNN.create_element(None, 0)
        ]

        test2 = [
            TestSkNN.create_element(None, 1),
            TestSkNN.create_element(None, 80),
            TestSkNN.create_element(None, 23),
            TestSkNN.create_element(None, 22)
        ]

        test3 = [
            TestSkNN.create_element(None, 1),
            TestSkNN.create_element(None, float('inf')),
            TestSkNN.create_element(None, 23),
            TestSkNN.create_element(None, 22)
        ]

        m.process_sequence(seq1)
        m.process_sequence(seq2)
        m.process_sequence(seq3)

        self.assertEqual(len(m.nodes[Model.INIT_LABEL].storage), 1)

        tagger = SkNN(m)

        res1, score1 = tagger.tag(test1)
        res2, score2 = tagger.tag(test2)
        res3, score3 = tagger.tag(test3)

        self.assertEqual(res3, None)

        self.assertEqual(len(res1), 4)
        self.assertEqual(len(res2), 4)

        self.assertEqual(res1[0].label, "l1")
        self.assertEqual(res1[1].label, "l2")
        self.assertEqual(res1[2].label, "l3")
        self.assertEqual(res1[3].label, "l3")

        self.assertEqual(res2[0].label, "l1")
        self.assertEqual(res2[1].label, "l4")
        self.assertEqual(res2[2].label, "l5")
        self.assertEqual(res2[3].label, "l5")
