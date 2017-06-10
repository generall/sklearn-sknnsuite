# sklearn-SkNNSuite

Wrapper which provides scikit-learn-compatible implementation of SkNN sequence labeling algorithm.

SkNN is a metric algorithm for labeling\classification sequential data.
It accepts for classification any type of sequence elements with only condition: you need to define distance function which can calculate distance between pair of elements.

Detailed description of SkNN could be found at [arXiv](https://arxiv.org/abs/1610.04718).

# Installation

```
TODO
```

# Usage

```$python
from sknn_suite import SkNNSuite

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

```

# References

* SkNN detailed [description](https://arxiv.org/abs/1610.04718)
* Named Entity Linking with SkNN - [https://github.com/generall/SkNN-NER](https://github.com/generall/SkNN-NER)
* Ruby implementation of SkNN [https://github.com/generall/SkNN-ruby](https://github.com/generall/SkNN-ruby)