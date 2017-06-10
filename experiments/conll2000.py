def load(fname):
    f = open(fname)

    res = []
    curr = []
    for line in f.readlines():
        line = line.strip()
        if line == "":
            res.append(curr)
            curr = []
        else:
            curr.append(line.split(' '))

    if len(curr) > 0:
        res.append(curr)

    return res


def split_labels(data):
    labels = []
    features = []
    for seq in data:
        l = []
        f = []
        for x in seq:
            l.append(x[-1])
            f.append(x[:-1])
        features.append(f)
        labels.append(l)
    return features, labels


def extend(sequence, n):
    seq_with_borders = [['BEGIN']] * n + sequence + [['END']] * n
    res = []
    for i in range(n, len(sequence) + n):
        extended_element = []
        for j in range(-n, n + 1):
            extended_element += seq_with_borders[i + j]
        res.append(extended_element)
    return res


data = load('/home/generall/data/CoNLL2000/train.txt')
features, labels = split_labels(data)
print(features[0])
print(extend(features[0], 1))
print(labels[0])
