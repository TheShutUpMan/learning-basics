from math import log


class DecisionNode:
    def __init__(self, feature=None, classification=None):
        self.feature = feature
        self.children = []
        self.classification = classification

    def learn(self, data, classification):
        classified = True
        example = classification[0]
        for i in classification:
            if example != i:
                classified = False
                break
        if classified:
            self.classification = classification[0]
            return self

        def entropy(data):
            positive = sum(data)
            negative = (len(data) - positive) / len(data)
            positive /= len(data)
            if positive == 0 or negative == 0:
                return 0
            rval = -(positive * log(positive, 2)) - (
                negative * log(negative, 2))
            return rval

        totalEntropy = entropy(classification)

        def gain(feature):
            featureValues = list(set(feature))
            numberedFeature = [(n, i) for n, i in enumerate(feature)]
            subsetNumbers = [[n for n, i in numberedFeature if i == j]
                             for j in featureValues]
            subsets = [[x for i, x in enumerate(classification) if i in s]
                       for s in subsetNumbers]
            subsetE = sum(
                (len(i) / len(classification)) * entropy(i) for i in subsets)
            return (totalEntropy - subsetE, subsetNumbers, featureValues)

        gains = [gain(i) for i in data]
        testFeature = gains.index(max(gains))
        (subsets, featureValues) = gains[testFeature][1:]
        self.feature = testFeature
        for n, value in enumerate(featureValues):
            print(n, value)
            newdata = [[
                val for i, val in enumerate(feature) if i in subsets[n]
            ] for feature in data]
            newclass = [
                x for i, x in enumerate(classification) if i in subsets[n]
            ]
            child = DecisionNode()
            child.learn(newdata, newclass)
            self.children.append((value, child))
        return self

    def classify(self, vec):
        if self.classification is not None:
            return self.classification
        feature = vec[self.feature]
        child = [i[1] for i in self.children if i[0] == feature][0]
        return child.classify(vec)


node = DecisionNode()

data = [
    "FFMCCCMFFCFMMC", "UUUDLLLDLDDDUD", "HHHHIIIHIIIHIH", "HIHHHIIHHHIIHI",
    [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]
]

data2 = [
    "SSORRROSSRSOOR", "HHHMCCCMCMMMHM", "HHHHNNNNHHHNNH", "WSWWWSSWWWSSWS",
    [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
]

node.learn(data[:-1], data[-1])
