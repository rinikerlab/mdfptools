class BasePredictor():
    """Null model
    """
    pass
class SolutionPredictor(BasePredictor):
    pass
class LiquidPredictor(BasePredictor):
    pass

from sklearn.tree import _tree
from sklearn.externals import joblib
import pickle
rfr  = joblib.load('/home/shuwang/SHAREPOINT/jonas/charge_models/F.model')
max(rfr.estimators_[0].tree_.feature)
max(rfr.estimators_[1].tree_.feature)
max(rfr.estimators_[11].tree_.feature)
max(rfr.estimators_[20].tree_.feature)
[max(i.tree_.feature) for i in rfr.estimators_]
rfr.estimators_[20].__dict__
x = rfr.estimators_[20].tree_
x.feature[10:20]
[i == _tree.TREE_UNDEFINED for i in x.feature]
rfr.estimators_[1].__dict__

def tree_to_code(tree, tree_number = 0, feature_names = ["var{}".format(i) for i in range(2048)]):
    """https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree"""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree{}({}):".format(tree_number, ", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))
    recurse(0,1)

print(tree_to_code(rfr.estimators_[0]))
tree_ = tree.tree_
len(tree_.feature)

from sklearn import tree
tree.export_graphviz(rfr.estimators_[0], "/home/shuwang/sandbox/tmp")
tree
import numpy as np
rfr.estimators_[0].predict([[0.1 for i in range(2048)]])

rfr.estimators_[0].__dict__

pickle.dump([i.predict for i in rfr.estimators_], open("/home/shuwang/sandbox/tmp.pickle", "wb"))
joblib.dump([i.predict for i in rfr.estimators_], open("/home/shuwang/sandbox/tmp.pickle", "wb"), compress = 9)
