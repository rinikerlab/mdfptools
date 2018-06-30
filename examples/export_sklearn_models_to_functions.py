from sklearn.tree import _tree
from sklearn.externals import joblib
import sys

def tree_to_code(tree, tree_number = 0, feature_names = ["var{}".format(i) for i in range(2048)]):
    """https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree"""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("  def tree{}({}):".format(tree_number, "feature_list"))
    print("    " + ",".join(feature_names) + " = feature_list")
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
            print("{}return {}".format(indent, tree_.value[node][0][0]))
    recurse(0,2)

# import os
# import pickle
elements = ["O", "N", "C", "H", "F", "P", "S", "Cl", "Br", "I"]
with open("../mdfptools/DDecMLChargers.py", "w") as f:
    sys.stdout = f
    print("""
    Method from paper 10.1021/acs.jcim.7b00663
    Currently only considering epsilon = 78
    """)
    for i in elements:
        rfr  = joblib.load('../mdfptools/data/{}.model'.format(i))
        # print("import pickle")
        print("class Charger{}():".format(i))
        [tree_to_code(val, idx) for idx, val in enumerate(rfr.estimators_)]
        # print("pickle.dump([ {} ], open('./tmp.pickle', 'wb'))".format(",".join(["tree{}".format(i) for i in range(100) ])))
    # import tmp
    # pickle.dump([val for key,val in tmp.__dict__.items() if key.startswith("tree")] , open("./tmp.pickle", "wb"))
    # os.system("python tmp.py")

    # os.remove("./tmp.py")
