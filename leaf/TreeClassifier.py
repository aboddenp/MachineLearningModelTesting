import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import math

best_tree_model_validation = DecisionTreeClassifier('gini', 'best', 10, 2, 1, 0, 12, 0,
                                                    48)  # best acurracy obtained by a tree classifier in validation stage
highest_score = 0


# best_tree_model_test = None # best acurracy obtained by a tree classifier in test stage


def load_data(ftrain='train_data.csv', ftest='test_data.csv', fvalidate='validation_data.csv'):
    train_data = pandas.read_csv(ftrain)
    test_data = pandas.read_csv(ftest)
    validate_data = pandas.read_csv(fvalidate)
    t1 = [train_data.iloc[:, 1:], train_data['class']]
    t2 = [test_data.iloc[:, 1:], test_data['class']]
    t3 = [validate_data.iloc[:, 1:], validate_data['class']]

    return t1, t2, t3


# def train(df_train, crit, split, max_depth, max_features, max_leaf,seed = 0):
#     tree_classifier = DecisionTreeClassifier(crit, split, max_depth, 2, 1, 0, max_features, seed, max_leaf)
#     class_labels = df_train[1]
#     train_data = df_train[0] # all except the class label
#     tree_classifier.fit(train_data,class_labels)
#     return tree_classifier

def validate(classifier, df_validate):
    ground_truth = df_validate[1]
    predicted_values = classifier.predict(df_validate[0])
    accuracy = accuracy_score(ground_truth, predicted_values)
    global highest_score
    if (accuracy > highest_score):
        highest_score = accuracy
        global best_tree_model_validation
        best_tree_model_validation = classifier
    return accuracy, predicted_values  # returns score plus the predicted values


def test(classifier, df_test):
    ground_truth = df_test[1]
    predicted_values = classifier.predict(df_test[0])
    accuracy = accuracy_score(ground_truth, predicted_values)
    return accuracy, predicted_values


df_train, df_test, df_validate = load_data()

#
#
# with no updates
tree_classifier = DecisionTreeClassifier(random_state=0)
class_labels = df_train[1]
train_data = df_train[0]  # all except the class label
tree_classifier.fit(train_data, class_labels)
print(tree_classifier.get_n_leaves())
print(f"with default values: {test(tree_classifier, df_test)[0]}")

criterion = ['entropy', 'gini']
split = ['best', 'random']
max_depth = [i for i in range(3, df_train[0].iloc[0].count(), 2)]
min_sample_leaves = [i for i in range(1, 10)]
max_feature = [i for i in range(1, df_train[0].iloc[0].count(), 1)]
weights = 'balanced'
max_leaves = [i for i in range(3, 60)]

adjust = False
if adjust == True:
    log = open("DecisionTreeLogs/training_log.csv", 'a')
    for crit in criterion:
        for spl in split:
            for depth in max_depth:
                for feature in max_feature:
                    for leaves in max_leaves:
                        for min_sample_leave in min_sample_leaves:
                            model = DecisionTreeClassifier(crit, spl, depth, 2, min_sample_leave, 0, feature, 0, leaves,
                                                           class_weight=weights)
                            model.fit(df_train[0], df_train[1])
                            accuracy, predictions = validate(model, df_validate)
                            print(accuracy)
                            log.write(
                                f'{accuracy:.7f},{crit},{spl},{depth},{feature},{leaves},{min_sample_leave},{weights}\n')
    log.close()

best_tree_model_validation = DecisionTreeClassifier(max_depth=9, min_samples_leaf=2, max_features=6, random_state=0,
                                                    max_leaf_nodes=44, class_weight='balanced')
best_tree_model_validation.fit(df_train[0], df_train[1])
print(f"best acurracy: {test(best_tree_model_validation, df_test)[0]}")  # acurracy is 71% with best model
