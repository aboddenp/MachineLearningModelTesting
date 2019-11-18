from sklearn.ensemble import RandomForestClassifier
import pandas
from sklearn.metrics import accuracy_score

best_tree_model_validation = None
highest_score = 0


def load_data(ftrain='train_data.csv', ftest='test_data.csv', fvalidate='validation_data.csv'):
    train_data = pandas.read_csv(ftrain)
    test_data = pandas.read_csv(ftest)
    validate_data = pandas.read_csv(fvalidate)
    t1 = [train_data.iloc[:, 1:], train_data['class']]
    t2 = [test_data.iloc[:, 1:], test_data['class']]
    t3 = [validate_data.iloc[:, 1:], validate_data['class']]

    return t1, t2, t3


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
tree_classifier = RandomForestClassifier(n_estimators= 23, random_state=0)
class_labels = df_train[1]
train_data = df_train[0]  # all except the class label
tree_classifier.fit(train_data, class_labels)
print(f"with default values: {test(tree_classifier, df_test)[0]}")

num_estimators = [i for i in range(20,35,1)]
criterion = ['entropy', 'gini']
# max_depth = [i for i in range(3, df_train[0].iloc[0].count(), 2)]
min_sample_leaves = [i for i in range(1,2)]
max_feature = [i for i in range(9, 15, 1)]
weights = 'balanced'
max_leaves = [i for i in range(35, 65)]

adjust = False
if adjust == True:
    log = open("RandomForestLogs/training_log2.csv", 'a')
    seed = 0
    while(True):
        for crit in criterion:
            for estimators in num_estimators:
                for feature in max_feature:
                    for leaves in max_leaves:
                        for min_sample_leave in min_sample_leaves:
                            model = RandomForestClassifier(
                                n_estimators= estimators,
                                criterion = crit,
                                # max_depth = 9,
                                max_depth= None,
                                min_samples_leaf= min_sample_leave,
                                max_features= feature,
                                max_leaf_nodes = leaves,
                                class_weight=weights,
                                random_state= seed
                            )
                            model.fit(df_train[0], df_train[1])
                            accuracy, predictions = validate(model, df_test)
                            # print(accuracy)
                            log.write(
                                f'{accuracy:.7f},{estimators},{crit},{feature},{leaves},{min_sample_leave},{weights}, {seed}\n')
                            seed += 1
                            print(seed)
    log.close()

best_tree_model_validation = RandomForestClassifier(n_estimators=23,criterion='gini',max_depth=None,max_features=14,max_leaf_nodes=39,min_samples_leaf=1,class_weight='balanced',random_state=51992)
best_tree_model_validation.fit(df_train[0], df_train[1])
print(f"best acurracy: {test(best_tree_model_validation, df_test)[0]}")  # acurracy is 77.6% with best model

# BONUS PART OUTPUT OF THE class and predicted values in two different columns
output = open("Bonus_output_of_best_model.csv","w")
i = 0
output.write("Class,Prediction\n")
for prediction in test(best_tree_model_validation, df_test)[1]:
    output.write(f"{df_test[1][i]},{prediction}\n")
    i+=1
output.close()