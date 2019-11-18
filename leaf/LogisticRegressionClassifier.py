import pandas
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore") # supresses warning regarding the changes of the defaults of the some of the methods used

best_logistic_model = None
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
        global best_logistic_model
        best_logistic_model = classifier
    return accuracy, predicted_values  # returns score plus the predicted values


def test(classifier, df_test):
    ground_truth = df_test[1]
    predicted_values = classifier.predict(df_test[0])
    accuracy = accuracy_score(ground_truth, predicted_values)
    return accuracy, predicted_values


def log_result(accuracy, classifier):
    str_classifier = str(classifier).split("\n")
    str_classifier[0] = (str_classifier[0])[19:-1]
    str_classifier[-1] = (str_classifier[-1])[:-1]
    csv_format = ""
    for string in str_classifier:
        params = string.split(",")
        for param in params:
            value = param.split('=')
            if (len(value) < 2):
                continue
            csv_format += value[1] + ","
    csv_format = csv_format[:-1]
    return str(accuracy) + "," + csv_format + '\n'


# get data
df_train, df_test, df_validate = load_data()

log = open('LogisticRegressionLogs/traininglog2.csv', 'a')

# train model without adujusting any Hyper parameters
default = LogisticRegression(solver='lbfgs', multi_class='auto')
class_labels = df_train[1]
train_data = df_train[0]  # all except the class label
default.fit(train_data, class_labels)
accurra = validate(default, df_test)[0]
# log.write(log_result(accurra, default))
print(f'the default values result with this accuracy: {accurra}')

# testing lbfds solver
p1 =  LogisticRegression(solver='lbfgs',  #maximum accuracy obtained is 74.6%
                         penalty='l2',
                         random_state=0,
                         max_iter =1000,
                         multi_class='ovr',
                         n_jobs=1,
                         class_weight='balanced',
                         dual = False,
                         C = 1000.0,
                         fit_intercept = True,
                         intercept_scaling = True,
                         tol = 0.0001,
                         verbose= 0,
                         l1_ratio= None
                         )

adjust = False
if adjust:
    p1.fit(train_data, class_labels)
    accuracy = validate(p1,df_validate)[0]
    print(f"adjusting p1 with   = {accuracy}")
    log.write(log_result(accuracy,p1))
else:
    p1.fit(train_data, class_labels)
    accuracy = validate(p1,df_test)[0]
    print(f"The acurracy is = {accuracy}")
    log.write(log_result(accuracy,p1))


log.close()
