import pandas
from sklearn.metrics import accuracy_score
from sklearn import linear_model  # remeber to use SGDREGRESSOR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

best_SGD_model = None
highest_score = 0


def load_data(ftrain='train_data.csv', ftest='test_data.csv', fvalidate='validation_data.csv'):
    train_data = pandas.read_csv(ftrain)
    test_data = pandas.read_csv(ftest)
    validate_data = pandas.read_csv(fvalidate)
    t1 = [train_data.iloc[:, 1:], train_data['mpg']]
    t2 = [test_data.iloc[:, 1:], test_data['mpg']]
    t3 = [validate_data.iloc[:, 1:], validate_data['mpg']]

    return t1, t2, t3


def validate(classifier, df_validate):
    ground_truth = df_validate[1]
    predicted_values = classifier.predict(df_validate[0])
    mse = mean_squared_error(ground_truth, predicted_values)
    abe = mean_absolute_error(ground_truth, predicted_values)
    global highest_score
    if (mse < highest_score or abe < highest_score):
        highest_score = accuracy
        global best_SGD_model
        best_SGD_model = classifier
    return (mse, abe), predicted_values  # returns score plus the predicted values


def test(classifier, df_test):
    ground_truth = df_test[1]
    predicted_values = classifier.predict(df_test[0])
    mse = mean_squared_error(ground_truth, predicted_values)
    abe = mean_absolute_error(ground_truth, predicted_values)
    return (mse, abe), predicted_values


def log_result(errors, classifier):
    str_classifier = str(classifier).split("\n")
    str_classifier[0] = (str_classifier[0])[13:-1]
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
    return f"{errors[0]},{errors[1]},{csv_format}\n"  # mean squared error, absolute error other values


# get data
df_train, df_test, df_validate = load_data()

log = open('LinearRegressionLogs/traininglog2.csv', 'a')

# train model without adujusting any Hyper parameters
default = linear_model.SGDRegressor(max_iter=1000, tol=0.001, random_state= 0)
print(default)
class_labels = df_train[1]
train_data = df_train[0]  # all except the class label
default.fit(train_data, class_labels)
accurra = validate(default, df_test)[0]
# log.write(log_result(accurra, default))
print(f'the default values result with this accuracy: {accurra}')

# testing lbfds solver
p1 = linear_model.SGDRegressor(
    learning_rate= 'adaptive',
    eta0 = 0.02, # initial learning rate with constant invscaling adaptive
    power_t = 0.3, # exponent for inverse scaling learning
    early_stopping = False, # stop learning when score
    validation_fraction = 0.2, # proportion of training to use for validation
    n_iter_no_change =4,
    loss = 'huber', #‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’
    penalty = 'elasticnet',
    alpha =0.0091 ,# can be set to optimal determines regularization term
    l1_ratio= 0.001, # elastic net mixing parameter
    fit_intercept= True, # whether intercept should be estimated
    max_iter= 1000, # number of passes over training
    tol =  0.001, # stoping criterion
    shuffle= True, # should training data be shuffled ?
    random_state=500,
    verbose=False, # verbosity level
    epsilon= 0.01, #  is ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive
    average = False
)

adjust = False
if adjust:
    p1.fit(train_data, class_labels)
    accuracy = validate(p1, df_validate)[0]
    print(f"adjusting p1 with    mean squared error = {accuracy[0]:10.2f}  absolute squared error= {accuracy[1]:10.2f}")
    log.write(log_result(accuracy, p1))

else:
    p1.fit(train_data, class_labels)
    accuracy = validate(p1, df_test)[0]
    print(f"adjusting p1 with    mean squared error = {accuracy[0]:10.2f}  absolute squared error= {accuracy[1]:10.2f}")
    log.write(log_result(accuracy, p1))

    output = open("Bonus_output_of_best_model.csv", "w")
    i = 0
    output.write("Class,Prediction\n")
    for prediction in test(p1, df_test)[1]:
        output.write(f"{df_test[1][i]:3.4f},{prediction:3.4f}\n")
        i += 1
    output.close()

log.close()
