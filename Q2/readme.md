Each .py file can simply be run in any python interpreter and the test() function predicts the categories for each document in test data.

The train-test split can be changed by changing the splitfrac paramter in load_data().
e.g:
    splitfrac = 0.7 means 70% data is used in training or 70-30 train-test ratio

Performance evaluators like number of correct predictions, accuracy, precision, recall, f1 score can be seen printed on output terminal.