"""
Sherlock analysis toolbox

Wirawan Purwanto
Created: 2020-02-14

This toolbox is a SPECIALIZED toolbox to streamline
machine learning codes, originally developed for DeapSECURE
training program.
You are free to modify and adapt this program to suit your needs,
subject to the license adopted by the DeapSECURE team.

In this toolbox we adopt a semi-OOP approach, where the 'class methods' are
coded as normal functions (i.e. we call them as FUNCTION(OBJ, ARGS...)
instead of OBJ.FUNCTION(ARGS, ...)).
We do so because the design of the workbench is always evolving, so it is
very hard to pin down using classic OOP paradigm from the beginning.


The key functions of this module are:

  summarize_dataset
  preprocess_sherlock_19F17C
  step0_label_features
  step_drop_columns
  step_select_columns
  report_nonnumeric_features
  step_onehot_encoding
  step_feature_scaling
  step_train_test_split
  model_decision_tree
  model_logistic
  model_train
  model_evaluate

"""

import os
import sys
import time

import pandas
import numpy
import matplotlib.pyplot
#import seaborn
import sklearn

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from argparse import Namespace

# short aliases
pd = pandas
np = numpy
plt = matplotlib.pyplot
pyplot = matplotlib.pyplot
#sns = seaborn

# Do this yourselves, y'all!
# %matplotlib inline

def get_time():
    """Gets elapsed wallclock time in units of seconds as floats,
    with sub-second resolution.
    Per definition, the origin of the time is not well specified;
    only difference between two times are meaningful."""
    # perf_counter is valid since Python 3.3 and we will use that
    # in favor of older function (clock) that is not well-specified
    # across different platforms.
    try:
        return time.perf_counter()
    except AttributeError:
        return time.clock()


class timer(object):
    '''A small timer class.'''
    def start(self):
        self.tm1 = get_time()
    def stop(self):
        self.tm2 = get_time()
        return (self.tm2 - self.tm1)
    def length(self):
        return self.tm2 - self.tm1

class InputData(Namespace):
    """Representation of input data:

    * df = original dataframe
    * df_features = feature matrix (most current version)
    * labels = label part of the df

    Optional components:

    * scaler = the scaler/normalizer object
    * df_features_unscaled = unscaled feature matrix
    * train_features = training subset, the feature matrix part
    * test_features = testing subset, the feature matrix part
    * train_labels = training subset, the label part
    * test_labels = testing subset, the label part
    """
    pass


class MLModel(Namespace):
    """Representation of machine learning model:
    * Input = an InputData structure that contains the input matrices
    * model = _the_ Classifier object
    * model_params = parameters fed to the model constructor
    """
    pass



# Inquire state of the packages

print("Pandas version:", pd.__version__)
print("Scikit-learn version: ", sklearn.__version__)

#df = pd.read_csv("sherlock_apps_yhe_test.csv")

# TODO add verification step to make sure the data file is what we suppose it is

def summarize_dataset(df, T_describe=False):
    print("* shape:", df.shape)
    print("* columns::\n",
          list(df.columns))
    print()
    print("* info::\n")
    df.info()
    print()
    print("* describe::\n")
    if T_describe:
        print(df.describe().T)
    else:
        print(df.describe())
    print()
    sys.stdout.flush()


def verify_sherlock_19F17C(df):
    #assert 
    pass


def preprocess_sherlock_19F17C(df):
    """Do standard preprocessing of a Sherlock 19F17C dataset.

    Returns the preprocessed dataframe, where all the obviously
    bad and missing data are removed.

    Args:
        df (DataFrame): Raw input DataFrame

    Returns: DataFrame
        DataFrame of the cleaned data, free from invalid/missing values
    """
    # Delete irrelevant feature(s)
    del_features_irrelevant = [
        'Unnamed: 0'  # mere integer index
    ]

    # Missing data or bad data
    del_features_bad = [
        'cminflt', # all-missing feature
        'guest_time', # all-flat feature
    ]
    del_features = del_features_irrelevant + del_features_bad

    df2 = df.drop(del_features, axis=1)
    print("Preprocessing:")
    print("- dropped %d columns: %s" % (len(del_features), del_features))
    #print()
    print("- remaining missing data:")
    isna_counts = df2.isna().sum()
    print(isna_counts[isna_counts > 0])
    #print()
    print("- dropping the rest of missing data")
    df2.dropna(inplace=True)
    print("- remaining shape: %s" % (df2.shape,))
    sys.stdout.flush()

    return df2


def step0_label_features(df):
    """Preliminary step: Separates labels
    from the rest of the features.
    NOTE: Returns a struct for further processing!

    This step needs to be done right after preprocessing
    but before other processing steps below.
    """
    R = InputData()
    R.df = df
    R.col_labels = 'ApplicationName'
    R.labels = df[R.col_labels]
    R.df_features = df.drop(R.col_labels, axis=1)

    return R


def step_drop_columns(R, cols):
    """Drops additional columns from the features DataFrame
    """
    print("Step: Drop additional columns")
    print("- dropped %d columns: %s" % (len(cols), cols))
    R.df_features = R.df_features.drop(cols, axis=1)
    return R


def step_select_columns(R, cols):
    """Selects only specified columns (and remove others)

    This is the converse of step_drop_columns() to explicitly
    specify which columns to actually enter into the machine learning model.
    """
    print("Step: Select columns")
    cols = list(cols)
    print("- selected only %d columns: %s" % (len(cols), cols))
    R.df_features = R.df_features[cols]
    return R


def report_nonnumeric_features(df):
    print("Non-numeric features:")
    for C in df.columns:
        kind = df[C].dtype.kind
        # https://stackoverflow.com/questions/19900202/how-to-determine-whether-a-column-variable-is-numeric-or-not-in-pandas-numpy/46423535
        if kind not in 'bfiuc':
            print("- %s  %s" % (df[C].dtype.kind, C))
    print()


def step_onehot_encoding(R):
    """Performs one-hot encoding for **all** categorical features."""
    print("Step: Converting all non-numerical features to one-hot encoding.")
    report_nonnumeric_features(R.df_features)
    R.df_features = pd.get_dummies(R.df_features)
    return R


def step_feature_scaling(R):
    """Step: Feature scaling using StandardScaler.
    """
    from sklearn import preprocessing
    print("Step: Feature scaling with StandardScaler")
    df = R.df_features
    R.df_features_unscaled = df
    R.scaler = preprocessing.StandardScaler()
    R.scaler.fit(df)
    # TODO: Show the means and scales
    # Recast the features still in a dataframe form
    R.df_features = pd.DataFrame(R.scaler.transform(df),
                                 columns=df.columns,
                                 index=df.index)
    return R


def step_train_test_split(R, test_size, random_state):
    """Step: Performs train-test split on the master dataset.
    This should be the last step before constructing & training the model.
    """
    from sklearn.model_selection import train_test_split
    print("Step: Train-test split  test_size=%s  random_state=%s" \
              % (test_size, random_state))
    R.train_features, R.test_features, R.train_labels, R.test_labels = \
        train_test_split(R.df_features, R.labels,
                         test_size=test_size, random_state=random_state)
    R.tt_split_params = dict(test_size=test_size, random_state=random_state)

    print("- training dataset: %d records" % (len(R.train_features),))
    print("- testing dataset:  %d records" % (len(R.test_features),))
    sys.stdout.flush()

    return R


def model_decision_tree(Input: InputData, model_params) -> MLModel:
    """Creates a Decision Tree classifier with a given input data
    (embodied in the InputData object)
    and hyperparameters.

    Args:

      Input (InputData): the class holding all information about the input data,
        primarily the features and labels, already separated into train & test
        subsets.

      model_params (dict): the key-value pairs of additional parameters to
        feed to the model upon constructing it. These are basically the hyperparameters.
    """
    from sklearn.tree import DecisionTreeClassifier

    M = MLModel()
    M.model_params = dict(model_params)
    M.model = DecisionTreeClassifier(**M.model_params)
    M.Input = Input
    print("model_decision_tree: Created a new model")
    print(M.model)
    print()
    return M


def model_logistic(Input: InputData, model_params) -> MLModel:
    """Creates a Decision Tree classifier with a given input data
    (embodied in the InputData object
    and hyperparameters.
    """
    from sklearn.linear_model import LogisticRegression

    M = MLModel()
    M.model_params = dict(model_params)
    M.model = LogisticRegression(**M.model_params)
    M.Input = Input
    print("model_logistics: Created a new model")
    print(M.model)
    print()
    return M


def model_train(M, validate=True):
    """Trains a recently constructed machine learning model.
    """
    R = M.Input
    model = M.model
    t1 = timer()
    print("Step: Training model: ", type(model).__name__)
    sys.stdout.flush()
    t1.start()
    model.fit(R.train_features, R.train_labels)
    print("- training completed: time=%s secs" % (t1.stop(),))
    sys.stdout.flush()

    return M


def model_evaluate(M, extra_metrics=['precision', 'recall']):
    """Evaluates the quality metrics of the model.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

    R, model = M.Input, M.model
    M.train_L_pred = model.predict(R.train_features)
    M.test_L_pred = model.predict(R.test_features)

    print("Step: Validating model:", type(model).__name__)
    sys.stdout.flush()
    print("Metrics using training dataset:")

    if True:
        M.train_accuracy = accuracy_score(R.train_labels, M.train_L_pred)
        M.train_accuracy_raw = accuracy_score(R.train_labels, M.train_L_pred,
                                              normalize=False)
        print("- accuracy_score[train]:  {:12.8f}  raw: {}".format(
                  M.train_accuracy,
                  M.train_accuracy_raw)
             )
    if "precision" in extra_metrics:
        M.train_precision_avg = precision_score(R.train_labels, M.train_L_pred,
                                                average='macro')
        M.train_precision_wavg = precision_score(R.train_labels, M.train_L_pred,
                                                 average='weighted')
        print("- precision_score[train]: {:12.8f}   {:12.8f}  (unweighted, weighted avg)".format(
                  M.train_precision_avg,
                  M.train_precision_wavg)
             )
    if "recall" in extra_metrics:
        M.train_recall_avg = recall_score(R.train_labels, M.train_L_pred,
                                          average='macro')
        M.train_recall_wavg = recall_score(R.train_labels, M.train_L_pred,
                                           average='weighted')
        print("- recall_score[train]:    {:12.8f}   {:12.8f}  (unweighted, weighted avg)".format(
                  M.train_recall_avg,
                  M.train_recall_wavg)
             )
    #print("- recall_score[train]:   ", M.train_recall)
    #print("precision_score:",precision_score(train_L, train_L_pred))
    #print("recall_score:",recall_score(train_L, train_L_pred))
    M.train_confmat = confusion_matrix(R.train_labels, M.train_L_pred)
    print("- confusion_matrix[train]::")
    print(M.train_confmat)

    # --------------------------------------------------------------------
    print()
    print("Metrics using testing dataset:")
    if True:
        M.test_accuracy = accuracy_score(R.test_labels, M.test_L_pred)
        M.test_accuracy_raw = accuracy_score(R.test_labels, M.test_L_pred,
                                             normalize=False)
        print("- accuracy_score[test]:   {:12.8f}  raw: {}".format(
                  M.test_accuracy,
                  M.test_accuracy_raw)
             )
    if "precision" in extra_metrics:
        M.test_precision_avg = precision_score(R.test_labels, M.test_L_pred,
                                               average='macro')
        M.test_precision_wavg = precision_score(R.test_labels, M.test_L_pred,
                                                average='weighted')
        print("- precision_score[test]:  {:12.8f}   {:12.8f}  (unweighted, weighted avg)".format(
                  M.test_precision_avg,
                  M.test_precision_wavg)
             )
    if "recall" in extra_metrics:
        M.test_recall_avg = recall_score(R.test_labels, M.test_L_pred,
                                         average='macro')
        M.test_recall_wavg = recall_score(R.test_labels, M.test_L_pred,
                                          average='weighted')
        print("- recall_score[test]:     {:12.8f}   {:12.8f}  (unweighted, weighted avg)".format(
                  M.test_recall_avg,
                  M.test_recall_wavg)
             )
    #print("- precision_score[test]:", M.test_precision)
    #print("- recall_score[test]:   ", M.test_recall)
    M.test_confmat = confusion_matrix(R.test_labels, M.test_L_pred)
    print("- confusion_matrix[test]::")
    print(M.test_confmat)
    sys.stdout.flush()
    return M


# For the sake of learners' notebooks:
model_evaluate2 = model_evaluate
