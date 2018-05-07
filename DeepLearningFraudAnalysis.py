import pandas as pd
import numpy as np
from keras import models, layers, regularizers
from numpy.random import seed
from tensorflow import set_random_seed
from sklearn.metrics import confusion_matrix
import pygal

# Set initial parameters
orig_data_path = 'data/PS_20174392719_1491204439457_log.csv'
metrics_path = 'data/model_metrics_results.csv'
new_data_path = 'data/PS_20174392719_subset.csv'
current_style = pygal.style.Style(major_label_font_size = 18.0)
seed(50)
set_random_seed(50)


# Pre-processing
def pre_process_kaggle_data(data_path, subset_path,record_size=150000):
    df = pd.read_csv(data_path)
    print('Are there any (explicitly) missing values in the dataset that need to be imputed? {}'.format(df.isnull().values.any()))

    # One-hot encoding the transactions types as features
    transaction_type_features = pd.get_dummies(df['type'])
    transaction_type_features.columns = map(str.lower, transaction_type_features.columns)

    df = df.join(transaction_type_features)
    df = df.drop('type', axis=1)

    # One-hot encoding customer or merchant information
    customer_or_merchant_orig = df['nameOrig'].apply(lambda x: int(x[0]=='C'))
    df = df.drop('nameOrig', axis=1)
    df = df.join(customer_or_merchant_orig)
    customer_or_merchant_new = df['nameDest'].apply(lambda x: int(x[0]=='C'))
    df = df.drop('nameDest', axis=1)
    df = df.join(customer_or_merchant_new)
    df = df.rename(index=str, columns={'nameOrig' : 'originatorCustomer', 'nameDest' : 'destinationCustomer'})
    print('Fraudulent Record Percentage: {}'.format(len(df[df['isFraud']==1])/ float(len(df))))

    # Note: To fulfill course maximum size restrictions, we're shrinking our dataset for this assignment.
    # To create a more balanced set we'll: undersample the majority class and oversample the minority class

    if record_size >= len(df[df['isFraud']==1]):
        subset_df =  pd.concat([df[df['isFraud']==0][:record_size-len(df[df['isFraud']==1])], df[df['isFraud']==1]]).sample(frac=1)
    else:
        subset_df = df[:record_size]
    subset_df.to_csv(subset_path,index=False)
    print('Creating a subset of data with {} records'.format(record_size))


def create_test_train_split(orig_path, train_percentage=.8, transform='none'):
    orig_df = pd.read_csv(orig_path)
    train_x = orig_df[:int(len(orig_df) * train_percentage)]
    train_y = train_x['isFraud'].as_matrix()
    train_x = train_x.drop(['isFraud'],axis=1).as_matrix()
    test_x = orig_df[int(len(orig_df) * (1 - train_percentage)):]
    test_y = test_x['isFraud'].as_matrix()
    test_x = test_x.drop(['isFraud'],axis=1).as_matrix()
    if transform=='log':
        train_x = np.log(train_x+1)
        test_x = np.log(test_x+1)

    if transform=='normal':
        full_df = orig_df.drop(['isFraud'],axis=1).as_matrix()
        full_mean = full_df.mean(axis=0)
        full_std = np.std(full_mean, axis=0)
        train_x = train_x - full_mean
        test_x = test_x - full_mean
        train_x = train_x / full_std
        test_x = test_x / full_std

    return train_x, train_y, test_x, test_y

def train_model(new_data_path, path_output):
    df_list = []

    # Note: Binary Cross Entropy is recommended for models that output a probability, so it is not modified in the search
    # Note: The RELU activation function is typically able to solve a wide range of problems, so it is not modified in the search
    # Note: RMSPROP is typically a good optimizer for many problems, so it is also not included in the search

    # Grid Search for parameters
    for transformed_data in ['none', 'log', 'normal']:
        train_x, train_y, test_x, test_y = create_test_train_split(new_data_path,transform=transformed_data)
        for current_batch_size in [150, 300, 600]:
            for current_epoch_size in [10, 40]:
                for units_per_layer in [14, 28]:
                    for current_regularization_weight in [0, 0.01, 0.05]:
                        fraud_model = models.Sequential()
                        fraud_model.add(layers.Dense(units_per_layer, activation='relu', kernel_regularizer=regularizers.l2(current_regularization_weight), input_shape=(14,)))
                        fraud_model.add(layers.Dense(units_per_layer, kernel_regularizer=regularizers.l2(current_regularization_weight), activation='relu'))
                        fraud_model.add(layers.Dense(1, activation='sigmoid'))
                        fraud_model.compile(optimizer ='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
                        fraud_model.fit(train_x, train_y, epochs=current_epoch_size, batch_size=current_batch_size)
                        train_results = fraud_model.evaluate(train_x,train_y)
                        results = fraud_model.evaluate(test_x,test_y)
                        model_predictions = fraud_model.predict_classes(test_x)
                        true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(test_y,model_predictions).ravel()
                        perf_row = {
                                     'Transform' : str(transformed_data),
                                     'Batch_size': current_batch_size,
                                     'Epochs' : current_epoch_size,
                                     'Units per layer' : units_per_layer,
                                     'L2 Regularization Weight' : current_regularization_weight,
                                     'Precision' : float(true_positives) / np.nextafter((false_positives + true_positives),1),
                                     'Recall': float(true_positives) / np.nextafter((false_negatives + true_positives),1),
                                     'Train Accuracy' : train_results[1],
                                     'Test Accuracy': results[1]
                                     }
                        df_list.append(perf_row)

    pd.DataFrame(df_list).to_csv(path_output,index=False)
    return pd.DataFrame(df_list)

def analyze_file_results(file_path):
    model_metric_df = pd.read_csv(file_path)
    analyze_results(model_metric_df)

def generate_chart(curr_df):
    y_labels = curr_df.columns
    bar_chart = pygal.Bar(title='Accuracy Metrics', y_title='Accuracy Metrics',x_title=curr_df.index.name,
                          style = current_style, x_labels_major_every = 1)
    bar_chart.x_labels = curr_df.index.tolist()
    for elm in y_labels:
        print(curr_df[elm].tolist())
        bar_chart.add(elm, curr_df[elm].tolist())
    bar_chart.render_to_file('{}.svg'.format(curr_df.index.name))

def analyze_results(model_metric_df):
    parameter_list = ['Transform', 'Batch_size', 'Epochs', 'Units per layer', 'L2 Regularization Weight']
    metric_list = ['Precision', 'Recall', 'Train Accuracy', 'Test Accuracy']
    for elm in parameter_list:
        curr_df = model_metric_df.groupby([elm]).mean()
        curr_df = curr_df[metric_list]
        generate_chart(curr_df)

# pre_process_kaggle_data(orig_data_path, new_data_path)
model_metric_df = train_model(new_data_path,metrics_path)
analyze_file_results(metrics_path)
