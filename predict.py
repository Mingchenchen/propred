import argparse
import os

import numpy as np

from keras import backend as K
from keras.models import load_model 
from util import * 

def main():
    data_dir, model_path, nthreads = parse_arguments()
    
    print('Loading tensors...')
    x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    x_test = [np.load(os.path.join(data_dir, 'x_test_casp10.npy')), np.load(os.path.join(data_dir, 'x_test_casp11.npy')), np.load(os.path.join(data_dir, 'x_test_cullpdb.npy'))]
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = [np.load(os.path.join(data_dir, 'y_test_casp10.npy')), np.load(os.path.join(data_dir, 'y_test_casp11.npy')), np.load(os.path.join(data_dir, 'y_test_cullpdb.npy'))]

    max_epochs = 100
    batch_size = 32
    patience = 10
    
    y_train_pred, y_test_pred = predict_accuracy(model_path, x_train, x_test, y_train, y_test, max_epochs=max_epochs, batch_size=batch_size, patience=patience)
    

def predict_accuracy(model_path, x_train, x_test, y_train, y_test,
                   max_epochs=1000, batch_size=32, patience=20):
    model = load_model(model_path)
    y_train_pred = model.predict(x_train)
    print('Train accuracy: {:.2f}%'.format(calculate_accuracy(y_train, y_train_pred) * 100.0))
    y_test_pred = []
    for i in range(3):  
        y_test_pred.append(model.predict(x_test[i]))
        print('Test accuracy: {:.2f}%'.format(calculate_accuracy(y_test[i], y_test_pred[i]) * 100.0))
    return y_train_pred, y_test_pred 

def get_f1_score(y_train_pred, y_test_pred, y_train, y_test):
    cm_train = generate_confusion_matrix(y_train, y_train_pred)
    calculate_recall_precision_f1_mcc(cm)
def parse_arguments():
    """
    Read directory containing data tensors and model path from command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, metavar='DATA_DIR', help='Directory containing .npy tensors')
    parser.add_argument('model_path', type=str, metavar='MODEL', help='Path to Keras model')
    parser.add_argument('-t', '--threads', type=int, metavar='NTHREADS', help='Number of parallel threads')
    args = parser.parse_args()

    return args.data_dir, args.model_path, args.threads

if __name__ == '__main__':
    main()