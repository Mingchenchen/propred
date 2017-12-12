#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, Bidirectional, TimeDistributed, Masking, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint

from util import calculate_accuracy


def blstm(x_train, x_val, x_test, y_train, y_val, y_test, out_dir,
          name='blstm_model', hidden_units=10, layers=1, max_epochs=1000, batch_size=32, patience=20,
          dropout=0.0, recurrent_dropout=0.0):
    """
    Bidirectional LSTM model for protein secondary structure prediction.
    """
    num_samples = x_train.shape[0]
    max_seq_len = x_train.shape[1]
    num_features = x_train.shape[2]
    num_classes = y_train.shape[2]
    
    # Build Keras model
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(max_seq_len, num_features)))
    model.add(Bidirectional(LSTM(hidden_units, return_sequences=True, input_shape=(max_seq_len, num_features),
                                 dropout=dropout, recurrent_dropout=recurrent_dropout)))
    if layers > 1:
        for _ in range(layers-1):
            model.add(Bidirectional(LSTM(hidden_units, return_sequences=True,
                                         dropout=dropout, recurrent_dropout=recurrent_dropout)))
    model.add(TimeDistributed(Dense(num_classes)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    # Train model. Use early-stopping on validation data to determine when to stop training.
    model_path = os.path.join(out_dir, name + '.h5')
    checkpointer = ModelCheckpoint(model_path, save_best_only=True)
    model.fit(x_train, y_train, epochs=max_epochs, batch_size=batch_size, verbose=2,
              validation_data=(x_val, y_val), callbacks=[EarlyStopping(patience=patience), checkpointer])

    model = load_model(model_path)  # Best model is not necessarily current model instance b/c patience != 0
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print('Train accuracy: {:.2f}%'.format(calculate_accuracy(y_train, y_train_pred) * 100.0))
    print('Test accuracy: {:.2f}%'.format(calculate_accuracy(y_test, y_test_pred) * 100.0))

    return model

def bgru(x_train, x_val, x_test, y_train, y_val, y_test, out_dir,
          name='bgru_model', hidden_units=10, layers=1, max_epochs=1000, batch_size=32, patience=20,
          dropout=0.0, recurrent_dropout=0.0):
    """
    Bidirectional GRU model for protein secondary structure prediction.
    """
    num_samples = x_train.shape[0]
    max_seq_len = x_train.shape[1]
    num_features = x_train.shape[2]
    num_classes = y_train.shape[2]
    
    # Build Keras model
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(max_seq_len, num_features)))
    model.add(Bidirectional(GRU(hidden_units, return_sequences=True, input_shape=(max_seq_len, num_features),
                                 dropout=dropout, recurrent_dropout=recurrent_dropout)))
    if layers > 1:
        for _ in range(layers-1):
            model.add(Bidirectional(GRU(hidden_units, return_sequences=True,
                                         dropout=dropout, recurrent_dropout=recurrent_dropout)))
    model.add(TimeDistributed(Dense(num_classes)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    # Train model. Use early-stopping on validation data to determine when to stop training.
    model_path = os.path.join(out_dir, name + '.h5')
    checkpointer = ModelCheckpoint(model_path, save_best_only=True)
    model.fit(x_train, y_train, epochs=max_epochs, batch_size=batch_size, verbose=1,
              validation_data=(x_val, y_val), callbacks=[EarlyStopping(patience=patience), checkpointer])

    model = load_model(model_path)  # Best model is not necessarily current model instance b/c patience != 0
    y_train_pred = model.predict(x_train)
    print('Train accuracy: {:.2f}%'.format(calculate_accuracy(y_train, y_train_pred) * 100.0))
    # Test set accuracy 
    y_test_pred = []
    for i in range(3):  
        y_test_pred.append(model.predict(x_test[i]))
        print('Test accuracy: {:.2f}%'.format(calculate_accuracy(y_test[i], y_test_pred[i]) * 100.0))

    return model
    
def breslstm(x_train, x_val, x_test, y_train, y_val, y_test, out_dir,
          name='breslstm_model', hidden_units=10, layers=1, max_epochs=1000, batch_size=32, patience=20,
          dropout=0.0, recurrent_dropout=0.0):
    """
    Bidirectional Residual LSTM model for protein secondary structure prediction.
    """
    num_samples = x_train.shape[0]
    max_seq_len = x_train.shape[1]
    num_features = x_train.shape[2]
    num_classes = y_train.shape[2]
    
    # Build Keras model
    inputs = Input(shape=(max_seq_len, num_features))

    x = Masking(mask_value=0, input_shape=(max_seq_len, num_features))(inputs)
    x = Bidirectional(LSTM(hidden_units, return_sequences=True, input_shape=(max_seq_len, num_features),
                                 dropout=dropout, recurrent_dropout=recurrent_dropout))(x)
    if layers > 1:
        for _ in range(layers-1):
            x_rnn = Bidirectional(LSTM(hidden_units, return_sequences=True,
                                     dropout=dropout, recurrent_dropout=recurrent_dropout))(x)
            x = add([x, x_rnn])

    x = TimeDistributed(Dense(num_classes))(x)
    outputs = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    # Train model. Use early-stopping on validation data to determine when to stop training.
    model_path = os.path.join(out_dir, name + '.h5')
    checkpointer = ModelCheckpoint(model_path, save_best_only=True)
    model.fit(x_train, y_train, epochs=max_epochs, batch_size=batch_size, verbose=1,
              validation_data=(x_val, y_val), callbacks=[EarlyStopping(patience=patience), checkpointer])

    model = load_model(model_path)  # Best model is not necessarily current model instance b/c patience != 0
    y_train_pred = model.predict(x_train)
    print('Train accuracy: {:.2f}%'.format(calculate_accuracy(y_train, y_train_pred) * 100.0))
    # Test set accuracy 
    y_test_pred = []
    for i in range(3):  
        y_test_pred.append(model.predict(x_test[i]))
        print('Test accuracy: {:.2f}%'.format(calculate_accuracy(y_test[i], y_test_pred[i]) * 100.0))

    return model
    
def load_and_train(model_path, x_train, x_val, x_test, y_train, y_val, y_test,
                   max_epochs=1000, batch_size=32, patience=20):
    """
    Load model and resume training.
    """
    model = load_model(model_path)
    checkpointer = ModelCheckpoint(model_path, save_best_only=True)
    model.fit(x_train, y_train, epochs=max_epochs, batch_size=batch_size, verbose=1,
              validation_data=(x_val, y_val), callbacks=[EarlyStopping(patience=patience), checkpointer])

    model = load_model(model_path)
    y_train_pred = model.predict(x_train)
    print('Train accuracy: {:.2f}%'.format(calculate_accuracy(y_train, y_train_pred) * 100.0))
    # print('Test accuracy: {:.2f}%'.format(calculate_accuracy(y_test, y_test_pred) * 100.0))
    
    y_test_pred = []
    for i in range(3):  
        y_test_pred.append(model.predict(x_test[i]))
        print('Test accuracy: {:.2f}%'.format(calculate_accuracy(y_test[i], y_test_pred[i]) * 100.0))

    return model
