#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, Bidirectional, TimeDistributed, Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint


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
    score_train = model.evaluate(x_train, y_train)[1]
    score_test = model.evaluate(x_test, y_test)[1]
    print('Train accuracy: {:.2f}%'.format(score_train * 100.0))
    print('Test accuracy: {:.2f}%'.format(score_test * 100.0))

    return model
