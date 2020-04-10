from keras import backend as K


def PER_loss(y_true, y_pred, importances):
    return K.mean(K.square(y_true - y_pred) * importances)
