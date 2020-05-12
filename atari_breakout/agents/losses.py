from keras import backend as K


def _Huber(y_true, y_pred):
    error = K.abs(y_true - y_pred)
    quadratic_part = K.clip(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    return 0.5 * K.square(quadratic_part) + linear_part


def Huber_loss(y_true, y_pred):
    return K.mean(_Huber(y_true, y_pred))


def PER_MSE_loss(y_true, y_pred, importances):
    error = K.abs(y_true - y_pred)
    L = K.square(error)
    return K.mean(L * importances)


def PER_Huber_loss(y_true, y_pred, importances):
    return K.mean(_Huber(y_true, y_pred) * importances)
