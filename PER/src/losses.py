from keras import backend as K


def PER_loss(importance):

    def loss(y_true, y_pred):
        return K.mean(K.square(y_true - y_pred) * importance.reshape(-1, 1))

    return loss
