

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     Imports     --
#···············································································
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
import numpy as np
import matplotlib.pyplot as plt
from undersample import X_train, y_train, original_Xtest, original_ytest
from sklearn.metrics import confusion_matrix
from oversample import Xsm_train, ysm_train
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
print("DONE")
## UNDERSAMPLING

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     Model     --
#···············································································
n_inputs = X_train.shape[1]

undersample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

undersample_model.summary()
undersample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
undersample_model.fit(X_train, y_train, validation_split=0.2, batch_size=25, epochs=20, shuffle=True, verbose=2)
undersample_predictions = undersample_model.predict(_X_test, batch_size=200, verbose=0)
predict_fp =undersample_model.predict(_X_test) 
undersample_fraud_predictions=np.argmax(predict_fp,axis=1)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                            --     Viz     --
#···············································································
import itertools

# Create a confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


undersample_cm = confusion_matrix(_y_test, undersample_fraud_predictions)
actual_cm = confusion_matrix(_y_test, _y_test)
labels = ['No Fraud', 'Fraud']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(undersample_cm, labels, title="Random UnderSample \n Confusion Matrix", cmap=plt.cm.Reds)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## OVERSAMPLING



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     Model     --
#···············································································
n_inputs = Xsm_train.shape[1]

oversample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
oversample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
oversample_model.fit(np.array(Xsm_train), np.array(ysm_train), validation_split=0.2, batch_size=300, epochs=20, shuffle=True, verbose=2)
oversample_predictions = oversample_model.predict(_X_test, batch_size=200, verbose=0)
predict_osp =oversample_model.predict(_X_test) 
oversample_fraud_predictions=np.argmax(predict_osp,axis=1)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                            --     Viz     --
#···············································································
oversample_smote = confusion_matrix(_y_test, oversample_fraud_predictions)
actual_cm = confusion_matrix(_y_test, original_ytest)
labels = ['No Fraud', 'Fraud']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(oversample_smote, labels, title="OverSample (SMOTE) \n Confusion Matrix", cmap=plt.cm.Oranges)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


