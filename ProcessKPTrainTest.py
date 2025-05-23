# 6. Preprocess Data and Create Labels and Features
# 7. Build and Train LSTM Neural Network
# 8. Make Predictions

from SetupTesting import *
from KPCreateFolders import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


def processData():
    label_map = {label: num for num, label in enumerate(actions)}

    sequences, labels = [], []
    for action in actions:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            window = []
            for frame_num in range(sequence_length):
                print(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)


    return X_train, X_test, y_train, y_test


def getModel(input_size = 1662):
    model = Sequential()
    #1662 with pose
    #126 only two hands
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(25, input_size)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model


def train(X_train, y_train, modelFile):
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    model = getModel(126)
    model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback])
    model.summary()
    model.save(modelFile)


def test(X_test, y_test, modelFile):
    model = getModel(126)
    model.load_weights(modelFile)
    yhat = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    print("Confusion Matrix:")
    print(multilabel_confusion_matrix(ytrue, yhat))

    print("Accuracy Score:")
    print(accuracy_score(ytrue, yhat))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = processData()
    train(X_train, y_train, modelFile="modelActions23-5.h5")
    test(X_test, y_test, modelFile="modelActions23-5.h5")
