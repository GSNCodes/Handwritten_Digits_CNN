from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

(X_Train,Y_Train),(X_Test,Y_Test) = mnist.load_data()

print(X_Train.shape)

print(X_Test.shape)

print(X_Train[10])

print(Y_Train[10])

plt.imshow(X_Train[10])

X_Train = X_Train.reshape(60000,28,28,1)
X_Test  = X_Test.reshape(10000,28,28,1)

#1-Hot Encoding

y_train_oneh = to_categorical(Y_Train)
y_test_oneh  = to_categorical(Y_Test)

print(y_train_oneh[10]) # Should be equivalent to the value we saw above

digit_model = Sequential()
digit_model.add(Conv2D(64,kernel_size=3,activation='relu',input_shape=(28,28,1)))
digit_model.add(Conv2D(32,kernel_size=3,activation='relu'))
digit_model.add(Flatten())
digit_model.add(Dense(10,activation='softmax'))

digit_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = digit_model.fit(X_Train,y_train_oneh, validation_data=(X_Test,y_test_oneh),epochs=10)

#Plot the Training and Validation Accuracy

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy of our Model")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['Train','Validation'], loc='upper left')
plt.show()

# Comparing with some data from the training set

predictions = digit_model.predict(X_Test[100:110])
print(np.argmax(predictions,axis=1))

print(Y_Test[100:110])

