
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

mnist = fetch_openml('mnist_784', version='active', as_frame=False)
mnist.keys()
X,Y = mnist["data"], mnist["target"]
some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)

# plt.imshow(some_digit_image,cmap= mpl.cm.binary, interpolation='nearest')
# plt.axis('off')
# plt.show()

Y= Y.astype(np.uint8)

X_train,X_test,y_train,y_test = X[:60000],X[60000:],Y[:60000],Y[60000:]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_5)

sgd_clf.predict([some_digit])

cross_val_score(sgd_clf,X_train,y_train_5, cv=3, scoring='accuracy')