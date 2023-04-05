# Perceptron-Logistic-Regression-for-Iris-Dataset-Classification
This program implements Perceptron and Logistic Regression classifiers for binary classification of the Iris dataset, using sepal and petal lengths and widths as features. The performance of each classifier is evaluated by plotting the number of misclassified samples and the decision boundary between the two classes

This program compares the performance of Perceptron Classifier and Logistic Regression Classifier for binary classification problem using the iris dataset. The program uses the scikit-learn and matplotlib libraries, which can be installed using the following command:

```
pip install scikit-learn matplotlib
```


## The program has the following parts:

```
Perceptron Classifier
a. Using only sepal and petal lengths, the program plots the number of misclassified samples at the end of each of 50 iterations.
b. With sepal and petal lengths as x1 and x2 coordinates, the program plots the sample data and separation regions after 50 iterations. It shows the final value of weight w.
c. The program repeats part b when all four features are included in xi.


Logistic Regression Classifier
d. Using only sepal and petal lengths, the program plots the number of misclassified samples at the end of each of 50 iterations.
e. With sepal and petal lengths as x1 and x2 coordinates, the program plots the sample data and separation regions after 50 iterations. It shows the final value of weight w.
f. The program repeats part e when all four features are included in xi.

The Perceptron algorithm is based on the following decision rule:

If w^T*x > 0, predict class 1
If w^T*x <= 0, predict class -1
where w is the weight vector, x is the input vector, and w^T denotes the transpose of w.
The update rule for the weights in the Perceptron algorithm is:

w(t+1) = w(t) + eta*y(t)*x(t)
where w(t) is the weight vector at iteration t, eta is the learning rate, y(t) is the true class label of the input vector x(t), and x(t) is the input vector at 
iteration t.
The Perceptron algorithm can be extended to handle nonlinearly separable data by using a variant called the "kernel Perceptron". The kernel Perceptron uses a kernel 
function to transform the input data into a higher-dimensional feature space where it is linearly separable.

The Perceptron algorithm is a type of binary classifier, meaning that it can only distinguish between two classes. It can be extended to handle multiclass 
classification problems using techniques such as "one-vs-all" and "one-vs-one" classification.

One limitation of the Perceptron algorithm is that it can get stuck in local minima and fail to converge to the global minimum. This can be addressed by using more 
advanced optimization techniques such as stochastic gradient descent.

The results show that both classifiers are based on linear separation of the feature space, with the logistic regression classifier being more accurate and adaptable 
than the perceptron classifier. The logistic classifier can learn complicated and nonlinear decision boundaries because of the gradient of the logistic function, which 
adds a nonlinear modification of the input information. The logistic classifier is more precise than the perceptron classifier and may reach minimal error after fewer
iterations, but it takes more compute and memory to retain the weights and input data. The trade-off between accuracy and complexity, as well as the unique application
needs, determine the classifier to choose.
```

The Outputs of the Matplotlib:
![image](https://user-images.githubusercontent.com/33391934/230037724-c06fd0b6-56d6-4c60-a084-f44b1a0dc48a.png)
![image](https://user-images.githubusercontent.com/33391934/230037856-ff693abd-3295-423b-bc08-b152ceca1032.png)
![image](https://user-images.githubusercontent.com/33391934/230038030-5268d8d0-98af-4759-a791-25a89172b726.png)
![image](https://user-images.githubusercontent.com/33391934/230038155-e51cac6f-a3b4-4e9c-adb1-1cedccd022e3.png)
![image](https://user-images.githubusercontent.com/33391934/230038286-d944b5dd-2c61-41f5-802c-c02b0a6923c1.png)






