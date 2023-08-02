# Vocabulary Gesture Recognition with the Leap Motion Sensor

![Leap Motion](/imgs/Leap_Motion.png)

The gesture vocabulary used is available in [this link](https://lttm.dei.unipd.it/downloads/gesture/).

The dataset (<code>db</code> folder) was collected with the Leap Motion Sensor and the LeapC API. The code used for the data collection is in the repo <code>LeapC-GR-Template</code>, available in [this link](https://github.com/Henrique-Shiguemoto/LeapC-GR-Template)

After the data collection, a kNN, SVM and Random Forest Classifier were used to recognize the gesture vocabulary.

Many configuration settings were tested for each classifier.

For kNN, different numbers of k were tested.
For SVM, different kernel functions were tested.
For Random Forest, different numbers of forest were tested.

Also, 4-Fold cross validation was used since 4 people participated in the creation of the gesture dataset.

# How to Run the Source Code

In the <code>param_search</code> folder, there are scripts for each classifier. These scripts tests many configurations for the algorithms and prints results such as accuracy, standard deviation and coefficient of variation (CV, for short).

Just type <code>python script_name.py</code> to run the source code.

In the <code>confusion_matrix</code> folder, there are also scripts for each classifier. For these scripts, the best configurations (from the corresponding script in the <code>param_search</code>) are tested again to create a confusion matrix for the classifier. A confusion matrix is rendered with matplotlib in each script.

Just type <code>python script_name.py</code> to run the source code.

# Technologies Used

- C programming language
- LeapC API (Leap Motion SDK)
- Visual Studio 2022
- Python 3.9
- Pandas
- Numpy
- Sklearn
- Matplotlib

# Results

Mean Accuracies:

- kNN: 81.45%
- SVM: 84.41%
- RF:  71.41%

# Confusion Matrices

kNN:

![kNN](/imgs/KNN_CM.png)

SVM:

![SVM](/imgs/SVM_CM.png)

Random Forest:

![RF](/imgs/RF_CM.png)

More results can be found in the <code>imgs</code> folder.