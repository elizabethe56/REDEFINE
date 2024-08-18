# REDEFINE : Reclassify Data Enhancement for Improved Normalization and Evaluation.

REDEFINE is an app that identifies potentially misclassified points in classification datasets by cross-examining results from supervised and unsupervised models. Points get highlighted if both of the models agree on a label that is different than the given label. The app has been designed especially for usability, so both data scientists and non-data scientists can make the most of the process. All current models are from the Scikit-Learn library.

### Data Requirements
* 200MB or smaller
* .CSV format
* Numeric data only; target and ID columns can be nonnumeric

### Models
* Supervised
  * [K Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html)
    * K-Nearest Neighbors (KNN) is a simple algorithm used for classification and regression tasks, determining the class of a new data point by comparing it to the majority class of its k nearest neighbors based on a chosen distance metric, like Euclidean distance, in the feature space. Essentially, it makes predictions by finding the most common class among the k nearest data points in a given dataset.
  * [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    * Random Forest is an ensemble learning method that builds multiple decision trees during training and merges their predictions to improve accuracy and prevent overfitting. It works by creating diverse trees through random sampling of both data points and features, and then aggregates their outputs via voting or averaging to make robust and accurate predictions for classification tasks.
* Unsupervised
  * [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    * KMeans is an iterative clustering algorithm that partitions a dataset into K clusters by minimizing the sum of squared distances between data points and the centroid of their assigned cluster. It starts by randomly initializing K centroids, assigns data points to the nearest centroid, recalculates centroids based on the mean of assigned points, and iterates until centroids stabilize or a specified number of iterations is reached.

### Outputs

| file | description |
|------|-------------|
| Results File | .CSV file with the columns: ID, original labels, supervised learning results, unsupervised learning results |
| Metadata File | .TXT file with the results and all of the information necessary for manual replication. |
| Parameters File | .JSON file with a dictionary containing all parameters from the models, including random seeds.  Can be used to generate replication. |
| Plot | There are two locations to download the active plot. To download the plot as a .PNG, click the save button ![Screenshot](save.png) in the plot. To download the plot as an .HTML, click the button underneath the plot. |

### Access
* Hosted: https://redefine-app.streamlit.app/
    * The app will sometimes deactivate due to inactivity. Simply click the button to wake the app up.
* Local: 
    * In the terminal, build the virtual environment:
        * `pipenv shell`
        * `pipenv install --ignore-pipfile`
    * Run the app:
        * ```streamlit run src/main.py```

### Next steps
The next goal is to make for a streamlined in-app replicability process.  This will involve the use of a .JSON download and upload, so the user can upload the file and retrieve the exact settings and results as before.
