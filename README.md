# REDEFINE
"Reclassify Enhanced Data for Improved Normalization and Evaluation."
This goal for this project is to develop a pipeline that uses supervised classifiers and unsupervised cluster algorithms to analyze data and determine which, if any, data points may be misclassified.  This has been done on a smaller scale previously, so my main focus is generalizing the pipeline and providing more user options and useability.

### The Data
I have provided data for this project, however, it is mostly to be used for demo and testing purposes only.  The idea is for users to have their own data they want to run through the pipeline.  The data needs to be relatively small, classification datasets, and at the moment, numerical only (we'll see if I have time to add cleaning functionality).  The Sci-Kit Learn Iris dataset has been added to the web app as a "demo" data capability, both for users and for my personal convenience.

### Next steps
I plan to finish the plot and provide the option to see the data with PCA or tSNE.  I also plan to add more helpful information about the inputs to provide for a more accessible app.  Lastly, I will streamline a way to input previous settings with a JSON file.

### Access:
Global access*: https://redefine-app.streamlit.app/
* This app will sometimes deactivate due to inactivity

Local access: run ```streamlit run app.py``` in the terminal
