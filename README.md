# REDEFINE : Reclassify Data Enhancement for Improved Normalization and Evaluation.

REDEFINE is an app that identifies potentially misclassified points in classification datasets by cross-examining results from supervised and unsupervised models. Points get highlighted if both of the models agree on a label that is different than the given label. The app has been designed especially for usability, so both data scientists and non-data scientists can make the most of the process. All current models are from the Scikit-Learn library.

### Data Requirements
* 200MB or smaller
* .CSV format
* Numeric data only; target and ID columns can be nonnumeric

### Outputs

| file | description |
|------|-------------|
| Results File | .CSV file with the columns: ID, original labels, supervised learning results, unsupervised learning results |
| Metadata File | .TXT file with the results and all of the information necessary for replication. |
| Plot | There are two locations to download the active plot. To download the plot as a .PNG, click the save button ![Screenshot](save.png) in the plot. To download the plot as an .HTML, click the button underneath the plot. |

### Access
* Hosted: https://redefine-app.streamlit.app/
    * The app will sometimes deactivate due to inactivity.
* Local: 
    * In the terminal, build the virtual environment:
        * `pipenv shell`
        * `pipenv install --ignore-pipfile`
    * Run the app:
        * ```streamlit run main.py```

### Next steps
The next goal is to make for a streamlined in-app replicability process.  This will involve the use of a .JSON download and upload, so the user can upload the file and retrieve the exact settings and results as before.
