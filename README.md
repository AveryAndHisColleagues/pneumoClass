To anyone who's interested in the project, 

please make sure you're on the branch chest_xray, which is ahead of the main branch by I don't know how many commits.

I'm way too lazy to keep the main branch updated since this is just a practice project. 

Take care of yourself and be well.

# Key Results
Model performance was evaluated on the official test set. Key findings are listed below:

- **Best Overall Performance:** ResNet18 achieved the best balance between sensitivity and specificity.
- **Pneumonia Recall:** All models kept a recall rate above 90% for pneumonia detection, minimizing missed diagnoses.
- **Threshold Analysis:** We conducted a threshold sweep to find optimal thresholds for clinical screening (see the `Threshold Sweep` chart in the reports folder).

# Explanations
`semifull_test.ipynb` handles data visualization. `check_kaggle.ipynb` re-splits the dataset, because the original validation set only has 8 normal and 8 pneumonia samples.
All model comparison plots can be found in the "results" folder.

# Todo
- Complete full tests and update results here (basically finished. MiniVGG was not tested on this Kaggle dataset, as it adds little value).
- Optimize the classification threshold (default is currently 0.5).
