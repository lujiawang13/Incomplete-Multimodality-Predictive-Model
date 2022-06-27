# Incomplete-Multimodality-Predictive-Model

(The example will be uploaded soon.)


Multi-modality datasets are becoming very common in practice, especially in health care. 

However, it is also common that subjects have incomplete modalities due to cost or other practical constraints.

This package aims to fuse incomplete-modality datasets to build a machine learning predictive model. 

To understand the incomplete-modality predictive model better, suppose there are three modalities in one dataset: MRI, CT, and EEG. In each modality, there are multiple features.

 Some records have all three modalities; while others have different kinds of incomplete modality patterns (e.g., some only have MRI, CT, no EEG, and some only have CT and EEG, no MRI). We want to build a model to predict 'y'.
 
There are two common strategies to address this issue:

(1) Removing those records with any missing modality.

(2) Using a missing data imputation technique as a pre-step to impute missing values.

The first one will miss a lot of information, and the second increased the model uncertainty . Incorrect imputation of missing values could lead to a wrong prediction. 

This package explores the information from modality missing data directly. 

Simply, the idea of Incomplete-Multimodality-Predictive-Model is shown below:

(1) Grouping each type of modality missing data. For example, group 1: MRI, CT, no EEG; group 2: CT and EEG, no MRI. group 3: MRI, CT and EEG.

(2) Assuming a common coefficient vector for all features in different groups, denoted by beta. Suppose there are p1, p2, p3 features in MRI, CT and EEG. The length of beta vector should be p=p1+p2+p3.

(3) Assuming different coefficient values for modalities in different groups. For example, group 1: alpha_11 for MRI, alpha_12 for CT; group 2: alpha_22 for CT, alpha_23 for EEG; group 3: alpha_31 for MRI, alpha_32 for CT, alpha_33 for EEG. Then it's easy to know the predictive function in the form of f(alpha*[MRT,CT,EEG]*beta).

(3) Building an objective optimization using a convex loss function (such as the least squares loss function, or the logistic loss function) and a regularization (such as LASSO, Ridge, ELastic net).

(4) Solving the optimziation to estimate alphas and beta.

Finally, with the estimated coefficients, we can do the prediction on any type of modality missings samples.


