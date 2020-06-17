# PD-EEG-SAI, Data and Models
This repository contains the electroencephalography (EEG), short-latency afferent inhibition (SAI), and cognitive data and associated models designed to reproduce and explain the results in *Electrophysiological and Cognitive Correlates of Short Latency Afferent Inhibition in Parkinson’s Disease* by Wapstra et al.

We tested: 1) whether including cognitive status improved the ability of QEEG metrics to differentiate between Parkinson's Disease (PD) and non-PD; and 2) whether quantitative EEG (QEEG) measures that distinguish PD from non-PD were predictive of SAI.

## Methods

Sixty-two participants with PD and 37 non-PD comparison participants received a resting EEG scan and a subset (23 PD, 20 non-PD) completed an SAI protocol using transcranial magnetic stimulation (TMS). Relative power, oscillation burst lifetime (OBL), detrended fluctuation exponent, central frequency, and bandwidth were calculated for 4 frequency bands (delta, theta, alpha, and beta) measured on two bipolar derivations (Fz-Cz and Pz-Oz). 

## Results

A set of 14 QEEG predictors correctly classified 63.5% of PD participants, and mild cognitive impairment (MCI) had little to no effect on predictivitive ability. QEEG measures that distinguish PD from non-PD were not predictive of SAI.

## Models

The five QEEG biomarkers were calculated for the four EEG frequency bands measured on two bipolar derivations, Fz-Cz and Pz-Oz.  Subsets of potential predictors from this full set of 40 biomarkers were used for the statistical tests described below. All variables were mean centered and scaled to unit variance, and the variance inflation factors (VIF) were examined to measure collinearity. VIF is the quotient of the variance in a model that quantifies the severity of multicollinearity and any VIF over 10 indicates a multicollinearity issue. Because the correlates were highly collinear, traditional logistic regression methods could not be used. To adjust for multicollinearity, partial least squares for logistic regression (PLSLR) and ridge regression were used for analysis for the EEG sample. Ridge regression and a feedforward neural network were used for analysis of the EEG-SAI subset. All statistical tests controlled for age and sex. 

**Group Comparison -- Group_Comparison.Rmd**

We used Student’s t-tests to examine the differences of electrophysiological measures localized to Fz-Cz and Pz-Oz electrodes and the differences in SAI between PD and non-PD participants.
A Pearson’s Chi-squared test with Yates’ continuity correction was used to examine the correlation between PD status and MCI status for all participants.

**Partial least squares for logistic regression (Disease Status) -- PD_Regression.Rmd**

This model was used to estimate the regression coefficients and their corresponding p-values for the EEG sample in predicting disease status. The R package “plsRglm” was used for implementation [1]. PLSLR creates a set of new features from linear combinations of the original set of predictors. Features are computed to maximize covariation with the outcome variable, in this case, the participant’s disease status (PD or non-PD). Feature computation also iterates so that each feature estimates the biomarker-outcome covariance that remains after subtracting the contribution of previous features. Confidence intervals around the regression coefficients and p-values on each of the original predictors were estimated via the bootstrap method. PLSLR was implemented using cross-validation to select the optimal number of features to minimize the misclassification rate and maximize the AUC. An optimal number of 5 features was selected, based on maximizing the AUC and being near-optimal for the misclassification rate, and they were used to estimate the coefficients and confidence intervals for the original set of predictors. A major advantage of using PLSLR is its ability to handle a strong degree of multicollinearity, retaining in the model only those predictors needed to obtain strongest explanatory power [1]. The subset of biomarker predictors with a p-value lesser than 0.05 in the PLSLR of PD status is hereafter denoted the PD subset. The profile of each linear combination reveals which subsets of original biomarkers carry the most weight in each of the new feature variables. Two PLSLR models were run, one including cognitive status and one without, to examine whether MCI affected the weight or significance of the PD subset of biomarker predictors in the diagnosis of PD status.

**Ridge Regression (Disease Status) -- PD_Regression.Rmd**

This model was used to examine and confirm the ability of QEEG predictors to predict disease status across all subjects with and without MCI. The R package “glmnet” was used for implementation [2]. Due to multicollinearity in the dataset, ridge regression was selected as the ideal model for strictly measuring predictive power of the predictors and confirming the classification findings from the PLSLR analysis [3]. Ridge regression works by minimizing mean-squared error and then adding a degree of bias to regression estimates. The degree of bias is represented by adding the L2 penalty which is equal to the square of the magnitude of the coefficients. The tuning parameter controls the strength of the penalty term. The degree of bias is selected based on choosing the tuning parameter that maximizes the cross-validation accuracy. Cross-validated prediction accuracy was assessed through calculating the misclassification rate and the binomial deviance.

**Ridge regression (SAI) -- SAI_Regression.Rmd**

This model was used to measure the predictive power of the EEG-SAI sample of QEEG variables to SAI. The PD subset of predictors and the set of all QEEG predictors, both with MCI included as a confounding variable, were used to examine predictive power. EEG-SAI sample had the same high degree of multicollinearity as the EEG sample and a smaller set of subjects. The PLS algorithm was applied to the data set, but the model with an intercept only was selected as optimal. The ridge regression model was selected for SAI analysis as it can utilize the full set of EEG predictors.

**Feedforward Neural Network (SAI) -- SAI_Regression.Rmd**

To verify the findings from the EEG-SAI ridge regression models, feedforward neural network models were implemented. The R package “nnet” was used for implementation [4]. Feed-forward neural networks work by using the gradient descent algorithm. Weights are selected through nonlinear optimization to minimize the mean squared error over a training set. Each weight is then iteratively changed proportionally to its effect on the error. The cross-validated MSE was calculated for various values of the decay parameter (regularization parame-ter to avoid over-fitting) and the number of units in hidden layer.

## References

[1]  Bastien, P., Vinzi V. E., & Tenenhaus, M. (2005). PLS generalised linear regression. Computational Statistics & Data Analysis 48,17-47.

[2]  Hastie, T., & Qian J. (2014). Glmnet Vignette. Retrieved from https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html

[3]  Bager, A., Roman, M., Algedih, M., & Mohammed, B. (2017, June). Addressing multicollinearity in regression models: a ridge regression application. Retrieved from https://mpra.ub.uni-muenchen.de/81390/.

[4]  Venables, W. N., & Ripley, B. D. (2002). Modern Applied Statistics with S. Fourth edition. Springer.

