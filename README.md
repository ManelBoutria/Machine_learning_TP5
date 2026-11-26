Machine Learning TP5: feature selection

Filter and wrapper feature selection on the Breast Cancer Wisconsin dataset, with logistic regression.

Run
bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn
python tp5.py

The figures are saved in figures/.

Results (test set, 171 patients)
Method	Features	Test accuracy
Baseline (all features)	30	0.9883
Correlation filter	20	0.9766
SelectKBest (F-score)	20	0.9766
SelectKBest (mutual information)	20	0.9708
Forward selection	4	0.9415
Backward elimination	3	0.9532
RFE	10	0.9708
RFECV	8	0.9591
Conclusion
No selection method beats the 30 features. RFE with 10 features loses only 3 patients out of 171.
Univariate filters keep redundant copies (radius, perimeter, area); the correlation filter, RFE and the wrappers keep one of each.
The F-score filter and RFE never picked a noise feature; forward and backward selection did, and they cost far more (1275 models for forward selection).
