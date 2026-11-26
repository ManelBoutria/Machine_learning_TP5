Practical session 5: feature selection (Breast Cancer Wisconsin)

Filter and wrapper feature selection with logistic regression on load_breast_cancer (569 patients, 30 features).


The script prints the tables and saves the figures in ./figures (about 7-8 minutes; set RUN_ROBUSTNESS = False in part D.1 to skip the extra checks and save about 4 minutes). A shorter version without the comments is Work_session_5_short.py (about 2 minutes).

Rules followed
One seed (SEED = 42) and one stratified 70 % / 30 % split used everywhere.
Scaling and selection are inside a scikit-learn Pipeline, so they are refitted on the training folds only.
Every choice (threshold, k, number of features) uses 5-fold stratified cross-validation on the training set only.
The test set is used for the final table of D.3. The baseline test score in A.4 and the test column of B.2 are shown only because the exercise asks for them and were never used to choose anything.
F1-score is for the positive class (benign).
Summary

On this dataset logistic regression with all 30 features is already excellent (test accuracy 0.9883), and no selection method beats it. A 10-feature subset (RFE) keeps 0.9708 and a 20-feature non-redundant subset 0.9766, and the differences (2 to 3 patients out of 171) are not significant. The selected features agree with the correlation structure: one representative of each block of redundant size measurements, plus the independent worst texture and worst smoothness. Filters and RFE never picked a noise feature in the noise experiment, the wrappers did.

Part A. Data inspection and baseline

Baseline row. 30 features, test accuracy 0.9883 (169 of 171 patients), test F1 0.9907, training time about 0.01 s (training accuracy 0.9874). The DummyClassifier("most_frequent") only reaches 0.627 (0.626 on the test set, F1 0.770), so the model really learns the task.

Why this dataset is a good candidate for feature selection

Strong redundancy. 15 of the 435 pairs of features have |correlation| > 0.95 and 23 have |correlation| > 0.90. Mean radius, perimeter and area and the worst radius, perimeter and area form one block of 6 near-copies of the same measurement; the three "error" versions form a second block, and concavity / concave points form a third one (threshold 0.90).
Many features carry almost no information alone. texture error, symmetry error, fractal dimension error and mean fractal dimension have F-scores close to 0 (B.3), so a much smaller model may be enough.
A reference to compare with. A simple linear model already reaches 98.8 % (against 62.7 % for the dummy), so we can measure exactly what is lost or gained with fewer features; fewer features also give a model that is easier to interpret and cheaper to measure.
Part B.1. Variance threshold

Result (unscaled training data). Threshold 0 removes 0 features (there is no constant feature), threshold 0.01 removes 16, threshold 0.1 removes 19.

Why it is meaningless after standardization. StandardScaler gives every feature a variance of exactly 1 (checked above: minimum = maximum = 1), so no threshold below 1 can remove anything: the filter would only compare identical numbers.

Why it is of limited use on this dataset, even before scaling. The variance depends on the units of a feature, not on its information. worst area has a variance of about 3·10⁵ and fractal dimension error about 8·10⁻⁶ simply because areas are measured in hundreds and fractal dimensions in hundredths. Thresholds 0.01 and 0.1 just delete every feature measured on a small scale (smoothness, compactness, symmetry, fractal dimension, ...) and keep all large-scale ones, including the six near-identical size features. Some deleted features are useful: worst smoothness (variance 0.0005) is selected by 6 of the 7 methods of D.3. The only safe use of this filter is threshold 0 (constant columns), and there are none here.

Part B.2. Removing correlated features

Result (selection repeated inside every fold; the test accuracy column is displayed only because the exercise asks for it and was not used to choose anything):

threshold	features kept	CV accuracy (training set)	test accuracy (display only)
none	30	0.9824 ± 0.0189	0.9883
0.95	23	0.9824 ± 0.0189	0.9825
0.90	20	0.9824 ± 0.0189	0.9766
0.80	12	0.9447 ± 0.0220	0.9357

Is removing redundancy enough to improve the accuracy? No. Down to 20 features (threshold 0.90) the accuracy is the same as with 30: useful for simplicity, but not an improvement. At 0.80 the filter becomes harmful (CV accuracy falls by almost 4 points): the rule only compares features pairwise, so it chains deletions and removes eight of the ten "mean" features (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points) together with their informative neighbours. In addition, logistic regression with its L2 penalty already copes reasonably with correlated inputs, so there is little to gain from deleting copies. Threshold 0.90 (20 features; it ties with 0.95 on CV accuracy and keeps fewer features) is used in D.3.

Part B.3. F-score, mutual information and SelectKBest

Comment on the two rankings. They are almost the same: Spearman correlation 0.966, and the two top-10 lists contain exactly the same 10 features (worst concave points, worst perimeter, worst radius, worst area, mean concave points, mean perimeter, mean radius, mean area, mean concavity, worst concavity). Both give almost zero to texture error, symmetry error, fractal dimension error and mean fractal dimension. Mutual information could capture non-linear dependences, but here the classes are separated mostly monotonically, so it brings nothing more than the F-test.

Comment on accuracy against k (5-fold CV, selection inside the pipeline):

k	2	5	10	15	20	25	30
F-score	0.935	0.952	0.952	0.960	0.987	0.982	0.982
mutual information	0.937	0.955	0.952	0.960	0.980	0.982	0.982

The accuracy is stuck around 0.95–0.96 up to k = 15 and jumps at k = 20. The smallest k within one standard deviation of the best is k = 20 for both scores (k = 15 is more than one standard deviation below the best).

Why can a univariate filter select nearly identical features? Each feature is scored alone against the target. radius, perimeter and area of the same nucleus are almost the same measurement (correlation > 0.95), so they all get a high score together, and the filter cannot know that after keeping one of them the others add nothing. Six of the 10 best F-score features belong to this family. The slots of a small k are therefore filled with copies, which is why the accuracy stays flat until k = 20. The complementary features that really add information (worst texture, worst smoothness) have only a mediocre univariate score (ranks 16 and 18) and enter much later.

Part C.1. Forward selection

Comments. The two implementations select the same 10 features (mean radius, mean texture, mean perimeter, mean compactness, texture error, smoothness error, worst radius, worst texture, worst perimeter, worst smoothness). This is expected: same estimator, same folds and the same tie rule (the first best candidate wins), so the greedy path is identical.

The CV accuracy rises quickly: 0.922 with 1 feature (worst radius), 0.967 with 2 (worst smoothness), 0.982 with 3 (worst texture), 0.985 with 4 (mean radius), and then stays on a plateau (0.985–0.990) up to 10 features. With the exercise's stopping rule (gain < ε = 0.001) the algorithm stops after 4 features (worst radius, worst smoothness, worst texture, mean radius), because the 5th feature brings a gain of 0. As the CV accuracy is an average over 398 patients, one more correct patient changes it by 0.0025, so ε = 0.001 means "stop as soon as no additional patient is gained". The cost of this rule is that it ignores the small gains that come later (+0.0025 at the 8th and the 10th feature).

It is also instructive that the second feature chosen is worst smoothness, which has a poor univariate score (rank 18 in B.3): it is selected because it is complementary to worst radius, which a filter cannot see.

Part C.2. Backward elimination

Comments (C.2.2). Again scikit-learn and my implementation retain the same 10 features (area error, smoothness error, fractal dimension error, worst radius, worst texture, worst perimeter, worst area, worst smoothness, worst concavity, worst symmetry). The curve starts at 0.9824 with 30 features, rises to 0.9899 when the redundant features are removed (24 down to 13 features), and then falls slowly (0.9874 with 12 and 11, 0.9849 with 10 features). The smallest size within one standard deviation of the best (0.9899 ± 0.0095) is 3 features (worst texture, worst perimeter, worst smoothness).

Forward versus backward (C.2.3). They share only 5 of 10 features (smoothness error, worst radius, worst texture, worst perimeter, worst smoothness; Jaccard similarity 0.33). The CV accuracy of the two subsets is 0.9899 ± 0.0095 for forward and 0.9849 ± 0.0123 for backward.

Why are they not identical? Both algorithms are greedy and they follow different paths. Forward starts from nothing and keeps forever what it added first: it takes one member of the size family (worst radius) and then adds complementary features (worst smoothness, worst texture, mean radius, ...). Backward starts with all 30 features and first discards those that are redundant given the others (mean radius, mean perimeter, mean texture are the first three to go), and a feature that has been removed can never come back. Moreover the accuracy moves in coarse steps of 0.0025, so many candidate moves tie and the first one in column order is taken; many different subsets of 10 features have almost the same CV accuracy, so the "best" subset is not unique.

Part C.3. Recursive feature elimination (RFE)

RFE with logistic regression (C.3.1). The 10 selected features (ranking 1) are mean area, mean concave points, area error, worst radius, worst texture, worst perimeter, worst area, worst smoothness, worst symmetry and worst concave points. The next ones are mean compactness (2), mean radius (3), radius error (4), worst concavity (5), and the first eliminated are worst compactness (21), mean fractal dimension (20), mean symmetry (19) and worst fractal dimension (18). The ranking keeps several members of the size family (worst radius, perimeter, area, mean area, area error): the L2-penalized model spreads the weight over correlated features, so none of them gets a weight small enough to be eliminated early. worst compactness, which has a high univariate score, is eliminated first, probably because its information is already carried by correlated features present in the model.

RFECV (C.3.2). The optimal number of features is 8 (mean area, area error, worst radius, worst texture, worst perimeter, worst area, worst smoothness, worst concave points) with a CV accuracy of about 0.985. The curve is flat (about 0.982) above 15 features, irregular between 9 and 14 and falls below 8; since the standard deviations are 0.015–0.03, the differences between 8 and 30 features are smaller than one standard deviation, so the optimum is only weakly determined.

Other estimators (C.3.3). The selections are not the same. The linear SVC shares 8 of its 10 features with logistic regression, the random forest only 6 (and 5 with the SVC). Only 4 features are chosen by all three: worst radius, worst perimeter, worst area, worst concave points. RFE eliminates according to the importance given by the estimator: linear models use |weight| on standardized features, so complementary variables such as worst texture or worst smoothness are kept; the random forest uses impurity-based importance, which is dominated by the strongest single features (size and concavity) and keeps mean radius, mean perimeter, mean concavity, but not texture or smoothness. A RFE selection is therefore relative to the model used.

RFE versus backward elimination (C.3.4). RFE removes the feature with the smallest importance (|coefficient|) of one fitted model, without cross-validation: for 30 → 10 features it trains about 21 models (20 eliminations + the final fit). Backward elimination removes the feature whose removal gives the best cross-validated accuracy: at each step it trains one model per candidate and per fold, 5 × (30 + 29 + ... + 11) = 2050 models for the same 30 → 10 path, about 100 times more. RFE is much cheaper but relies on a proxy (the weights) instead of the real validation performance; the two 10-feature subsets share 7 features.

Part C.4. Cost of the wrapper search

Forward selection with p = 30, k = 10 and 5-fold cross-validation: at step i = 0, ..., 9 the algorithm evaluates the 30 − i remaining features, each with 5 models, so

5 × (30 + 29 + ... + 21) = 5 × 255 = 1275 models (255 candidate subsets).

Exhaustive search: all non-empty subsets, (2³⁰ − 1) × 5 = 1 073 741 823 × 5 = 5 368 709 115 models; only the subsets of exactly 10 features, C(30,10) × 5 = 30 045 015 × 5 = 150 225 075 models.

Check against the measured time. My implementation counts exactly 1275 models. One model (fit + score on a fold) takes about 4 ms, which predicts about 5 s for the forward search; the measured time is about 5–6 s (own implementation and scikit-learn), so the count is consistent (the small excess is overhead). With the same time per model, the exhaustive search would need about 250 days for all subsets and about 7 days for the subsets of exactly 10 features on one core, which is why wrappers are always greedy.

Part C.5. Extra check: is the greedy result optimal?

With 3 features the exhaustive search is feasible (4060 subsets). The best CV accuracy is 0.9824, reached by {worst radius, worst texture, worst smoothness}: this is exactly the subset obtained by forward selection at step 3 (rank 1 of 4060). Backward elimination at 3 features gives {worst texture, worst perimeter, worst smoothness}, which also reaches 0.9824 (two subsets tie for the best). So on this dataset the greedy searches find an optimal subset for k = 3. This is an observation, not a guarantee (see question 4 of the final discussion).

Part D.1. Noise injection

Answer D.1 (10 features kept by each method, 20 noise features available):

seed 42, noise columns placed last: SelectKBest (F-score) 0 noise features, RFE 0, Forward Selection 2 (noise_00, noise_05), Backward Elimination 4 (noise_00, noise_12, noise_17, noise_18).
three other noise draws (seeds 1, 2, 3): F-score 0, 0, 0 | RFE 0, 0, 0 | Forward 2, 0, 2 | Backward 3, 3, 5.
same experiment with the 50 columns in random order (seeds 42, 1, 2): F-score 0, 0, 0 | RFE 0, 0, 0 | Forward 3, 4, 3 | Backward 2, 1, 3.
Most robust: the F-score filter and RFE with logistic regression: never a noise feature in any of the 7 runs. Most likely fooled: the two wrappers: over the 7 runs forward selected 0 to 4 noise features (mean 2.3) and backward 1 to 5 (mean 3.0) out of 10.
Why: a noise feature has an F-score close to 1 (real features: tens to hundreds) and a weight close to 0 in the logistic regression, so univariate scoring and weight-based elimination rank it last. The CV accuracy used by forward/backward is a noisy and coarse criterion (steps of 0.0025 on a plateau around 0.98): adding or keeping a noise feature can gain one patient by chance and is then accepted, and nothing in the criterion penalizes useless features.
Which wrapper is worse depends on the column order: with the noise columns last, backward is worse; with the columns in random order, forward is worse. The reason is the tie-breaking: when several candidates give the same CV accuracy, scikit-learn takes the lowest column index, so the order of the columns decides which feature is kept among tied candidates. So the difference between forward and backward is partly an artefact; the robust conclusion is filter/RFE-weights < wrappers in noise sensitivity.
Part D.2. Leakage

Result. On the 50-feature training set, with SelectKBest(F-score, k = 10) + logistic regression:

protocol	5-fold CV accuracy
incorrect (selection on all training rows, then CV)	0.9523
correct (scaler + selector + classifier in a Pipeline)	0.9523

The gap is 0.000, and averaged over 30 different fold assignments it is only +0.0003 (standard deviation 0.0016): the leak is invisible here.

How information leaks. In the incorrect protocol the F-scores are computed with the labels of all training patients, including those that will later be the validation fold. The 10 chosen features are therefore the ones that best separate the validation patients too, so the validation fold is no longer "unseen" and the accuracy is optimistic. In the Pipeline, the scaler and SelectKBest are refitted on the 4 training folds at each split, and the validation fold is only used for scoring.

Why the gap is small here. (i) The signal is strong: the same top features (size and concavity) are chosen whichever rows are used, so adding the validation rows hardly changes the selected set; in D.1 the F-score filter never selected a noise feature. (ii) There are 398 observations for 50 features, so the chance correlation of a noise feature with the label is tiny compared with the true signal.

When it becomes large. With few observations, many features and a weak signal (genomics, imaging, text). Demonstration above: 60 observations, 2000 features of pure noise, random labels. The incorrect protocol gives 0.860 accuracy (it looks like a good classifier!), the correct one 0.507, which is chance level and the truth. The more flexible the selection (wrappers, tuning k), the larger the effect.

Part D.3. Final comparison on the test set

Results on the test set (each pipeline is fitted on the training set and evaluated once; selection times are those of this run and only indicative):

Method	Nb features	Test accuracy	Test F1	Selection time
Baseline (all features)	30	0.9883	0.9907	–
Filter: correlation (B.2, 0.90)	20	0.9766	0.9813	about 0.01 s
Filter: SelectKBest, F-score	20	0.9766	0.9813	about 0.005 s
Filter: SelectKBest, mutual information	20	0.9708	0.9770	about 0.06 s
Wrapper: Forward Selection	4	0.9415	0.9528	about 2 s
Wrapper: Backward Elimination	3	0.9532	0.9626	about 12 s
Wrapper: RFE	10	0.9708	0.9767	about 0.06 s
Wrapper: RFECV	8	0.9591	0.9671	about 0.7 s

Comments. (1) No selection method beats the baseline: with 171 test patients, 0.9883 means 169 correct. The filters keep 20 features and lose 2–3 patients, RFE keeps 10 and loses 3, RFECV 8 and loses 5, backward (3 features) loses 6 and forward (4 features) loses 8. (2) The very small subsets of forward and backward are the worst on the test set although their CV accuracy was about 0.98: their sizes were chosen by rules that cut the plateau (ε = 0.001, one standard deviation), and a subset picked as the best on the same folds has an optimistic CV score. (3) Filters and RFE are fast (milliseconds); forward and backward take seconds and tens of seconds.

Presence matrix and consistency with Part A. Only worst texture is selected by all 7 methods; worst perimeter and worst smoothness are selected by 6 of 7. This is consistent with the correlation structure of Part A:

The univariate filters (F-score, mutual information) keep all 6 members of the radius / perimeter / area block and all 3 error copies, whereas the correlation filter keeps 1 of each, backward 1 and 0, forward 2 and 0, RFE 4 and 1.
The features that almost every method agrees on are the ones that are not part of any redundant block (worst texture, worst smoothness): they carry information that nothing else carries. For the size block, the methods agree that one representative is needed but not which one (worst radius, worst perimeter, worst area, mean area, ...), because the members are interchangeable.
Final discussion

1. Which method gives the best test accuracy with the fewest features? Is the difference with the baseline significant? No reduced model beats the baseline: it has the best test accuracy (0.9883, 30 features). The best compromise is RFE with 10 features (0.9708), which loses 3 patients of 171; the filters with 20 features (0.9766 for correlation and F-score) lose 2. These differences are not significant: one patient is 0.58 points of accuracy, the standard error of an accuracy of 0.988 on 171 patients is about 0.8 points, and McNemar's exact test gives p = 0.5 (correlation, F-score), 0.25 (mutual information, RFE) and 0.06 (RFECV). Only forward selection (4 features, 8 patients lost, p = 0.008) and backward elimination (3 features, 6 patients lost, p = 0.031) are lower than the baseline, and with 7 comparisons these p-values are borderline after correction.

2. Advantages and limits of filters compared with wrappers. Filters are very fast (milliseconds), independent of any model, and scale to thousands of features. But they score each feature alone: they ignore redundancy (6 of the 10 best F-scores are copies of the size feature, and accuracy stays around 0.95–0.96 for k ≤ 15) and interactions (worst smoothness has a mediocre F-score, rank 18, yet forward selection picks it second because it complements worst radius). Wrappers evaluate subsets with the real model and metric, so they handle redundancy and complementarity and are tuned to the model, but they are slower and the result depends on the model (RFE with a random forest shares only 6 of 10 features with RFE with logistic regression) and on the noisy CV estimate.

3. Why are wrappers more expensive? They must train the model for every candidate subset and every fold. Forward selection with p features, k selected and F folds trains F × (p + (p − 1) + ... + (p − k + 1)) = F × (k·p − k(k − 1)/2) models: 1275 for p = 30, k = 10, F = 5 (measured: seconds, against 5 ms for the F-score filter), and about 99 000 for p = 1000 and k = 20. An exhaustive search would need 2ᵖ − 1 subsets, i.e. more than 10⁹ for p = 30.

4. Do forward selection and backward elimination guarantee the optimal subset? No. They are greedy: forward can never remove a feature it has added, backward can never bring back a feature it has removed, so they cannot find a subset that needs a swap. In our results the 10-feature subsets of the two methods differ (5 features in common) and have different CV accuracies (0.9899 for forward, 0.9849 for backward), so at least the backward subset is not optimal for size 10. For k = 3, where exhaustive search is possible (C.5), both greedy searches did reach the optimum (0.9824), so greedy methods often work well in practice, but without any guarantee. Many subsets also tie because the accuracy changes by steps of 0.0025.

5. Behaviour in front of the 20 noise features. The F-score filter and RFE (logistic regression) never selected a noise feature (0 of 10 in all 7 runs of D.1). Forward selection selected 0 to 4 and backward elimination 1 to 5 noise features out of 10, depending on the noise draw and on the column order. Filters and weight-based RFE rely on a statistic with a strong signal-to-noise ratio (F-score near 1 or weight near 0 for a noise feature), so they are robust here. Wrappers optimize a noisy, coarse CV accuracy on a plateau, so a useless feature can be accepted by chance (and ties are broken by column order): they are the family most likely to be fooled. Which of forward and backward is worse changes with the column order, so no conclusion should be drawn between those two.

6. Why must feature selection be performed inside the cross-validation loop, and how does a Pipeline prevent leakage? Selecting features is part of the model fitting: it uses the labels. If it is done before cross-validation on all rows, the validation folds have already influenced the model (the chosen features are the ones that happen to separate those patients), and the CV score is biased upwards, as shown in D.2 (0.860 against 0.507 on random labels). A Pipeline bundles scaler, selector and classifier into one estimator, so cross_val_score refits all the steps on the training folds only and applies them to the validation fold; the same holds for the final test set, which is only transformed, never used for fitting.

7. In which situation would you choose each family?

Thousands of features: filters (univariate scores, mutual information, correlation pruning) as a first screening, possibly followed by RFE or an embedded method on the reduced set; wrappers such as forward selection would need hundreds of thousands of models.
Small dataset: simple filters or strongly regularized models, with the whole selection inside a (nested) cross-validation; flexible wrappers overfit the CV estimate (here the 3–4 feature subsets had 0.98 CV accuracy but 0.94–0.95 on the test set) and leakage is largest.
Need for interpretability: a short, non-redundant list: correlation pruning followed by a filter, or RFE/forward selection with a fixed small number of features, accepting a small loss of accuracy.
Limited computing time: filters (milliseconds) or RFE (about 20 fits); avoid forward/backward searches.
Moderate number of features (tens) and a fixed model whose accuracy matters most: wrappers (RFECV, forward/backward) are affordable and tune the subset to the model.

Take-away for this dataset. Logistic regression with all 30 features is already excellent (98.8 %) and feature selection does not improve it, but a 10-feature subset (RFE) keeps 97.1 % and a 20-feature non-redundant subset 97.7 %, and the selection is consistent with the correlation structure: one representative per block of redundant size measurements plus the independent texture and smoothness features.

All numbers come from a single run with SEED = 42 (scikit-learn 1.9.1); small differences may appear with another version. Selection times are indicative.