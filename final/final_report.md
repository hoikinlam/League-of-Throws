# Predicting “Throws” in League of Legends Using Machine Learning
**Author:** Hoikin Lam  
**Date:** Dec 1, 2025

## Introduction: 
In competitive multiplayer games such as League of Legends (LoL), gaining an early lead does not guarantee victory. Even high-ranked teams frequently “throw” their games. In other words, they lose after establishing a measurable advantage. Previous analytics and community discussions tend to focus on win prediction, but far less attention is given to throw prediction: identifying whether a team with the upper hand is at risk of losing.

This paper presents a machine-learning classifier that predicts whether a leading team will “throw” their advantage and ultimately lose. Using \~200,000 matches across Master, GrandMaster, and Challenger ranks, I engineered advantage-based features, conducted correlation-driven reduction, evaluated multiple models, and applied hyperparameter tuning to develop a high-performing Random Forest throw classifier. Lastly, this paper will summarize the data preparation, exploratory analysis, model development, design choices, and final performance results.   
## Dataset Overview:
The dataset contains match statistics from the three highest competitive ranks in LoL. Each row represents one fully completed game, with attributes describing:   
* Team statistics (gold, kills, assists, deaths)
* Game objectives (first blood, first tower, dragon, etc.)
* Map control indicators (minions, wards)
* Win outcome (blueWins or redWins)
* Engineered throw labels

Across Master, GrandMaster, and Challenger, the dataset contains \~200,000 games, with approximately 10.3% labeled as throws. The rationale behind choosing these ranks is due to consistency and strategic depth among top-level players. Since higher ranked players tend to take the game more seriously, there is less randomness in gameplay which would skew the data.
## Definition: What is a throw?
In this project, a throw is defined as a team that secures a significant early advantage (≥3 early objectives) but still loses the match. The early objectives include: first blood, first tower, first dragon, first baron, and first inhibitor. However, there were clear limitations with the label and the data such as the data only containing the overall game duration rather than time stamps, which makes it harder to identify leads. Also, the throws only make up ~10% of the data, which causes class imbalances.  
## Exploratory Data Analysis & Feature Engineering:
From the eda notebook:   
* Throws spike during mid-games (~20 minutes), where volatility is highest.
* Moderate leads are more throw-prone than overwhelming leads.
* Differences between blue team and red team totals are more informative than raw totals.
* Throw behavior is consistent across ranks (Master vs GM vs Challenger).

To reflect relative performance rather than absolute totals, several engineered features were created:   
* Difference metrics: gold_diff, kills_diff, towers_diff, dragons_diff, barons_diff
* Ratio metrics: gold_ratio, kills_ratio, towers_ratio, dragons_ratio, barons_ratio
* Normalized metrics (per minute): gold_diff_norm, kills_diff_norm, objective_diff_norm, etc

These features were shown to outperform raw stats, reducing noise and focusing on advantage magnitude.
## Unsuccessful Feature Attempts:
The following features were tested but ultimately removed due to poor performance.

Ward-based map control features  
* Hypothesis: More wards → better map control → fewer throws.
* Reality: Wards strongly correlated with gold and kills; redundant.

Early-lead based scores  
* Attempted to create a weighted advantage score.
* Correlated too strongly with the throw label resulting in overfitting.
## Modeling Approach: 
I trained baseline and advanced models using an 80/20 split:   
**Models Evaluated**  
* Dummy Classifier (“Always No Throw”)
* Logistic Regression
* K-Nearest Neighbors (K=15)
* Decision Tree
* Random Forest (baseline)
* Gradient Boosting
* Random Forest with hyperparameter tuning

Each model was evaluated with:   
* Accuracy
* F1(throw) — main metric
* Macro F1
* ROC-AUC
* PR-AUC (critical for imbalanced data)

Overall, the random forest model performed the strongest due to its ability to capture non-linear relationships. Though, accuracy tends to be misleading potentially due to class imbalance. The random forest model has been tuned with various parameters in n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf. Then, the model is cross-validated and resulted in a F1 score of 0.627. Results from each model are included within the repository. Additionally, as noted before, to attempt to address the class imbalance, I explored various class weights and threshold tuning. 
## Conclusion:
This project demonstrates that predicting “throws” in high-ranked League of Legends matches is both feasible and informative. Despite throws making up only about 10% of games, the advantage-based engineered features such as gold ratio, tower ratio, and normalized objective differences, captured the underlying competitive dynamics well. These features represented meaningful shifts in map pressure and resource control that precede a thrown lead. Simpler linear models such as Logistic Regression struggled to learn these nonlinear interactions, while tree-based models consistently outperformed them. After hyperparameter tuning, the final Random Forest model achieved strong performance on the imbalanced task, including an F1 of 0.688, ROC-AUC of 0.952, and PR-AUC of 0.765, showing that it successfully recovered many true throws without over-predicting them.

One persistent challenge was the inherent class imbalance, which required careful model selection and class weighting to prevent the classifier from defaulting to the majority class. The confusion matrix still reflects this imbalance, but the model’s precision-recall and F1 scores confirm meaningful predictive ability. Throughout development, several additional ideas such as vision-control features, alternative early-lead formulas, and timing-based objective metrics were explored but excluded due to redundancy, instability, overfitting, or risk of leaking information from the outcome. Their consideration highlights the broader investigation and experimental depth of the project. While the dataset’s lack of event timestamps limited the ability to model leads over time, the results show that even static game-summary data contain measurable patterns associated with throwing. The final Random Forest model provides a strong foundation for identifying at-risk leads. Lastly, some considerations for future work could involve around champion data since different champions have different abilities and stat scalings that could influence gameplay (particularly the longer game durations), a temporal based match data could yield even richer insights by distinguishing clear early, mid, late game decision-making and momentum shifts.
