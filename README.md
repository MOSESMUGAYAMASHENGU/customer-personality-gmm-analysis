🛍 STRATEGIC CUSTOMER SEGMENTATION USING GAUSSIAN MIXTURE MODELS (GMM)

📌 Executive Overview
This project develops a probabilistic customer segmentation framework using a Gaussian Mixture Model (GMM) to identify high-value behavioral segments based on product spending patterns.
Rather than applying basic clustering (e.g., K-Means), this analysis:
- Uses probabilistic soft clustering.
- Selects model complexity via Bayesian Information Criterion (BIC).
- Quantifies assignment confidence.
- Translates segmentation into actionable revenue strategy.

The objective is not just to cluster customers — but to create a segmentation system that directly informs marketing investment, loyalty design, and revenue optimization.

🎯 Business Problem
Retail businesses often apply broad, undifferentiated marketing strategies that fail to account for:
a) Spending intensity differences.
b) Category preference specialization.
c) Revenue concentration across customers.
d) Behavioral upgrade opportunities

This project answers:

How can we identify statistically distinct customer spending behaviors and convert them into strategic marketing actions?

📊 Dataset Overview
-Dataset: marketing_campaign.csv
-Observations: 2,240 customers
-Total Variables: 29 features

For segmentation, we focus on spending behavior across six product categories:
-MntWines
-MntFruits
-MntMeatProducts
-MntFishProducts
-MntSweetProducts
-MntGoldProds

These variables represent direct purchasing intensity and form the behavioral foundation of the segmentation model.

🧹 Data Preparation
The dataset was loaded and validated for segmentation suitability:
```python
import pandas as pd

df = pd.read_csv('marketing_campaign.csv', sep='\t')
print(df.shape)
print(df.isnull().sum())
```

Key Findings:
- Dataset shape: (2240, 29)
- Only 24 missing values in Income
- No missing values in the selected spending features
- Spending features required no imputation

Preparation Steps:
- Selected six spending variables
- Standardized features prior to modeling
- Verified distribution patterns and zero-spending frequency
- Ensured no multicollinearity distortions affecting clustering

### Feature Distributions After Scaling

![All Product Features After Scaling](PLOTS/All%20Product%20Features%20After%20Scaling.png)

⚙️ Modeling Approach
Why Gaussian Mixture Model?
Gaussian Mixture Models were selected because:
- Real customer segments are rarely spherical
- Soft clustering captures uncertainty
- Probabilistic assignments enable risk-aware targeting
- BIC provides principled model selection

Unlike K-Means, GMM provides:
- Flexible covariance structures
- Assignment probability per customer
- Statistical framework for model complexity control

📉 Model Selection:
GMM models were fit for K = 1 to 10 components.

Model evaluation used Bayesian Information Criterion (BIC):
```python
K=1:  BIC=33,405
K=2:  BIC=15,798
K=3:  BIC=11,459
K=4:  BIC=10,841
K=5:  BIC=9,662
K=6:  BIC=8,506
K=7:  BIC=8,279
K=8:  BIC=7,931
K=9:  BIC=7,898  ← Optimal
K=10: BIC=7,926
```
### Model Selection: BIC & AIC Curve

![Finding Optimal K using BIC & AIC](PLOTS/Finding%20Optimal%20K%20using%20BIC%20%26%20AIC.png)

✅ Optimal Model: 9 Components

The lowest BIC occurs at K = 9, indicating the best balance between fit and model complexity.


📊 Cluster Characteristics
Each customer receives:
- A most-likely cluster label
- A probability score (confidence level)

Average assignment confidence: ~99.6%

Below are simplified mean spending profiles:
```python
| Cluster | Wines | Fruits | Meat | Fish | Sweets | Gold |
| ------- | ----- | ------ | ---- | ---- | ------ | ---- |
| 0       | 1043  | 17     | 756  | 116  | 19     | 138  |
| 1       | 23    | 3      | 19   | 4    | 2      | 9    |
| 2       | 352   | 21     | 139  | 31   | 20     | 39   |
| 3       | 33    | 4      | 22   | 6    | 3      | 11   |
| 4       | 227   | 27     | 143  | 51   | 27     | 41   |
| 5       | 603   | 48     | 389  | 69   | 49     | 61   |
| 6       | 76    | 6      | 33   | 9    | 5      | 17   |
| 7       | 801   | 58     | 502  | 92   | 58     | 84   |
| 8       | 115   | 10     | 55   | 14   | 9      | 23   |
```
### Spending Distributions Per Product

![Spending Distributions Per Product](PLOTS/Spending%20Distributions%20Per%20Product.png)

### Clustering Results of Spending Patterns

![Clustering Results of the Spending Patterns](PLOTS/Clustering%20Results%20of%20the%20Spending%20Patterns.png)

🧠 Strategic Insights:
1️⃣ Revenue Is Highly Concentrated;
- Premium clusters represent a minority of customers but contribute a disproportionate share of total revenue.

Implication:
- Retention of premium clusters should be prioritized over broad acquisition.

2️⃣ Category Affinity Is Behaviorally Structured; 
Customers exhibit clear specialization patterns:
- Wine-dominant buyers
- Meat-focused customers
- Balanced premium consumers
- Ultra-low engagement shoppers

Implication:
Marketing should move from product-level promotions to segment-level targeting.

3️⃣ Mid-Tier Clusters Represent Growth Potential;
Moderate clusters:
- Spend consistently
- Show cross-category activity
- Have not yet reached premium intensity

Implication:
Design spend-escalation pathways to migrate mid-tier customers into premium segments.

4️⃣ Low-Spending Segments Require Controlled Strategy;
Low-value clusters may reflect:
- Price sensitivity
- Low engagement
- Early lifecycle stage
- Churn risk

Implication:
Avoid blanket discounting. Use targeted reactivation based on behavioral signals.

5️⃣ Probabilistic Targeting Reduces Marketing Risk;
Using GMM probability scores:
- High-confidence customers → Precision targeting
- Lower-confidence customers → Broader campaigns

This minimizes misallocation of marketing spend.

💼 Strategic Recommendations
🎯 1. Budget Allocation by Value Tier
```python
| Tier      | Strategy                | Budget Priority |
| --------- | ----------------------- | --------------- |
| Premium   | Retention & Exclusivity | High            |
| Mid-Tier  | Upsell & Expansion      | Medium-High     |
| Low-Spend | Selective Reactivation  | Controlled      |
```
🏆 2. Loyalty Architecture Redesign
Implement tier-based loyalty aligned to cluster structure:
- Gold Tier → Premium clusters
- Silver Tier → Mid-tier clusters
- Entry Tier → General population

Incorporate spend thresholds to encourage upward migration.

📈 3. Cross-Selling Intelligence
Behavior-driven product pairing:
- Wine-heavy → Premium meat pairings
- Meat-heavy → Curated wine bundles
- Balanced → High-margin gold product upsell

Move from descriptive segmentation to prescriptive personalization.

📊 4. Campaign Framework
Each campaign should include:
- Cluster targeting rule
- Confidence threshold (e.g., >95%)
 -Revenue uplift hypothesis
- Post-campaign lift evaluation

Segmentation becomes a measurable revenue engine.


🚀 Operationalization Roadmap
1) Integrate cluster labels into CRM system
2) Automate probability-based targeting rules
3) Monitor cluster migration quarterly
4) Track revenue uplift by segment
5) Refit GMM annually for behavioral drift


🧪 Technical Competencies Demonstrated
1) Probabilistic modeling (Gaussian Mixture Models)
2) Model selection via BIC
3) Behavioral feature engineering
4) Soft clustering & uncertainty quantification
5) Business translation of unsupervised learning
6) Strategic revenue optimization framing

📌 Future Enhancements
- Incorporate demographic variables (Income, Age, Marital Status)
- Add Recency & campaign response data
- Compare against K-Means and Hierarchical clustering
- Build uplift models per cluster
- Evaluate long-term customer lifetime value per segment

👤 Author

MOSES MUGAYA MASHENGU
Machine Learning & Strategic Analytics


🏁 Final Executive Summary
This project identifies nine statistically distinct customer segments using Gaussian Mixture Modeling with BIC-driven model selection.
The segmentation framework reveals:
- Revenue concentration among premium clusters
- Structured category preference behavior
- Upgrade potential in mid-tier customers
 -Operational risk in low-engagement segments

Most importantly, the model translates directly into:
- Revenue-weighted marketing allocation
- Loyalty program design
- Precision targeting using probabilistic confidence
- Measurable campaign optimization

This is not descriptive clustering.

It is a deployable customer intelligence framework designed for strategic revenue impact.
