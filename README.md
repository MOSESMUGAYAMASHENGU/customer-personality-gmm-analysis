# Customer Personality Analysis: From Spending Patterns to Strategic Growth

This case study is a deep-dive customer segmentation project using **Gaussian Mixture Models (GMM)** to understand customer spending behavior and propose targeted marketing strategies. The analysis follows a structured workflow: **Problem Scoping → Data Preparation → Modeling → Analysis → Recommendations**.

## Tech Stack

**Python:**
- pandas (data manipulation)
- numpy (numerical operations)
- scikit-learn (GMM, PCA, StandardScaler)
- matplotlib / seaborn (visualizations)

---

## Business Task

Design data-driven marketing strategies to increase customer lifetime value by understanding natural spending segments and activating high-potential customer groups.

## Data Source

**Customer Personality Analysis Dataset** (Kaggle)
- 2,240 customers
- 29 attributes including demographics, spending across 6 product categories, and campaign responses

---

## Ask

- How do customers naturally segment based on their spending patterns across product categories?
- What distinguishes high-value customers from low-value ones?
- Which customers are "uncertain" and ready to upgrade?
- How can we target each segment with personalized marketing strategies?

---

## Prepare

The dataset was loaded and assessed for quality:

```python
import pandas as pd
df = pd.read_csv('marketing_campaign.csv', sep='\t')
print(df.shape)  # (2240, 29)
print(df.isnull().sum())  # Only Income has 24 missing values (1.07%)

Key preparation steps:

    Selected 6 spending columns for segmentation: MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds

    Confirmed no missing values in spending columns

    Analyzed zero-spending patterns to understand category penetration

Process
Data Processing Pipeline

1. Exploratory Analysis

    Identified zero-spending patterns: Meat (0% zeros), Wine (0.6% zeros) are universal; Fruits (17.9%), Fish (17.1%), Sweets (18.7%) have significant non-buyers

    Detected outliers (keeping them as premium customer signals)

https://PLOTS/Spending%2520Distributions%2520Per%2520Product.png

2. Feature Scaling (Critical for GMM)

    Applied StandardScaler to normalize all 6 spending features to mean=0, std=1

https://PLOTS/All%2520Product%2520Features%2520After%2520Scaling.png

3. Dimensionality Reduction for Validation

    Applied PCA to reduce 6D → 2D while preserving 68.4% of variance

    PC1 (56.1%): Spending volume

    PC2 (12.3%): Specialist vs generalist orientation

4. Model Selection

    Tested K=1 through K=10 using BIC and AIC

    BIC suggested K=9 (conservative), AIC suggested K=10

    Selected K=9 to avoid overfitting

K=1: BIC=33,405  |  K=6: BIC=8,506
K=2: BIC=15,798  |  K=7: BIC=8,279
K=3: BIC=11,459  |  K=8: BIC=7,931
K=4: BIC=10,841  |  K=9: BIC=7,898 ← OPTIMAL
K=5: BIC=9,662   |  K=10: BIC=7,926

https://plots/Finding%2520Optimal%2520K%2520using%2520BIC%2520%2526%2520AIC.png

5. Final GMM Training

    n_init=10 (multiple random starts to avoid local optima)

    covariance_type='full' (allow ellipsoidal clusters)

    Converged in 82 iterations

    Log-likelihood: -1.32

6. Confidence Scoring

    Generated probabilistic assignments for every customer

    Mean confidence: 0.910 | Median: 0.973

    83.2% have confidence > 0.8

Analysis & Share
Cluster Profiles (Average Spending)
Cluster	Name	Size	Defining Characteristics	Confidence
0	"Almost Nothings"	17.5%	75-96% below avg across ALL categories	0.949
1	"Light Shoppers"	11.1%	58-72% below avg, but buy some Gold	0.902
2	"Meat & Wine Lovers"	6.3%	Wine +71%, Meat +163%, Gold +142%	0.845
3	"Wine & Gold Light"	10.1%	Buy only Wine/Gold, avoid food	0.930
4	"Ultra-Premium All-Rounders"	11.7%	+67% to +242% across ALL categories	0.913
5	"Premium No-Gold"	11.7%	Highest wine (+99%) but avoid gold	0.834
6	"True Zeros"	12.0%	Zero on Fruits/Sweets, near-zero everything	0.958
7	"Wine Specialists"	8.7%	Wine +74%, average elsewhere	0.871
8	"Wine & Gold Heavy"	10.9%	Wine +65%, Gold +83%, avoid food	0.926

https://PLOTS/Clustering%2520Results%2520of%2520the%2520Spending%2520Patterns.png
Revenue Concentration (The 40/20 Rule)

Cluster 4 (11.7% of customers) drives:

    39.9% of Sweets revenue

    35.7% of Fish revenue

    34.4% of Fruits revenue

    32.4% of Gold revenue

    28.4% of Meat revenue

    19.6% of Wine revenue

Key Behavioral Patterns

1. Specialist vs Generalist Spectrum

    Generalists: Cluster 4 (all categories)

    Specialists: Cluster 2 (meat+wine), Cluster 7 (wine), Cluster 8 (wine+gold), Cluster 3 (wine+gold light)

2. The Uncertainty Opportunity

    16.8% (376 customers) have confidence < 0.8

    These customers sit between segments—actively considering expanding their purchases

    Clusters 2,5,7 show lowest confidence → prime for upselling

3. Category Penetration Patterns

    Wine and Meat are staples (near-universal purchase)

    Fruits, Fish, Sweets are premium indicators (only high-spenders buy consistently)

    Gold creates a specialist split (some premium customers buy, some avoid)

Act
How do the 9 customer segments differ in their spending behavior?

    Ultra-Premium segment (Cluster 4) shows balanced high spending across all categories—they are "all-rounders" who value the full product range

    Specialist segments (Clusters 2,7,8) concentrate spending in 1-2 categories—they have clear preferences but untapped potential in others

    Low-value segments (Clusters 0,6) spend 75-99% below average—they represent either new customers, budget-conscious shoppers, or mismatched prospects

Why would low-value customers upgrade?

    Value realization - Customers in Cluster 1 (Light Shoppers) who gradually increase spending may realize they'd save with targeted bundles

    Category discovery - Specialists (Clusters 2,7,8) who try new categories may discover broader value

    Convenience - Premium features (early access, personal shopping) appeal to Cluster 4

    Social proof - VIP program exclusivity drives aspiration in Clusters 5 and 2

    Seasonal triggers - Holiday periods may convert gift-buyers (Clusters 3,8) into regulars

How can the company use digital media to influence upgrades?

    Geo-targeted advertisements - Focus on communities with high concentration of "Uncertain" segments (Clusters 2,5,7)

    Pre-holiday campaigns - Launch gold campaigns before gifting seasons for Clusters 5 and 2

    Usage Pattern Messaging - Show Meat & Wine Lovers (Cluster 2) how much they'd save with "Butcher's Block" subscription

    Specialist packages - Wine-only weekend packages for Cluster 7

    Trial programs - One-month "premium experience" for Light Shoppers (Cluster 1)

    Bundle discounts - Wine + Gold bundles for Cluster 5 (Premium No-Gold)

    Category discovery content - Recipe ideas pairing wine with fish/seafood for Cluster 2

Recommendations Summary
Segment	Strategy	Offer
Cluster 4 (Ultra-Premium)	Retain & deepen	VIP "Black Card" program
Cluster 5 (Premium No-Gold)	Convert on gold	"Complete Your Collection" campaign
Cluster 2 (Meat & Wine)	Lock in subscription	"Butcher's Block" monthly box
Cluster 7 (Wine Specialists)	Expand categories	Wine & Food pairing events
Clusters 3,8 (Wine+Gold)	Maintain & cross-sell	Gift bundles for holidays
Cluster 1 (Light Shoppers)	Nurture & educate	Category discovery content
Clusters 0,6 (Zeros)	Reactivate or suppress	Win-back offers + surveys
About

Capstone project demonstrating advanced unsupervised learning techniques for customer segmentation. Uses GMM for soft clustering and PCA for dimensional validation to uncover actionable business insights.

Topics: python scikit-learn gaussian-mixture-models pca customer-segmentation unsupervised-learning data-science marketing-analytics
