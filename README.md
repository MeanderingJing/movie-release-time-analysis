# movie-release-time-analysis
## Objective
Investigates the relationship between a movie's release timing and its box office profitability
## Key Methods
- Chi-square test
- association rule mining
- multinomial logistic regression
## Dataset
- Sourced from Kaggle 
- Movies from 2003-2023, 2750 records
## Exporatory Data Analysis 
![image](https://github.com/user-attachments/assets/b3e2c94e-aad7-4e04-b540-f628d42b8c8f)

![image](https://github.com/user-attachments/assets/f764564d-43d8-41a4-8017-c91edf39aac3)
## Statistical Analysis (Chi-square Test of Independence)
### Statistically Significant Findings
- Chi-square statistic: 80.306
- p-value: < 0.05 (significant)
### Key Takeaway 
Release month and box office profitability (ROI) are statistically associated.
## Assocoation Rule Mining
![image](https://github.com/user-attachments/assets/1effecaf-b47f-41d0-9ae6-8cdff6a40681)
- Support: The frequency of an itemset appearing in the dataset
- Confidence: The confidence of a rule A → B is the likelihood that B appears in transactions where A appears.
- Lift:  Lift measures the strength of a rule over the random occurrence of B, given A. A lift value >1 means A and B are more likely to appear together than expected by chance, suggesting a meaningful association.












