from typing import List
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Data preprocessing and machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
)
import statsmodels.api as sm

def preprocess_movie_dataset(movie_df:pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess movie dataset for data mining analysis.
    
    Parameters
    -----------
    movie_df: pd.DataFrame
        Input movie dataset
    
    Returns
    -------
        pd.DataFrame: Preprocessed and cleaned dataset
    """
    ############################# Subset creation ##############################
    # Select movie data between 2003 and 2023
    start_date = "1983-01-01"
    end_date = "2023-12-31"

    # Convert production date early to ensure consistent datetime handling
    movie_df['production_date'] = pd.to_datetime(movie_df['production_date'])
    # Filter out invalid dates first
    movie_df = movie_df.dropna(subset=['production_date'])
    # Sort and filter by date range
    filtered_df = movie_df[
        (movie_df['production_date'] >= start_date) &
        (movie_df['production_date'] <= end_date)
    ].sort_values(by='production_date')

    ######################## Feature selection and renaming #########################
    # Confirmed with the Internet that production_date here refers to initial release date
    selected_features = ['movie_title','production_date','genres', 'movie_averageRating', 'Production budget $', 'Domestic gross $', 'Worldwide gross $']
    filtered_df = filtered_df[selected_features]
    filtered_df.columns = ['movie_title', 'release_date', 'genre', 'avg_rating',
       'prod_budget', 'domestic_gross', 'global_gross']
    # print(filtered_df.head(10))

    ############################ Data Cleaning ##############################
    # Remove rows with any duplicates or missing values
    df_cleaned = filtered_df.drop_duplicates()
    df_cleaned = df_cleaned.dropna()

    # Validate budget and gross columns 
    df_cleaned = df_cleaned[
        (df_cleaned['prod_budget'] > 0) & (df_cleaned['domestic_gross'] > 0)
    ]

    ##################### Feature transformation & creation ##################
    # Simplify genre by keeping only the first genre
    df_cleaned['genre'] = df_cleaned['genre'].str.split(',').str[0].str.lower().str.strip()

    # Convert release_date to datetime and extract month
    df_cleaned['release_month'] = df_cleaned['release_date'].dt.strftime('%b')
    # df_cleaned['release_month'] = df_cleaned['release_date'].dt.month
    # print(f"df_cleaned['release_month'] is {df_cleaned['release_month']}")


    # Create ROI feature
    df_cleaned['ROI'] = (df_cleaned['global_gross'] - df_cleaned['prod_budget'])/df_cleaned['prod_budget']
    # print(df_cleaned[['movie_title', 'ROI', 'prod_budget', 'domestic_gross']])

    # Convert the 'global_gross' column to millions
    df_cleaned['global_gross_millions'] = df_cleaned['global_gross'] / 1_000_000
    # Format the column to display 2 decimal places for readability
    df_cleaned['global_gross_millions'] = df_cleaned['global_gross_millions'].round(2)

    # Convert the 'prod_budget' column to millions
    df_cleaned['prod_budget_millions'] = df_cleaned['prod_budget'] / 1_000_000
    # Format the column to display 2 decimal places for readability
    df_cleaned['prod_budget_millions'] = df_cleaned['prod_budget_millions'].round(2)

    ###################### Categorize continous variables ######################
    _categorize_features(df_cleaned)
    # Validate categorization
    categorical_cols = ['budget_category', 'rating_category', 'roi_category']
    original_cols = ['prod_budget', 'avg_rating', 'ROI']
    # _validate_categorization(df_cleaned, categorical_cols, original_cols)

    # Basic validation
    # print(f"Rows after cleaning: {len(df_cleaned)}")
    # print("\nGenre distribution:")
    # print(df_cleaned['genre'].value_counts())
    # print("\nRelease month distribution:")
    # print(df_cleaned['release_month'].value_counts())

    df_cleaned.to_csv('processed_movie_data.csv')
    return df_cleaned

def _categorize_features(df: pd.DataFrame, budget_col:str='prod_budget_millions', rating_col:str='avg_rating', roi_col:str='ROI', n_categories:int=3):
    """
    Helper function to categorize continous variables in the provided DataFrame and 
    add the category features to the DataFrame. 
    """
    # Categorize budget
    df['budget_category'] = pd.qcut(
        df[budget_col], 
        q=n_categories, 
        labels=['Low Budget', 'Moderate Budget', 'High Budget']
    )

    # Categorize rating
    df['rating_category'] = pd.qcut(
        df[rating_col], 
        q=n_categories, 
        labels=['Low Rating', 'Moderate Rating', 'High Rating']
    )

    # Categorize ROI
    df['roi_category'] = pd.qcut(
        df[roi_col], 
        q=n_categories, 
        labels=['Low ROI', 'Moderate ROI', 'High ROI']
    )
    # print(df[['prod_budget', 'budget_category', 'avg_rating', 'rating_category', 'ROI', 'roi_category']].head(20))

def _validate_categorization(df:pd.DataFrame, categorical_cols: List[str], original_cols: List[str]):
    """
    Helper function for statistical validation of the categorizations

    The Kruskal-Wallis H-test helps us understand if our categorizations are meaningful by 
    statistically testing whether the groups have significantly different median returns.
    """
    # Distribution analysis
    for col in categorical_cols:
        print(f"\n{col} Distribution:")
        print(df[col].value_counts(normalize=True))
    
    # Statistical test (ANOVA or Kruskal-Wallis)
    for col, orig_col in zip(categorical_cols, original_cols):
        # Group original continuous values by categories
        groups = [group[orig_col].values 
                  for name, group in df.groupby(col, observed=False)]
        
        # Kruskal-Wallis H-test (non-parametric ANOVA)
        h_statistic, p_value = stats.kruskal(*groups)
        
        print(f"\n{col} Significance Test:")
        print(f"H-statistic: {h_statistic}")
        print(f"p-value: {p_value}")

def plot_global_monthly_revenues(df: pd.DataFrame):
    """
    Create a line graph to compare the median and mean domestic revenues across each release month.
    """
    # Group by month and calculate mean and median revenues
    monthly_revenues = df.groupby('release_month').agg({
        'global_gross_millions': ['mean', 'median']
    }).reset_index()
    
    # Flatten column names
    monthly_revenues.columns = ['month', 'mean_revenue', 'median_revenue']
    print(monthly_revenues)

    # Create line plot
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_revenues['month'], monthly_revenues['mean_revenue'], 
             marker='o', color='blue', label='Mean Revenue')
    plt.plot(monthly_revenues['month'], monthly_revenues['median_revenue'], 
             marker='o', color='red', label='Median Revenue')
    
    plt.title('Mean and Median Movie Global Gross Revenues by Release Month')
    plt.xlabel('')
    plt.ylabel('Global Gross (MM$)')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
def plot_monthly_roi_boxplot(df:pd.DataFrame):
    """
    Plot the ROI distribution across release month using a boxplot.
    """
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='release_month', y='ROI', data=df, 
                boxprops=dict(alpha=.7),
                whiskerprops=dict(color='black'),
                medianprops=dict(color='red'),
                showfliers=True,  # Show all outliers
                flierprops=dict(marker='o', markerfacecolor='red', markersize=3, 
                                linestyle='none'))
    
    # Set y-axis to log scale
    plt.yscale('symlog')  # Symmetrical log scale
    
    plt.title('Movie ROI Distribution by Release Month (Log Scale)')
    plt.xlabel('Month')
    plt.ylabel('Return on Investment % (Log Scale)')
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.tight_layout()
    plt.show()


def chi_square_test_for_release_month_roi(df:pd.DataFrame):
    """
    Use Chi-square test to determine whether a statistically significant association exists between 
    the release month and box office profitability.
    """
    # Create contingency table
    contingency_table = pd.crosstab(df['release_month'], df['roi_category'])

    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    print("\nChi-Square Statistic:", chi2)
    print("p-value:", p_value)
    print("Degrees of Freedom:", dof)

    # Interpret results
    alpha = 0.05
    if p_value < alpha:
        print("\nSignificant association exists between release month and ROI.")
    else:
        print("\nNo statistically significant association between release month and ROI.")

def plot_month_roi_heatmap(df:pd.DataFrame):
    """
    Plot a heatmap for movie counts by release month and ROI category.
    """
    # Define month order
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Create pivot table with ordered months
    pivot = df.groupby(['release_month', 'roi_category'], observed=False).size().unstack(fill_value=0)
    print(pivot)
    
    # Reindex to ensure chronological order
    pivot = pivot.reindex(month_order)

    # Visualize the relationship
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot.T, annot=True, cmap='YlGnBu', fmt='g')
    plt.title('Movie Count by Release Month and ROI Category')
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

def mine_association_rules(df:pd.DataFrame, min_support:float=0.02, min_confidence:float=0.4):
    """
    Generate association rules between release month and ROI categories.
    """
    # Prepare transactions
    transactions = df.apply(lambda row: [row['release_month'], row['roi_category']], axis=1).tolist()

    # Convert transactions to one-hot encoded DataFrame
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    associations_df = pd.DataFrame(te_ary, columns=te.columns_)

    # Generate frequent itemsets
    frequent_itemsets = apriori(associations_df, min_support=min_support, use_colnames=True)
    # print(frequent_itemsets.sort_values(by='support'))

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=None)

    print(rules.sort_values("lift", ascending=False)[['antecedents','consequents', 'support', 'confidence', 'lift']].head())

class MulticlassMoviePerformancePredictor:
    def __init__(self, data):
        """
        Initialize the predictor with movie performance data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing movie performance features
        """
        self.data = data
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None 
        self.y_train = None
        self.y_test = None
        self.model = None
        self.numeric_features = ['prod_budget', 'avg_rating']
        self.categorical_features = ['release_month', 'genre']

    def train_logistic_regression_model(self, test_size=0.2) -> Pipeline:

        # Separate features and target
        self.X = self.data[self.categorical_features + self.numeric_features]
        self.y = self.data['roi_category']

        # Create preprocessing pipeline for feature transformation
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ])

        # Split data and train logistic regression model
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, 
            self.y, 
            test_size=test_size, 
            stratify=self.y
        )

        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                solver='lbfgs',
                class_weight='balanced',  # Handle class imbalance
                max_iter=1000  # Increase iterations for convergence
            ))
        ])

        # Train the model
        self.model = pipeline.fit(self.X_train, self.y_train)
        return self.model

    def evaluate_model(self) -> dict:
        """
        Comprehensive model evaluation for multiclass classification

        Returns:
        --------
        dict : Model performance metrics
        """
        # Predictions 
        y_pred = self.model.predict(self.X_test) #result ndarray ['High ROI', 'Low ROI'...]
        y_pred_proba = self.model.predict_proba(self.X_test)

        # Get the correct class labels
        labels = list(self.model.classes_)

        # Performance metrics
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, 
                    yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        # Cross-Validation
        cv = StratifiedKFold(n_splits=5)
        # Performs k-fold cross-validation
        cv_scores = cross_val_score(
            self.model, 
            self.X, 
            self.y, 
            cv=cv, 
            scoring='f1_weighted'
        )
        print(f"cross_validation_scores is {cv_scores}, mean_cv_score is {cv_scores.mean()}")

        return {
                'classification_report': classification_report(self.y_test, y_pred),
                'cross_validation_scores': cv_scores,
                'mean_cv_score': cv_scores.mean()
            }

        # feature_importance(model)      

    def month_performance_analysis(self):
        """
        Perform month performance analysis with statistical significance testing 
        for multiclass logistic regression.

        Returns:
        --------
        dict
            Comprehensive statistical analysis of month performance
        """
        # months in calendar order
        calendar_month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        encoded_month_features_in_order = sorted(calendar_month_order)

        # Retrieve all coefficients from the pipeline
        classifier = self.model.named_steps['classifier']
        classes = classifier.classes_  # ['High ROI' 'Low ROI' 'Moderate ROI'] ndarray
        coefficients = classifier.coef_ # A list of 3 lists of coefficients for each feature

        # Find the start index of month features
        preprocessor = model.named_steps['preprocessor']
        categorical_encoder = preprocessor.named_transformers_['cat']
        genre_feature_count = len(categorical_encoder.categories_[1])
        numeric_feature_count = len(self.numeric_features)

        # Prepare data for standard error estimation
        X = self.model.named_steps['preprocessor'].transform(self.data) # <'scipy.sparse._csr.csr_matrix'>

        # Estimate standard errors
        standard_errors = self._calculate_multiclass_standard_errors(X, coefficients)

        # Initialize result dictionaries
        month_coefficients = {}
        month_confidence_intervals = {}
        month_p_values = {}

        for i, cls in enumerate(classes):
            # Extract month-specific coefficients
            start_idx = numeric_feature_count + genre_feature_count
            end_idx = start_idx + len(encoded_month_features_in_order)

            cls_month_coeffs = coefficients[i][start_idx:end_idx]
            cls_month_std_errors = standard_errors[start_idx:end_idx]

            # Calculate confidence intervals (95% confidence level)
            confidence_level = 0.95
            degrees_of_freedom = len(self.data) - X.shape[1]
            t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

            confidence_intervals = [
                (coeff - t_value * std_err, coeff + t_value * std_err) 
                for coeff, std_err in zip(cls_month_coeffs, cls_month_std_errors)
            ]

            # Calculate p-values using t-test
            p_values = [
                2 * (1 - stats.t.cdf(abs(coeff / std_err), degrees_of_freedom))
                for coeff, std_err in zip(cls_month_coeffs, cls_month_std_errors)
            ]

            month_coefficients[cls] = pd.Series(
                cls_month_coeffs, 
                index=encoded_month_features_in_order
            ).reindex(calendar_month_order)


            month_confidence_intervals[cls] = pd.Series(
                confidence_intervals, 
                index=encoded_month_features_in_order
            )

            month_p_values[cls] = pd.Series(
                p_values, 
                index=encoded_month_features_in_order
            )

        # Prepare comprehensive results
        results = {
            'coefficients': month_coefficients,
            'confidence_intervals': month_confidence_intervals,
            'p_values': month_p_values
        }
        # print(results)
        self._plot_month_coefficient_by_ROI(month_coefficients)
        # self._interpret_month_impact(month_coefficients)
        # self.interpret_monthly_significance(results)
        
        return results
    
    def _calculate_multiclass_standard_errors(self, X:scipy.sparse._csr.csr_matrix, coefficients: dict):
        """
        Helper function to calculate standard errors for multiclass logistic regression coefficients.
        Uses the observed information matrix (inverse of Hessian) to estimate standard errors.
        """
        try:
            # Check if the model has an attribute for the Hessian or information matrix
            if hasattr(self.model, 'hessian_') or hasattr(self.model, 'information_'):
                # If the model provides the information matrix directly
                information_matrix = self.model.hessian_ if hasattr(self.model, 'hessian_') else self.model.information_
                standard_errors = np.sqrt(np.diag(np.linalg.inv(information_matrix)))
                return standard_errors
            # Alternative method: Use numerical approximation of standard errors
            from scipy.optimize import approx_fprime
            # Get the original coefficients
            original_coef = self.model.coef_.ravel()
            # Define a small perturbation
            eps = 1e-8
            # Compute numerical approximation of the Hessian
            def cost_func(coef):
                # Temporarily set the model's coefficients
                self.model.coef_ = coef.reshape(self.model.coef_.shape)
                # Compute log-likelihood or some measure of fit
                return -np.sum(np.log(self.model.predict_proba(X)))
            # Compute the Hessian numerically
            hessian = approx_fprime(original_coef, 
                                    lambda x: approx_fprime(x, cost_func, eps), 
                                    eps)
            # Invert the Hessian and take square root of diagonal for standard errors
            standard_errors = np.sqrt(np.diag(np.linalg.inv(hessian)))
            return standard_errors
        except Exception as e:
            print(f"Warning: Could not calculate standard errors. Using approximation. Error: {e}")
            # Fallback: basic standard error approximation
            return np.std(coefficients, axis=0)
    
    def _plot_month_coefficient_by_ROI(self, month_coefficients:dict):
        classes = self.model.named_steps['classifier'].classes_
        plt.figure(figsize=(12, 6))
        for i, (cls, coefs) in enumerate(month_coefficients.items(), 1):
            plt.subplot(1, len(classes), i)
            coefs.plot(kind='bar')
            plt.title(f'Month Coefficients for {cls}')
            plt.xlabel('Month')
            plt.ylabel('Coefficient Value')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    
    def interpret_monthly_significance(self, results, significance_threshold=0.05):
        """
        Interpret the statistical significance of monthly performance and generate 
        a comprehensive report of significant month-ROI category associations.

        Parameters:
        -----------
        results : dict
            Results from month_performance_analysis
        significance_threshold : float, optional
            P-value threshold for statistical significance (default: 0.05)

        Returns:
        --------
        dict
            Detailed interpretation of statistical significance
        """
        # Initialize detailed interpretation
        detailed_interpretation = {
            'significant_associations': [],
            'non_significant_months': [],
            'summary': {
                'total_significant_associations': 0,
                'significant_months': set(),
                'significant_categories': set()
            }
        }

        # Iterate through each ROI category (class)
        for cls, p_values in results['p_values'].items():
            for month, p_value in p_values.items():
                # Determine significance level
                if p_value < significance_threshold:
                    # Identify the direction of the effect using the coefficient
                    coefficient = results['coefficients'][cls][month]

                    # Categorize the effect direction
                    if coefficient > 0:
                        effect_direction = "positively associated"
                    else:
                        effect_direction = "negatively associated"

                    # Confidence interval
                    ci_lower, ci_upper = results['confidence_intervals'][cls][month]

                    # Create a detailed association description
                    association = {
                        'category': cls,
                        'month': month,
                        'p_value': p_value,
                        'coefficient': coefficient,
                        'confidence_interval': (ci_lower, ci_upper),
                        'effect_description': f"{month} is {effect_direction} with {cls} ROI category"
                    }

                    # Add to significant associations
                    detailed_interpretation['significant_associations'].append(association)

                    # Update summary
                    detailed_interpretation['summary']['total_significant_associations'] += 1
                    detailed_interpretation['summary']['significant_months'].add(month)
                    detailed_interpretation['summary']['significant_categories'].add(cls)

        # Sort significant associations by p-value
        detailed_interpretation['significant_associations'].sort(key=lambda x: x['p_value'])

        # Print comprehensive report
        def print_interpretation(interpretation):
            print("\n===== Monthly ROI Category Performance Analysis =====")

            # Summary statistics
            print("\nSummary:")
            print(f"Total Significant Associations: {interpretation['summary']['total_significant_associations']}")
            print(f"Significant Months: {', '.join(interpretation['summary']['significant_months'])}")
            print(f"Significant Categories: {', '.join(interpretation['summary']['significant_categories'])}")

            # Detailed Significant Associations
            print("\nDetailed Significant Associations:")
            for assoc in interpretation['significant_associations']:
                print(f"\n{assoc['month']} - {assoc['category']} ROI Category:")
                print(f"  Effect: {assoc['effect_description']}")
                print(f"  P-value: {assoc['p_value']:.4f}")
                print(f"  Coefficient: {assoc['coefficient']:.4f}")
                print(f"  95% Confidence Interval: [{assoc['confidence_interval'][0]:.4f}, {assoc['confidence_interval'][1]:.4f}]")

        # Print the interpretation
        print_interpretation(detailed_interpretation)

        return detailed_interpretation

    
    def _interpret_month_impact(self, month_coefficients):
        """
        Provide interpretable insights about month impact on a movie's financial performance.
        """
        interpretations = {}
        for cls, coeffs in month_coefficients.items():
            # Sort coefficients to identify most influential months
            sorted_coeffs = coeffs.abs().sort_values(ascending=False)

            # Prepare interpretation
            interpretation = []
            for month, coef in sorted_coeffs.items():
                direction = "positive" if coeffs[month] > 0 else "negative"
                significance = (
                    "strong" if coef > 1 else 
                    "moderate" if coef > 0.5 else 
                    "weak"
                )

                interpretation.append(
                    f"{month}: {significance} {direction} association with {cls} "
                    f"(coefficient: {coeffs[month]:.3f})"
                )

            interpretations[cls] = interpretation

            # Print interpretations
            print("\nMonthly Performance Interpretations:")
            for cls, interpretations1 in interpretations.items():
                print(f"\nFor {cls} category:")
                for interpretation in interpretations1:
                    print(f"- {interpretation}")

        return interpretations

 
movie_df = pd.read_csv("movie_statistic_dataset.csv")
df_preprocessed = preprocess_movie_dataset(movie_df)

# plot_global_monthly_revenues(df_preprocessed)
# plot_roi_by_release_month_voilin(df_preprocessed)
# plot_monthly_roi_boxplot(df_preprocessed)
# chi_square_test_for_release_month_roi(df_preprocessed)
# mine_association_rules(df_preprocessed)
# plot_month_roi_heatmap(df_preprocessed)
predictor = MulticlassMoviePerformancePredictor(df_preprocessed)
model = predictor.train_logistic_regression_model(0.1)
predictor.evaluate_model()
predictor.month_performance_analysis()
# predictor.month_coefficient_analysis()

# predictor.calculate_month_statistical_metrics()












# def feature_importance(model):
#     """
#     Analyze feature importance for multinomial classification
    
#     Returns:
#     --------
#     feature_importance : dict of pandas.Series
#     """
#     # Get feature names after preprocessing
#     feature_names = (
#         model.named_steps['preprocessor']
#         .named_transformers_['num'].get_feature_names_out(
#             ['prod_budget', 'avg_rating']
#         ).tolist() + 
#         model.named_steps['preprocessor']
#         .named_transformers_['cat'].get_feature_names_out(
#             ['release_month', 'genre']
#         ).tolist()
#     )
    
#     # Get coefficients for each class
#     coefficients = model.named_steps['classifier'].coef_
#     class_names = model.named_steps['classifier'].classes_
    
#     # Create feature importance for each class
#     feature_importance = {}
#     for i, cls in enumerate(class_names):
#         importance = pd.Series(
#             coefficients[i], 
#             index=feature_names
#         ).abs().sort_values(ascending=False)
#         feature_importance[cls] = importance
    
#     # Visualize feature importance for each class
#     plt.figure(figsize=(15,10))
#     for i, (cls, importance) in enumerate(feature_importance.items(), 1):
#         plt.subplot(1, 3, i)
#         importance.plot(kind='bar')
#         plt.title(f'Feature Importance for {cls}')
#         plt.xlabel('Features')
#         plt.ylabel('Absolute Coefficient Value')
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
    
#     plt.show()
    
#     return feature_importance

# Get the feature names manually
# onehot_encoder = model.named_steps['preprocessor'].named_transformers_['cat']
# print(onehot_encoder.categories_)

# coeff_df = pd.DataFrame(coefficients, columns=feature_names)

    # # Extract coefficients related to 'release_month'
    # month_coeffs = coeff_df.filter(like='release_month')
    # month_coeffs = month_coeffs.T  # Transpose for readability
    # month_coeffs.columns = ['High ROI', 'Low ROI', 'Moderate ROI']  # Update if needed
    # print(month_coeffs)

#    # Filter for release_month-related features
#     month_features = [name for name in feature_names if 'release_month' in name]
#     print("Encoded release_month features:", month_features)


# # Remove outliers in numeric columns (optional)
# def remove_outliers(series: pd.Series) -> pd.Series:
#     Q1 = series.quantile(0.25)
#     Q3 = series.quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     # Replace outliers with NaN
#     return series.where((series >= lower_bound) & (series <= upper_bound))

