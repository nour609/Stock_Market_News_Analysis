from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from scipy import stats

class DataQualityChecker:
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data quality and generate a comprehensive report
        """
        report = {
            'total_rows': len(df),
            'date_range': {
                'start': df['Date'].min(),
                'end': df['Date'].max()
            },
            'missing_values': df.isnull().sum().to_dict(),
            'label_distribution': df['Label'].value_counts().to_dict(),
            'duplicates': len(df) - len(df.drop_duplicates()),
            'news_columns_quality': {}
        }
        
        # Check news columns quality
        for col in df.filter(like='Top'):
            non_null = df[col].notna().sum()
            unique_values = df[col].nunique()
            report['news_columns_quality'][col] = {
                'non_null_count': non_null,
                'coverage_percentage': (non_null / len(df)) * 100,
                'unique_values': unique_values
            }
        
        return report

class BaselineModel:
    def __init__(self):
        self.model = DummyClassifier(strategy='stratified', random_state=42)
        
    def fit_and_evaluate(self, X_train: np.ndarray, X_test: np.ndarray,
                        y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Fit a baseline model and evaluate its performance
        """
        self.model.fit(X_train, y_train)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        baseline_auc = roc_auc_score(y_test, y_pred_proba)
        
        return {
            'baseline_auc': baseline_auc,
            'baseline_accuracy': self.model.score(X_test, y_test)
        }

class StatisticalAnalyzer:
    @staticmethod
    def perform_statistical_tests(actual_perf: float, baseline_perf: float,
                                y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """
        Perform statistical tests to validate model performance
        """
        # Chi-square test for independence between predictions and actual values
        contingency_table = pd.crosstab(y_true, y_pred)
        chi2, chi2_p_value = stats.chi2_contingency(contingency_table)[:2]
        
        # Calculate performance improvement over baseline
        perf_improvement = actual_perf - baseline_perf
        
        # Calculate confidence intervals using bootstrap
        n_bootstrap = 1000
        bootstrap_scores = []
        for _ in range(n_bootstrap):
            indices = np.random.randint(0, len(y_true), len(y_true))
            bootstrap_scores.append(roc_auc_score(y_true[indices], y_pred[indices]))
        
        ci_lower = np.percentile(bootstrap_scores, 2.5)
        ci_upper = np.percentile(bootstrap_scores, 97.5)
        
        return {
            'chi2_statistic': chi2,
            'chi2_p_value': chi2_p_value,
            'performance_improvement': perf_improvement,
            'confidence_interval': (ci_lower, ci_upper)
        }

class NewsRankingAnalyzer:
    @staticmethod
    def analyze_news_importance(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the relationship between news position (ranking) and prediction impact
        """
        news_columns = [col for col in df.columns if col.startswith('Top')]
        news_impacts = {}
        
        for col in news_columns:
            # Calculate correlation between news presence and market movement
            news_present = ~df[col].isna()

            # Guard against constant input arrays which cause scipy to warn/error
            # (e.g., column is all NaN or all non-NaN, or label is constant)
            try:
                if news_present.nunique() <= 1 or df['Label'].nunique() <= 1:
                    corr_coef = float('nan')
                    p_val = float('nan')
                else:
                    # pointbiserialr expects numeric input for the binary variable
                    res = stats.pointbiserialr(news_present.astype(int), df['Label'])
                    # scipy returns an object with attributes 'statistic'/'pvalue' or 'correlation'/'pvalue'
                    corr_coef = getattr(res, 'correlation', getattr(res, 'statistic', float('nan')))
                    p_val = getattr(res, 'pvalue', float('nan'))
            except Exception:
                corr_coef = float('nan')
                p_val = float('nan')

            # Calculate how often this position contains significant news
            coverage = (news_present.sum() / len(df)) * 100

            news_impacts[col] = {
                'correlation_coefficient': corr_coef,
                'p_value': p_val,
                'coverage_percentage': coverage
            }
        
        # Analyze if earlier positions (higher ranked news) have more impact
        positions = list(range(1, len(news_columns) + 1))
        correlations = [news_impacts[f'Top{i}']['correlation_coefficient'] 
                       for i in positions]
        rank_correlation = stats.spearmanr(positions, correlations)
        
        return {
            'position_impacts': news_impacts,
            'rank_correlation': rank_correlation.correlation,
            'rank_correlation_p_value': rank_correlation.pvalue
        }