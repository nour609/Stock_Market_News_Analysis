from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from datetime import datetime

class TimeSeriesAnalyzer:
    def __init__(self):
        self.train_start = '2008-08-08'
        self.train_end = '2014-12-31'
        self.test_start = '2015-01-02'
        self.test_end = '2016-07-01'
        
    def split_by_time(self, df):
        """
        Split the data into training and test sets based on the paper's specified dates
        """
        df['Date'] = pd.to_datetime(df['Date'])
        
        train_mask = (df['Date'] >= self.train_start) & (df['Date'] <= self.train_end)
        test_mask = (df['Date'] >= self.test_start) & (df['Date'] <= self.test_end)
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        return train_df, test_df
    
    def analyze_temporal_patterns(self, df):
        """
        Analyze patterns in market movements over time
        """
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        
        # Yearly patterns
        yearly_stats = df.groupby('Year')['Label'].agg(['mean', 'count']).round(3)
        
        # Monthly patterns
        monthly_stats = df.groupby('Month')['Label'].agg(['mean', 'count']).round(3)
        
        # Day of week patterns
        dow_stats = df.groupby('DayOfWeek')['Label'].agg(['mean', 'count']).round(3)
        
        return {
            'yearly_patterns': yearly_stats,
            'monthly_patterns': monthly_stats,
            'day_of_week_patterns': dow_stats
        }
    
    def get_market_volatility_periods(self, df):
        """
        Identify periods of high market volatility
        """
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Calculate 30-day rolling window of label changes
        df['rolling_volatility'] = (
            df['Label'].rolling(window=30, min_periods=1)
            .apply(lambda x: (x != x.shift(1)).sum())
            .fillna(0)
        )
        
        # Find high volatility periods (top 25% of volatility)
        volatility_threshold = df['rolling_volatility'].quantile(0.75)
        high_volatility_periods = df[df['rolling_volatility'] >= volatility_threshold]
        
        return high_volatility_periods[['Date', 'rolling_volatility']]