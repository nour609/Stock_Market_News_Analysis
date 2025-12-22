from src.data_loader import StockNewsDataLoader
from src.news_analyzer import NewsAnalyzer
from src.analysis_utils import DataQualityChecker, BaselineModel, StatisticalAnalyzer, NewsRankingAnalyzer
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# ensure images directory exists
os.makedirs('images', exist_ok=True)

def main():
    # Set style for all plots
    plt.style.use('default')
    sns.set_theme()  # This will set seaborn's default theme
    
    # Initialize the data loader
    data_loader = StockNewsDataLoader()
    
    try:
        # Load the data
        print("\nLoading news and market data...")
        df = data_loader.load_data("Combined_News_DJIA.csv")
        
        # Check data quality
        print("\nPerforming data quality checks...")
        quality_checker = DataQualityChecker()
        quality_report = quality_checker.check_data_quality(df)
        
        print(f"\nData Quality Report:")
        print(f"  Total rows: {quality_report['total_rows']}")
        print(f"  Date range: {quality_report['date_range']['start']} to {quality_report['date_range']['end']}")
        print(f"  Missing values found: {sum(quality_report['missing_values'].values())}")
        print(f"  Duplicate rows: {quality_report['duplicates']}")
        
        # Analyze news ranking impact
        print("\nAnalyzing news ranking impact...")
        ranking_analyzer = NewsRankingAnalyzer()
        ranking_analysis = ranking_analyzer.analyze_news_importance(df)
        
        print("\nNews Position Impact Analysis:")
        print(f"  Rank correlation: {ranking_analysis['rank_correlation']:.3f}")
        print(f"  P-value: {ranking_analysis['rank_correlation_p_value']:.3f}")
        
        print("\nSample headlines from the dataset:")
        # Show first 3 headlines from a random day
        sample_day = df.iloc[0]
        for i in range(1, 4):  # Top1, Top2, Top3
            headline = sample_day[f'Top{i}']
            if isinstance(headline, bytes):
                headline = headline.decode('utf-8')
            print(f"  {i}. {headline}")
        
        print("\nMarket movement distribution:")
        label_dist = df['Label'].value_counts()
        print(f"  Market Up/Same Days: {label_dist.get(1, 0)} ({label_dist.get(1, 0)/len(df)*100:.1f}%)")
        print(f"  Market Down Days:    {label_dist.get(0, 0)} ({label_dist.get(0, 0)/len(df)*100:.1f}%)")
        
        # Initialize the news analyzer
        print("\nAnalyzing market data and news patterns...")
        analyzer = NewsAnalyzer()
        
        # Analyze temporal patterns first
        temporal_patterns = analyzer.time_analyzer.analyze_temporal_patterns(df)
        print("\nTemporal Analysis:")
        for pattern_type, stats in temporal_patterns.items():
            print(f"\n{pattern_type.replace('_', ' ').title()}:")
            print(stats)
        
        # Prepare and train the model using time-based split
        print("\nPreparing data with time-based split (2008-2014 train, 2015-2016 test)...")
        X_train, X_test, y_train, y_test = analyzer.prepare_data(df)
        
        print("Training market movement predictor...")
        analyzer.train_model(X_train, y_train)
        
        # Evaluate the model with enhanced metrics
        print("\nTraining and evaluating baseline model...")
        baseline = BaselineModel()
        baseline_performance = baseline.fit_and_evaluate(X_train, X_test, y_train, y_test)
        
        print("\nBaseline Model Performance:")
        print(f"  AUC Score: {baseline_performance['baseline_auc']:.3f}")
        print(f"  Accuracy:  {baseline_performance['baseline_accuracy']:.3f}")
        
        print("\nEvaluating main model performance...")
        performance = analyzer.analyze_performance(X_train, X_test, y_train, y_test)
        
        # Perform statistical analysis
        y_pred = (analyzer.model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        stats_analyzer = StatisticalAnalyzer()
        stats_results = stats_analyzer.perform_statistical_tests(
            performance['metrics']['auc'],
            baseline_performance['baseline_auc'],
            y_pred,
            y_test
        )
        
        print("\nModel Performance Metrics:")
        metrics = performance['metrics']
        print(f"  AUC Score:         {metrics['auc']:.3f}")
        print(f"  Accuracy:          {metrics['accuracy']:.3f}")
        print(f"  Precision:         {metrics['precision']:.3f}")
        print(f"  Recall:            {metrics['recall']:.3f}")
        print(f"  F1 Score:          {metrics['f1_score']:.3f}")
        
        print("\nCross-validation Performance:")
        print(f"  Mean AUC:          {performance['cv_mean']:.3f}")
        print(f"  Std Dev:           {performance['cv_std']:.3f}")
        
        print("\nStatistical Analysis:")
        print(f"  Chi-square statistic: {stats_results['chi2_statistic']:.3f}")
        print(f"  Chi-square p-value:   {stats_results['chi2_p_value']:.3f}")
        print(f"  Performance improvement over baseline: {stats_results['performance_improvement']:.3f}")
        print(f"  95% Confidence Interval: ({stats_results['confidence_interval'][0]:.3f}, {stats_results['confidence_interval'][1]:.3f})")
        
        # Analyze high volatility periods
        volatility_periods = analyzer.time_analyzer.get_market_volatility_periods(df)
        print("\nHigh Volatility Periods:")
        print(volatility_periods.head())
        
        # Create enhanced visualizations
        print("\nGenerating visualizations...")
        
        # 1. Word influence visualization with word clouds and detailed bar charts
        analyzer.plot_word_influence(10)
        
        # 2. Topic distribution visualization
        topics = analyzer.analyze_topic_distribution()
        
        # Create a bar plot of topic distributions
        topic_sizes = {topic: sum(freq for _, freq in terms) 
                      for topic, terms in topics.items()}
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(topic_sizes)), list(topic_sizes.values()),
               color=plt.cm.Set3(np.linspace(0, 1, len(topic_sizes))))
        plt.title('Distribution of News Topics')
        plt.xticks(range(len(topic_sizes)), list(topic_sizes.keys()), rotation=45)
        plt.ylabel('Total Term Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join('images', 'topic_distribution.png'))
        plt.close()
        
        print("\nDetailed Topic Analysis:")
        for topic, terms in topics.items():
            print(f"\n  {topic.title()} ({len(terms)} terms)")
            # Show top 5 most frequent terms per topic
            for term, freq in terms[:5]:
                print(f"    {term:<25} {freq:>6.0f} occurrences")
        
        # Plot temporal patterns
        plt.figure(figsize=(15, 10))
        volatility_periods = analyzer.time_analyzer.get_market_volatility_periods(df)
        plt.plot(volatility_periods['Date'], volatility_periods['rolling_volatility'])
        plt.title('Market Volatility Over Time')
        plt.xlabel('Date')
        plt.ylabel('30-Day Rolling Volatility')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join('images', 'market_volatility.png'))
        plt.close()
        
    except FileNotFoundError:
        print("Please download the dataset first!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()