import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
from nltk.tokenize import word_tokenize
from datetime import datetime, timedelta
import re
from src.data_loader import StockNewsDataLoader
from src.news_analyzer import NewsAnalyzer
from src.analysis_utils import (
    DataQualityChecker, BaselineModel, 
    StatisticalAnalyzer, NewsRankingAnalyzer
)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page config
st.set_page_config(
    page_title="Stock Market Prediction Analysis",
    page_icon="📈",
    layout="wide"
)

def run_analysis():
    """Run the complete analysis and return results"""
    # Initialize components
    data_loader = StockNewsDataLoader()
    analyzer = NewsAnalyzer()
    quality_checker = DataQualityChecker()
    ranking_analyzer = NewsRankingAnalyzer()
    
    # Load data
    df = data_loader.load_data("Combined_News_DJIA.csv")
    
    # Get quality report
    quality_report = quality_checker.check_data_quality(df)
    
    # Get news ranking analysis
    ranking_analysis = ranking_analyzer.analyze_news_importance(df)
    
    # Get temporal patterns
    temporal_patterns = analyzer.time_analyzer.analyze_temporal_patterns(df)
    
    # Prepare and train model
    X_train, X_test, y_train, y_test = analyzer.prepare_data(df)
    analyzer.train_model(X_train, y_train)
    
    # Get baseline performance
    baseline = BaselineModel()
    baseline_performance = baseline.fit_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Get model performance
    performance = analyzer.analyze_performance(X_train, X_test, y_train, y_test)
    
    # Get statistical analysis
    y_pred = (analyzer.model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    stats_analyzer = StatisticalAnalyzer()
    stats_results = stats_analyzer.perform_statistical_tests(
        performance['metrics']['auc'],
        baseline_performance['baseline_auc'],
        y_pred,
        y_test
    )
    
    # Get topic analysis
    topics = analyzer.analyze_topic_distribution()
    
    # Get volatility analysis
    volatility_periods = analyzer.time_analyzer.get_market_volatility_periods(df)
    
    return {
        'df': df,
        'quality_report': quality_report,
        'ranking_analysis': ranking_analysis,
        'temporal_patterns': temporal_patterns,
        'baseline_performance': baseline_performance,
        'model_performance': performance,
        'stats_results': stats_results,
        'topics': topics,
        'volatility_periods': volatility_periods,
        'analyzer': analyzer
    }

def main():
    # Title and description
    st.title("📈 Stock Market Prediction Analysis")
    st.markdown("""
    This dashboard presents a comprehensive analysis of stock market prediction using news headlines.
    The analysis includes data quality assessment, temporal patterns, model performance, and topic analysis.
    """)
    
    # Initialize session state for results
    if 'results' not in st.session_state:
        with st.spinner('Running analysis...'):
            st.session_state.results = run_analysis()
    
    results = st.session_state.results
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview",
        "📈 Market Analysis",
        "🤖 Model Performance",
        "📰 News Analysis",
        "📑 Statistical Insights"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.header("Data Quality Overview")
        
        # Date range selector
        df = results['df'].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # Filter data based on date range
        mask = (df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))
        filtered_df = df[mask]
        
        # Display metrics for filtered data
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(filtered_df))
        with col2:
            missing = filtered_df.isna().sum().sum()
            st.metric("Missing Values", missing)
        with col3:
            duplicates = len(filtered_df) - len(filtered_df.drop_duplicates())
            st.metric("Duplicate Rows", duplicates)
        with col4:
            up_ratio = (filtered_df['Label'] == 1).mean() * 100
            st.metric("Up Days %", f"{up_ratio:.1f}%")
        
        # Interactive headline browser
        st.subheader("Headlines Browser")
        selected_date = st.selectbox(
            "Select Date",
            filtered_df['Date'].dt.date.unique(),
            format_func=lambda x: x.strftime('%Y-%m-%d')
        )
        
        selected_day = filtered_df[filtered_df['Date'].dt.date == selected_date].iloc[0]
        for i in range(1, 26):  # Show all 25 headlines
            headline = selected_day[f'Top{i}']
            if isinstance(headline, bytes):
                headline = headline.decode('utf-8')
            st.write(f"{i}. {headline}")
        
        # Interactive market movement visualization
        st.subheader("Market Movement Analysis")
        
        # Create daily returns
        filtered_df['DayOfWeek'] = filtered_df['Date'].dt.day_name()
        daily_stats = filtered_df.groupby('DayOfWeek')['Label'].agg(['mean', 'count']).round(3)
        daily_stats.columns = ['Up Ratio', 'Count']
        
        # Plot using plotly
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]])
        
        # Pie chart of up/down distribution
        labels = ['Up/Same', 'Down']
        values = [
            (filtered_df['Label'] == 1).sum(),
            (filtered_df['Label'] == 0).sum()
        ]
        fig.add_trace(
            go.Pie(labels=labels, values=values, name="Market Direction"),
            row=1, col=1
        )
        
        # Bar chart of daily patterns
        fig.add_trace(
            go.Bar(x=daily_stats.index, y=daily_stats['Up Ratio'], 
                  name="Up Day Ratio", marker_color='lightgreen'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, title_text="Market Movement Patterns")
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Market Analysis
    with tab2:
        st.header("Temporal Analysis")
        
        # Time granularity selector
        granularity = st.selectbox(
            "Select Time Granularity",
            ["Daily", "Weekly", "Monthly", "Yearly"]
        )
        
        # Calculate market patterns based on selected granularity
        df = results['df'].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        if granularity == "Daily":
            df['Period'] = df['Date']
        elif granularity == "Weekly":
            df['Period'] = df['Date'].dt.to_period('W').astype(str)
        elif granularity == "Monthly":
            df['Period'] = df['Date'].dt.to_period('M').astype(str)
        else:  # Yearly
            df['Period'] = df['Date'].dt.year
        
        patterns = df.groupby('Period').agg({
            'Label': ['mean', 'count'],
            'Date': 'first'  # For sorting
        }).reset_index()
        patterns.columns = ['Period', 'Up_Ratio', 'Count', 'Date']
        patterns = patterns.sort_values('Date')
        
        # Create interactive time series plot
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=patterns['Period'], y=patterns['Up_Ratio'],
                      name="Up Day Ratio", line=dict(color='green')),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Bar(x=patterns['Period'], y=patterns['Count'],
                  name="Number of Days", marker_color='lightblue'),
            secondary_y=True
        )
        
        fig.update_layout(
            title_text=f"Market Patterns ({granularity} View)",
            xaxis_title="Time Period",
            height=500
        )
        
        fig.update_yaxes(title_text="Up Day Ratio", secondary_y=False)
        fig.update_yaxes(title_text="Number of Days", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volatility analysis
        st.subheader("Market Volatility Analysis")
        
        # Add volatility calculation options
        vol_window = st.slider("Volatility Window (Days)", 5, 60, 30)
        
        # Calculate rolling volatility
        volatility = df.copy()
        volatility['rolling_vol'] = (
            volatility['Label']
            .rolling(window=vol_window)
            .std()
            .fillna(0)
        )
        
        # Plot volatility
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=volatility['Date'],
            y=volatility['rolling_vol'],
            mode='lines',
            name=f'{vol_window}-Day Volatility',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=f"{vol_window}-Day Rolling Volatility",
            xaxis_title="Date",
            yaxis_title="Volatility",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display high volatility periods
        st.subheader("High Volatility Periods")
        threshold = st.select_slider(
            "Volatility Threshold (percentile)",
            options=range(75, 100, 5),
            value=90
        )
        
        vol_threshold = volatility['rolling_vol'].quantile(threshold/100)
        high_vol_periods = volatility[volatility['rolling_vol'] >= vol_threshold]
        
        st.dataframe(
            high_vol_periods[['Date', 'rolling_vol']]
            .sort_values('rolling_vol', ascending=False)
            .head(10)
            .style.format({'rolling_vol': '{:.3f}'})
        )
    
    # Tab 3: Model Performance
    with tab3:
        st.header("Model Evaluation")
        
        # Model performance metrics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Metrics")
            metrics = results['model_performance']['metrics']
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = metrics['auc'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "AUC Score"},
                gauge = {
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.7], 'color': "lightgreen"},
                        {'range': [0.7, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.5
                    }
                }
            ))
            st.plotly_chart(fig)
            
        with col2:
            st.subheader("Performance Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                'Value': [
                    metrics['accuracy'],
                    metrics['precision'],
                    metrics['recall'],
                    metrics['f1_score']
                ]
            })
            
            fig = px.bar(metrics_df, x='Metric', y='Value',
                        color='Value',
                        color_continuous_scale='Viridis',
                        range_y=[0, 1])
            st.plotly_chart(fig)
        
        # Interactive model testing
        st.subheader("Test the Model")
        st.write("Enter up to 5 news headlines to get a market movement prediction")
        
        headlines = []
        for i in range(5):
            headline = st.text_input(f"Headline {i+1}", key=f"headline_{i}")
            if headline:
                headlines.append(headline)
        
        if headlines:
            # Prepare the input data
            test_data = pd.DataFrame({
                f'Top{i+1}': headlines[i] if i < len(headlines) else ''
                for i in range(25)
            }, index=[0])
            
            # Make prediction
            X = results['analyzer'].vectorizer.transform(
                test_data.fillna('').apply(lambda x: ' '.join(x), axis=1)
            )
            prob = results['analyzer'].model.predict_proba(X)[0, 1]
            
            # Display prediction
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Probability of Market Rise"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "green" if prob > 0.5 else "red"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightcoral"},
                            {'range': [50, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                st.plotly_chart(fig)
            
            with col2:
                prediction = "Market is likely to go UP" if prob > 0.5 else "Market is likely to go DOWN"
                confidence = abs(prob - 0.5) * 2  # Scale to 0-1
                st.markdown(f"### Prediction")
                st.markdown(f"**{prediction}**")
                st.markdown(f"Confidence: {confidence:.1%}")
        
        # Model performance visualizations
        st.subheader("Performance Visualization")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(
                title="ROC Curve",
                labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
            )
            
            from sklearn.metrics import roc_curve
            y_pred_proba = results['analyzer'].model.predict_proba(results['analyzer'].X_test)[:, 1]
            fpr, tpr, _ = roc_curve(results['analyzer'].y_test, y_pred_proba)
            
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC = {metrics["auc"]:.3f})')
            )
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash'))
            )
            
            st.plotly_chart(fig)
            
        with col2:
            from sklearn.metrics import confusion_matrix
            y_pred = results['analyzer'].model.predict(results['analyzer'].X_test)
            cm = confusion_matrix(results['analyzer'].y_test, y_pred)
            
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual"),
                x=['Down', 'Up'],
                y=['Down', 'Up'],
                text_auto=True,
                color_continuous_scale='RdYlGn',
                title="Confusion Matrix"
            )
            
            st.plotly_chart(fig)
    
    # Tab 4: News Analysis
    with tab4:
        st.header("News Content Analysis")
        
        # Interactive topic analysis
        st.subheader("Topic Analysis")
        
        # Topic selector
        selected_topics = st.multiselect(
            "Select Topics to Analyze",
            list(results['topics'].keys()),
            default=list(results['topics'].keys())[:2]
        )
        
        # Create interactive topic visualization
        topic_data = []
        for topic in selected_topics:
            terms = results['topics'][topic]
            for term, freq in terms:
                topic_data.append({
                    'Topic': topic,
                    'Term': term,
                    'Frequency': freq
                })
        
        topic_df = pd.DataFrame(topic_data)
        
        fig = px.treemap(
            topic_df,
            path=[px.Constant("Topics"), 'Topic', 'Term'],
            values='Frequency',
            color='Frequency',
            color_continuous_scale='RdBu'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Word impact analysis
        st.subheader("Interactive Word Impact Analysis")
        
        # Get word impact data
        pos_words, neg_words = results['analyzer'].get_most_influential_words(20)
        
        # Create word impact dataframe
        impact_data = []
        for words in [pos_words, neg_words]:
            for word, score, freq in words:
                impact_data.append({
                    'Word': word,
                    'Impact Score': score,
                    'Frequency': freq,
                    'Type': 'Positive' if score > 0 else 'Negative'
                })
        
        impact_df = pd.DataFrame(impact_data)
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            min_freq = st.slider("Minimum Word Frequency", 
                               int(impact_df['Frequency'].min()),
                               int(impact_df['Frequency'].max()),
                               int(impact_df['Frequency'].min()))
        
        with col2:
            impact_type = st.multiselect(
                "Impact Type",
                ['Positive', 'Negative'],
                default=['Positive', 'Negative']
            )
        
        # Filter data
        filtered_impact = impact_df[
            (impact_df['Frequency'] >= min_freq) &
            (impact_df['Type'].isin(impact_type))
        ]
        
        # Create interactive scatter plot
        fig = px.scatter(
            filtered_impact,
            x='Impact Score',
            y='Frequency',
            color='Type',
            hover_data=['Word'],
            size='Frequency',
            color_discrete_map={'Positive': 'green', 'Negative': 'red'},
            title='Word Impact vs Frequency'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # News headline pattern analysis
        st.subheader("Headline Pattern Analysis")
        
        # Get all headlines
        all_headlines = []
        for col in results['df'].filter(like='Top'):
            headlines = results['df'][col].dropna()
            all_headlines.extend([h.decode() if isinstance(h, bytes) else h 
                               for h in headlines])
        
        # Tokenize and analyze headline patterns
        tokens = []
        for headline in all_headlines:
            text = (headline or '').lower()
            try:
                # Use NLTK tokenizer when available
                tokens.extend(word_tokenize(text))
            except LookupError:
                # Fallback: simple regex-based tokenization if NLTK data missing
                tokens.extend(re.findall(r"\b\w+\b", text))
        
        # Calculate word frequencies
        word_freq = pd.Series(tokens).value_counts()
        
        # Plot word frequency distribution
        fig = px.bar(
            x=word_freq.head(20).index,
            y=word_freq.head(20).values,
            title='Most Common Words in Headlines',
            labels={'x': 'Word', 'y': 'Frequency'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Statistical Insights
    with tab5:
        st.header("Statistical Analysis")
        # Safely display statistical results (handle missing or NaN values)
        stats_res = results.get('stats_results')
        model_perf = results.get('model_performance')

        if not stats_res or not model_perf:
            st.info("Statistical results are not available. Run the analysis or check the logs for errors.")
            # Offer an easy one-click rerun of the analysis for debugging
            if st.button("Re-run analysis"):
                with st.spinner("Re-running full analysis..."):
                    st.session_state.results = run_analysis()
                    results = st.session_state.results
                    stats_res = results.get('stats_results')
                    model_perf = results.get('model_performance')
                    st.success("Analysis re-run complete. Scroll down to see updated values.")
        else:
            def fmt(x):
                try:
                    if x is None:
                        return "N/A"
                    if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)):
                        return f"{x:.3f}"
                    return str(x)
                except Exception:
                    return "N/A"

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Hypothesis Testing")
                st.metric("Chi-square Statistic", fmt(stats_res.get('chi2_statistic')))
                st.metric("P-value", fmt(stats_res.get('chi2_p_value')))

            with col2:
                st.subheader("Confidence Intervals")
                ci = stats_res.get('confidence_interval')
                if ci and len(ci) == 2:
                    st.write(f"95% CI: ({fmt(ci[0])}, {fmt(ci[1])})")
                else:
                    st.write("95% CI: N/A")

            st.subheader("Cross-validation Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean AUC", fmt(model_perf.get('cv_mean')))
            with col2:
                st.metric("Standard Deviation", fmt(model_perf.get('cv_std')))

            # Debugging helper: allow developer/user to inspect raw stats
            if st.checkbox("Show raw statistical results (debug)"):
                st.subheader("Raw stats_results")
                st.json(stats_res)
                st.subheader("Raw model_performance")
                st.json(model_perf)

if __name__ == "__main__":
    main()