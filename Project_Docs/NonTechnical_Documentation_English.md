# Project Overview — Non-Technical Summary

This project studies the relationship between news, online discussions, and the stock market. It collects news articles and Reddit posts, combines them with stock market data, and analyzes these sources to help people understand whether—and how—public information may influence market movements. The goal is to turn complex data into clear, visual reports that anyone can read and learn from.

## Who this project is for

* Everyday investors who want to understand whether news headlines or online discussions are linked to market changes.
* Business students and researchers who need an accessible report on how news impacts financial markets.
* Analysts and managers who want quick visual summaries without working directly with raw data.

## What the project does (plain language)

* **Gathers information**: Brings together news articles, Reddit posts, and historical stock market prices.
* **Looks for signals**: Examines the language and timing of news and discussions to see how they relate to market movements.
* **Measures performance**: Uses models to estimate how well news-related signals explain or predict market changes.
* **Creates reports**: Produces easy-to-read web pages and images that summarize key findings and trends.

## Key features (simple list)

* **Data collection**: Uses prepared news and Reddit data combined with historical stock price information.
* **News analysis**: Identifies common words, topics, and sentiment (positive or negative tone) in news articles and Reddit posts.
* **Time-series analysis**: Compares news and discussion patterns with market price changes over time.
* **Model evaluation**: Tests simple predictive models and summarizes their performance.
* **Automated reporting**: Generates user-friendly HTML reports and charts that can be opened in a web browser.

## Common outputs you will see

* Interactive-style HTML reports in the `report_html` folder, presenting results and charts.
* Images and charts showing trends and comparisons in the `report_html/images` and `images` folders.
* A CSV table for uploads or deeper inspection (for technical users) in the `data` folder.

## How to view the results (for non-technical users)

Open the main report file named `report_html/index.html` in a web browser. The pages are designed so you can read conclusions, explore charts, and understand the main takeaways without performing any technical steps.

## Why this is useful

This project turns large amounts of information into straightforward insights: which types of news tend to appear before market movements, whether online discussions are generally optimistic or concerned, and how reliable these signals are. This helps people make more informed decisions and better understand the link between public information and market behavior.
