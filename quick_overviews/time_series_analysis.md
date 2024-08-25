## Time Series Analysis

1. **Time Series**: A sequence of data points collected or recorded at specific time intervals, often used to analyze trends, patterns, and forecasting.

2. **Seasonality**: A repeating pattern or fluctuation in a time series that occurs at regular intervals, such as daily, monthly, or yearly.

3. **Trend**: The long-term movement or direction in a time series, indicating an overall increase, decrease, or stability over time.

4. **Stationarity**: A property of a time series where its statistical properties (mean, variance, autocorrelation) are constant over time.

5. **Autocorrelation**: The correlation of a time series with its own past values, indicating how past data points are related to current values.

6. **Partial Autocorrelation**: The correlation between a time series and its lagged values, controlling for the values at intervening lags.

7. **Lag**: The time difference between observations in a time series, often used in autocorrelation and moving average calculations.

8. **Moving Average (MA)**: A technique used to smooth time series data by averaging adjacent observations over a specified number of periods.

9. **Exponential Smoothing**: A forecasting technique that applies exponentially decreasing weights to past observations, giving more importance to recent data.

10. **Holt-Winters Method**: A time series forecasting method that accounts for seasonality, trend, and level, using exponential smoothing.

11. **Differencing**: A method used to transform a non-stationary time series into a stationary one by subtracting consecutive observations.

12. **Autoregressive (AR) Model**: A model where the current value of a time series is regressed on its previous values (lags).

13. **Moving Average (MA) Model**: A model where the current value of a time series is expressed as a linear combination of past forecast errors.

14. **ARMA Model**: A combination of Autoregressive (AR) and Moving Average (MA) models used to describe stationary time series data.

15. **ARIMA Model**: An extension of the ARMA model that includes differencing to handle non-stationary time series data.

16. **SARIMA Model**: An extension of the ARIMA model that incorporates seasonality, making it suitable for seasonal time series forecasting.

17. **Seasonal Decomposition of Time Series (STL)**: A method for decomposing a time series into its seasonal, trend, and residual components.

18. **Box-Jenkins Methodology**: A systematic approach to identifying, estimating, and validating ARIMA models for time series forecasting.

19. **Lag Operator**: A mathematical operator that shifts a time series back by a specified number of periods (lags).

20. **White Noise**: A time series with a constant mean, constant variance, and no autocorrelation, often used as a model for random fluctuations.

21. **Random Walk**: A time series model where the current value is the previous value plus a random error, often used to model stock prices.

22. **Unit Root**: A characteristic of a time series that indicates non-stationarity, where shocks have a permanent effect on the level of the series.

23. **Dickey-Fuller Test**: A statistical test used to determine whether a time series has a unit root and is therefore non-stationary.

24. **KPSS Test**: A statistical test used to assess the stationarity of a time series, particularly for testing the null hypothesis of stationarity.

25. **Cointegration**: A statistical property of a set of time series where a linear combination of them is stationary, despite each being non-stationary.

26. **Granger Causality**: A statistical hypothesis test used to determine whether one time series can predict another.

27. **Impulse Response Function (IRF)**: A function that describes the response of a time series to a shock or impulse in another series or in its own past.

28. **Vector Autoregression (VAR)**: A multivariate time series model where each variable is modeled as a linear function of its own past values and the past values of other variables.

29. **Vector Error Correction Model (VECM)**: A multivariate time series model used when variables are cointegrated, capturing both short-term dynamics and long-term relationships.

30. **Seasonal Adjustment**: The process of removing seasonal effects from a time series to analyze underlying trends and cycles.

31. **Spectral Analysis**: A method used to examine the frequency components of a time series, often using Fourier transforms.

32. **Fourier Transform**: A mathematical transform used to decompose a time series into its frequency components.

33. **Periodogram**: A graphical representation of the frequency spectrum of a time series, showing the strength of different frequency components.

34. **Autoregressive Integrated Moving Average (ARIMA) with Exogenous Variables (ARIMAX)**: An ARIMA model that includes external variables to improve forecasting accuracy.

35. **Prophet**: A time series forecasting tool developed by Facebook that is robust to missing data and handles seasonality and holidays.

36. **Exogenous Variables**: External variables that influence a time series but are not influenced by it, often included in models to improve accuracy.

37. **Kalman Filter**: An algorithm that provides estimates of unknown variables in a time series, particularly useful for handling noise and missing data.

38. **State Space Model**: A mathematical model that represents a time series as a set of hidden states, often used in conjunction with the Kalman filter.

39. **Maximum Likelihood Estimation (MLE)**: A method used to estimate the parameters of a time series model by maximizing the likelihood function.

40. **Akaike Information Criterion (AIC)**: A metric used to compare the goodness-of-fit of different time series models, penalizing model complexity.

41. **Bayesian Information Criterion (BIC)**: Similar to AIC, but with a stronger penalty for model complexity, often used for model selection.

42. **Out-of-Sample Forecasting**: The process of evaluating a time series model's performance by making predictions on data not used in model fitting.

43. **Backtesting**: The process of testing a time series forecasting model on historical data to assess its accuracy and robustness.

44. **Ensemble Forecasting**: Combining multiple time series models to improve forecasting accuracy by averaging their predictions.

45. **Rolling Forecast Origin**: A forecasting method where the origin of the forecast is continuously rolled forward, allowing for ongoing model evaluation.

46. **Sliding Window**: A method used to analyze a subset of a time series over a moving window of fixed size, often used in rolling forecasts.

47. **Holt’s Linear Trend Model**: A time series forecasting method that accounts for both level and trend using exponential smoothing.

48. **Triple Exponential Smoothing (TES)**: Also known as Holt-Winters method, this technique extends exponential smoothing to handle seasonality.

49. **Residuals**: The differences between the observed values and the values predicted by a time series model, used to evaluate model accuracy.

50. **Autocovariance**: The covariance of a time series with its own past values, used to measure the degree of dependence between different time points.

51. **Time Series Decomposition**: The process of separating a time series into its underlying components (trend, seasonality, and residuals) to better understand its behavior.


### Important Keywords for "Association Rule" in Business Analytics

1. **Association Rule Mining**: A data mining technique used to find associations or relationships between data items in large datasets.
   - **Extended Definition**: This technique explores large volumes of data to discover hidden patterns, correlations, and casual structures among sets of items. It's particularly useful in retail, e-commerce, and market basket analysis.

2. **Support**: Indicates how frequently an itemset appears in the dataset.
   - **Formula**: $(\text{Support}(A) = \frac{\text{Frequency of } A}{\text{Total Transactions}})$
   - **Explanation**: Support measures the proportion of transactions that include the itemset. It's a crucial metric for determining the relevance and significance of an association rule.
   - **Extended Example**: If itemset A appears in 30 out of 100 transactions, its support is 30%.

3. **Confidence**: Measures the likelihood that a rule is true for a transaction that contains the antecedent.
   - **Formula**: $(\text{Confidence}(A \rightarrow B) = \frac{\text{Support}(A \cap B)}{\text{Support}(A)})$
   - **Explanation**: Confidence indicates the probability that the presence of item A leads to the presence of item B. It quantifies the strength of the implication.
   - **Extended Example**: If Support(A ∩ B) = 20% and Support(A) = 30%, then Confidence(A → B) = 20% / 30% = 66.67%.

4. **Lift**: Evaluates the strength of a rule over the random co-occurrence of the antecedent and the consequent.
   - **Formula**: $(\text{Lift}(A \rightarrow B) = \frac{\text{Support}(A \cap B)}{\text{Support}(A) \times \text{Support}(B)})$
   - **Explanation**: Lift compares the observed support with what would be expected if A and B were independent. It measures how much more likely B is to be purchased when A is purchased, compared to B's usual purchase rate.
   - **Interpretation**: 
     - Lift > 1: Positive correlation
     - Lift = 1: Independence
     - Lift < 1: Negative correlation

5. **Antecedent (LHS)**: The left-hand side of the rule, representing the condition or "if" part of the rule.
   - **Example**: In the rule $(A \rightarrow B)$, A is the antecedent.
   - **Extended Explanation**: The antecedent is the item or itemset that, when present in a transaction, implies the possible presence of the consequent.

6. **Consequent (RHS)**: The right-hand side of the rule, representing the outcome or "then" part of the rule.
   - **Example**: In the rule $(A \rightarrow B)$, B is the consequent.
   - **Extended Explanation**: The consequent is the item or itemset that is implied to be present in transactions containing the antecedent.

7. **Itemset**: A collection of one or more items.
   - **Frequent Itemset**: An itemset that meets a minimum support threshold.
   - **Extended Explanation**: Itemsets are the building blocks of association rules. The process of finding frequent itemsets is often the most computationally intensive part of association rule mining.

8. **Apriori Algorithm**: A classic algorithm used for mining frequent itemsets and generating association rules.
   - **Explanation**: It uses a bottom-up approach, where frequent subsets are extended one item at a time.
   - **Key Principle**: If an itemset is frequent, then all of its subsets must also be frequent (known as the Apriori principle).
   - **Process Overview**:
     1. Find frequent 1-itemsets
     2. Generate candidate k-itemsets from frequent (k-1)-itemsets
     3. Prune candidates using the Apriori principle
     4. Scan the database to determine frequent k-itemsets
     5. Repeat steps 2-4 until no more frequent itemsets are found

9. **Eclat Algorithm**: An algorithm for mining frequent itemsets that uses a depth-first search strategy.
   - **Explanation**: It focuses on the vertical data format and explores intersections of itemsets.
   - **Key Feature**: Uses tid-lists (transaction ID lists) to represent itemsets, which can be more efficient for certain types of datasets.

10. **FP-Growth Algorithm**: An efficient and scalable algorithm for mining the frequent itemset without candidate generation.
    - **Explanation**: It uses a divide-and-conquer approach based on the FP-tree structure.
    - **Key Advantages**: 
      - Eliminates the need for candidate generation
      - Requires only two database scans
      - Often faster than Apriori for dense datasets

11. **Rule Generation**: The process of generating association rules from frequent itemsets.
    - **Explanation**: Rules are generated from frequent itemsets that meet the minimum confidence threshold.
    - **Process**: For each frequent itemset L, generate all nonempty subsets of L. For each nonempty subset S of L, output the rule S → (L - S) if its confidence is at least the minimum confidence threshold.

12. **Pruning**: The process of removing infrequent itemsets or rules that do not meet the required support or confidence thresholds.
    - **Explanation**: Pruning helps to reduce the search space and focus on the most relevant associations.
    - **Methods**: 
      - Support-based pruning: Remove itemsets with support below the minimum threshold
      - Confidence-based pruning: Remove rules with confidence below the minimum threshold

13. **Transaction Database**: The database or dataset that contains transactions, each consisting of a set of items.
    - **Example**: A supermarket database where each transaction is a customer's purchase.
    - **Structure**: Typically represented as a set of transactions, where each transaction is a set of items.

14. **Minimum Support Threshold**: A user-defined threshold that determines the minimum support an itemset must have to be considered frequent.
    - **Explanation**: Itemsets with support below this threshold are considered infrequent and are pruned.
    - **Trade-off**: Lower thresholds can uncover more patterns but increase computational cost and may produce trivial rules.

15. **Minimum Confidence Threshold**: A user-defined threshold that determines the minimum confidence a rule must have to be considered strong.
    - **Explanation**: Rules with confidence below this threshold are not considered strong enough.
    - **Trade-off**: Higher thresholds produce stronger rules but may miss some interesting patterns.

16. **Closed Itemset**: An itemset is closed if none of its immediate supersets have the same support.
    - **Explanation**: Closed itemsets represent the largest sets with the same support.
    - **Importance**: Mining closed itemsets can be more efficient than mining all frequent itemsets while still preserving complete information.

17. **Maximal Itemset**: An itemset is maximal if none of its immediate supersets are frequent.
    - **Explanation**: Maximal itemsets are the largest frequent itemsets.
    - **Relationship to Closed Itemsets**: Every maximal itemset is closed, but not every closed itemset is maximal.

### Mathematical Formulas with Explanations

1. **Support**: 
   $$
   \text{Support}(A) = \frac{\text{Frequency of } A}{\text{Total Transactions}}
   $$
   - Indicates the proportion of transactions that include itemset A.
   - Used to filter out infrequent itemsets and determine the relevance of rules.

2. **Confidence**: 
   $$
   \text{Confidence}(A \rightarrow B) = \frac{\text{Support}(A \cap B)}{\text{Support}(A)}
   $$
   - Measures the likelihood of itemset B occurring when itemset A is present.
   - Provides a measure of the reliability of the inference made by a rule.

3. **Lift**: 
   $$
   \text{Lift}(A \rightarrow B) = \frac{\text{Support}(A \cap B)}{\text{Support}(A) \times \text{Support}(B)}
   $$
   - Evaluates the strength of the rule by comparing the observed support with the expected support if A and B were independent.
   - A lift greater than 1 indicates that A and B appear together more often than expected by chance.

4. **Conviction**:
   $$
   \text{Conviction}(A \rightarrow B) = \frac{1 - \text{Support}(B)}{1 - \text{Confidence}(A \rightarrow B)}
   $$
   - Measures the implication strength of a rule.
   - Conviction is infinite for rules that always hold (100% confidence) and 1 for independent items.

These keywords, formulas, and their extended explanations form the foundation of understanding and applying association rule mining in business analytics. They provide the necessary tools for discovering meaningful patterns in transactional data, which can be used to inform business strategies, optimize operations, and enhance customer experiences.