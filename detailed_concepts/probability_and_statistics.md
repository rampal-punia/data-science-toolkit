# Probability & Statistics

1. **Probability**: The measure of the likelihood that an event will occur, ranging from 0 to 1.

2. **Random Variable**: 

- A **random variable** is a variable whose possible values are numerical outcomes of a random phenomenon. It is a function that assigns a real number to each outcome of a random experiment or event. Random variables can be classified into:

- **Discrete random variables**: Take on a countable number of distinct values (e.g., the number of heads in a series of coin tosses).
- **Continuous random variables**: Take on an uncountably infinite number of values within a given range (e.g., the height of individuals in a population).

The probability distribution of a random variable provides the probabilities of all possible values it can take.

3. **Probability Distribution**: A function that describes the likelihood of obtaining the possible values that a random variable can take.

A **probability distribution** describes how the values of a random variable are distributed. It provides the probabilities of occurrence of different possible outcomes in an experiment. There are two main types of probability distributions:

- **Discrete Probability Distribution**: Associated with discrete random variables, it lists the probabilities of each possible discrete value. For example, in a roll of a die, the probability distribution assigns a probability of $ \frac{1}{6} $ to each of the six possible outcomes.

- **Continuous Probability Distribution**: Associated with continuous random variables, it is described by a probability density function (PDF). The probability of a specific value is zero; instead, probabilities are calculated over intervals. For example, the normal distribution is a common continuous distribution.

Probability distributions are fundamental in statistics, as they model the likelihood of different outcomes and help in making predictions and inferences.

4. **Normal Distribution**: A probability distribution that is symmetric about the mean, showing that data near the mean are more frequent in occurrence. It is a continuous probability distribution that is symmetric about its mean, depicting the familiar bell-shaped curve. It is defined by two parameters:

- **Mean (μ)**: The center of the distribution, where the highest point of the curve is located.
- **Standard deviation (σ)**: Measures the spread or dispersion of the distribution. A smaller standard deviation results in a steeper curve, while a larger one results in a flatter curve.

The **Empirical Rule** or the **68-95-99.7 Rule** or the Key properties of a normal distribution include:

- Approximately 68% of the data falls within one standard deviation of the mean.
- Approximately 95% of the data falls within two standard deviations of the mean.
- Approximately 99.7% of the data falls within three standard deviations of the mean.

This rule is called "empirical" because it's based on observations and is a good approximation for many real-world phenomena that follow a normal distribution. The normal distribution is widely used in statistics, as many natural phenomena and measurement errors are normally distributed.

### Bernoulli Distribution

The **Bernoulli distribution** is a discrete probability distribution for a random variable that has exactly two possible outcomes: success (usually coded as 1) and failure (usually coded as 0). It is defined by a single parameter, $ p $, which represents the probability of success.

- **Example**: Flipping a coin, where the outcome can be either heads (success) or tails (failure).

- **Usage**: The Bernoulli distribution is often used to model binary outcomes, such as predicting whether a customer will buy a product (yes/no) or whether an email is spam (spam/not spam).

## Binomial Distribution

The **Binomial distribution** is a discrete probability distribution that describes the number of successes in a fixed number of independent Bernoulli trials, each with the same probability of success. It is defined by two parameters:

- **n**: The number of trials or experiments.
- **p**: The probability of success in each trial.

The probability of observing exactly `k` successes in `n` trials is given by the binomial probability formula:

$$ P(X = k) = \binom{n}{k} p^k (1-p)^{n-k} $$

where:
- $ \binom{n}{k} $ is the binomial coefficient, which represents the number of ways to choose $ k $ successes out of $ n $ trials.
- $ p^k $ is the probability of having $ k $ successes.
- $ (1-p)^{n-k} $ is the probability of having $ n-k $ failures.

### Key Characteristics:
- The trials are independent, meaning the outcome of one trial does not affect the outcome of another.
- Each trial has only two possible outcomes: success or failure.
- The probability of success $ p $ remains constant in each trial.

### Example:
If you roll a die 10 times and want to calculate the probability of rolling exactly 3 sixes, you can use the binomial distribution with $ n = 10 $ and $ p = \frac{1}{6} $.

### Usage:

The binomial distribution is commonly used in binary classification problems, hypothesis testing, and any scenario where the interest is in the number of successes over a series of trials. It's foundational in models like logistic regression and in calculating confidence intervals for proportions.

## Poisson distribution

The **Poisson distribution** is a discrete probability distribution that expresses the probability of a given number of events occurring within a fixed interval of time or space, provided that these events happen with a known constant mean rate and independently of the time since the last event.

### Key Characteristics:
- **Parameter**: The Poisson distribution is characterized by a single parameter, $ \lambda $ (lambda), which represents the average number of occurrences (mean rate) of the event in the given interval.
- **Independence**: The occurrence of one event does not affect the occurrence of another.
- **No upper limit**: The number of events can be any non-negative integer, starting from zero.

The probability of observing exactly $ k $ events in a given interval is given by the Poisson probability formula:

$$ P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!} $$

where:
- $ k $ is the number of events.
- $ \lambda $ is the average number of events.
- $ e $ is the base of the natural logarithm (approximately 2.71828).
- $ k! $ is the factorial of $ k $.

### Example:
If a call center receives an average of 10 calls per hour ($ \lambda = 10 $), the Poisson distribution can be used to calculate the probability of receiving exactly 8 calls in the next hour.

### Usage:
The Poisson distribution is often used to model the number of times an event occurs in a fixed interval of time or space. Common applications include:
- **Modeling rare events**: Like the number of emails received per hour, the number of earthquakes in a region, or the arrival of customers at a store.
- **Event prediction**: It is used in predictive modeling where the goal is to estimate the likelihood of a certain number of events happening in a given period.

The Poisson distribution is particularly useful in scenarios where events occur randomly and independently, and it's a fundamental distribution in queuing theory, reliability engineering, and many fields within data science and machine learning.

## Exponential Distribution

The **Exponential distribution** is a continuous probability distribution that describes the time between events in a Poisson process. It is used to model the time until the next event occurs, given that events happen continuously and independently at a constant average rate.

### Key Characteristics:
- **Parameter**: The Exponential distribution is defined by a single parameter $ \lambda $ (lambda), which represents the rate at which events occur. The mean of the distribution is $ \frac{1}{\lambda} $.
- **Memoryless Property**: The distribution is memoryless, meaning that the probability of an event occurring in the future is independent of how much time has already elapsed.

The probability density function (PDF) of the Exponential distribution is given by:

$$ f(x; \lambda) = \lambda e^{-\lambda x} $$

where:
- $ x $ is the time until the next event.
- $ \lambda $ is the rate parameter (the inverse of the mean).

The cumulative distribution function (CDF), which gives the probability that the time until the next event is less than or equal to $ x $, is:

$$ F(x; \lambda) = 1 - e^{-\lambda x} $$

### Example:
If a light bulb has an average lifetime of 1,000 hours, the time until the bulb burns out can be modeled using an Exponential distribution with $ \lambda = \frac{1}{1000} $.

### Usage:
The Exponential distribution is useful in various applications:
- **Reliability Analysis**: Modeling the time until a system failure or the lifetime of products.
- **Queuing Theory**: Estimating the time between arrivals of customers in a service system.
- **Survival Analysis**: Used in survival models to assess time-to-event data.

The Exponential distribution is foundational in understanding stochastic processes and in fields where the timing of events is of interest.

## Uniform Distribution

The **Uniform distribution** is a continuous probability distribution where all outcomes in a given range are equally likely. It is characterized by two parameters, which define the range of possible values.

### Key Characteristics:
- **Parameters**: 
  - **$ a $**: The minimum value of the range.
  - **$ b $**: The maximum value of the range.
  
- **Probability Density Function (PDF)**: For a continuous uniform distribution, the PDF is constant between $ a $ and $ b $, and is given by:

  $$ f(x; a, b) = \frac{1}{b - a} \quad \text{for } a \leq x \leq b $$

  Outside this interval, the PDF is zero.

- **Cumulative Distribution Function (CDF)**: The CDF, which gives the probability that a random variable $ X $ is less than or equal to $ x $, is:

  $$ F(x; a, b) = \frac{x - a}{b - a} \quad \text{for } a \leq x \leq b $$

  For values outside this interval, $ F(x; a, b) $ is 0 if $ x < a $ and 1 if $ x > b $.

### Example:
If a random variable $ X $ is uniformly distributed between 2 and 5, then every value between 2 and 5 is equally likely. The PDF for $ X $ is $ \frac{1}{5 - 2} = \frac{1}{3} $ within this range.

### Usage:
The Uniform distribution is often used in simulations and algorithms where a uniform spread of values is required. Applications include:
- **Random Sampling**: Generating random numbers within a specific range for simulations or initializing algorithms.
- **Algorithm Testing**: Used to test algorithms and models with evenly distributed data.
- **Normalization**: In some cases, data may be transformed to a uniform distribution as part of preprocessing or normalization steps.

The Uniform distribution is simple and intuitive, making it a useful tool for various applications in probability and statistics.


### Null Hypothesis (H0)

- **Definition**: The null hypothesis is a statement that there is no effect, no difference, or no relationship between variables. It represents a default or baseline assumption that is tested against the data.

- **Purpose**: It serves as the starting point for statistical testing. The goal of hypothesis testing is to determine whether there is enough evidence to reject the null hypothesis in favor of the alternative hypothesis.

- **Form**: It is often formulated as a statement of equality or no effect. For example:
  - **In a test comparing means**: $ H_0: \mu_1 = \mu_2 $ (The means of two groups are equal.)
  - **In a test of proportions**: $ H_0: p = p_0 $ (The proportion is equal to a specified value.)

### Alternative Hypothesis (H1 or Ha)
- **Definition**: The alternative hypothesis is a statement that indicates the presence of an effect, a difference, or a relationship between variables. It represents what the researcher aims to provide evidence for.
- **Purpose**: It represents the hypothesis that is accepted if there is sufficient evidence to reject the null hypothesis.
- **Form**: It is often formulated as a statement of inequality or effect. For example:
  - **In a test comparing means**: $ H_1: \mu_1 \ne \mu_2 $ (The means of two groups are not equal.)
  - **In a test of proportions**: $ H_1: p \ne p_0 $ (The proportion is different from a specified value.)

### Example Scenario

**Study Objective**: Determine if a new teaching method improves student performance compared to a traditional method.

- **Null Hypothesis (H0)**: There is no difference in average test scores between students taught with the new method and those taught with the traditional method. (e.g., $ H_0: \mu_{\text{new}} = \mu_{\text{traditional}} $)
- **Alternative Hypothesis (H1)**: There is a difference in average test scores between the two methods. (e.g., $ H_1: \mu_{\text{new}} \ne \mu_{\text{traditional}} $)

### Testing Process
1. **Formulate Hypotheses**: Define $ H_0 $ and $ H_1 $.
2. **Collect Data**: Gather sample data relevant to the hypothesis.
3. **Conduct Statistical Test**: Perform an appropriate statistical test to calculate a test statistic and p-value.
4. **Compare p-value to Significance Level**: Decide whether to reject or fail to reject $ H_0 $ based on the p-value and significance level ($ \alpha $).

If the evidence in the data is strong enough to reject the null hypothesis, you accept the alternative hypothesis, suggesting that there is a statistically significant effect or difference.

## p-Value

The **p-value** (probability value) is a statistical measure that helps determine the significance of your results in hypothesis testing. It quantifies the probability of obtaining an effect at least as extreme as the one observed in your sample, assuming that the null hypothesis is true.

### Key Points:

- **Null Hypothesis (H0)**: The default assumption that there is no effect or no difference. For example, it might state that there is no difference between two groups or no association between two variables.

- **Alternative Hypothesis (H1 or Ha)**: The hypothesis that there is an effect or a difference.

- **Interpretation of p-value**:
  - **Low p-value (typically ≤ 0.05)**: Indicates that the observed result is unlikely to have occurred under the null hypothesis. This often leads to rejecting the null hypothesis in favor of the alternative hypothesis.
  - **High p-value (> 0.05)**: Suggests that the observed result is consistent with the null hypothesis, and there is not enough evidence to reject it.

### Calculation and Usage:

1. **Calculate the p-value**: Based on the test statistic (e.g., t-statistic, z-score) from your data, calculate the probability of observing a result as extreme as or more extreme than the one obtained, assuming the null hypothesis is true.

2. **Compare with Significance Level ($ \alpha $)**: The significance level ($ \alpha $) is a threshold set before conducting the test (commonly 0.05). Compare the p-value to $ \alpha $:
   - If $ \text{p-value} \leq \alpha $, reject the null hypothesis.
   - If $ \text{p-value} > \alpha $, fail to reject the null hypothesis.

### Example:

Suppose you conduct a study to determine if a new drug has a different effect compared to a placebo. Your null hypothesis is that there is no difference in effect between the drug and the placebo. If the p-value from your test is 0.03, and you have set your significance level at 0.05, you would reject the null hypothesis, indicating that the observed difference is statistically significant.

### Cautions:

- **Not a Measure of Effect Size**: A small p-value indicates that the observed result is unlikely under the null hypothesis but does not measure the size or importance of the effect.
- **Dependent on Sample Size**: Large sample sizes can lead to very small p-values even for trivial effects. Therefore, p-values should be considered alongside other measures, such as confidence intervals and effect sizes.

The p-value is a central concept in hypothesis testing and is widely used in research to assess the strength of evidence against the null hypothesis.

## Critical Value

A **critical value** is a threshold used in hypothesis testing to determine whether to reject the null hypothesis. It is a point (or points) on the scale of the test statistic that separates the region where the null hypothesis is not rejected from the region where it is rejected.

### Key Characteristics:

- **Significance Level ($ \alpha $)**: The critical value is determined based on the significance level, which is the probability of rejecting the null hypothesis when it is actually true (Type I error). Common significance levels are 0.05, 0.01, and 0.10.

- **Test Statistic**: The critical value is compared to the test statistic obtained from your sample data. The test statistic measures how far your sample statistic is from the null hypothesis.

- **Regions of Rejection**: The critical value defines the boundary of the rejection region (or regions) in the distribution of the test statistic. If the test statistic falls in this region, the null hypothesis is rejected.

### Types of Critical Values:

1. **Z-Critical Value**: Used in z-tests, which are based on the standard normal distribution. For a significance level of 0.05 in a two-tailed test, the z-critical values are approximately ±1.96.

2. **T-Critical Value**: Used in t-tests, which are based on the Student's t-distribution. The t-critical value depends on the significance level and the degrees of freedom (related to sample size). For example, for a significance level of 0.05 and 20 degrees of freedom, the t-critical value might be approximately ±2.086.

3. **Chi-Square Critical Value**: Used in chi-square tests, based on the chi-square distribution. The critical value depends on the significance level and degrees of freedom. For example, for a significance level of 0.05 and 10 degrees of freedom, the chi-square critical value might be approximately 18.307.

### Example:

Suppose you are conducting a z-test with a significance level of 0.05 in a two-tailed test. The critical values for this significance level are approximately ±1.96. If your calculated z-test statistic is 2.5, which exceeds the critical value of 1.96, you would reject the null hypothesis, indicating that the result is statistically significant.

In summary, critical values are essential for decision-making in hypothesis testing, helping to determine whether the observed data provide sufficient evidence to reject the null hypothesis.

## Z-Score

A **z-score** is a measure that describes how many standard deviations a data point is from the mean of a distribution. It is a standardized score used to compare data points from different distributions or to understand the relative position of a data point within its distribution.

### Key Characteristics:

- **Formula**:
  
  $$   z = \frac{X - \mu}{\sigma} $$
  
  where:
  - $ X $ is the data point.
  - $ \mu $ is the mean of the distribution.
  - $ \sigma $ is the standard deviation of the distribution.

- **Interpretation**:
  - **$ z = 0 $**: The data point is exactly at the mean.
  - **$ z > 0 $**: The data point is above the mean.
  - **$ z < 0 $**: The data point is below the mean.
  - **Magnitude**: The larger the absolute value of the z-score, the further away the data point is from the mean.

### Example:

Suppose the mean test score of a class is 70, with a standard deviation of 10. If a student scores 85, the z-score is calculated as:

$$ z = \frac{85 - 70}{10} = 1.5 $$

This z-score of 1.5 means the student's score is 1.5 standard deviations above the mean.

### Usage in Data Science/ML:

- **Standardization**: Z-scores are used in data preprocessing to standardize data. Standardizing transforms data to have a mean of 0 and a standard deviation of 1, which is useful for many machine learning algorithms that assume normally distributed data.
  
- **Outlier Detection**: Z-scores help in identifying outliers. Typically, data points with z-scores greater than ±2 or ±3 are considered outliers.

- **Comparison**: Z-scores allow comparison of data points from different distributions or scales by converting them to a common scale.

The z-score is a fundamental concept in statistics and helps in interpreting and comparing data points relative to the distribution from which they originate.

## Central Limit Theorem

The **Central Limit Theorem (CLT)** is a fundamental theorem in statistics that describes the distribution of the sum (or average) of a large number of independent, identically distributed random variables. It states that, under certain conditions, the distribution of the sum (or average) of these variables approaches a normal distribution, regardless of the original distribution of the variables.

### Key Aspects:

- **Independence**: The random variables must be independent, meaning the outcome of one variable does not affect the outcome of another.

- **Identically Distributed**: The random variables must have the same probability distribution and the same mean and variance.

- **Sample Size**: The theorem applies when the sample size is sufficiently large. Although "sufficiently large" depends on the shape of the original distribution, a common rule of thumb is that a sample size of 30 or more is often adequate for the CLT to hold.

- **Normal Distribution**: As the sample size increases, the distribution of the sample mean (or sum) will approximate a normal distribution, even if the original data is not normally distributed.

### Formula:
If $ X_1, X_2, \ldots, X_n $ are independent random variables with mean $ \mu $ and variance $ \sigma^2 $, the sample mean $ \bar{X} $ is given by:

$$ \bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i $$

According to the CLT, the distribution of $ \bar{X} $ approaches a normal distribution with mean $ \mu $ and variance $ \frac{\sigma^2}{n} $ as $ n $ becomes large.

### Example:
Suppose you have a population of students with a non-normal distribution of test scores. If you take multiple random samples of size 50 and calculate the mean score for each sample, the distribution of these sample means will approximate a normal distribution, even if the original population distribution is not normal.

### Usage in Data Science/ML:

- **Inferential Statistics**: The CLT justifies the use of normal distribution approximations for hypothesis testing and confidence intervals, even when the underlying data distribution is unknown.

- **Sampling Distribution**: It allows for the application of statistical methods to estimate population parameters and test hypotheses based on sample data.

- **Modeling**: Many statistical and machine learning algorithms assume that the data follows a normal distribution, and the CLT provides a theoretical foundation for these assumptions.

The Central Limit Theorem is crucial in statistics because it simplifies the process of making inferences about populations based on sample data and supports the use of various statistical techniques.

## Confidence Interval

A **confidence interval** is a range of values, derived from sample data, that is likely to contain the true population parameter (such as the mean or proportion) with a certain level of confidence. It provides an estimate of the uncertainty associated with a sample statistic.

### Key Characteristics:

- **Point Estimate**: The central value of the confidence interval is typically the sample statistic, such as the sample mean ($ \bar{X} $) or proportion ($ \hat{p} $).

- **Confidence Level**: The confidence level, usually expressed as a percentage (e.g., 90%, 95%, 99%), indicates the probability that the confidence interval contains the true population parameter. A 95% confidence level, for example, means that if we were to take 100 different samples and compute a confidence interval from each, about 95 of those intervals would be expected to contain the true population parameter.

- **Margin of Error**: The confidence interval is calculated by adding and subtracting a margin of error from the point estimate. The margin of error depends on the standard error of the sample statistic and the critical value (z-score or t-score) associated with the desired confidence level.

### Formula for a Confidence Interval:

For a population mean, the confidence interval is typically calculated as:

$$ \text{Confidence Interval} = \bar{X} \pm Z \times \frac{\sigma}{\sqrt{n}} $$

Where:
- $ \bar{X} $ is the sample mean.
- $ Z $ is the critical value from the standard normal distribution corresponding to the desired confidence level (e.g., 1.96 for 95% confidence).
- $ \sigma $ is the population standard deviation (or sample standard deviation if the population standard deviation is unknown).
- $ n $ is the sample size.

For a population proportion, the confidence interval is:

$$ \text{Confidence Interval} = \hat{p} \pm Z \times \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} $$

Where:
- $ \hat{p} $ is the sample proportion.
- $ Z $ is the critical value from the standard normal distribution.
- $ n $ is the sample size.

### Example:

Suppose you collect a sample of 100 students' test scores with a mean score of 75 and a standard deviation of 10. To calculate a 95% confidence interval for the population mean:

- Point estimate ($ \bar{X} $) = 75
- Standard error = $ \frac{10}{\sqrt{100}} = 1 $
- Critical value for 95% confidence ($ Z $) ≈ 1.96

The confidence interval is:

$$ 75 \pm 1.96 \times 1 = 75 \pm 1.96 = [73.04, 76.96] $$

This means you can be 95% confident that the true population mean lies between 73.04 and 76.96.

### Usage in Data Science/ML:

- **Estimation**: Confidence intervals are used to estimate population parameters (such as mean, variance, proportion) based on sample data.
  
- **Model Evaluation**: Confidence intervals can be applied to assess the reliability of model performance metrics, such as accuracy or error rates.

- **Hypothesis Testing**: Confidence intervals provide an alternative to hypothesis testing, offering a range of values for the parameter of interest rather than a simple reject/accept decision.

Confidence intervals are a powerful tool in inferential statistics, helping to quantify the uncertainty in estimates derived from sample data and providing a range within which the true population parameter is likely to fall.


## Type I & Type II Errors

In the context of hypothesis testing, **Type I** and **Type II** errors are two kinds of errors that can occur when making a decision about whether to reject or fail to reject the null hypothesis.

### Type I Error (False Positive)

- **Definition**: A Type I error occurs when the null hypothesis ($ H_0 $) is true, but it is incorrectly rejected. In other words, it is the mistake of concluding that there is an effect or a difference when, in reality, there is none.

- **Significance Level ($ \alpha $)**: The probability of making a Type I error is denoted by the significance level ($ \alpha $). Common values for $ \alpha $ are 0.05, 0.01, or 0.10, meaning there is a 5%, 1%, or 10% chance of rejecting the null hypothesis when it is actually true.

- **Example**: If a medical test incorrectly indicates that a healthy person has a disease, that’s a Type I error.

### Type II Error (False Negative)

- **Definition**: A Type II error occurs when the null hypothesis ($ H_0 $) is false, but it is incorrectly failed to be rejected. In other words, it is the mistake of failing to detect an effect or difference when one actually exists.

- **Probability ($ \beta $)**: The probability of making a Type II error is denoted by $ \beta $. The power of a test (1 - $ \beta $) is the probability of correctly rejecting a false null hypothesis.

- **Example**: If a medical test incorrectly indicates that a person with a disease is healthy, that’s a Type II error.

### Relationship Between Type I and Type II Errors

- **Trade-off**: There is often a trade-off between Type I and Type II errors. Reducing the chance of a Type I error (by lowering $ \alpha $) can increase the chance of a Type II error, and vice versa. This is because making a test more stringent (to avoid false positives) might make it less sensitive (more likely to miss true effects).

- **Balancing the Errors**: The choice of $ \alpha $ depends on the context and the consequences of making each type of error. For example, in medical testing, where a Type I error might mean unnecessary treatment but a Type II error could mean missing a serious condition, the balance is critical.

### Summary:

- **Type I Error** (False Positive): Rejecting $ H_0 $ when it is actually true. Probability = $ \alpha $.
- **Type II Error** (False Negative): Failing to reject $ H_0 $ when it is actually false. Probability = $ \beta $.

Understanding Type I and Type II errors is crucial in hypothesis testing to ensure that decisions are made with appropriate consideration of the risks and consequences of incorrect conclusions.

## Power of a Test

The **power of a test** is a key concept in hypothesis testing that measures the test's ability to correctly reject the null hypothesis when it is false. In other words, it quantifies the likelihood that a test will detect an effect or difference when one actually exists.

### Key Characteristics:

- **Definition**: The power of a test is defined as the probability that the test correctly rejects a false null hypothesis ($ H_0 $). It is mathematically represented as $ 1 - \beta $, where $ \beta $ is the probability of making a Type II error (failing to reject a false null hypothesis).

- **Interpretation**:
  - A **high power** (close to 1) means there is a strong likelihood of detecting an effect if one exists, reducing the risk of a Type II error.
  - A **low power** (close to 0) means the test is less likely to detect an effect, increasing the risk of a Type II error.

- **Factors Affecting Power**:
  1. **Sample Size**: Larger sample sizes generally increase the power of a test because they provide more information about the population, making it easier to detect true effects.
  2. **Effect Size**: The magnitude of the difference or effect being tested. Larger effect sizes increase the power, as the effect is more likely to be detected.
  3. **Significance Level ($ \alpha $)**: Setting a higher significance level (e.g., 0.10 instead of 0.05) increases the power of the test but also raises the risk of a Type I error.
  4. **Variance**: Lower variance within the data increases power because the data points are more tightly clustered, making it easier to detect differences.

- **Example**:
  Suppose a pharmaceutical company is testing a new drug. The null hypothesis is that the drug has no effect. The power of the test is the probability that the study will detect a beneficial effect of the drug if it truly has one. If the power of the test is 0.80 (or 80%), there is an 80% chance of correctly rejecting the null hypothesis if the drug is effective.

### Importance in Data Science/ML:

- **Study Design**: Power analysis is often used in the planning phase of experiments to determine the sample size needed to achieve a desired power level, ensuring that the study is likely to detect meaningful effects.

- **Model Evaluation**: In machine learning, the power of a test can relate to the model’s ability to detect true positives (e.g., correctly classifying a positive instance), which is crucial for evaluating model performance.

- **Balancing Errors**: Understanding the power of a test helps in balancing the risks of Type I and Type II errors, making it possible to design tests that are both sensitive and reliable.

In summary, the power of a test is a critical metric in hypothesis testing that reflects the test’s ability to identify true effects or differences, thereby reducing the likelihood of missing significant findings.


## Bayes' Theorem

**Bayes' Theorem** is a fundamental concept in probability theory and statistics that describes how to update the probability of a hypothesis based on new evidence or information. It provides a mathematical framework for revising existing beliefs (prior probabilities) in light of new data (likelihood), resulting in updated beliefs (posterior probabilities).

### Bayes' Theorem Formula:

$$ P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)} $$

Where:
- $ P(H|E) $ is the **posterior probability**: The probability of the hypothesis $ H $ given the evidence $ E $.
- $ P(E|H) $ is the **likelihood**: The probability of observing the evidence $ E $ given that the hypothesis $ H $ is true.
- $ P(H) $ is the **prior probability**: The initial probability of the hypothesis $ H $ before seeing the evidence $ E $.
- $ P(E) $ is the **marginal likelihood** or **evidence**: The total probability of observing the evidence $ E $ under all possible hypotheses.

### Key Concepts:

- **Prior Probability ($ P(H) $)**: This represents your initial belief or estimate of the probability of the hypothesis before observing the new evidence. For example, if you believe that a person is generally healthy, you might assign a low prior probability to them having a rare disease.

- **Likelihood ($ P(E|H) $)**: This represents how likely the observed evidence is, assuming that the hypothesis is true. For example, if a medical test is accurate, the likelihood of testing positive if someone has the disease will be high.

- **Posterior Probability ($ P(H|E) $)**: This is the updated probability of the hypothesis after considering the new evidence. It reflects how the prior belief has been adjusted in light of the data.

- **Marginal Likelihood ($ P(E) $)**: This is the total probability of the evidence under all possible scenarios. It acts as a normalizing factor to ensure that the posterior probabilities sum to 1.

### Example:

Imagine you're testing for a rare disease that affects 1% of the population. The test is 99% accurate, meaning it has a 99% chance of correctly identifying someone with the disease (true positive) and a 99% chance of correctly identifying someone without the disease (true negative).

- **Hypothesis $ H $**: The person has the disease.
- **Evidence $ E $**: The person tested positive.

Using Bayes' Theorem:

1. **Prior Probability ($ P(H) $)**: The probability that the person has the disease before testing = 0.01.
2. **Likelihood ($ P(E|H) $)**: The probability of testing positive if the person has the disease = 0.99.
3. **Marginal Likelihood ($ P(E) $)**: The overall probability of testing positive, which includes both true positives and false positives.

Now, calculate the posterior probability ($ P(H|E) $):

$$ P(H|E) = \frac{0.99 \times 0.01}{(0.99 \times 0.01) + (0.01 \times 0.99)} \approx 0.50 $$

This result means that even if the person tests positive, the probability that they actually have the disease is only about 50%, because the disease is so rare.

### Usage in Data Science/ML:

- **Naive Bayes Classifier**: A popular machine learning algorithm that applies Bayes' Theorem with the assumption that features are independent. It’s used in spam detection, sentiment analysis, and more.
  
- **Bayesian Inference**: Bayes' Theorem is fundamental to Bayesian statistics, where it is used to update the probability of a hypothesis as more data becomes available.

- **Decision Making**: It’s used in decision-making processes under uncertainty, where you need to update your beliefs based on new evidence.

Bayes' Theorem is a powerful tool for reasoning under uncertainty, allowing you to systematically update your beliefs in light of new evidence. It forms the basis of Bayesian statistics and is widely used in various fields, including data science, machine learning, and artificial intelligence.

### Likelihood

**Likelihood** is a fundamental concept in statistics and probability theory that measures how well a statistical model explains observed data. It is a function of the parameters of a model, given a set of observed data. The likelihood function tells us the probability of the observed data under different possible values of the model parameters.

### Key Points:

- **Function of Parameters**: Unlike the probability of data given parameters (which is a fixed value), the likelihood is viewed as a function of the parameters, with the observed data held constant.
  
- **Purpose**: The goal is to find the parameter values that maximize the likelihood, meaning that they make the observed data most probable under the model.

- **Formula**: If you have a set of data points $ X = \{x_1, x_2, \dots, x_n\} $ and a model with parameters $ \theta $, the likelihood function $ L(\theta) $ is often expressed as:

$$ L(\theta) = P(X|\theta) = P(x_1|\theta) \times P(x_2|\theta) \times \dots \times P(x_n|\theta) $$

For continuous distributions, this is typically the product of the probability density function (PDF) evaluated at each data point.

### Example:

Consider a coin with an unknown probability $ \theta $ of landing heads. If you flip the coin 10 times and observe 7 heads and 3 tails, the likelihood of observing this data can be calculated as:

$$ L(\theta) = \theta^7 \times (1 - \theta)^3 $$

This likelihood function can be used to determine the most likely value of $ \theta $ that explains the observed data.

### Maximum Likelihood Estimator (MLE)

**Maximum Likelihood Estimation (MLE)** is a method for estimating the parameters of a statistical model. The idea is to find the parameter values that maximize the likelihood function, meaning these values make the observed data most probable.

### Key Points:

- **Objective**: The objective of MLE is to find the parameter $ \hat{\theta} $ that maximizes the likelihood function $ L(\theta) $.

- **Procedure**:
  1. **Write the Likelihood Function**: Express the likelihood function $ L(\theta) $ for the observed data.
  2. **Log-Likelihood**: For mathematical convenience, especially when dealing with products, the log-likelihood function is often used, as it transforms products into sums. The log-likelihood is given by:

  $$ \text{Log-Likelihood} = \log L(\theta) = \sum_{i=1}^{n} \log P(x_i|\theta) $$

  3. **Maximize**: Find the parameter $ \hat{\theta} $ that maximizes the log-likelihood function. This is usually done by taking the derivative of the log-likelihood function with respect to $ \theta $, setting it equal to zero, and solving for $ \theta $.

- **Example**:
  
  Continuing with the coin flip example, the log-likelihood function for $ \theta $ would be:

  $$ \log L(\theta) = 7 \log(\theta) + 3 \log(1 - \theta) $$

  To find the MLE, take the derivative with respect to $ \theta $, set it to zero, and solve for $ \theta $. In this case, the MLE $ \hat{\theta} $ is the proportion of heads observed, which is $ 7/10 = 0.7 $.

### Usage in Data Science/ML:

- **Model Fitting**: MLE is a common method for fitting statistical models to data. It's used in linear regression, logistic regression, and many other models.
  
- **Parameter Estimation**: MLE provides a way to estimate the parameters of a model that best explain the data, making it a critical tool in both supervised and unsupervised learning.

- **Probabilistic Models**: In machine learning models that involve probabilities, such as Naive Bayes classifiers, Hidden Markov Models, and others, MLE is often used to estimate parameters.

### Summary:

- **Likelihood** measures how probable the observed data is under different parameter values of a model.
- **Maximum Likelihood Estimation (MLE)** is the process of finding the parameter values that maximize this likelihood, providing the best fit for the data.

MLE is widely used because it has desirable properties, such as consistency (it gives the true parameter as the sample size increases) and efficiency (it uses the data effectively to provide accurate estimates).

20. **Overfitting**: A modeling error that occurs when a model is too complex and captures noise along with the underlying pattern.

21. **Underfitting**: A modeling error that occurs when a model is too simple to capture the underlying pattern of the data.

## Bias

Bias refers to the error introduced by approximating a real-world problem with a simplified model. It's the difference between the expected predictions of our model and the true values.

High bias means the model makes strong assumptions about the data, leading to an oversimplified model that fails to capture important patterns (underfitting).

## Variance 

Variance is the model's sensitivity to small fluctuations in the training data. It measures how much the predictions would change if we used a different training dataset.

High variance means the model is overly complex and captures noise in the training data, leading to poor generalization on new data (overfitting).

### Bias-Variance Trade-off

The bias-variance trade-off is the balance between underfitting (high bias) and overfitting (high variance). The goal is to find the sweet spot where the model complexity is just right to capture the underlying patterns without fitting to noise.

As model complexity increases:
- Bias tends to decrease (the model can fit the training data better)
- Variance tends to increase (the model becomes more sensitive to variations in the training data)

The optimal point balances these two sources of error to minimize overall prediction error.

### Underfitting

Underfitting occurs when a model is too simple to capture the underlying patterns in the data. Signs of underfitting include:
- High bias
- Low variance
- Poor performance on both training and test data

### Overfitting

Overfitting happens when a model learns the training data too well, including its noise and peculiarities. An overfit model will perform well on training data but poorly on new, unseen data. Signs of overfitting include:
- Low bias
- High variance
- Excellent performance on training data, but poor performance on test data


24. **Covariance**: A measure of how much two random variables change together, indicating the direction of their linear relationship.

25. **Correlation**: A standardized measure of the linear relationship between two variables, ranging from -1 to 1.

26. **Linear Regression**: A method to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation.

27. **Logistic Regression**: A statistical method for modeling binary outcome variables using a logistic function.

28. **ANOVA (Analysis of Variance)**: A statistical method used to compare means across multiple groups and determine if they are significantly different.

29. **Chi-Square Test**: A statistical test used to determine if there is a significant association between two categorical variables.

30. **T-Test**: A statistical test used to compare the means of two groups to determine if they are significantly different from each other.

31. **Maximum Likelihood Estimation (MLE)**: A method used to estimate the parameters of a statistical model that maximizes the likelihood of the observed data.

32. **Prior Probability**: The probability assigned to an event or hypothesis before any new evidence is considered.

33. **Posterior Probability**: The updated probability of an event or hypothesis after considering new evidence, calculated using Bayes' Theorem.

34. **Marginal Probability**: The probability of an event occurring irrespective of the outcomes of other variables.

35. **Joint Probability**: The probability of two or more events occurring simultaneously.

36. **Conditional Probability**: The probability of an event occurring given that another event has already occurred.

37. **Monte Carlo Simulation**: A computational technique that uses random sampling to estimate complex mathematical or physical systems.

38. **Bootstrap Method**: A statistical technique that involves resampling with replacement to estimate the sampling distribution of a statistic.

39. **Markov Chain**: A stochastic process where the probability of transitioning to any future state depends only on the current state and not on the sequence of events that preceded it.

40. **Expectation-Maximization (EM) Algorithm**: A method used to find maximum likelihood estimates of parameters in models with latent variables.

41. **Principal Component Analysis (PCA)**: A dimensionality reduction technique that transforms data into a set of orthogonal components that capture the maximum variance.

42. **Eigenvalues and Eigenvectors**: Mathematical concepts used in PCA and other linear transformations, where eigenvectors represent directions, and eigenvalues represent magnitudes.

43. **Entropy**: A measure of uncertainty or randomness in a probability distribution, often used in information theory.

44. **Mutual Information**: A measure of the amount of information obtained about one random variable through another random variable.

45. **Gini Coefficient**: A measure of inequality or diversity, often used in decision trees to evaluate splits.

46. **Kurtosis**: A measure of the "tailedness" of the probability distribution of a real-valued random variable.

47. **Skewness**: A measure of the asymmetry of the probability distribution of a real-valued random variable.

48. **Sample Space**: The set of all possible outcomes in a probability experiment.

49. **Law of Large Numbers**: A principle stating that as the size of a sample increases, the sample mean will approach the population mean.

50. **Homoscedasticity**: A condition in which the variance of errors or the residuals is constant across all levels of the independent variable(s).

51. **Heteroscedasticity**: A condition where the variance of errors or residuals is not constant across all levels of the independent variable(s).
