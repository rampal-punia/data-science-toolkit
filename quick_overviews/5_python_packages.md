## Numpy 

Here’s a list of 21 important commands, methods, and functions in NumPy and Pandas, including examples where relevant:

### **NumPy**

1. **`numpy.array()`**
   - **Definition**: Creates a NumPy array from a list or tuple.
   - **Code**: 
     ```python
     import numpy as np
     arr = np.array([1, 2, 3])
     ```

2. **`numpy.zeros()`**
   - **Definition**: Creates an array filled with zeros.
   - **Code**: 
     ```python
     arr = np.zeros((2, 3))
     ```

3. **`numpy.ones()`**
   - **Definition**: Creates an array filled with ones.
   - **Code**: 
     ```python
     arr = np.ones((2, 3))
     ```

4. **`numpy.arange()`**
   - **Definition**: Creates an array with evenly spaced values within a specified range.
   - **Code**: 
     ```python
     arr = np.arange(0, 10, 2)
     ```

5. **`numpy.linspace()`**
   - **Definition**: Creates an array with evenly spaced values over a specified interval.
   - **Code**: 
     ```python
     arr = np.linspace(0, 1, 5)
     ```

6. **`numpy.reshape()`**
   - **Definition**: Changes the shape of an array without changing its data.
   - **Code**: 
     ```python
     arr = np.arange(6).reshape((2, 3))
     ```

7. **`numpy.transpose()`**
   - **Definition**: Permutes the dimensions of an array.
   - **Code**: 
     ```python
     arr = np.array([[1, 2], [3, 4]])
     transposed_arr = np.transpose(arr)
     ```

8. **`numpy.dot()`**
   - **Definition**: Computes the dot product of two arrays.
   - **Code**: 
     ```python
     arr1 = np.array([1, 2])
     arr2 = np.array([3, 4])
     dot_product = np.dot(arr1, arr2)
     ```

9. **`numpy.sum()`**
   - **Definition**: Computes the sum of array elements.
   - **Code**: 
     ```python
     arr = np.array([1, 2, 3])
     total = np.sum(arr)
     ```

10. **`numpy.mean()`**
    - **Definition**: Computes the mean of array elements.
    - **Code**: 
      ```python
      arr = np.array([1, 2, 3])
      mean_value = np.mean(arr)
      ```

11. **`numpy.std()`**
    - **Definition**: Computes the standard deviation of array elements.
    - **Code**: 
      ```python
      arr = np.array([1, 2, 3])
      std_dev = np.std(arr)
      ```

12. **`numpy.max()`**
    - **Definition**: Finds the maximum value in an array.
    - **Code**: 
      ```python
      arr = np.array([1, 2, 3])
      max_value = np.max(arr)
      ```

13. **`numpy.min()`**
    - **Definition**: Finds the minimum value in an array.
    - **Code**: 
      ```python
      arr = np.array([1, 2, 3])
      min_value = np.min(arr)
      ```

14. **`numpy.argsort()`**
    - **Definition**: Returns the indices that would sort an array.
    - **Code**: 
      ```python
      arr = np.array([3, 1, 2])
      sorted_indices = np.argsort(arr)
      ```

15. **`numpy.where()`**
    - **Definition**: Returns indices of elements that satisfy a condition.
    - **Code**: 
      ```python
      arr = np.array([1, 2, 3])
      indices = np.where(arr > 1)
      ```

16. **`numpy.unique()`**
    - **Definition**: Finds unique elements in an array.
    - **Code**: 
      ```python
      arr = np.array([1, 2, 2, 3])
      unique_elements = np.unique(arr)
      ```

17. **`numpy.concatenate()`**
    - **Definition**: Joins a sequence of arrays along an existing axis.
    - **Code**: 
      ```python
      arr1 = np.array([1, 2])
      arr2 = np.array([3, 4])
      concatenated_arr = np.concatenate((arr1, arr2))
      ```

18. **`numpy.histogram()`**
    - **Definition**: Computes the histogram of a set of data.
    - **Code**: 
      ```python
      data = np.array([1, 2, 1, 3, 4])
      hist, bins = np.histogram(data, bins=4)
      ```

19. **`numpy.corrcoef()`**
    - **Definition**: Computes the correlation coefficient matrix.
    - **Code**: 
      ```python
      arr1 = np.array([1, 2, 3])
      arr2 = np.array([4, 5, 6])
      correlation = np.corrcoef(arr1, arr2)
      ```

20. **`numpy.random.rand()`**
    - **Definition**: Generates an array of random numbers between 0 and 1.
    - **Code**: 
      ```python
      arr = np.random.rand(3, 2)
      ```

21. **`numpy.random.randn()`**
    - **Definition**: Generates an array of random numbers from a standard normal distribution.
    - **Code**: 
      ```python
      arr = np.random.randn(3, 2)
      ```

## Pandas

Here are 51 important commands, methods, and functions in Pandas, along with examples where relevant:

### **1. `pd.DataFrame()`**
- **Definition**: Creates a DataFrame from various data structures like dictionaries, lists, or arrays.
- **Code**: 
  ```python
  import pandas as pd
  df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
  ```

### **2. `pd.Series()`**
- **Definition**: Creates a Series object from a list, array, or dictionary.
- **Code**: 
  ```python
  series = pd.Series([1, 2, 3])
  ```

### **3. `DataFrame.head()`**
- **Definition**: Returns the first n rows of the DataFrame.
- **Code**: 
  ```python
  df.head(2)
  ```

### **4. `DataFrame.tail()`**
- **Definition**: Returns the last n rows of the DataFrame.
- **Code**: 
  ```python
  df.tail(2)
  ```

### **5. `DataFrame.info()`**
- **Definition**: Provides a concise summary of the DataFrame, including the number of non-null entries and data types.
- **Code**: 
  ```python
  df.info()
  ```

### **6. `DataFrame.describe()`**
- **Definition**: Generates descriptive statistics of the DataFrame, including mean, standard deviation, and percentiles.
- **Code**: 
  ```python
  df.describe()
  ```

### **7. `DataFrame.loc[]`**
- **Definition**: Accesses a group of rows and columns by labels or boolean arrays.
- **Code**: 
  ```python
  df.loc[0:1, 'A']
  ```

### **8. `DataFrame.iloc[]`**
- **Definition**: Accesses a group of rows and columns by integer positions.
- **Code**: 
  ```python
  df.iloc[0:2, 0]
  ```

### **9. `DataFrame.groupby()`**
- **Definition**: Groups DataFrame using a mapper or by a series of columns.
- **Code**: 
  ```python
  grouped = df.groupby('A').sum()
  ```

### **10. `DataFrame.merge()`**
- **Definition**: Merges DataFrame objects with a database-style join.
- **Code**: 
  ```python
  df1 = pd.DataFrame({'key': ['A', 'B'], 'value': [1, 2]})
  df2 = pd.DataFrame({'key': ['A', 'B'], 'value': [3, 4]})
  merged_df = pd.merge(df1, df2, on='key')
  ```

### **11. `DataFrame.concat()`**
- **Definition**: Concatenates DataFrames along a particular axis.
- **Code**: 
  ```python
  df1 = pd.DataFrame({'A': [1, 2]})
  df2 = pd.DataFrame({'A': [3, 4]})
  concatenated_df = pd.concat([df1, df2])
  ```

### **12. `DataFrame.pivot_table()`**
- **Definition**: Creates a pivot table from the DataFrame.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': ['foo', 'bar'], 'B': [1, 2], 'C': [3, 4]})
  pivot_table = df.pivot_table(values='C', index='A', columns='B')
  ```

### **13. `DataFrame.drop()`**
- **Definition**: Removes rows or columns by labels.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
  df_dropped = df.drop(columns=['B'])
  ```

### **14. `DataFrame.rename()`**
- **Definition**: Renames labels of rows or columns.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
  df_renamed = df.rename(columns={'A': 'X'})
  ```

### **15. `DataFrame.sort_values()`**
- **Definition**: Sorts the DataFrame by the values of one or more columns.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [2, 1], 'B': [4, 3]})
  sorted_df = df.sort_values(by='A')
  ```

### **16. `DataFrame.sort_index()`**
- **Definition**: Sorts the DataFrame by its index.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [2, 1]}, index=['b', 'a'])
  sorted_df = df.sort_index()
  ```

### **17. `DataFrame.fillna()`**
- **Definition**: Fills NA/NaN values using a specified method or value.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, None], 'B': [None, 2]})
  filled_df = df.fillna(0)
  ```

### **18. `DataFrame.dropna()`**
- **Definition**: Removes missing values.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, None], 'B': [None, 2]})
  df_dropped = df.dropna()
  ```

### **19. `DataFrame.isna()`**
- **Definition**: Detects missing values.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, None], 'B': [None, 2]})
  na_df = df.isna()
  ```

### **20. `DataFrame.notna()`**
- **Definition**: Detects non-missing values.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, None], 'B': [None, 2]})
  notna_df = df.notna()
  ```

### **21. `DataFrame.apply()`**
- **Definition**: Applies a function along the axis of the DataFrame.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
  result = df.apply(lambda x: x + 1)
  ```

### **22. `DataFrame.map()`**
- **Definition**: Maps values of a Series according to an input mapping or function.
- **Code**: 
  ```python
  df = pd.Series([1, 2, 3])
  mapped_series = df.map({1: 'one', 2: 'two', 3: 'three'})
  ```

### **23. `DataFrame.applymap()`**
- **Definition**: Applies a function to each element of the DataFrame.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
  result = df.applymap(lambda x: x * 2)
  ```

### **24. `DataFrame.to_csv()`**
- **Definition**: Writes the DataFrame to a CSV file.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
  df.to_csv('output.csv')
  ```

### **25. `pd.read_csv()`**
- **Definition**: Reads a CSV file into a DataFrame.
- **Code**: 
  ```python
  df = pd.read_csv('input.csv')
  ```

### **26. `DataFrame.to_excel()`**
- **Definition**: Writes the DataFrame to an Excel file.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
  df.to_excel('output.xlsx')
  ```

### **27. `pd.read_excel()`**
- **Definition**: Reads an Excel file into a DataFrame.
- **Code**: 
  ```python
  df = pd.read_excel('input.xlsx')
  ```

### **28. `DataFrame.value_counts()`**
- **Definition**: Returns a Series containing counts of unique values.
- **Code**: 
  ```python
  df = pd.Series([1, 1, 2, 2, 2])
  counts = df.value_counts()
  ```

### **29. `DataFrame.corr()`**
- **Definition**: Computes the pairwise correlation of columns.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
  correlation = df.corr()
  ```

### **30. `DataFrame.cumsum()`**
- **Definition**: Computes the cumulative sum of DataFrame elements.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
  cumsum_df = df.cumsum()
  ```

Certainly, here’s the continuation:

### **31. `DataFrame.cumprod()`**
- **Definition**: Computes the cumulative product of DataFrame elements.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
  cumprod_df = df.cumprod()
  ```

### **32. `DataFrame.rolling()`**
- **Definition**: Provides rolling window calculations on DataFrame.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2, 3, 4]})
  rolling_mean = df.rolling(window=2).mean()
  ```

### **33. `DataFrame.expanding()`**
- **Definition**: Provides expanding window calculations on DataFrame.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2, 3, 4]})
  expanding_sum = df.expanding().sum()
  ```

### **34. `DataFrame.shift()`**
- **Definition**: Shifts the data in the DataFrame along the specified axis.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2, 3, 4]})
  shifted_df = df.shift(1)
  ```

### **35. `DataFrame.pivot()`**
- **Definition**: Reshapes data based on column values.
- **Code**: 
  ```python
  df = pd.DataFrame({'Date': ['2024-01-01', '2024-01-02'],
                     'City': ['New York', 'Los Angeles'],
                     'Temperature': [30, 70]})
  pivoted_df = df.pivot(index='Date', columns='City', values='Temperature')
  ```

### **36. `DataFrame.stack()`**
- **Definition**: Stacks the columns of a DataFrame into a Series.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
  stacked_df = df.stack()
  ```

### **37. `DataFrame.unstack()`**
- **Definition**: Unstacks the Series from rows to columns.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
  unstacked_df = df.stack().unstack()
  ```

### **38. `DataFrame.query()`**
- **Definition**: Queries the DataFrame using a query string.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
  queried_df = df.query('A > 1')
  ```

### **39. `DataFrame.sample()`**
- **Definition**: Randomly samples items from an axis of the DataFrame.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2, 3, 4]})
  sample_df = df.sample(n=2)
  ```

### **40. `DataFrame.replace()`**
- **Definition**: Replaces values given in a DataFrame with new values.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2, 1, 3]})
  replaced_df = df.replace({1: 10})
  ```

### **41. `DataFrame.query()`**
- **Definition**: Queries the DataFrame using a query string.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
  result = df.query('A > 1')
  ```

### **42. `DataFrame.duplicated()`**
- **Definition**: Returns a boolean Series denoting duplicate rows.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2, 1, 3]})
  duplicated_rows = df.duplicated()
  ```

### **43. `DataFrame.drop_duplicates()`**
- **Definition**: Removes duplicate rows from the DataFrame.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2, 1, 3]})
  unique_df = df.drop_duplicates()
  ```

### **44. `DataFrame.set_index()`**
- **Definition**: Sets the DataFrame index using existing columns.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
  indexed_df = df.set_index('A')
  ```

### **45. `DataFrame.reset_index()`**
- **Definition**: Resets the index of the DataFrame.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2]}, index=['a', 'b'])
  reset_df = df.reset_index()
  ```

### **46. `DataFrame.agg()`**
- **Definition**: Aggregates the DataFrame using one or more operations.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
  agg_df = df.agg({'A': 'sum', 'B': 'mean'})
  ```

### **47. `DataFrame.transform()`**
- **Definition**: Applies a function to each group of the DataFrame.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2, 1, 2], 'B': [3, 4, 5, 6]})
  transformed_df = df.groupby('A').transform(lambda x: x - x.mean())
  ```

### **48. `DataFrame.to_dict()`**
- **Definition**: Converts the DataFrame to a dictionary.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
  dict_representation = df.to_dict()
  ```

### **49. `DataFrame.to_numpy()`**
- **Definition**: Converts the DataFrame to a NumPy array.
- **Code**: 
  ```python
  df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
  numpy_array = df.to_numpy()
  ```

### **50. `DataFrame.to_sql()`**
- **Definition**: Writes the DataFrame to a SQL database.
- **Code**: 
  ```python
  from sqlalchemy import create_engine
  engine = create_engine('sqlite:///:memory:')
  df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
  df.to_sql('table_name', con=engine)
  ```

### **51. `pd.concat()`**
- **Definition**: Concatenates DataFrames along a particular axis.
- **Code**: 
  ```python
  df1 = pd.DataFrame({'A': [1, 2]})
  df2 = pd.DataFrame({'A': [3, 4]})
  concatenated_df = pd.concat([df1, df2])
  ```

These commands and functions cover a broad range of data manipulation and analysis tasks in Pandas, essential for Data Science.

## Matplotlib

Here are 21 important commands, methods, and functions in Matplotlib, along with examples where relevant:

### **1. `import matplotlib.pyplot as plt`**
- **Definition**: Imports the `pyplot` module from Matplotlib, commonly used for creating plots.
- **Code**: 
  ```python
  import matplotlib.pyplot as plt
  ```

### **2. `plt.plot()`**
- **Definition**: Plots lines and/or markers to the axes.
- **Code**: 
  ```python
  plt.plot([1, 2, 3], [4, 5, 6])
  plt.show()
  ```

### **3. `plt.scatter()`**
- **Definition**: Creates a scatter plot of y vs. x.
- **Code**: 
  ```python
  plt.scatter([1, 2, 3], [4, 5, 6])
  plt.show()
  ```

### **4. `plt.bar()`**
- **Definition**: Creates a bar plot.
- **Code**: 
  ```python
  plt.bar(['A', 'B', 'C'], [4, 5, 6])
  plt.show()
  ```

### **5. `plt.hist()`**
- **Definition**: Creates a histogram of the data.
- **Code**: 
  ```python
  plt.hist([1, 2, 2, 3, 3, 3, 4, 5])
  plt.show()
  ```

### **6. `plt.boxplot()`**
- **Definition**: Creates a box plot to visualize the distribution of data.
- **Code**: 
  ```python
  plt.boxplot([1, 2, 2, 3, 3, 3, 4, 5])
  plt.show()
  ```

### **7. `plt.pie()`**
- **Definition**: Creates a pie chart.
- **Code**: 
  ```python
  plt.pie([10, 20, 30], labels=['A', 'B', 'C'])
  plt.show()
  ```

### **8. `plt.title()`**
- **Definition**: Adds a title to the plot.
- **Code**: 
  ```python
  plt.plot([1, 2, 3], [4, 5, 6])
  plt.title('Sample Plot')
  plt.show()
  ```

### **9. `plt.xlabel()`**
- **Definition**: Adds a label to the x-axis.
- **Code**: 
  ```python
  plt.plot([1, 2, 3], [4, 5, 6])
  plt.xlabel('X-axis Label')
  plt.show()
  ```

### **10. `plt.ylabel()`**
- **Definition**: Adds a label to the y-axis.
- **Code**: 
  ```python
  plt.plot([1, 2, 3], [4, 5, 6])
  plt.ylabel('Y-axis Label')
  plt.show()
  ```

### **11. `plt.legend()`**
- **Definition**: Adds a legend to the plot.
- **Code**: 
  ```python
  plt.plot([1, 2, 3], [4, 5, 6], label='Line')
  plt.legend()
  plt.show()
  ```

### **12. `plt.grid()`**
- **Definition**: Adds a grid to the plot.
- **Code**: 
  ```python
  plt.plot([1, 2, 3], [4, 5, 6])
  plt.grid(True)
  plt.show()
  ```

### **13. `plt.xticks()`**
- **Definition**: Sets the position and labels of the x-axis ticks.
- **Code**: 
  ```python
  plt.plot([1, 2, 3], [4, 5, 6])
  plt.xticks([1, 2, 3], ['A', 'B', 'C'])
  plt.show()
  ```

### **14. `plt.yticks()`**
- **Definition**: Sets the position and labels of the y-axis ticks.
- **Code**: 
  ```python
  plt.plot([1, 2, 3], [4, 5, 6])
  plt.yticks([4, 5, 6], ['Low', 'Medium', 'High'])
  plt.show()
  ```

### **15. `plt.xlim()`**
- **Definition**: Sets the limits for the x-axis.
- **Code**: 
  ```python
  plt.plot([1, 2, 3], [4, 5, 6])
  plt.xlim(0, 4)
  plt.show()
  ```

### **16. `plt.ylim()`**
- **Definition**: Sets the limits for the y-axis.
- **Code**: 
  ```python
  plt.plot([1, 2, 3], [4, 5, 6])
  plt.ylim(3, 7)
  plt.show()
  ```

### **17. `plt.subplots()`**
- **Definition**: Creates a figure and a set of subplots.
- **Code**: 
  ```python
  fig, ax = plt.subplots()
  ax.plot([1, 2, 3], [4, 5, 6])
  plt.show()
  ```

### **18. `plt.subplot()`**
- **Definition**: Creates a single subplot in a grid of subplots.
- **Code**: 
  ```python
  plt.subplot(2, 1, 1)
  plt.plot([1, 2, 3], [4, 5, 6])
  plt.subplot(2, 1, 2)
  plt.bar([1, 2, 3], [4, 5, 6])
  plt.show()
  ```

### **19. `plt.savefig()`**
- **Definition**: Saves the current figure to a file.
- **Code**: 
  ```python
  plt.plot([1, 2, 3], [4, 5, 6])
  plt.savefig('plot.png')
  ```

### **20. `plt.tight_layout()`**
- **Definition**: Adjusts subplots to fit into the figure area.
- **Code**: 
  ```python
  fig, ax = plt.subplots(2, 2)
  ax[0, 0].plot([1, 2, 3], [4, 5, 6])
  ax[0, 1].scatter([1, 2, 3], [4, 5, 6])
  ax[1, 0].bar([1, 2, 3], [4, 5, 6])
  ax[1, 1].hist([1, 2, 2, 3, 3, 3])
  plt.tight_layout()
  plt.show()
  ```

### **21. `plt.colorbar()`**
- **Definition**: Adds a colorbar to the plot.
- **Code**: 
  ```python
  import numpy as np
  data = np.random.rand(10, 10)
  plt.imshow(data, cmap='viridis')
  plt.colorbar()
  plt.show()
  ```

These commands and functions cover a wide range of plotting capabilities in Matplotlib, useful for data visualization in Data Science.

## Seaborn

Here are 21 important commands, methods, and functions in Seaborn, along with examples where relevant:

### **1. `import seaborn as sns`**
- **Definition**: Imports the Seaborn library, used for statistical data visualization.
- **Code**: 
  ```python
  import seaborn as sns
  ```

### **2. `sns.set()`**
- **Definition**: Sets the aesthetic style of the plots.
- **Code**: 
  ```python
  sns.set(style='whitegrid')
  ```

### **3. `sns.plotting_context()`**
- **Definition**: Sets the context of the plot for different environments (e.g., paper, notebook).
- **Code**: 
  ```python
  sns.set_context('notebook')
  ```

### **4. `sns.load_dataset()`**
- **Definition**: Loads a built-in dataset from Seaborn.
- **Code**: 
  ```python
  df = sns.load_dataset('iris')
  ```

### **5. `sns.histplot()`**
- **Definition**: Plots a histogram.
- **Code**: 
  ```python
  sns.histplot(df['sepal_length'])
  ```

### **6. `sns.kdeplot()`**
- **Definition**: Plots a Kernel Density Estimate (KDE) of the data.
- **Code**: 
  ```python
  sns.kdeplot(df['sepal_length'])
  ```

### **7. `sns.scatterplot()`**
- **Definition**: Plots a scatter plot.
- **Code**: 
  ```python
  sns.scatterplot(x='sepal_length', y='sepal_width', data=df)
  ```

### **8. `sns.lineplot()`**
- **Definition**: Plots a line plot.
- **Code**: 
  ```python
  sns.lineplot(x='sepal_length', y='sepal_width', data=df)
  ```

### **9. `sns.barplot()`**
- **Definition**: Creates a bar plot with mean values and confidence intervals.
- **Code**: 
  ```python
  sns.barplot(x='species', y='sepal_length', data=df)
  ```

### **10. `sns.boxplot()`**
- **Definition**: Creates a box plot to visualize the distribution of data.
- **Code**: 
  ```python
  sns.boxplot(x='species', y='sepal_length', data=df)
  ```

### **11. `sns.violinplot()`**
- **Definition**: Creates a violin plot to show the distribution of the data.
- **Code**: 
  ```python
  sns.violinplot(x='species', y='sepal_length', data=df)
  ```

### **12. `sns.heatmap()`**
- **Definition**: Creates a heatmap of the data.
- **Code**: 
  ```python
  corr = df.corr()
  sns.heatmap(corr, annot=True)
  ```

### **13. `sns.pairplot()`**
- **Definition**: Creates a matrix of scatter plots for each pair of features.
- **Code**: 
  ```python
  sns.pairplot(df, hue='species')
  ```

### **14. `sns.jointplot()`**
- **Definition**: Creates a joint plot with a scatter plot and marginal histograms.
- **Code**: 
  ```python
  sns.jointplot(x='sepal_length', y='sepal_width', data=df)
  ```

### **15. `sns.regplot()`**
- **Definition**: Creates a scatter plot with a linear regression line.
- **Code**: 
  ```python
  sns.regplot(x='sepal_length', y='sepal_width', data=df)
  ```

### **16. `sns.catplot()`**
- **Definition**: Creates a categorical plot, useful for visualizing categorical data.
- **Code**: 
  ```python
  sns.catplot(x='species', y='sepal_length', data=df, kind='bar')
  ```

### **17. `sns.lmplot()`**
- **Definition**: Creates a linear model plot with regression line and confidence intervals.
- **Code**: 
  ```python
  sns.lmplot(x='sepal_length', y='sepal_width', data=df, hue='species')
  ```

### **18. `sns.pointplot()`**
- **Definition**: Creates a point plot with points and confidence intervals.
- **Code**: 
  ```python
  sns.pointplot(x='species', y='sepal_length', data=df)
  ```

### **19. `sns.stripplot()`**
- **Definition**: Creates a strip plot with jittered points.
- **Code**: 
  ```python
  sns.stripplot(x='species', y='sepal_length', data=df, jitter=True)
  ```

### **20. `sns.swarmplot()`**
- **Definition**: Creates a swarm plot with non-overlapping points.
- **Code**: 
  ```python
  sns.swarmplot(x='species', y='sepal_length', data=df)
  ```

### **21. `sns.set_palette()`**
- **Definition**: Sets the color palette for the plots.
- **Code**: 
  ```python
  sns.set_palette('husl')
  ```

These commands and functions cover a wide range of visualization capabilities in Seaborn, useful for exploratory data analysis and presenting insights in Data Science.

## Sciekit-Learn

Here are 31 important commands, methods, and functions in scikit-learn, along with examples where relevant:

### **1. `import sklearn`**
- **Definition**: Imports the scikit-learn library.
- **Code**: 
  ```python
  import sklearn
  ```

### **2. `from sklearn.model_selection import train_test_split`**
- **Definition**: Splits the dataset into training and testing sets.
- **Code**: 
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  ```

### **3. `from sklearn.linear_model import LinearRegression`**
- **Definition**: Imports the Linear Regression model.
- **Code**: 
  ```python
  from sklearn.linear_model import LinearRegression
  model = LinearRegression()
  ```

### **4. `from sklearn.ensemble import RandomForestClassifier`**
- **Definition**: Imports the Random Forest Classifier model.
- **Code**: 
  ```python
  from sklearn.ensemble import RandomForestClassifier
  model = RandomForestClassifier()
  ```

### **5. `from sklearn.svm import SVC`**
- **Definition**: Imports the Support Vector Classification model.
- **Code**: 
  ```python
  from sklearn.svm import SVC
  model = SVC()
  ```

### **6. `from sklearn.neighbors import KNeighborsClassifier`**
- **Definition**: Imports the k-Nearest Neighbors Classifier model.
- **Code**: 
  ```python
  from sklearn.neighbors import KNeighborsClassifier
  model = KNeighborsClassifier()
  ```

### **7. `from sklearn.metrics import accuracy_score`**
- **Definition**: Computes the accuracy of a classification model.
- **Code**: 
  ```python
  from sklearn.metrics import accuracy_score
  accuracy = accuracy_score(y_test, y_pred)
  ```

### **8. `from sklearn.metrics import confusion_matrix`**
- **Definition**: Computes the confusion matrix.
- **Code**: 
  ```python
  from sklearn.metrics import confusion_matrix
  cm = confusion_matrix(y_test, y_pred)
  ```

### **9. `from sklearn.metrics import classification_report`**
- **Definition**: Generates a classification report with precision, recall, and F1-score.
- **Code**: 
  ```python
  from sklearn.metrics import classification_report
  report = classification_report(y_test, y_pred)
  ```

### **10. `from sklearn.metrics import mean_squared_error`**
- **Definition**: Computes the mean squared error for regression models.
- **Code**: 
  ```python
  from sklearn.metrics import mean_squared_error
  mse = mean_squared_error(y_test, y_pred)
  ```

### **11. `from sklearn.preprocessing import StandardScaler`**
- **Definition**: Standardizes features by removing the mean and scaling to unit variance.
- **Code**: 
  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  ```

### **12. `from sklearn.preprocessing import MinMaxScaler`**
- **Definition**: Scales features to a given range, usually [0, 1].
- **Code**: 
  ```python
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  X_scaled = scaler.fit_transform(X)
  ```

### **13. `from sklearn.decomposition import PCA`**
- **Definition**: Performs Principal Component Analysis for dimensionality reduction.
- **Code**: 
  ```python
  from sklearn.decomposition import PCA
  pca = PCA(n_components=2)
  X_reduced = pca.fit_transform(X)
  ```

### **14. `from sklearn.feature_selection import SelectKBest`**
- **Definition**: Selects the top k features based on a score function.
- **Code**: 
  ```python
  from sklearn.feature_selection import SelectKBest, f_classif
  selector = SelectKBest(score_func=f_classif, k=3)
  X_new = selector.fit_transform(X, y)
  ```

### **15. `from sklearn.pipeline import Pipeline`**
- **Definition**: Constructs a pipeline to streamline data transformations and model fitting.
- **Code**: 
  ```python
  from sklearn.pipeline import Pipeline
  pipeline = Pipeline([
      ('scaler', StandardScaler()),
      ('model', RandomForestClassifier())
  ])
  pipeline.fit(X_train, y_train)
  ```

### **16. `from sklearn.model_selection import GridSearchCV`**
- **Definition**: Performs an exhaustive search over specified parameter values for an estimator.
- **Code**: 
  ```python
  from sklearn.model_selection import GridSearchCV
  param_grid = {'n_estimators': [10, 50, 100]}
  grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
  grid_search.fit(X_train, y_train)
  ```

### **17. `from sklearn.model_selection import RandomizedSearchCV`**
- **Definition**: Performs a randomized search over specified parameter values for an estimator.
- **Code**: 
  ```python
  from sklearn.model_selection import RandomizedSearchCV
  param_dist = {'n_estimators': [10, 50, 100]}
  random_search = RandomizedSearchCV(RandomForestClassifier(), param_dist, n_iter=10, cv=5)
  random_search.fit(X_train, y_train)
  ```

### **18. `from sklearn.metrics import roc_auc_score`**
- **Definition**: Computes the area under the Receiver Operating Characteristic (ROC) curve.
- **Code**: 
  ```python
  from sklearn.metrics import roc_auc_score
  auc = roc_auc_score(y_test, y_prob)
  ```

### **19. `from sklearn.metrics import roc_curve`**
- **Definition**: Computes the Receiver Operating Characteristic (ROC) curve.
- **Code**: 
  ```python
  from sklearn.metrics import roc_curve
  fpr, tpr, thresholds = roc_curve(y_test, y_prob)
  ```

### **20. `from sklearn.metrics import precision_score`**
- **Definition**: Computes the precision of a classification model.
- **Code**: 
  ```python
  from sklearn.metrics import precision_score
  precision = precision_score(y_test, y_pred)
  ```

### **21. `from sklearn.metrics import recall_score`**
- **Definition**: Computes the recall of a classification model.
- **Code**: 
  ```python
  from sklearn.metrics import recall_score
  recall = recall_score(y_test, y_pred)
  ```

### **22. `from sklearn.metrics import f1_score`**
- **Definition**: Computes the F1 score, which is the harmonic mean of precision and recall.
- **Code**: 
  ```python
  from sklearn.metrics import f1_score
  f1 = f1_score(y_test, y_pred)
  ```

### **23. `from sklearn.preprocessing import LabelEncoder`**
- **Definition**: Encodes categorical labels as numbers.
- **Code**: 
  ```python
  from sklearn.preprocessing import LabelEncoder
  encoder = LabelEncoder()
  y_encoded = encoder.fit_transform(y)
  ```

### **24. `from sklearn.preprocessing import OneHotEncoder`**
- **Definition**: Converts categorical values into a one-hot encoded format.
- **Code**: 
  ```python
  from sklearn.preprocessing import OneHotEncoder
  encoder = OneHotEncoder()
  X_encoded = encoder.fit_transform(X)
  ```

### **25. `from sklearn.metrics import mean_absolute_error`**
- **Definition**: Computes the mean absolute error for regression models.
- **Code**: 
  ```python
  from sklearn.metrics import mean_absolute_error
  mae = mean_absolute_error(y_test, y_pred)
  ```

### **26. `from sklearn.metrics import mean_absolute_error`**
- **Definition**: Computes the mean absolute error for regression models.
- **Code**: 
  ```python
  from sklearn.metrics import mean_absolute_error
  mae = mean_absolute_error(y_test, y_pred)
  ```

### **27. `from sklearn.metrics import mean_squared_error`**
- **Definition**: Computes the mean squared error for regression models.
- **Code**: 
  ```python
  from sklearn.metrics import mean_squared_error
  mse = mean_squared_error(y_test, y_pred)
  ```

### **28. `from sklearn.metrics import r2_score`**
- **Definition**: Computes the coefficient of determination (R²) for regression models.
- **Code**: 
  ```python
  from sklearn.metrics import r2_score
  r2 = r2_score(y_test, y_pred)
  ```

### **29. `from sklearn.ensemble import GradientBoostingClassifier`**
- **Definition**: Imports the Gradient Boosting Classifier model.
- **Code**: 
  ```python
  from sklearn.ensemble import GradientBoostingClassifier
  model = GradientBoostingClassifier()
  ```

### **30. `from sklearn.ensemble import AdaBoostClassifier`**
- **Definition**: Imports the AdaBoost Classifier model.
- **Code**: 
  ```python
  from sklearn.ensemble import AdaBoostClassifier
  model = AdaBoostClassifier()
  ```

### **31. `from sklearn.decomposition import TruncatedSVD`**
- **Definition**: Performs Truncated Singular Value Decomposition for dimensional

## Pytorch

Here are 21 important commands, methods, and functions in PyTorch, along with examples where relevant:

### **1. `import torch`**
- **Definition**: Imports the core PyTorch library.
- **Code**: 
  ```python
  import torch
  ```

### **2. `torch.tensor()`**
- **Definition**: Creates a tensor from a list or array.
- **Code**: 
  ```python
  tensor = torch.tensor([1.0, 2.0, 3.0])
  ```

### **3. `torch.zeros()`**
- **Definition**: Creates a tensor filled with zeros.
- **Code**: 
  ```python
  zeros = torch.zeros((2, 3))
  ```

### **4. `torch.ones()`**
- **Definition**: Creates a tensor filled with ones.
- **Code**: 
  ```python
  ones = torch.ones((2, 3))
  ```

### **5. `torch.eye()`**
- **Definition**: Creates a 2D tensor with ones on the diagonal and zeros elsewhere (identity matrix).
- **Code**: 
  ```python
  identity = torch.eye(3)
  ```

### **6. `torch.randn()`**
- **Definition**: Creates a tensor filled with random numbers from a normal distribution.
- **Code**: 
  ```python
  random_tensor = torch.randn((2, 3))
  ```

### **7. `torch.add()`**
- **Definition**: Adds two tensors.
- **Code**: 
  ```python
  a = torch.tensor([1, 2])
  b = torch.tensor([3, 4])
  result = torch.add(a, b)
  ```

### **8. `torch.matmul()`**
- **Definition**: Performs matrix multiplication between two tensors.
- **Code**: 
  ```python
  a = torch.randn((2, 3))
  b = torch.randn((3, 2))
  result = torch.matmul(a, b)
  ```

### **9. `torch.nn.Module`**
- **Definition**: Base class for all neural network modules.
- **Code**: 
  ```python
  import torch.nn as nn
  class MyModel(nn.Module):
      def __init__(self):
          super(MyModel, self).__init__()
          self.fc = nn.Linear(10, 1)
      
      def forward(self, x):
          return self.fc(x)
  ```

### **10. `torch.optim.SGD`**
- **Definition**: Stochastic Gradient Descent optimizer.
- **Code**: 
  ```python
  import torch.optim as optim
  model = MyModel()
  optimizer = optim.SGD(model.parameters(), lr=0.01)
  ```

### **11. `torch.nn.CrossEntropyLoss()`**
- **Definition**: Applies the Cross Entropy Loss function.
- **Code**: 
  ```python
  criterion = nn.CrossEntropyLoss()
  ```

### **12. `torch.nn.ReLU()`**
- **Definition**: Applies the ReLU (Rectified Linear Unit) activation function.
- **Code**: 
  ```python
  relu = nn.ReLU()
  ```

### **13. `torch.autograd.grad()`**
- **Definition**: Computes the gradient of tensors with respect to some scalar value.
- **Code**: 
  ```python
  x = torch.tensor([1.0, 2.0], requires_grad=True)
  y = x**2
  grad = torch.autograd.grad(outputs=y.sum(), inputs=x)
  ```

### **14. `torch.nn.Sequential`**
- **Definition**: A sequential container for stacking multiple layers.
- **Code**: 
  ```python
  model = nn.Sequential(
      nn.Linear(10, 50),
      nn.ReLU(),
      nn.Linear(50, 1)
  )
  ```

### **15. `torch.save()`**
- **Definition**: Saves a tensor or model to a file.
- **Code**: 
  ```python
  torch.save(model.state_dict(), 'model.pth')
  ```

### **16. `torch.load()`**
- **Definition**: Loads a tensor or model from a file.
- **Code**: 
  ```python
  model.load_state_dict(torch.load('model.pth'))
  ```

### **17. `torch.no_grad()`**
- **Definition**: Context manager that disables gradient calculation.
- **Code**: 
  ```python
  with torch.no_grad():
      output = model(input)
  ```

### **18. `torch.device()`**
- **Definition**: Specifies the device (CPU or GPU) for tensor operations.
- **Code**: 
  ```python
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  tensor = tensor.to(device)
  ```

### **19. `torch.cuda.is_available()`**
- **Definition**: Checks if a GPU is available.
- **Code**: 
  ```python
  if torch.cuda.is_available():
      print('CUDA is available!')
  ```

### **20. `torch.nn.functional`**
- **Definition**: Contains functions that operate on tensors and are used to build layers.
- **Code**: 
  ```python
  import torch.nn.functional as F
  output = F.relu(input)
  ```

### **21. `torch.utils.data.DataLoader`**
- **Definition**: Provides an iterable over the dataset with batching, shuffling, and multi-threading.
- **Code**: 
  ```python
  from torch.utils.data import DataLoader, TensorDataset
  dataset = TensorDataset(X, y)
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
  ```

These commands and methods cover a range of functionalities for creating, manipulating, and training models using PyTorch.

## Keras

Here are 21 important commands, methods, and functions in Keras, along with examples where relevant:

### **1. `import keras`**
- **Definition**: Imports the Keras library.
- **Code**: 
  ```python
  import keras
  ```

### **2. `from keras.models import Sequential`**
- **Definition**: Imports the Sequential model class for building a linear stack of layers.
- **Code**: 
  ```python
  from keras.models import Sequential
  model = Sequential()
  ```

### **3. `from keras.layers import Dense`**
- **Definition**: Imports the Dense layer, a fully connected layer in a neural network.
- **Code**: 
  ```python
  from keras.layers import Dense
  model.add(Dense(units=64, activation='relu', input_shape=(784,)))
  ```

### **4. `model.compile()`**
- **Definition**: Configures the model for training, specifying the optimizer, loss function, and metrics.
- **Code**: 
  ```python
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  ```

### **5. `model.fit()`**
- **Definition**: Trains the model on the provided data.
- **Code**: 
  ```python
  model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
  ```

### **6. `model.evaluate()`**
- **Definition**: Evaluates the model on test data.
- **Code**: 
  ```python
  loss, accuracy = model.evaluate(X_test, y_test)
  ```

### **7. `model.predict()`**
- **Definition**: Generates predictions from the model.
- **Code**: 
  ```python
  predictions = model.predict(X_test)
  ```

### **8. `from keras.layers import Dropout`**
- **Definition**: Imports the Dropout layer, used for regularization to prevent overfitting.
- **Code**: 
  ```python
  from keras.layers import Dropout
  model.add(Dropout(0.5))
  ```

### **9. `from keras.layers import Conv2D`**
- **Definition**: Imports the 2D convolutional layer for building Convolutional Neural Networks (CNNs).
- **Code**: 
  ```python
  from keras.layers import Conv2D
  model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
  ```

### **10. `from keras.layers import MaxPooling2D`**
- **Definition**: Imports the Max Pooling layer, used to downsample the spatial dimensions of the input.
- **Code**: 
  ```python
  from keras.layers import MaxPooling2D
  model.add(MaxPooling2D(pool_size=(2, 2)))
  ```

### **11. `from keras.layers import Flatten`**
- **Definition**: Flattens the input, used for transitioning from convolutional layers to fully connected layers.
- **Code**: 
  ```python
  from keras.layers import Flatten
  model.add(Flatten())
  ```

### **12. `from keras.layers import LSTM`**
- **Definition**: Imports the Long Short-Term Memory (LSTM) layer for building Recurrent Neural Networks (RNNs).
- **Code**: 
  ```python
  from keras.layers import LSTM
  model.add(LSTM(units=50, input_shape=(timesteps, features)))
  ```

### **13. `from keras.preprocessing.image import ImageDataGenerator`**
- **Definition**: Imports the ImageDataGenerator class for real-time data augmentation.
- **Code**: 
  ```python
  from keras.preprocessing.image import ImageDataGenerator
  datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=20)
  ```

### **14. `model.save()`**
- **Definition**: Saves the entire model (architecture, weights, and optimizer) to a file.
- **Code**: 
  ```python
  model.save('model.h5')
  ```

### **15. `from keras.models import load_model`**
- **Definition**: Loads a saved Keras model from a file.
- **Code**: 
  ```python
  from keras.models import load_model
  model = load_model('model.h5')
  ```

### **16. `from keras.callbacks import EarlyStopping`**
- **Definition**: Imports the EarlyStopping callback, which stops training when a monitored metric has stopped improving.
- **Code**: 
  ```python
  from keras.callbacks import EarlyStopping
  early_stopping = EarlyStopping(monitor='val_loss', patience=5)
  model.fit(X_train, y_train, epochs=50, callbacks=[early_stopping])
  ```

### **17. `from keras.callbacks import ModelCheckpoint`**
- **Definition**: Imports the ModelCheckpoint callback, which saves the model at regular intervals.
- **Code**: 
  ```python
  from keras.callbacks import ModelCheckpoint
  checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
  model.fit(X_train, y_train, epochs=50, callbacks=[checkpoint])
  ```

### **18. `from keras.layers import BatchNormalization`**
- **Definition**: Imports the BatchNormalization layer for normalizing activations in the network.
- **Code**: 
  ```python
  from keras.layers import BatchNormalization
  model.add(BatchNormalization())
  ```

### **19. `from keras.layers import Activation`**
- **Definition**: Imports the Activation layer for applying activation functions.
- **Code**: 
  ```python
  from keras.layers import Activation
  model.add(Dense(units=64))
  model.add(Activation('relu'))
  ```

### **20. `keras.utils.to_categorical()`**
- **Definition**: Converts a class vector (integers) to binary class matrix (one-hot encoding).
- **Code**: 
  ```python
  from keras.utils import to_categorical
  y_train_one_hot = to_categorical(y_train)
  ```

### **21. `from keras.optimizers import Adam`**
- **Definition**: Imports the Adam optimizer, which is widely used for training neural networks.
- **Code**: 
  ```python
  from keras.optimizers import Adam
  optimizer = Adam(learning_rate=0.001)
  model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  ```

These commands and methods are essential for building, training, and evaluating models using Keras, which is now integrated into TensorFlow as `tf.keras`.

## TensorFlow

Here are 21 important commands, methods, and functions in TensorFlow, along with examples where relevant:

### **1. `import tensorflow as tf`**
- **Definition**: Imports the TensorFlow library.
- **Code**: 
  ```python
  import tensorflow as tf
  ```

### **2. `tf.constant()`**
- **Definition**: Creates a constant tensor with a fixed value.
- **Code**: 
  ```python
  constant_tensor = tf.constant([1.0, 2.0, 3.0])
  ```

### **3. `tf.Variable()`**
- **Definition**: Creates a variable tensor that can be updated.
- **Code**: 
  ```python
  variable_tensor = tf.Variable([1.0, 2.0, 3.0])
  ```

### **4. `tf.placeholder()`** (Deprecated)
- **Definition**: Defines a placeholder for feeding data into the computation graph (replaced by `tf.function`).
- **Code**: 
  ```python
  # Note: This is deprecated in TensorFlow 2.x
  placeholder = tf.placeholder(tf.float32, shape=[None, 3])
  ```

### **5. `tf.function()`**
- **Definition**: Converts a Python function into a TensorFlow graph function for optimization.
- **Code**: 
  ```python
  @tf.function
  def add(a, b):
      return a + b
  ```

### **6. `tf.keras.models.Sequential`**
- **Definition**: Imports the Sequential model class for building a linear stack of layers.
- **Code**: 
  ```python
  from tensorflow.keras.models import Sequential
  model = Sequential()
  ```

### **7. `tf.keras.layers.Dense`**
- **Definition**: Imports the Dense layer, a fully connected layer in a neural network.
- **Code**: 
  ```python
  from tensorflow.keras.layers import Dense
  model.add(Dense(units=64, activation='relu', input_shape=(784,)))
  ```

### **8. `model.compile()`**
- **Definition**: Configures the model for training, specifying the optimizer, loss function, and metrics.
- **Code**: 
  ```python
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  ```

### **9. `model.fit()`**
- **Definition**: Trains the model on the provided data.
- **Code**: 
  ```python
  model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
  ```

### **10. `model.evaluate()`**
- **Definition**: Evaluates the model on test data.
- **Code**: 
  ```python
  loss, accuracy = model.evaluate(X_test, y_test)
  ```

### **11. `model.predict()`**
- **Definition**: Generates predictions from the model.
- **Code**: 
  ```python
  predictions = model.predict(X_test)
  ```

### **12. `tf.data.Dataset`**
- **Definition**: Creates a dataset for efficient data loading and preprocessing.
- **Code**: 
  ```python
  dataset = tf.data.Dataset.from_tensor_slices((X, y))
  dataset = dataset.batch(32).shuffle(1000)
  ```

### **13. `tf.image.resize()`**
- **Definition**: Resizes images to a specified shape.
- **Code**: 
  ```python
  resized_image = tf.image.resize(image, [128, 128])
  ```

### **14. `tf.keras.callbacks.EarlyStopping`**
- **Definition**: Stops training when a monitored metric has stopped improving.
- **Code**: 
  ```python
  from tensorflow.keras.callbacks import EarlyStopping
  early_stopping = EarlyStopping(monitor='val_loss', patience=5)
  model.fit(X_train, y_train, epochs=50, callbacks=[early_stopping])
  ```

### **15. `tf.keras.callbacks.ModelCheckpoint`**
- **Definition**: Saves the model at regular intervals or when it improves.
- **Code**: 
  ```python
  from tensorflow.keras.callbacks import ModelCheckpoint
  checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
  model.fit(X_train, y_train, epochs=50, callbacks=[checkpoint])
  ```

### **16. `tf.keras.layers.Conv2D`**
- **Definition**: 2D convolutional layer for building Convolutional Neural Networks (CNNs).
- **Code**: 
  ```python
  from tensorflow.keras.layers import Conv2D
  model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
  ```

### **17. `tf.keras.layers.MaxPooling2D`**
- **Definition**: Max Pooling layer to downsample the spatial dimensions.
- **Code**: 
  ```python
  from tensorflow.keras.layers import MaxPooling2D
  model.add(MaxPooling2D(pool_size=(2, 2)))
  ```

### **18. `tf.keras.layers.Flatten`**
- **Definition**: Flattens the input, used for transitioning from convolutional layers to fully connected layers.
- **Code**: 
  ```python
  from tensorflow.keras.layers import Flatten
  model.add(Flatten())
  ```

### **19. `tf.keras.layers.LSTM`**
- **Definition**: Long Short-Term Memory layer for building Recurrent Neural Networks (RNNs).
- **Code**: 
  ```python
  from tensorflow.keras.layers import LSTM
  model.add(LSTM(units=50, input_shape=(timesteps, features)))
  ```

### **20. `tf.keras.models.load_model`**
- **Definition**: Loads a saved Keras model from a file.
- **Code**: 
  ```python
  from tensorflow.keras.models import load_model
  model = load_model('model.h5')
  ```

### **21. `tf.keras.utils.to_categorical`**
- **Definition**: Converts class vectors to binary class matrices (one-hot encoding).
- **Code**: 
  ```python
  from tensorflow.keras.utils import to_categorical
  y_train_one_hot = to_categorical(y_train)
  ```

These commands and methods are crucial for building, training, and evaluating machine learning models using TensorFlow, especially with its Keras API.

## Transformers

Here are 21 important commands, methods, and functions in Hugging Face's Transformers library, along with examples where relevant:

### **1. `from transformers import pipeline`**
- **Definition**: Imports the `pipeline` function to easily access various pre-built models.
- **Code**:
  ```python
  from transformers import pipeline
  classifier = pipeline('sentiment-analysis')
  ```

### **2. `from transformers import AutoTokenizer`**
- **Definition**: Imports the `AutoTokenizer` class to load tokenizers.
- **Code**:
  ```python
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
  ```

### **3. `from transformers import AutoModel`**
- **Definition**: Imports the `AutoModel` class to load pre-trained models.
- **Code**:
  ```python
  from transformers import AutoModel
  model = AutoModel.from_pretrained('bert-base-uncased')
  ```

### **4. `tokenizer.encode()`**
- **Definition**: Encodes text into token IDs.
- **Code**:
  ```python
  encoded_input = tokenizer.encode('Hello, world!', return_tensors='pt')
  ```

### **5. `tokenizer.decode()`**
- **Definition**: Decodes token IDs back into text.
- **Code**:
  ```python
  decoded_output = tokenizer.decode(encoded_input[0])
  ```

### **6. `model.forward()`**
- **Definition**: Performs a forward pass through the model to get predictions.
- **Code**:
  ```python
  outputs = model(encoded_input)
  ```

### **7. `from transformers import BertForSequenceClassification`**
- **Definition**: Imports a specific model class for sequence classification tasks.
- **Code**:
  ```python
  from transformers import BertForSequenceClassification
  model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
  ```

### **8. `from transformers import Trainer`**
- **Definition**: Imports the `Trainer` class for training and evaluating models.
- **Code**:
  ```python
  from transformers import Trainer, TrainingArguments
  training_args = TrainingArguments(output_dir='./results', num_train_epochs=1)
  trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
  trainer.train()
  ```

### **9. `from transformers import TrainingArguments`**
- **Definition**: Imports the `TrainingArguments` class to configure training.
- **Code**:
  ```python
  from transformers import TrainingArguments
  training_args = TrainingArguments(output_dir='./results', per_device_train_batch_size=8)
  ```

### **10. `from transformers import pipeline`**
- **Definition**: Imports the `pipeline` function for high-level model interfaces.
- **Code**:
  ```python
  from transformers import pipeline
  generator = pipeline('text-generation', model='gpt2')
  generated_text = generator('Hello, world!', max_length=50)
  ```

### **11. `from transformers import DistilBertTokenizer`**
- **Definition**: Imports the tokenizer for DistilBERT models.
- **Code**:
  ```python
  from transformers import DistilBertTokenizer
  tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
  ```

### **12. `from transformers import DistilBertForSequenceClassification`**
- **Definition**: Imports the model class for DistilBERT sequence classification.
- **Code**:
  ```python
  from transformers import DistilBertForSequenceClassification
  model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
  ```

### **13. `model.save_pretrained()`**
- **Definition**: Saves the model to a specified directory.
- **Code**:
  ```python
  model.save_pretrained('./my_model')
  ```

### **14. `tokenizer.save_pretrained()`**
- **Definition**: Saves the tokenizer to a specified directory.
- **Code**:
  ```python
  tokenizer.save_pretrained('./my_tokenizer')
  ```

### **15. `from transformers import T5Tokenizer`**
- **Definition**: Imports the tokenizer for T5 models.
- **Code**:
  ```python
  from transformers import T5Tokenizer
  tokenizer = T5Tokenizer.from_pretrained('t5-small')
  ```

### **16. `from transformers import T5ForConditionalGeneration`**
- **Definition**: Imports the T5 model class for conditional generation tasks.
- **Code**:
  ```python
  from transformers import T5ForConditionalGeneration
  model = T5ForConditionalGeneration.from_pretrained('t5-small')
  ```

### **17. `tokenizer.batch_encode_plus()`**
- **Definition**: Encodes multiple text sequences into token IDs.
- **Code**:
  ```python
  encoded_inputs = tokenizer.batch_encode_plus(['Hello', 'world'], padding=True, return_tensors='pt')
  ```

### **18. `model.generate()`**
- **Definition**: Generates text sequences from the model.
- **Code**:
  ```python
  generated_ids = model.generate(input_ids=encoded_input, max_length=50)
  generated_text = tokenizer.decode(generated_ids[0])
  ```

### **19. `from transformers import RobertaTokenizer`**
- **Definition**: Imports the tokenizer for RoBERTa models.
- **Code**:
  ```python
  from transformers import RobertaTokenizer
  tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
  ```

### **20. `from transformers import RobertaForSequenceClassification`**
- **Definition**: Imports the RoBERTa model class for sequence classification.
- **Code**:
  ```python
  from transformers import RobertaForSequenceClassification
  model = RobertaForSequenceClassification.from_pretrained('roberta-base')
  ```

### **21. `from transformers import AutoModelForTokenClassification`**
- **Definition**: Imports a model class for token classification tasks.
- **Code**:
  ```python
  from transformers import AutoModelForTokenClassification
  model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased')
  ```

These commands and methods are essential for working with Hugging Face Transformers, covering tasks from loading and using models to training and generating predictions.

## NLTK

Here are 21 important commands, methods, and functions in the NLTK (Natural Language Toolkit) Python library, along with example code where relevant:

### **1. `import nltk`**
- **Definition**: Imports the NLTK library.
- **Code**:
  ```python
  import nltk
  ```

### **2. `nltk.download('punkt')`**
- **Definition**: Downloads the Punkt tokenizer models, necessary for tokenization.
- **Code**:
  ```python
  nltk.download('punkt')
  ```

### **3. `nltk.tokenize.word_tokenize()`**
- **Definition**: Tokenizes text into individual words.
- **Code**:
  ```python
  from nltk.tokenize import word_tokenize
  text = "Hello, world!"
  tokens = word_tokenize(text)
  ```

### **4. `nltk.tokenize.sent_tokenize()`**
- **Definition**: Tokenizes text into sentences.
- **Code**:
  ```python
  from nltk.tokenize import sent_tokenize
  text = "Hello world. This is NLTK."
  sentences = sent_tokenize(text)
  ```

### **5. `nltk.corpus.stopwords.words()`**
- **Definition**: Returns a list of stopwords in a specified language.
- **Code**:
  ```python
  from nltk.corpus import stopwords
  nltk.download('stopwords')
  stop_words = stopwords.words('english')
  ```

### **6. `nltk.stem.WordNetLemmatizer()`**
- **Definition**: Lemmatizes words to their base form.
- **Code**:
  ```python
  from nltk.stem import WordNetLemmatizer
  lemmatizer = WordNetLemmatizer()
  lemmatized_word = lemmatizer.lemmatize('running')
  ```

### **7. `nltk.stem.PorterStemmer()`**
- **Definition**: Stems words to their root form using the Porter algorithm.
- **Code**:
  ```python
  from nltk.stem import PorterStemmer
  stemmer = PorterStemmer()
  stemmed_word = stemmer.stem('running')
  ```

### **8. `nltk.pos_tag()`**
- **Definition**: Tags words with their part-of-speech.
- **Code**:
  ```python
  from nltk import pos_tag
  tokens = word_tokenize("This is an example sentence.")
  tagged = pos_tag(tokens)
  ```

### **9. `nltk.chunk.ne_chunk()`**
- **Definition**: Performs named entity recognition (NER) on tagged words.
- **Code**:
  ```python
  from nltk import ne_chunk
  chunked = ne_chunk(tagged)
  ```

### **10. `nltk.FreqDist()`**
- **Definition**: Computes the frequency distribution of words.
- **Code**:
  ```python
  from nltk import FreqDist
  fdist = FreqDist(tokens)
  ```

### **11. `nltk.collocations.BigramCollocationFinder()`**
- **Definition**: Finds bigrams (pairs of consecutive words) in text.
- **Code**:
  ```python
  from nltk.collocations import BigramCollocationFinder
  finder = BigramCollocationFinder.from_words(tokens)
  bigrams = finder.ngram_fd
  ```

### **12. `nltk.metrics.ConfusionMatrix()`**
- **Definition**: Creates a confusion matrix for classification tasks.
- **Code**:
  ```python
  from nltk.metrics import ConfusionMatrix
  cm = ConfusionMatrix([1, 2, 3], [1, 2, 1])
  ```

### **13. `nltk.corpus.words()`**
- **Definition**: Returns a list of words in the corpus.
- **Code**:
  ```python
  from nltk.corpus import words
  word_list = words.words()
  ```

### **14. `nltk.tokenize.RegexpTokenizer()`**
- **Definition**: Tokenizes text based on a regular expression pattern.
- **Code**:
  ```python
  from nltk.tokenize import RegexpTokenizer
  tokenizer = RegexpTokenizer(r'\w+')
  tokens = tokenizer.tokenize("Hello, world!")
  ```

### **15. `nltk.corpus.reader.PlaintextCorpusReader()`**
- **Definition**: Reads plain text files as a corpus.
- **Code**:
  ```python
  from nltk.corpus.reader import PlaintextCorpusReader
  corpus = PlaintextCorpusReader('corpus_directory', '.*\.txt')
  ```

### **16. `nltk.corpus.reader.tagged.TaggedCorpusReader()`**
- **Definition**: Reads tagged corpora with POS tags.
- **Code**:
  ```python
  from nltk.corpus.reader import TaggedCorpusReader
  tagged_corpus = TaggedCorpusReader('corpus_directory', '.*\.txt', tagset='universal')
  ```

### **17. `nltk.parse.ChartParser()`**
- **Definition**: Parses sentences using a chart parser.
- **Code**:
  ```python
  from nltk.parse import ChartParser
  from nltk import CFG
  grammar = CFG.fromstring("""
    S -> NP VP
    NP -> Det N
    VP -> V NP
    Det -> 'a' | 'an' | 'the'
    N -> 'man' | 'dog' | 'cat'
    V -> 'saw' | 'ate' | 'walked'
  """)
  parser = ChartParser(grammar)
  ```

### **18. `nltk.tokenize.TreebankWordDetokenizer()`**
- **Definition**: Detokenizes text by joining tokens.
- **Code**:
  ```python
  from nltk.tokenize import TreebankWordDetokenizer
  detokenizer = TreebankWordDetokenizer()
  text = detokenizer.detokenize(tokens)
  ```

### **19. `nltk.corpus.reader.sents()`**
- **Definition**: Returns sentences from a corpus.
- **Code**:
  ```python
  from nltk.corpus import brown
  sentences = brown.sents()
  ```

### **20. `nltk.util.bigrams()`**
- **Definition**: Generates bigrams from a list of tokens.
- **Code**:
  ```python
  from nltk.util import bigrams
  bigrams_list = list(bigrams(tokens))
  ```

### **21. `nltk.classify.NaiveBayesClassifier()`**
- **Definition**: Creates a Naive Bayes classifier.
- **Code**:
  ```python
  from nltk.classify import NaiveBayesClassifier
  train_data = [({'feature': 'value'}, 'class')]
  classifier = NaiveBayesClassifier.train(train_data)
  ```

These commands and methods cover various aspects of text processing, tokenization, classification, and parsing in NLTK, providing essential tools for natural language processing tasks.

## Gensim


Here are Important commands, methods, and functions in the Gensim Python library, along with example code where relevant:

### **1. `import gensim`**
- **Definition**: Imports the Gensim library.
- **Code**:
  ```python
  import gensim
  ```

### **2. `from gensim.models import Word2Vec`**
- **Definition**: Imports the Word2Vec model class for word embeddings.
- **Code**:
  ```python
  from gensim.models import Word2Vec
  ```

### **3. `Word2Vec(sentences=..., vector_size=..., window=..., min_count=..., workers=...)`**
- **Definition**: Initializes and trains a Word2Vec model.
- **Code**:
  ```python
  model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
  ```

### **4. `model.save()`**
- **Definition**: Saves the trained model to a file.
- **Code**:
  ```python
  model.save('word2vec.model')
  ```

### **5. `Word2Vec.load()`**
- **Definition**: Loads a previously saved Word2Vec model.
- **Code**:
  ```python
  model = Word2Vec.load('word2vec.model')
  ```

### **6. `model.wv.most_similar()`**
- **Definition**: Finds the most similar words to a given word.
- **Code**:
  ```python
  similar_words = model.wv.most_similar('example')
  ```

### **7. `from gensim.models import FastText`**
- **Definition**: Imports the FastText model class.
- **Code**:
  ```python
  from gensim.models import FastText
  ```

### **8. `FastText(sentences=..., vector_size=..., window=..., min_count=..., workers=...)`**
- **Definition**: Initializes and trains a FastText model.
- **Code**:
  ```python
  model = FastText(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
  ```

### **9. `model.wv.get_vector()`**
- **Definition**: Retrieves the vector for a given word.
- **Code**:
  ```python
  vector = model.wv.get_vector('example')
  ```

### **10. `from gensim.models import LdaModel`**
- **Definition**: Imports the LDA model class for topic modeling.
- **Code**:
  ```python
  from gensim.models import LdaModel
  ```

### **11. `LdaModel(corpus=..., id2word=..., num_topics=..., passes=...)`**
- **Definition**: Initializes and trains an LDA model.
- **Code**:
  ```python
  lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=10, passes=15)
  ```

### **12. `lda_model.show_topics()`**
- **Definition**: Displays the topics discovered by the LDA model.
- **Code**:
  ```python
  topics = lda_model.show_topics(num_topics=5)
  ```

### **13. `from gensim.corpora import Dictionary`**
- **Definition**: Imports the Dictionary class for mapping words to IDs.
- **Code**:
  ```python
  from gensim.corpora import Dictionary
  ```

### **14. `Dictionary(documents=...)`**
- **Definition**: Initializes a dictionary from a collection of documents.
- **Code**:
  ```python
  dictionary = Dictionary(documents)
  ```

### **15. `dictionary.doc2bow()`**
- **Definition**: Converts a document into the bag-of-words representation.
- **Code**:
  ```python
  bow = dictionary.doc2bow('example text')
  ```

### **16. `from gensim.models import TfidfModel`**
- **Definition**: Imports the TF-IDF model class.
- **Code**:
  ```python
  from gensim.models import TfidfModel
  ```

### **17. `TfidfModel(corpus=...)`**
- **Definition**: Initializes and trains a TF-IDF model.
- **Code**:
  ```python
  tfidf_model = TfidfModel(corpus)
  ```

### **18. `tfidf_model[corpus]`**
- **Definition**: Applies the TF-IDF model to the corpus.
- **Code**:
  ```python
  tfidf_corpus = tfidf_model[corpus]
  ```

### **19. `from gensim.similarities import Similarity`**
- **Definition**: Imports the Similarity class for similarity queries.
- **Code**:
  ```python
  from gensim.similarities import Similarity
  ```

### **20. `Similarity(corpus=..., index=..., num_best=...)`**
- **Definition**: Initializes a similarity index for efficient similarity queries.
- **Code**:
  ```python
  index = Similarity('index', tfidf_model[corpus], num_features=len(dictionary))
  ```

### **21. `index[query_bow]`**
- **Definition**: Retrieves the most similar documents to a query.
- **Code**:
  ```python
  similarities = index[query_bow]
  ```

These commands and methods cover various aspects of text processing, topic modeling, and similarity calculations using Gensim, providing essential tools for natural language processing and machine learning tasks.

## Numpy, Pandas and matplotlib Questions

## NumPy Questions

### Question: Introduction to NumPy

What is NumPy and why is it important for scientific computing in Python? Can you explain the main features that make NumPy efficient?
Creating Arrays

### Question: 

How do you create a NumPy array from a list? Can you create a 3x3 matrix filled with random numbers and explain different methods to initialize arrays (e.g., zeros, ones, arange, linspace)?

### Question: Array Indexing and Slicing

How do you access elements and subarrays in a NumPy array? Can you provide examples of basic indexing, slicing, and boolean indexing?
Array Operations

### Question: 
What are some common array operations in NumPy? Can you demonstrate element-wise operations, matrix multiplication, and using universal functions (ufuncs)?

### Question: Broadcasting

What is broadcasting in NumPy, and how does it work? Can you provide an example where broadcasting is used to perform operations on arrays of different shapes?

## pandas Questions

### Question: Introduction to pandas

What is pandas and why is it used in data analysis? Can you explain the difference between Series and DataFrame objects?


### Question: Creating DataFrames

How do you create a DataFrame in pandas? Can you create a DataFrame from a dictionary of lists and explain different ways to initialize a DataFrame?

### Question: DataFrame Indexing and Selection

How do you select rows and columns in a DataFrame? Can you explain the difference between loc and iloc for indexing?

### Question: Handling Missing Data

How do you handle missing data in a pandas DataFrame? Can you demonstrate techniques for detecting, filling, and dropping missing values?

### Question: DataFrame Operations

What are some common DataFrame operations in pandas? Can you explain operations like merging, concatenation, group by, and pivot tables with examples?

## Matplotlib Questions

### Question: Introduction to Matplotlib

What is Matplotlib, and why is it used in data visualization? Can you explain the basic anatomy of a Matplotlib plot (e.g., figure, axes, labels)?

### Question: Creating Plots

How do you create a simple line plot in Matplotlib? Can you provide an example of plotting a sine wave and customizing the plot with titles, labels, and legends?

### Question: Subplots

What are subplots in Matplotlib, and how do you create them? Can you demonstrate how to create a 2x2 grid of subplots and share axes?

### Question: Customizing Plots

How do you customize the appearance of plots in Matplotlib? Can you explain how to change colors, line styles, markers, and use different colormaps?

### Question: Histograms and Bar Charts

How do you create histograms and bar charts in Matplotlib? Can you provide examples of visualizing the distribution of a dataset and comparing categorical data?

## Advanced Level Questions

### Question: NumPy Advanced Indexing

What are advanced indexing techniques in NumPy? Can you provide examples of using integer array indexing and multi-dimensional indexing?

### Question: Vectorization and Performance

How does vectorization improve performance in NumPy? Can you demonstrate the performance difference between vectorized operations and traditional loops?

### Question: Time Series Analysis with pandas

How do you handle time series data in pandas? Can you explain resampling, shifting, and rolling operations with examples?

### Question: Matplotlib Advanced Customization

How do you create complex plots in Matplotlib? Can you demonstrate advanced customization techniques such as annotations, secondary y-axes, and interactive plots?

### Question: Integration of pandas and Matplotlib

How do you integrate pandas and Matplotlib for data visualization? Can you provide examples of plotting data directly from a DataFrame and using pandas plotting capabilities?

### Question: Handling Large Datasets

How do you handle large datasets in NumPy and pandas? Can you discuss techniques like chunking, memory optimization, and using Dask for out-of-core computation?

These questions are designed to assess a candidate's comprehensive understanding of NumPy, pandas, and Matplotlib, covering foundational concepts, practical usage, and advanced techniques for scientific computing and data visualization in Python.