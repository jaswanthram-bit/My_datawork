#!/usr/bin/env python
# coding: utf-8

# # Problem 1 S&P 500

# In[1]:


import pandas as pd
file_path= 'C:/Users/jaswa/OneDrive/Desktop/733/Home Work 3/SP500_ticker.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')
df.head()


# A)Fit a PCA model to log returns (log return = log( Price [t+1]/Price [t]) derived from stock price data and complete the following tasks

# In[3]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px


# Import necessary libraries for data preprocessing and analysis
import pandas as pd
import numpy as np

# Import libraries for visualization
import matplotlib.pyplot as plt
import plotly.express as px

# Import libraries for machine learning
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# In[4]:


# Load the data with the first column as the index
raw_price_data = pd.read_csv("SP500_close_price_no_missing.csv", index_col=0)
ticker_info = pd.read_csv("SP500_ticker.csv", encoding="latin1")

# Display basic information about the closing prices dataframe
print("raw_price_data DataFrame:")
print(raw_price_data.info())
print("\nSummary Statistics:")
print(raw_price_data.describe())

# Display basic information about the closing prices dataframe
print("ticker_info DataFrame:")
print(ticker_info.info())
print("\nSummary Statistics:")
print(ticker_info.describe())


# #A 1.Derive log returns from the raw stock price dataset

# In[5]:


# Derive log returns using the specified formula
log_returns = np.log(raw_price_data.shift(-1) / raw_price_data).dropna()

# Standardize the log returns
scaler = StandardScaler()
standardized_returns = scaler.fit_transform(log_returns)

# Fit PCA model
pca = PCA()
pca.fit(standardized_returns)

# Print log_returns for verification
print(log_returns)


# #A 2. Plot a scree plot which shows the distribution of variance contained in subsequent principal components sorted by their eigenvalues.

# In[7]:


# Scree plot with logarithmic y-axis
eigenvalues = pca.explained_variance_
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue (log scale)')
plt.show()


# In[8]:


# Scree plot with annotations
eigenvalues = pca.explained_variance_
explained_variances = pca.explained_variance_ratio_

plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue (log scale)')

# Annotate the points with explained variance
for i, (eig, exp_var) in enumerate(zip(eigenvalues, explained_variances), start=1):
    plt.annotate(f'PC{i}\nExplained Variance: {exp_var:.2%}', (i, eig), textcoords="offset points", xytext=(0, 10), ha='center')

plt.show()


# #A 3. Create a second plot showing cumulative variance retained if top N components are kept after dimensionality reduction (i.e. the horizontal axis will show the number of components kept, the vertical axis will show the cumulative percentage of variance retained).

# In[10]:


# Cumulative variance plot
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_) * 100

plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.title('Cumulative Percentage of Variance Retained')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Percentage of Variance Retained')
plt.show()


# In[11]:


# Cumulative variance plot for top 10 components
top_n_components = 10
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_[:top_n_components]) * 100

plt.plot(range(1, top_n_components + 1), cumulative_variance_ratio, marker='o')
plt.title(f'Cumulative Percentage of Variance Retained for Top {top_n_components} Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Percentage of Variance Retained')
plt.show()


# #A 4.How many principal components must be retained in order to capture at least 80% of the total variance in data?

# In[13]:


# Scree plot with annotations and interpretations
eigenvalues = pca.explained_variance_
explained_variances = pca.explained_variance_ratio_
cumulative_variances = np.cumsum(explained_variances)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-', color='b', label='Eigenvalue')
plt.plot(range(1, len(eigenvalues) + 1), cumulative_variances, marker='o', linestyle='--', color='r', label='Cumulative Variance')

plt.yscale('log')  # Set y-axis to logarithmic scale
plt.title('Scree Plot with Cumulative Variance')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue (log scale) / Cumulative Variance')
plt.legend()

# Annotate the points with explained variance and interpretations
for i, (eig, exp_var, cum_var) in enumerate(zip(eigenvalues, explained_variances, cumulative_variances), start=1):
    plt.annotate(f'PC{i}\nExplained Variance: {exp_var:.2%}\nCumulative Variance: {cum_var:.2%}', 
                 (i, eig), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

# Highlight the point where cumulative variance reaches 80%
target_cumulative_variance = 0.80
highlight_index = np.argmax(cumulative_variances >= target_cumulative_variance) + 1
plt.axvline(x=highlight_index, color='g', linestyle='--', label=f'80% Cumulative Variance (PC {highlight_index})')
plt.legend()

# Add grid lines for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.show()

# Interpretation
print(f"Number of Principal Components to retain 80% of Cumulative Variance: PC{highlight_index}")


# B. Analysis of principal components and weights

# 1.Compute and plot the time series of the 1st principal component and observe temporal patterns. Identify the date with the lowest value for this component and conduct a quick research on the Internet to see if you can identify event(s) that might explain the observed behavior.

# In[14]:


import plotly.express as px

# Extract the 1st principal component time series
pc1_time_series = pca.transform(standardized_returns)[:, 0]

# Create a DataFrame for Plotly
plotly_data = pd.DataFrame({'Date': log_returns.index, '1st Principal Component': pc1_time_series})

# Plot the time series of the 1st principal component using Plotly
fig = px.line(plotly_data, x='Date', y='1st Principal Component', labels={'1st Principal Component': 'Principal Component Value'},
              title='Time Series of the 1st Principal Component')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Principal Component Value')

# Customize legend
fig.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=1.02,
    xanchor='right',
    x=1
))

# Show the plot
fig.show()

# Identify the date with the lowest value for the 1st principal component
min_pc1_date = log_returns.index[np.argmin(pc1_time_series)]
print(f"\nDate with the lowest value for the 1st principal component: {min_pc1_date}")


# 2.Extract the weights from the PCA model for 1st and 2nd principal components.

# In[16]:


# Extract weights for the 1st and 2nd principal components
weights_pc1 = pca.components_[0, :]
weights_pc2 = pca.components_[1, :]

# Create DataFrames to display the weights
weights_df = pd.DataFrame({'Ticker': log_returns.columns, 'PC1_Weights': weights_pc1, 'PC2_Weights': weights_pc2})

# Display the weights DataFrame
print("\nWeights for the 1st and 2nd Principal Components:")
print(weights_df)


# 3.Create a plot to show weights of the 1st principal component grouped by the industry sector (for example, you may draw a bar plot of mean weight per sector). Observe the distribution of weights (magnitudes, signs). Based on your observation, what kind of information do you think the 1st principal component might have captured?

# In[17]:


# Create a DataFrame with tickers and corresponding sectors
ticker_info_df = pd.read_csv("SP500_ticker.csv", encoding='latin1')

# Merge the ticker information with weights
weights_df = pd.DataFrame({'Ticker': log_returns.columns, 'PC1_Weights': weights_pc1})
weights_df = pd.merge(weights_df, ticker_info_df[['ticker', 'sector']], left_on='Ticker', right_on='ticker')

# Group by sector and calculate the mean weight for the 1st principal component
mean_weights_by_sector = weights_df.groupby('sector')['PC1_Weights'].mean().sort_values()

# Plot the bar plot
plt.figure(figsize=(12, 6))
mean_weights_by_sector.plot(kind='bar', color='blue')
plt.xlabel('Industry Sector')
plt.ylabel('Mean Weight for 1st PC')
plt.title('Mean Weights of the 1st Principal Component by Industry Sector')
plt.xticks(rotation=45, ha='right')
plt.show()

# Display the DataFrame with mean weights by sector
print("\nMean Weights of the 1st Principal Component by Industry Sector:")
print(mean_weights_by_sector)


# 4.Make a similar plot for the 2nd principal component. What kind of information do you think does this component reveal?

# In[18]:


# Create a DataFrame with tickers and corresponding sectors for the 2nd principal component
weights_df_pc2 = pd.DataFrame({'Ticker': log_returns.columns, 'PC2_Weights': weights_pc2})
weights_df_pc2 = pd.merge(weights_df_pc2, ticker_info_df[['ticker', 'sector']], left_on='Ticker', right_on='ticker')

# Group by sector and calculate the mean weight for the 2nd principal component
mean_weights_by_sector_pc2 = weights_df_pc2.groupby('sector')['PC2_Weights'].mean().sort_values()

# Plot the bar plot for the 2nd principal component
plt.figure(figsize=(12, 6))
mean_weights_by_sector_pc2.plot(kind='bar', color='green')
plt.xlabel('Industry Sector')
plt.ylabel('Mean Weight for 2nd PC')
plt.title('Mean Weights of the 2nd Principal Component by Industry Sector')
plt.xticks(rotation=45, ha='right')
plt.show()

# Display the DataFrame with mean weights by sector for the 2nd principal component
print("\nMean Weights of the 2nd Principal Component by Industry Sector:")
print(mean_weights_by_sector_pc2)


# 5.Suppose we wanted to construct a new stock index using one principal component to track the overall market tendencies. Which of the two components would you prefer to use for this purpose, the 1st or the 2nd? Why?

# In[20]:


# Get explained variances for each principal component
explained_variances = pca.explained_variance_ratio_

# Display the explained variances for the 1st and 2nd principal components
print("Explained Variances:")
print(f"1st Principal Component: {explained_variances[0]:.4f}")
print(f"2nd Principal Component: {explained_variances[1]:.4f}")

# Choose the component with higher explained variance for constructing the new stock index
if explained_variances[0] > explained_variances[1]:
    chosen_component = 1
else:
    chosen_component = 2

print(f"\nPreferred Component for New Stock Index: {chosen_component} (based on higher explained variance)")


# The qualities and information recorded by each component influence the choice of the first and second primary components for generating a new stock index.
# 
# 1st Principal Compnent (PC1): The first principal component represents the direction of largest variance in the data. It is frequently taken as an indicator of broader market movement or market tendencies. PC1 is expected to capture common factors affecting a wide range of stocks in the context of stock returns.
# 
# 2nd Principal Component (PC2): PC2 represents the highest variance direction orthogonal to PC1. It captures variability that PC1 does not explain. PC2 may capture more particular or idiosyncratic characteristics that affect various groups of stocks differently in the context of stock returns.
# 
# PC1 is often preferred for constructing a new stock index that tracks overall market tendencies and general market movements. PC1 is intended to capture the most significant and common patterns in data, making it an appropriate choice for portraying the whole market.
# 
# However, the proportion of variance explained by each component must be considered. If PC1 explains a much greater fraction of total variation than PC2, the case for choosing PC1 becomes stronger. Furthermore, the interpretation of the weights assigned to particular stocks in PC1 should be scrutinized to ensure that they are consistent with expectations.
# 
# In conclusion, if PC1 captures a significant share of overall market movement and is easily interpretable in the context of stock returns, it would be the best alternative for developing a new stock index to track market tendencies.

# # Problem 2 -BMI

# In[21]:


import pandas as pd
data=pd.read_csv("C:/Users/jaswa/OneDrive/Desktop/733/Home Work 3/BMI.csv")


# In[22]:


data.head()


# In[23]:


data.tail()


# In[24]:


data.shape


# In[25]:


data.info()


# In[26]:


data.describe()


# In[27]:


get_ipython().system('pip install pandas-profiling')


# In[28]:


get_ipython().system('pip install pandas ydata-profiling')


# In[29]:


import pandas as pd
from ydata_profiling import ProfileReport
df = pd.read_csv("C:/Users/jaswa/OneDrive/Desktop/733/Home Work 3/BMI.csv")
profile = ProfileReport(df)
profile


# In[30]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the data
data_path = "C:\\Users\\jaswa\\OneDrive\\Desktop\\733\\Home Work 3\\BMI.csv"
data = pd.read_csv(data_path)

# Separate features and target variable
X = data.drop('fatpctg', axis=1)
y = data['fatpctg']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Method 1: Feature Importance from Tree-based models (e.g., Random Forest)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Select features based on importance
sfm = SelectFromModel(rf_model, threshold=0.1)
sfm.fit(X_train, y_train)
selected_features_rf = X_train.columns[sfm.get_support()]

# Method 2: Recursive Feature Elimination (RFE)
lr_model = LinearRegression()
rfe = RFE(lr_model, n_features_to_select=5)
rfe.fit(X_train, y_train)
selected_features_rfe = X_train.columns[rfe.ranking_ == 1]

# Method 3: Mutual Information
mi_scores = mutual_info_regression(X_train, y_train)
selected_features_mi = X_train.columns[mi_scores > 0.1]

# Print the selected features from each method
print("Selected features using Random Forest:", selected_features_rf)
print("Selected features using RFE:", selected_features_rfe)
print("Selected features using Mutual Information:", selected_features_mi)

# Visualize feature importances from Random Forest
feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Features Importance (Random Forest)')
plt.show()


# A) Wrapper Method : Search for the best set of features using backward and forward stepwise regression

# In[32]:


import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
data_path = "C:\\Users\\jaswa\\OneDrive\\Desktop\\733\\Home Work 3\\BMI.csv"
data = pd.read_csv(data_path)

# Separate features and target variable
X = data.drop('fatpctg', axis=1)
y = data['fatpctg']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a constant to the features for the statsmodels regression
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Backward Stepwise Regression
def backward_stepwise_selection(X, y):
    features = X.columns.tolist()
    while len(features) > 0:
        X_temp = X[features]
        model = sm.OLS(y, X_temp).fit()
        max_p_value = max(model.pvalues)
        if max_p_value > 0.05:  # Adjust the significance level as needed
            excluded_feature = model.pvalues.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features

selected_features_backward = backward_stepwise_selection(X_train, y_train)
print("Selected features using Backward Stepwise Regression:", selected_features_backward)

# Forward Stepwise Regression
def forward_stepwise_selection(X, y):
    features = []
    while True:
        remaining_features = [feature for feature in X.columns if feature not in features]
        if len(remaining_features) == 0:
            break
        best_pvalue = np.inf
        best_feature = None
        for feature in remaining_features:
            X_temp = X[features + [feature]]
            model = sm.OLS(y, X_temp).fit()
            p_value = model.pvalues[-1]
            if p_value < best_pvalue:
                best_pvalue = p_value
                best_feature = feature
        if best_pvalue < 0.05:  # Adjust the significance level as needed
            features.append(best_feature)
        else:
            break
    return features

selected_features_forward = forward_stepwise_selection(X_train, y_train)
print("Selected features using Forward Stepwise Regression:", selected_features_forward)


# B) Filter Method: Output a ranking of features using correlation statistics (i.e. between any of the input variables and output)

# In[33]:


import pandas as pd

# Load the data
data_path = "C:\\Users\\jaswa\\OneDrive\\Desktop\\733\\Home Work 3\\BMI.csv"
data = pd.read_csv(data_path)

# Separate features and target variable
X = data.drop('fatpctg', axis=1)
y = data['fatpctg']

# Calculate correlation between each feature and the target variable
correlation_with_target = X.apply(lambda x: x.corr(y))

# Sort features based on absolute correlation values
feature_ranking = correlation_with_target.abs().sort_values(ascending=False)

# Print the feature rankings
print("Feature Rankings based on Correlation with Target:")
print(feature_ranking)


# C)Embedded Method
# 
# 1.Lasso Regression

# In[34]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Load the data
data_path = "C:\\Users\\jaswa\\OneDrive\\Desktop\\733\\Home Work 3\\BMI.csv"
data = pd.read_csv(data_path)

# Separate features and target variable
X = data.drop('fatpctg', axis=1)
y = data['fatpctg']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Lasso Regression model
lasso_model = Lasso(alpha=0.1)  # You can adjust the alpha parameter
lasso_model.fit(X_train_scaled, y_train)

# Get selected features based on non-zero coefficients
selected_features_lasso = X.columns[lasso_model.coef_ != 0]

# Print the selected features from Lasso Regression
print("Selected features using Lasso Regression:", selected_features_lasso)


# 2.Random Forest(Importance)

# In[35]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load the data
data_path = "C:\\Users\\jaswa\\OneDrive\\Desktop\\733\\Home Work 3\\BMI.csv"
data = pd.read_csv(data_path)

# Separate features and target variable
X = data.drop('fatpctg', axis=1)
y = data['fatpctg']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame with feature names and their importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort features based on importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the feature importance ranking
print("Feature Importance Ranking (Random Forest):")
print(feature_importance_df)


# In[ ]:




