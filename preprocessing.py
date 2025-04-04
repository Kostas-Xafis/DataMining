import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

bankrupt_data = None
verbose = False

training_data = pd.read_csv('./data/training_companydata.csv', na_values=['?'])

# ======== Data Preprocessing ========
def rm_dups(df):
    # ===== Duplicate Values =====
    # Drop duplicate rows
    return df.drop_duplicates()

def rm_high_nan(df, threshold=0.1):
    # ===== Missing Values =====
    # Drop Rows with High Missing Values
    null_perc = df.isnull().mean(axis=1)
    to_drop_missing = null_perc.where(null_perc > threshold).dropna().index
    print("Rows with high missing values: ", to_drop_missing.tolist()) if verbose else None
    return df.drop(index=to_drop_missing)

def rm_empty(df, threshold=0.1):
    # ===== Missing Values =====
    # Drop Columns with High Missing Values
    null_perc = df.isnull().mean()
    to_drop_missing = null_perc.where(null_perc > threshold).dropna().index
    print("Columns with high missing values: ", to_drop_missing.tolist()) if verbose else None
    return df.drop(columns=to_drop_missing)

def fill_nan(df, method='mean'):
    if method == 'mean':
        # Fill missing values with the mean of the column
        df.fillna(df.mean(), inplace=True)
    elif method == 'median':
        # Fill missing values with the median of the column
        df.fillna(df.median(), inplace=True)
    elif method == 'zero':
        # Fill missing values with 0
        df.fillna(0, inplace=True)

    return df

def binning(df, threshold=0.1, bins=3):
    # =====  Binning =====
    # Bin the columns with high amount of 0 values
    to_bin = df.columns[df.eq(0).mean().ge(threshold)]
    print("Columns to bin: ", to_bin) if verbose else None
    for column in to_bin:
        column_name = column + "_bin"
        df[column_name] = pd.cut(df[column], bins, labels=False)
        df = df.drop(columns=[column])
    return df

def high_zero(df, threshold=0.1):
    # ===== 0 Values =====
    # Drop columns with high amount of 0 values
    zero_perc = df.isin([0]).mean()
    to_drop_zero = zero_perc.where(zero_perc > threshold).dropna().index
    print("Columns with high 0 values: ", to_drop_zero.tolist()) if verbose else None
    return df.drop(columns=to_drop_zero)

def correlation(df, threshold=0.999, method='pearson'):
    # ===== Correlation =====
    # Drop the columns with high correlation
    correlation_matrix = df.corr(method).abs()
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))

    # Find two highly correlated columns
    high_corr_pairs = [(column, upper[column].idxmax()) for column in upper.columns if any(upper[column] >= threshold)]

    # Visualize the data of 2 correlated columns
    # for pair in high_corr_pairs:
    #     col1, col2 = pair
    #     print(f"Visualizing columns: {col1} and {col2}")

    #     plt.scatter(df[col1], df[col2])
    #     plt.xlabel(col1)
    #     plt.ylabel(col2)
    #     plt.title(f'Scatter plot of {col1} vs {col2}')
    #     plt.show()
    #     plt.pause(1)
    # else:
    #     print("No highly correlated columns found.")

    # Drop the columns that are found in pairs the least thus keeping the most important ones
    pair_counts = {}
    for cols in high_corr_pairs:
        for col in cols:
            if col in pair_counts:
                pair_counts[col] = pair_counts.get(col, 0) + 1
            else:
                pair_counts[col] = 1

    # Find columns that are part of multiple pairs
    columns_to_drop = [col for col, count in pair_counts.items() if count == 1]
    print("Columns found in multiple pairs: ", columns_to_drop) if verbose else None
    return df.drop(columns=columns_to_drop)

def deskew(df):
    # ===== Deskewing =====
    # Skewness of the data
    skewness = df.skew()
    skewness = skewness[abs(skewness) > 10]
    # print("Skewness of the data: ", skewness)

    e = 0.00001
    skewed_features = skewness.index
    for feature in skewed_features:
        if "bin" in feature:
            continue
        # Normalize the feature to be within the range (0, 1)
        min_val = df[feature].min()
        max_val = df[feature].max()
        df[feature] = (df[feature] - min_val) / (max_val - min_val + e)
        # Implement this formula: x = log(x) - log(1 - x)
        # df[feature] = np.log(df[feature] + e) - np.log((1-e) - df[feature])

        print(f"Deskewed feature: {feature}", " with new skewness: ", df[feature].skew()) if verbose else None
    return df

def normalize(df, method='z-score'):
    # ===== Normalization =====
    if method == 'minmax':
        # MinMax Data normalization: Performs worse than Z-score normalization
        df = (df-df.min()) / (df.max()-df.min())
    elif method == 'z-score':
        # Z-score Data normalization
        df = (df-df.mean()) / df.std()
    elif method == 'robust':
        # Robust Data normalization
        df = (df-df.median()) / (df.quantile(0.75) - df.quantile(0.25))
    elif method == 'scaler':
        # Scaler Data normalization
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
    return df

def data_manipulation(df, method):
    global bankrupt_data
    if method == 'Poly':
        # ===== Polynomial Expansion =====
        poly = PolynomialFeatures(degree=2)
        poly_features = poly.fit_transform(df)
        df_poly = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(df.columns), index=df.index).drop(columns=['1'])
        return df_poly
    elif method == 'BestK':
        # ===== Feature Selection =====
        kbest = SelectKBest(score_func=f_classif, k=10)
        kbest.fit(df, bankrupt_data)
        kbest_cols = df.columns[kbest.get_support(indices=True)]
        print("Best K features: ", kbest_cols)
        return df[kbest_cols]
    elif method == 'PCA':
        # ===== PCA =====
        pca = PCA(n_components=5)
        pca_features = pca.fit_transform(df)
        df_pca = pd.DataFrame(pca_features, columns=[f'PC{i+1}' for i in range(5)], index=df.index)
        return pd.concat([df, data_manipulation(df_pca, 'Poly')], axis=1)
    else:
        return df

default_args = {'rm_empty': {'threshold': 0.1}, 'smote': 'off', 'fill_nan': {'method': 'zero'}, 'binning': 'off', 'high_zero': 'off', 'correlation': 'off', 'outliers': {'threshold': 3.0}, 'deskew': 'on', 'normalize': {'method': 'robust'}}
# default_args = {"rm_empty": {'threshold': 0.2},"binning": 'off',"fill_nan": {'method': 'zero'},"high_zero": 'off',"correlation": {'threshold': 0.999, 'method': 'kendall'},"outliers": {'threshold': 2.5},"deskew": 'off',"normalize": {'method': 'z-score'}}
def get_training_data(df: pd.DataFrame =None, args=None, ret=False):
    if args is None:
        raise ValueError("Args cannot be None")
    if df is None:
        global bankrupt_data
        df = training_data.copy()
        df = rm_high_nan(rm_dups(df), 0.1)
        bankrupt_data = df['X65']
        df = df.drop('X65', axis=1)
    else:
        df = rm_high_nan(rm_dups(df), 0.1)
    
    for key, value in args.items():
        if value == 'off':
            continue
        if key == 'rm_empty':
            df = rm_empty(df, value['threshold'])
        elif key == 'fill_nan':
            df = fill_nan(df, value['method'])
        elif key == 'binning':
            """ Bining got.... binned! """
            # df = binning(df, value['threshold'], value['bins'])
        elif key == 'high_zero':
            """ High_ly_ Ineffective"""
            # df = high_zero(df, value['threshold'])
        elif key == 'correlation':
            df = correlation(df, value['threshold'])
        elif key == 'deskew':
            df = deskew(df)
        elif key == 'normalize':
            df = normalize(df, value['method'])
        elif key == 'data_manipulation':
            df = data_manipulation(df, value['method'])

    df = pd.concat([df, bankrupt_data], axis=1)
    if ret:
        return df
    df.to_csv('./data/training_companydata_prepped.csv', index=False, float_format='%.5f')

def outliers(df: pd.DataFrame, target, threshold=2.5):
    # ===== Outliers =====
    # Remove outliers using Z-score
    z_scores = np.abs((df - df.mean()) / df.std())
    outliers = z_scores > threshold
    # print("Number of outliers: ", outliers.sum().sum()) if verbose else None
    target = target[~outliers.any(axis=1)]
    df = df[~outliers.any(axis=1)]

    # Put the outliers to the mean of the column
    # df.mask(outliers, df.mean(), inplace=True, axis=1)
    return df, target

def smote(df, target):
    # ===== SMOTE =====
    # Synthetic Minority
    return SMOTE(sampling_strategy='minority', random_state=39).fit_resample(df, target)

def prepare_training_data(x, y, args=None):
    if args is None:
        return x, y
    for key, value in args.items():
        if value == 'off':
            continue
        match key:
            case 'outliers':
                x, y = outliers(x, y, value['threshold'])
            case 'smote':
                x, y = smote(x, y)
    return x, y

def get_testing_data(labels, args, ret=False):
    df = pd.read_csv('./data/test_unlabeled.csv', na_values=['?', ''], names=[f'X{i}' for i in range(1, 65)])
    df = df.drop(columns=[col for col in df.columns if col not in labels])

    # Allow only value changing preprocessing
    for key, value in args.items():
        if value == 'off':
            continue
        if key == 'fill_nan':
            df = fill_nan(df, value['method'])
        elif key == 'normalize':
            df = normalize(df, value['method'])
        elif key == 'deskew':
            df = deskew(df)

    if ret:
        return df
    df.to_csv('./data/unlabeled_data_prepped.csv', index=False, float_format='%.5f')

__all__ = ['get_training_data', 'prepare_training_data']

if __name__ == '__main__':
    verbose = True
    get_training_data()
