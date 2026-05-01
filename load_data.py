import pandas as pd
import numpy as np

df = pd.read_csv('original_copy.csv')
# print(df.head())
# print(df.columns)
# print(df.describe())
# print(df.isna().sum())
# print(df.duplicated().sum())
# print(df.info())
# print(df.corr(numeric_only=True ))
# print(df.skew(numeric_only=True))
# print(df['oldpeak'])
# df['trestbps'] = df['trestbps'].replace(0, np.nan)
# df['chol'] = df['chol'].replace(0, np.nan)
# # print(df.columns)
# df.to_csv('replaced_0_with_nan.csv', index=False)
# print(df.isna().mean().sort_values(ascending=False))
# print((df == 0).mean())
#
# for col in df.columns:
#     if pd.api.types.is_numeric_dtype(df[col]):
#         print(f"{col}: {df[col].skew()}")
#     else:
#         print(f"{col}: Skipping (not numeric)")

# summary = pd.DataFrame({
#     'zeros': (df == 0).sum(),
#     'nan': df.isna().sum()
# })
categories = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
# print(summary)
# df[categories] = df[categories].fillna("Unknown")
# df.to_csv('original_copy.csv', index=False)

df['num'] = (df['num'] > 0).astype(int)

base_df = df.copy()

def handle_categorical(df):
    cat_cols = df.select_dtypes(include=['object', 'string']).columns
    return df.assign(**{
        col: df[col].fillna("Unknown") for col in cat_cols
    })

def impute_numeric(df):
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df

df_a = base_df.copy()
df_a = handle_categorical(df_a)
df_a = impute_numeric(df_a)

df_b = base_df.copy()
suspect_zero_cols = ['chol', 'fbs', 'exang', 'oldpeak']

for col in suspect_zero_cols:
    df_b[col] = df_b[col].replace(0, np.nan)

df_b = handle_categorical(df_b)
df_b = impute_numeric(df_b)
df_a, df_b = df_a.align(df_b, join='left', axis=1, fill_value=0)
df_c = base_df.copy()
df_c = df_c.drop(['ca', 'thal'], axis=1)

df_c = handle_categorical(df_c)
df_c = impute_numeric(df_c)