import pandas as pd
import numpy as np
from feature_engine.encoding import OneHotEncoder

def process_features(input_file, output_file):
    # Load the cleaned data
    df = pd.read_csv(input_file)

    # Implementing Feature Creation
    df['price_x_promotion'] = df['checkout_price'] * df['emailer_for_promotion']
    df['week_sin'] = np.sin((df['week'] - 1) * (2. * np.pi / 52))
    df['week_cos'] = np.cos((df['week'] - 1) * (2. * np.pi / 52))
    df['price_ratio'] = df['checkout_price'] / df['base_price']
    df['discount'] = df['base_price'] - df['checkout_price']
    df['is_discounted'] = (df['discount'] > 0).astype(int)
    df['promo_interaction'] = df['emailer_for_promotion'] * df['homepage_featured']
    df['cuisine_category'] = df['cuisine'] + '_' + df['category']

    # Rolling averages
    df['SMA_4_weeks'] = df['num_orders'].rolling(window=4).mean()
    df['EWMA_4_weeks'] = df['num_orders'].ewm(span=4).mean()
    df['rolling_average_4_weeks'] = df['num_orders'].rolling(window=4, min_periods=1).mean()

    df.fillna(0, inplace=True)

    # One-hot encoding of categorical variables
    categorical_columns = list(df.select_dtypes(include=['object']).columns)
    encoder = OneHotEncoder(variables=categorical_columns, drop_last=False)
    df_encoded = encoder.fit_transform(df)

    # Save the processed data
    df_encoded.to_csv(output_file, index=False)

if __name__ == "__main__":
    process_features('clean_data.csv', 'processed_data.csv')

