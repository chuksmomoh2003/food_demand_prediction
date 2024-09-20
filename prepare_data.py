import pandas as pd

def load_and_clean_data():
    # Load the datasets
    train_food_demand = pd.read_csv('train_food_demand.csv')
    meal_info = pd.read_csv('meal_info.csv')
    fulfilment_center_info = pd.read_csv('fulfilment_center_info.csv')
    
    # Merging the datasets
    merged_df_1 = pd.merge(train_food_demand, meal_info, on='meal_id', how='left')
    df = pd.merge(merged_df_1, fulfilment_center_info, on='center_id', how='left')

    # Calculate the 95th percentile of 'num_orders'
    percentile_95 = df['num_orders'].quantile(0.95)

    # Calculate the median of 'num_orders'
    median_num_orders = df['num_orders'].median()

    # Replace values above the 95th percentile with the median
    df.loc[df['num_orders'] > percentile_95, 'num_orders'] = median_num_orders

    # Save the cleaned and merged data
    df.to_csv('clean_data.csv', index=False)

if __name__ == "__main__":
    load_and_clean_data()

