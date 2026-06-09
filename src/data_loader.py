from datasets import load_dataset

def load_adult_income():
    dataset = load_dataset("jlh/uci-adult-income")
    train_df = dataset['train'].to_pandas()
    return train_df

if __name__ == "__main__":
    df = load_adult_income()
    print(df.shape)
    print(df.head())