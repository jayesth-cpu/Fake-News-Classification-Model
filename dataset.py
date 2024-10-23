import pandas as pd
true_news = pd.read_csv('True.csv')
fake_news = pd.read_csv('Fake.csv')

true_news['label'] = 1
fake_news['label'] = 0

all_news = pd.concat([true_news, fake_news], ignore_index=True)

print(all_news.shape)  # Shows the number of rows and columns
print(all_news.columns)  # Shows the column names
print(all_news['label'].value_counts())  # Shows the distribution of true and fake news
print(all_news.head())  # Shows the first few rows of your data