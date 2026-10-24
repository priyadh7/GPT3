import pandas as pd

df = pd.read_excel("Email_Assistant_Dataset_100.xlsx")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(3).to_string())

df.to_csv("data/emails.csv", index=False)
print("\nDataset saved to data/emails.csv")
