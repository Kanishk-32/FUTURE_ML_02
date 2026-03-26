import pandas as pd

from src.preprocess import clean_text
from src.model import train_model
from src.evaluation import evaluate

df = pd.read_csv("data/new_data.csv")

print("Dataset shape:", df.shape)
print("\nColumns:\n", df.columns)

df = df.rename(columns={
    "Document": "Ticket Description",
    "Topic_group": "Ticket Type"
})

df["Ticket Subject"] = ""
df["Ticket Priority"] = "Medium"

df["text"] = df["Ticket Subject"] + " " + df["Ticket Description"]

df["clean_text"] = df["text"].apply(clean_text)

print("\nSample cleaned text:\n")
print(df["clean_text"].head())

y_category = df["Ticket Type"]

model, X_test, y_test, y_pred = train_model(
    df["clean_text"], y_category
)

print("\n==============================")
print(" CATEGORY MODEL RESULTS ")
print("==============================")

evaluate(y_test, y_pred)

results = pd.DataFrame({
    "Actual Category": y_test,
    "Predicted Category": y_pred
})

results.to_csv("outputs/predictions.csv", index=False)

print("\nPredictions saved to outputs/predictions.csv")