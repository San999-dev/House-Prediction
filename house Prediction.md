```python
# ==========================================
# Step 1: Import Libraries
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import tkinter as tk
from tkinter import messagebox

# ==========================================
# Step 2: Load Dataset
# ==========================================
df = pd.read_csv("HousingPrices.csv")  # Make sure CSV is in same folder

# Features & Target
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# ==========================================
# Step 3: Train Model
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "house_price_model.pkl")

# ==========================================
# Step 4: Prediction Function
# ==========================================
def predict_price():
    try:
        # Get input values from entry boxes
        values = [float(entry.get()) for entry in entries]
        df_input = pd.DataFrame([values], columns=X.columns)
        price = model.predict(df_input)[0]
        messagebox.showinfo("Predicted Price", f"Estimated Price: ${price:,.2f}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")

# ==========================================
# Step 5: Build GUI
# ==========================================
root = tk.Tk()
root.title("House Price Prediction")

entries = []
for col in X.columns:
    frame = tk.Frame(root)
    frame.pack(pady=5)
    tk.Label(frame, text=col + ":").pack(side=tk.LEFT)
    entry = tk.Entry(frame)
    entry.pack(side=tk.LEFT)
    entries.append(entry)

tk.Button(root, text="Predict Price", command=predict_price, bg="green", fg="white").pack(pady=20)

root.mainloop()

```


```python

```


```python

```
