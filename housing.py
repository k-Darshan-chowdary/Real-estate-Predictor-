import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
print("📂 Loading dataset...")
FILE_NAME = r"C:\Users\User\OneDrive\文档\info project\USA_Housing.csv"
df = pd.read_csv(FILE_NAME)

print("✅ Data loaded successfully!")
print("📊 Columns available:", df.columns.tolist(), "\n")

# ---------------------------
# Step 1: Select features
# ---------------------------
features = ['bed', 'bath', 'acre_lot', 'house_size', 'city', 'state']
target = 'price'

# Drop rows with missing values
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# ---------------------------
# Step 2: Preprocessing
# ---------------------------
categorical_features = ['city', 'state']
numeric_features = ['bed', 'bath', 'acre_lot', 'house_size']

# OneHotEncode city & state
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)

# Build pipeline with preprocessing + Linear Regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# ---------------------------
# Step 3: Train/Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Step 4: Train Model
# ---------------------------
print("🤖 Training model...")
model.fit(X_train, y_train)

# ---------------------------
# Step 5: Evaluate Model
# ---------------------------
y_pred = model.predict(X_test)

print("\n📊 Model Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# ---------------------------
# Step 6: Take User Input
# ---------------------------
print("\n🏠 Enter your house details to predict the price:")

bed = int(input("Number of bedrooms: "))
bath = int(input("Number of bathrooms: "))
acre_lot = float(input("Lot size (in acres): "))
house_size = float(input("House size (in sq ft): "))
city = input("City: ")
state = input("State: ")

example = pd.DataFrame({
    'bed': [bed],
    'bath': [bath],
    'acre_lot': [acre_lot],
    'house_size': [house_size],
    'city': [city],
    'state': [state]
})

# ---------------------------
# Step 7: Prediction
# ---------------------------
pred_price = model.predict(example)[0]
print(f"\n💰 Predicted price for {example.iloc[0].to_dict()}: ${pred_price:,.2f}")
