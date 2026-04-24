import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline

# Wczytanie danych
df = pd.read_csv('goodreads_with_genres.csv')
df = df[df['My Rating'] > 0].copy()
df['Liked'] = df['My Rating'].apply(lambda x: 1 if x >= 4 else 0)

# Inżynieria bazowych cech i obsługa braków
def categorize_community_rating(rating):
    if rating <= 3.5: return 'Low'
    elif 3.5 < rating <= 3.8: return 'Below Average'
    elif 3.8 < rating <= 4.0: return 'Average'
    elif 4.0 < rating <= 4.2: return 'Above Average'
    else: return 'High'

df['Rating_Band'] = df['Average Rating'].apply(categorize_community_rating)
df['Number of Pages'] = df['Number of Pages'].fillna(df['Number of Pages'].median())
df['Original Publication Year'] = df['Original Publication Year'].fillna(df['Year Published'])
df['Original Publication Year'] = df['Original Publication Year'].fillna(df['Original Publication Year'].median())

# Obsługa braków dla nowej cechy OpenLibrary_Rating
median_ol_rating = df['OpenLibrary_Rating'].median()
if pd.isna(median_ol_rating):
    median_ol_rating = 3.0
df['OpenLibrary_Rating'] = df['OpenLibrary_Rating'].fillna(median_ol_rating)

# Przetwarzanie cech kategorycznych (Gatunki z OpenLibrary)
df['OpenLibrary_Genre'] = df['OpenLibrary_Genre'].fillna('Unknown')
genres_df = df['OpenLibrary_Genre'].str.get_dummies(sep=', ')

# Łączenie wszystkich cech w jeden zbiór danych
X_base = df[['Rating_Band', 'Number of Pages', 'Original Publication Year', 'OpenLibrary_Rating']]
X_base = pd.get_dummies(X_base, columns=['Rating_Band'], drop_first=True)

X = pd.concat([X_base, genres_df], axis=1)
y = df['Liked']

# Klasyczny podział dla sprawdzenia na pojedynczym zbiorze testowym
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- MODEL 1 (RANDOM FOREST) ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Predykcje i Prawdopodobieństwa
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

# Oceny
rf_acc = accuracy_score(y_test, rf_pred)
rf_roc = roc_auc_score(y_test, rf_proba)

# --- MODEL 2 (LOGISTIC REGRESSION) ---
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Predykcje i Prawdopodobieństwa
lr_pred = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# Oceny
lr_acc = accuracy_score(y_test, lr_pred)
lr_roc = roc_auc_score(y_test, lr_proba)

# --- WALIDACJA KRZYŻOWA (CROSS-VALIDATION) DLA REGRESJI LOGISTYCZNEJ ---
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

# Liczymy dwie metryki osobno
cv_roc_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
cv_acc_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

# --- WYNIKI ---
print(f"--- FINALNE WYNIKI PROJEKTU ---")
print(f"Baza danych: {len(X)} książek, użyto {X.shape[1]} cech (w tym OpenLibrary_Rating).")

print("\n1. RANDOM FOREST (Test Set):")
print(f"   Accuracy: {rf_acc:.2f}")
print(f"   ROC AUC:  {rf_roc:.2f}")

print("\n2. LOGISTIC REGRESSION (Test Set):")
print(f"   Accuracy: {lr_acc:.2f}")
print(f"   ROC AUC:  {lr_roc:.2f}")

print("\n3. LOGISTIC REGRESSION (Cross-Validation):")
print(f"   ROC AUC (5 iteracji):  {[round(score, 2) for score in cv_roc_scores]}")
print(f"   Średni ROC AUC:        {cv_roc_scores.mean():.2f}")
print(f"   Accuracy (5 iteracji): {[round(score, 2) for score in cv_acc_scores]}")
print(f"   Średnia Dokładność:    {cv_acc_scores.mean():.2f}")