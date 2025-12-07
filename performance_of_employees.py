# data_analysis_pipeline_with_scalers_and_models.py
# Usage: update DATA_PATH if needed, then run:
#    python data_analysis_pipeline_with_scalers_and_models.py

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from collections import Counter
import time

# ---------- USER CONFIG ----------
DATA_PATH = Path("C:\\Users\\Rahul\\Desktop\\internship project\\Test_data.csv")
OUTPUT_DIR = Path('C:\\Users\\Rahul\\Desktop\\internship project\\report_output')
PLOTS_DIR = Path ('C:\\Users\\Rahul\\Desktop\\internship project\\report_output\\plot')
PLOTS_DIR = OUTPUT_DIR /"plot"
SUMMARY_CSV = OUTPUT_DIR / "scaler_model_summary.csv"
RANDOM_STATE = 42

# Create folders
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid", palette="pastel")

# ---------- Utility functions ----------
def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at {path}")
    return pd.read_csv(path)

def basic_inspect(df):
    print("\n---- df.info() ----")
    df.info()
    print("\n---- df.head() ----")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nMissing counts:")
    print(df.isnull().sum())

def fill_nulls(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns

    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_cols:
        if df[col].mode().size > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna("", inplace=True)
    return df

def drop_duplicates(df):
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Dropped {before-after} duplicate rows.")
    return df

def detect_target_column(df):
    # common candidates
    candidates = ['target','Target','label','Label','y','Y','outcome']
    for c in candidates:
        if c in df.columns:
            return c
    # else use last column
    return df.columns[-1]

def prepare_features(df, target_col=None):
    if target_col is None:
        target_col = detect_target_column(df)
    print(f"Using target column: {target_col}")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # encode target if non-numeric
    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        print(f"Target encoded with LabelEncoder. Classes: {list(le.classes_)}")
    else:
        # ensure integer labels for some libraries
        if pd.api.types.is_float_dtype(y):
            # if obvious integers stored as floats, convert
            if np.allclose(y, y.astype(int)):
                y = y.astype(int)

    # Identify columns
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Ordinal encode categorical features (necessary before scaling)
    if cat_cols:
        enc = OrdinalEncoder(dtype=float, handle_unknown='use_encoded_value', unknown_value=-1)
        X[cat_cols] = enc.fit_transform(X[cat_cols].astype(str))
        print(f"Ordinal-encoded categorical cols: {cat_cols}")

    return X, y, num_cols, cat_cols, target_col

# ---------- Scaler function ----------
def apply_scalers(X: pd.DataFrame, numeric_cols: list):
    """
    Returns dict: scaler_name -> scaled DataFrame (copy of X with numeric_cols scaled)
    """
    scalers = {
        "RobustScaler": RobustScaler(),
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "MaxAbsScaler": MaxAbsScaler()
    }
    scaled_versions = {}
    for name, scaler in scalers.items():
        Xs = X.copy()
        if len(numeric_cols) > 0:
            Xs[numeric_cols] = scaler.fit_transform(Xs[numeric_cols])
        scaled_versions[name] = Xs
    return scaled_versions

# ---------- Model factory (with fallbacks) ----------
def get_models(random_state=RANDOM_STATE):
    """
    Returns dict: model_name -> estimator, and notes dict describing which libs were used.
    Uses light xgboost/catboost/lgb with sensible default n_estimators to keep runtime reasonable.
    """
    models = {}
    notes = {}
    # XGBoost
    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                              n_estimators=150, random_state=random_state, n_jobs=4)
        notes['XGBoost'] = "xgboost"
    except Exception:
        models['XGBoost'] = GradientBoostingClassifier(n_estimators=150, random_state=random_state)
        notes['XGBoost'] = "fallback: sklearn.GradientBoostingClassifier"

    # CatBoost
    try:
        from catboost import CatBoostClassifier
        models['CatBoost'] = CatBoostClassifier(verbose=0, iterations=150, random_state=random_state)
        notes['CatBoost'] = "catboost"
    except Exception:
        models['CatBoost'] = HistGradientBoostingClassifier(max_iter=150, random_state=random_state)
        notes['CatBoost'] = "fallback: sklearn.HistGradientBoostingClassifier"

    # LightGBM
    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMClassifier(n_estimators=150, random_state=random_state, n_jobs=4)
        notes['LightGBM'] = "lightgbm"
    except Exception:
        models['LightGBM'] = GradientBoostingClassifier(n_estimators=150, random_state=random_state)
        notes['LightGBM'] = "fallback: sklearn.GradientBoostingClassifier"

    return models, notes

# ---------- Training / Evaluation ----------
def safe_train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE):
    # stratify only if every class has at least 2 samples and more than 1 class
    counts = Counter(y)
    if len(counts) > 1 and min(counts.values()) >= 2:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        print("Stratify disabled (too few samples in at least one class or single-class target).")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_and_eval(model, X, y, model_name, scaler_name, out_dir):
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y)
    start = time.time()
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error training {model_name} with {scaler_name}: {e}")
        return None
    train_time = time.time() - start
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    # Save confusion matrix image
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{model_name} + {scaler_name}\nAcc={acc:.4f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    cm_path = out_dir / f"cm_{model_name}_{scaler_name}.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {
        "model": model_name,
        "scaler": scaler_name,
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
        "cm_path": str(cm_path),
        "train_time_sec": train_time
    }

# ---------- Main pipeline ----------
def main():
    print("Loading data from:", DATA_PATH)
    df = load_data(DATA_PATH)

    # 1) Inspect
    basic_inspect(df)

    # 2) Treat NULLs
    df = fill_nulls(df)

    # 3) Remove duplicates
    df = drop_duplicates(df)

    # 4) Identify features/target
    X, y, numeric_cols, cat_cols, target_col = prepare_features(df, target_col=None)
    print("Numeric columns:", numeric_cols)
    print("Categorical (encoded) columns:", cat_cols)

    # 5) Generate plots for EDA (count plots + histograms) and save them
    # Categorical count plots
    cat_cols_original = df.select_dtypes(include=['object','category','bool']).columns.tolist()
    for col in cat_cols_original:
        plt.figure(figsize=(8,5))
        vc = df[col].value_counts().iloc[:30]
        sns.barplot(x=vc.values, y=vc.index)
        plt.title(f"Count plot: {col}")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"count_{col}.png", dpi=150)
        plt.close()


    # 6) Apply scalers
    scaled_versions = apply_scalers(X, numeric_cols)
    print("Applied scalers:", list(scaled_versions.keys()))

    # 7) Get models
    models, notes = get_models()
    print("Models discovered and/or fallback notes:")
    for k,v in notes.items():
        print(f" - {k}: {v}")

    # 8) Iterate over (scaler x model)
    results = []
    for scaler_name, X_scaled in scaled_versions.items():
        for model_name, model in models.items():
            print(f"\nTraining {model_name} with {scaler_name} ...")
            try:
                res = train_and_eval(model, X_scaled, y, model_name, scaler_name, PLOTS_DIR)
                if res is not None:
                    results.append(res)
                    print(f" -> Accuracy: {res['accuracy']:.4f}, time(s): {res['train_time_sec']:.2f}")
            except Exception as e:
                print("Training failed for", model_name, scaler_name, ":", e)

    # 9) Summarize and save CSV
    if results:
        summary = pd.DataFrame([{
            'model': r['model'],
            'scaler': r['scaler'],
            'accuracy': r['accuracy'],
            'cm_path': r['cm_path'],
            'train_time_sec': r['train_time_sec']
        } for r in results])
        summary = summary.sort_values(by='accuracy', ascending=False).reset_index(drop=True)
        summary.to_csv(SUMMARY_CSV, index=False)
        print("\nSummary saved to:", SUMMARY_CSV)
        print("Top results:")
        print(summary.head(10))
        best = summary.iloc[0]
        print(f"\nBest: Model={best['model']}, Scaler={best['scaler']}, Accuracy={best['accuracy']:.4f}")
        print("Best confusion matrix image:", best['cm_path'])
    else:
        print("No successful training runs.")

if __name__ == "__main__":
    main()
