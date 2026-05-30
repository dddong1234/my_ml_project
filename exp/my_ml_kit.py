# my_ml_kit.py

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# my_ml_kit.py

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

def make_data_split(X, y, **split_kwargs):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        **split_kwargs
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

def make_pipe(preprocessor, model):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def get_default_scoring(task_type):
    if task_type == "classification":
        return ["accuracy", "f1", "precision", "recall", "roc_auc"]

    elif task_type == "regression":
        return [
            "neg_mean_squared_error",
            "neg_root_mean_squared_error",
            "neg_mean_absolute_error",
            "r2",
        ]

    else:
        raise ValueError("task_type must be 'classification' or 'regression'")


def clean_metric_name(metric):
    if metric.startswith("neg_"):
        return metric.replace("neg_", "")
    return metric


def summarize_cv_scores(cv_scores, scoring):
    result = {}

    for metric in scoring:
        score_key = f"test_{metric}"

        if score_key not in cv_scores:
            continue

        score_mean = cv_scores[score_key].mean()
        score_std = cv_scores[score_key].std()

        clean_name = clean_metric_name(metric)

        if metric.startswith("neg_"):
            score_mean = -score_mean

        result[f"cv_{clean_name}"] = score_mean
        result[f"cv_{clean_name}_std"] = score_std

    return result


def evaluate_classification_test(pipe, X_test, y_test, y_pred):
    result = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
    }

    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
        result["test_roc_auc"] = roc_auc_score(y_test, y_proba)
    except AttributeError:
        result["test_roc_auc"] = None

    return result


def evaluate_regression_test(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)

    return {
        "test_mse": mse,
        "test_rmse": np.sqrt(mse),
        "test_mae": mean_absolute_error(y_test, y_pred),
        "test_r2": r2_score(y_test, y_pred),
    }


def evaluate_test_metrics(task_type, pipe, X_test, y_test, y_pred):
    if task_type == "classification":
        return evaluate_classification_test(
            pipe=pipe,
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
        )

    elif task_type == "regression":
        return evaluate_regression_test(
            y_test=y_test,
            y_pred=y_pred,
        )

    else:
        raise ValueError("task_type must be 'classification' or 'regression'")


def evaluate_model(
    name,
    pipe,
    X_train,
    y_train,
    X_test,
    y_test,
    task_type="classification",
    results=None,
    cv=5,
    scoring=None,
):
    if scoring is None:
        scoring = get_default_scoring(task_type)

    cv_scores = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
    )

    cv_result = summarize_cv_scores(
        cv_scores=cv_scores,
        scoring=scoring,
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    test_result = evaluate_test_metrics(
        task_type=task_type,
        pipe=pipe,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
    )

    result = {
        "name": name,
        "task_type": task_type,
        **cv_result,
        **test_result,
    }

    if results is not None:
        results.append(result)

    return result, pipe, y_pred

def compare_results(results, sort_by=None, ascending=None):
    results_df = pd.DataFrame(results)

    if sort_by is None:
        task_type = results_df["task_type"].iloc[0]

        if task_type == "classification":
            sort_by = "cv_f1"
            ascending = False

        elif task_type == "regression":
            sort_by = "cv_root_mean_squared_error"
            ascending = True

    return results_df.sort_values(
        by=sort_by,
        ascending=ascending
    ).reset_index(drop=True)

def print_test_report(task_type, y_test, y_pred):
    if task_type == "classification":
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    elif task_type == "regression":
        mse = mean_squared_error(y_test, y_pred)

        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {np.sqrt(mse):.4f}")
        print(f"MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"R2   : {r2_score(y_test, y_pred):.4f}")

    else:
        raise ValueError("task_type must be 'classification' or 'regression'")

def run_model_candidates(
    models,
    preprocessor,
    X_train,
    y_train,
    X_test,
    y_test,
    task_type="classification",
    results=None,
    cv=5,
    scoring=None,
):
    trained_pipes = {}
    predictions = {}

    if results is None:
        results = []

    for name, model in models.items():
        print(f"Running: {name}")

        pipe = make_pipe(
            preprocessor=preprocessor,
            model=model,
        )

        result, trained_pipe, y_pred = evaluate_model(
            name=name,
            pipe=pipe,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task_type=task_type,
            results=results,
            cv=cv,
            scoring=scoring,
        )

        trained_pipes[name] = trained_pipe
        predictions[name] = y_pred

    return results, trained_pipes, predictions