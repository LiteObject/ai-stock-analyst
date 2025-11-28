"""
Hyperparameter Optimization with Optuna.

This module provides GPU-accelerated hyperparameter optimization for:
- XGBoost
- LightGBM
- CatBoost
- Ensemble configurations

Supports:
- Multi-objective optimization (accuracy vs interpretability)
- Time-series aware cross-validation
- GPU utilization for faster training
- Pruning for early stopping of bad trials
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Import optuna with proper type stubs
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler

HAS_OPTUNA = True

# Import ML libraries
try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    xgb = None  # type: ignore[assignment]

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    lgb = None  # type: ignore[assignment]

try:
    import catboost as cb

    HAS_CB = True
except ImportError:
    HAS_CB = False
    cb = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""

    best_params: Dict[str, Any]
    best_score: float
    best_trial_number: int
    optimization_history: List[Dict[str, Any]]
    feature_importance: Optional[Dict[str, float]] = None
    cv_scores: Optional[List[float]] = None


@dataclass
class GPUConfig:
    """GPU configuration for training."""

    enabled: bool = True
    device_id: int = 0
    memory_fraction: float = 0.9

    @property
    def xgb_params(self) -> Dict[str, Any]:
        """XGBoost GPU parameters."""
        if self.enabled:
            return {
                "tree_method": "gpu_hist",
                "device": f"cuda:{self.device_id}",
            }
        return {"tree_method": "hist"}

    @property
    def lgb_params(self) -> Dict[str, Any]:
        """LightGBM GPU parameters."""
        if self.enabled:
            return {
                "device": "gpu",
                "gpu_device_id": self.device_id,
            }
        return {"device": "cpu"}

    @property
    def cb_params(self) -> Dict[str, Any]:
        """CatBoost GPU parameters."""
        if self.enabled:
            return {
                "task_type": "GPU",
                "devices": str(self.device_id),
            }
        return {"task_type": "CPU"}


class HyperparameterOptimizer:
    """
    GPU-accelerated hyperparameter optimization using Optuna.

    Supports XGBoost, LightGBM, and CatBoost with GPU acceleration.
    Uses time-series aware cross-validation.
    """

    def __init__(
        self,
        n_trials: int = 100,
        n_cv_splits: int = 5,
        gpu_config: Optional[GPUConfig] = None,
        random_state: int = 42,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        early_stopping_rounds: int = 50,
    ):
        """
        Initialize the optimizer.

        Args:
            n_trials: Number of Optuna trials
            n_cv_splits: Number of time-series CV splits
            gpu_config: GPU configuration (auto-detected if None)
            random_state: Random state for reproducibility
            timeout: Optimization timeout in seconds
            n_jobs: Number of parallel jobs (1 for GPU)
            early_stopping_rounds: Early stopping patience
        """
        if not HAS_OPTUNA:
            raise ImportError("Optuna is required. Install with: pip install optuna")

        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.gpu_config = gpu_config or self._detect_gpu()
        self.random_state = random_state
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds

        logger.info(f"GPU enabled: {self.gpu_config.enabled}")

    def _detect_gpu(self) -> GPUConfig:
        """Auto-detect GPU availability."""
        try:
            import torch  # type: ignore[import-unresolved]

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"Detected GPU: {gpu_name}")
                return GPUConfig(enabled=True, device_id=0)
        except ImportError:
            pass

        # Fallback: check CUDA directly
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                gpu_name = result.stdout.strip()
                logger.info(f"Detected GPU: {gpu_name}")
                return GPUConfig(enabled=True, device_id=0)
        except Exception:
            pass

        logger.info("No GPU detected, using CPU")
        return GPUConfig(enabled=False)

    def optimize_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = "logloss",
        direction: str = "minimize",
    ) -> OptimizationResult:
        """
        Optimize XGBoost hyperparameters.

        Args:
            X: Feature matrix
            y: Target variable
            metric: Optimization metric (logloss, auc, error)
            direction: minimize or maximize

        Returns:
            OptimizationResult with best parameters
        """
        if not HAS_XGB:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")

        def objective(trial: "optuna.Trial") -> float:
            params = {
                # Core parameters
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                # Regularization
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
                # GPU settings
                **self.gpu_config.xgb_params,
                "random_state": self.random_state,
                "objective": "binary:logistic",
                "eval_metric": metric,
                "early_stopping_rounds": self.early_stopping_rounds,
            }

            return self._cv_score_xgb(X, y, params, metric, trial)

        study = self._create_study(f"xgboost_{metric}", direction)
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )

        return self._create_result(study)

    def optimize_lightgbm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = "binary_logloss",
        direction: str = "minimize",
    ) -> OptimizationResult:
        """
        Optimize LightGBM hyperparameters.

        Args:
            X: Feature matrix
            y: Target variable
            metric: Optimization metric
            direction: minimize or maximize

        Returns:
            OptimizationResult with best parameters
        """
        if not HAS_LGB:
            raise ImportError(
                "LightGBM is required. Install with: pip install lightgbm"
            )

        def objective(trial: "optuna.Trial") -> float:
            params = {
                # Core parameters
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                # Regularization
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 1e-3, 10.0, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                # LightGBM specific
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                # GPU settings
                **self.gpu_config.lgb_params,
                "random_state": self.random_state,
                "objective": "binary",
                "metric": metric,
                "verbosity": -1,
                "force_col_wise": True,  # Faster for tall datasets
            }

            return self._cv_score_lgb(X, y, params, metric, trial)

        study = self._create_study(f"lightgbm_{metric}", direction)
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )

        return self._create_result(study)

    def optimize_catboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = "Logloss",
        direction: str = "minimize",
    ) -> OptimizationResult:
        """
        Optimize CatBoost hyperparameters.

        Args:
            X: Feature matrix
            y: Target variable
            metric: Optimization metric
            direction: minimize or maximize

        Returns:
            OptimizationResult with best parameters
        """
        if not HAS_CB:
            raise ImportError(
                "CatBoost is required. Install with: pip install catboost"
            )

        def objective(trial: "optuna.Trial") -> float:
            params = {
                # Core parameters
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                # Regularization
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
                "bagging_temperature": trial.suggest_float(
                    "bagging_temperature", 0.0, 10.0
                ),
                "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
                # CatBoost specific
                "border_count": trial.suggest_int("border_count", 32, 255),
                "grow_policy": trial.suggest_categorical(
                    "grow_policy", ["SymmetricTree", "Depthwise"]
                ),
                # GPU settings
                **self.gpu_config.cb_params,
                "random_seed": self.random_state,
                "loss_function": "Logloss",
                "eval_metric": metric,
                "verbose": False,
                "early_stopping_rounds": self.early_stopping_rounds,
            }

            return self._cv_score_cb(X, y, params, metric, trial)

        study = self._create_study(f"catboost_{metric}", direction)
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )

        return self._create_result(study)

    def optimize_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        base_models: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> OptimizationResult:
        """
        Optimize ensemble weights and configurations.

        Args:
            X: Feature matrix
            y: Target variable
            base_models: Pre-trained model configs

        Returns:
            OptimizationResult with optimal ensemble configuration
        """

        def objective(trial: "optuna.Trial") -> float:
            # Optimize model weights
            xgb_weight = trial.suggest_float("xgb_weight", 0.0, 1.0)
            lgb_weight = trial.suggest_float("lgb_weight", 0.0, 1.0)
            cb_weight = trial.suggest_float("cb_weight", 0.0, 1.0)

            # Normalize weights
            total = xgb_weight + lgb_weight + cb_weight
            if total == 0:
                return float("inf")

            weights = {
                "xgboost": xgb_weight / total,
                "lightgbm": lgb_weight / total,
                "catboost": cb_weight / total,
            }

            # Optimize meta-learner
            meta_type = trial.suggest_categorical(
                "meta_learner", ["logistic", "xgboost", "lightgbm"]
            )

            return self._cv_score_ensemble(X, y, weights, meta_type, trial)

        study = self._create_study("ensemble", "minimize")
        study.optimize(
            objective,
            n_trials=min(self.n_trials, 50),  # Fewer trials for ensemble
            timeout=self.timeout,
            show_progress_bar=True,
        )

        return self._create_result(study)

    def _create_study(self, name: str, direction: str) -> "optuna.Study":
        """Create an Optuna study with configured samplers and pruners."""
        sampler = TPESampler(seed=self.random_state)
        pruner = HyperbandPruner(
            min_resource=1,
            max_resource=self.n_cv_splits,
            reduction_factor=3,
        )

        study = optuna.create_study(
            study_name=name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
        )

        return study

    def _cv_score_xgb(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict[str, Any],
        metric: str,
        trial: "optuna.Trial",
    ) -> float:
        """Calculate cross-validation score for XGBoost."""
        assert xgb is not None, "XGBoost is required"
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        scores: List[float] = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=params.get("n_estimators", 100),
                evals=[(dval, "val")],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False,
            )

            preds = model.predict(dval)

            if metric in ["logloss", "binary_logloss"]:
                from sklearn.metrics import log_loss

                score = float(log_loss(y_val, preds))
            elif metric == "auc":
                from sklearn.metrics import roc_auc_score

                score = float(-roc_auc_score(y_val, preds))  # Negative for minimization
            else:
                from sklearn.metrics import accuracy_score

                preds_binary = (np.asarray(preds) > 0.5).astype(int)
                score = float(-accuracy_score(y_val, preds_binary))

            scores.append(score)

            # Pruning
            trial.report(float(np.mean(scores)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    def _cv_score_lgb(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict[str, Any],
        metric: str,
        trial: optuna.Trial,
    ) -> float:
        """Calculate cross-validation score for LightGBM."""
        assert lgb is not None, "LightGBM is required"
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        scores: List[float] = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]

            model = lgb.train(
                params,
                train_data,
                num_boost_round=params.get("n_estimators", 100),
                valid_sets=[val_data],
                callbacks=callbacks,
            )

            preds = np.asarray(model.predict(X_val))

            if metric in ["logloss", "binary_logloss"]:
                from sklearn.metrics import log_loss

                score = float(log_loss(y_val, preds))
            elif metric == "auc":
                from sklearn.metrics import roc_auc_score

                score = float(-roc_auc_score(y_val, preds))
            else:
                from sklearn.metrics import accuracy_score

                preds_binary = (preds > 0.5).astype(int)
                score = float(-accuracy_score(y_val, preds_binary))

            scores.append(score)

            trial.report(float(np.mean(scores)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    def _cv_score_cb(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict[str, Any],
        metric: str,
        trial: optuna.Trial,
    ) -> float:
        """Calculate cross-validation score for CatBoost."""
        assert cb is not None, "CatBoost is required"
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        scores: List[float] = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_pool = cb.Pool(X_train, label=y_train)
            val_pool = cb.Pool(X_val, label=y_val)

            model = cb.CatBoost(params)
            model.fit(train_pool, eval_set=val_pool, verbose=False)

            preds = model.predict(val_pool, prediction_type="Probability")[:, 1]

            if metric in ["Logloss", "logloss", "binary_logloss"]:
                from sklearn.metrics import log_loss

                score = float(log_loss(y_val, preds))
            elif metric in ["AUC", "auc"]:
                from sklearn.metrics import roc_auc_score

                score = float(-roc_auc_score(y_val, preds))
            else:
                from sklearn.metrics import accuracy_score

                preds_binary = (np.asarray(preds) > 0.5).astype(int)
                score = float(-accuracy_score(y_val, preds_binary))

            scores.append(score)

            trial.report(float(np.mean(scores)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    def _cv_score_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        weights: Dict[str, float],
        meta_type: str,
        trial: optuna.Trial,
    ) -> float:
        """Calculate cross-validation score for ensemble."""
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        scores: List[float] = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train base models and get predictions
            preds: Dict[str, np.ndarray] = {}

            if weights.get("xgboost", 0) > 0 and HAS_XGB and xgb is not None:
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)
                model = xgb.train(
                    {**self.gpu_config.xgb_params, "objective": "binary:logistic"},
                    dtrain,
                    num_boost_round=100,
                )
                preds["xgboost"] = model.predict(dval)

            if weights.get("lightgbm", 0) > 0 and HAS_LGB and lgb is not None:
                train_data = lgb.Dataset(X_train, label=y_train)
                model = lgb.train(
                    {
                        **self.gpu_config.lgb_params,
                        "objective": "binary",
                        "verbosity": -1,
                    },
                    train_data,
                    num_boost_round=100,
                )
                preds["lightgbm"] = np.asarray(model.predict(X_val))

            if weights.get("catboost", 0) > 0 and HAS_CB and cb is not None:
                train_pool = cb.Pool(X_train, label=y_train)
                model = cb.CatBoost(
                    {**self.gpu_config.cb_params, "iterations": 100, "verbose": False}
                )
                model.fit(train_pool)
                preds["catboost"] = model.predict(
                    cb.Pool(X_val), prediction_type="Probability"
                )[:, 1]

            # Weighted ensemble
            ensemble_pred = np.zeros(len(X_val))
            for name, pred in preds.items():
                ensemble_pred += weights[name] * pred

            from sklearn.metrics import log_loss

            score = float(log_loss(y_val, ensemble_pred))
            scores.append(score)

            trial.report(float(np.mean(scores)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    def _create_result(self, study: optuna.Study) -> OptimizationResult:
        """Create optimization result from study."""
        history = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append(
                    {
                        "trial_number": trial.number,
                        "params": trial.params,
                        "value": trial.value,
                    }
                )

        return OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            best_trial_number=study.best_trial.number,
            optimization_history=history,
        )


def quick_optimize(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgboost",
    n_trials: int = 50,
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """
    Quick hyperparameter optimization for a single model.

    Args:
        X: Feature matrix
        y: Target variable
        model_type: xgboost, lightgbm, or catboost
        n_trials: Number of optimization trials
        use_gpu: Whether to use GPU

    Returns:
        Dictionary with best parameters
    """
    gpu_config = GPUConfig(enabled=use_gpu)
    optimizer = HyperparameterOptimizer(
        n_trials=n_trials,
        gpu_config=gpu_config,
    )

    if model_type.lower() == "xgboost":
        result = optimizer.optimize_xgboost(X, y)
    elif model_type.lower() == "lightgbm":
        result = optimizer.optimize_lightgbm(X, y)
    elif model_type.lower() == "catboost":
        result = optimizer.optimize_catboost(X, y)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return result.best_params
