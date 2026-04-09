import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Union, Literal

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, clone
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE


class BaseSelector(ABC):
    """Abstract base class for wrapper method feature selectors."""

    def __init__(self, top_k: int = 20, model_params: Optional[Dict[str, Any]] = None, random_state: int = 42):
        self.top_k = top_k
        self.model_params = model_params if model_params else {}
        self.random_state = random_state
        self.logger = logging.getLogger(f"FeatureSelection.{self.__class__.__name__}")
        self.selected_features_: List[str] = []
        self.feature_importance_: pd.DataFrame = pd.DataFrame()

    def _validate_and_prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and preprocess input data."""
        X_processed = X.copy()
        if X_processed.isnull().any().any():
            self.logger.debug("Input data contains NaNs, filling with 0.")
            X_processed = X_processed.fillna(0)

        n_features = X_processed.shape[1]
        if self.top_k > n_features:
            self.logger.warning("Requested top_k (%d) > available features (%d). Resetting top_k to %d.",
                                self.top_k, n_features, n_features)
            self.top_k = n_features
        return X_processed

    def _finalize_selection(self, importance_df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """Sort feature importance and truncate to Top K."""
        self.feature_importance_ = importance_df.sort_values(
            by='importance', ascending=False
        ).reset_index(drop=True)

        self.selected_features_ = self.feature_importance_.head(self.top_k)['feature'].tolist()
        self.logger.info("Successfully selected %d features.", len(self.selected_features_))
        return self.selected_features_, self.feature_importance_

    @abstractmethod
    def select(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], pd.DataFrame]:
        """Execute feature selection."""
        raise NotImplementedError


class ShapSelector(BaseSelector):
    """SHAP selector based on Tree Models.

    Optimized for battery life prediction scenarios:
    Uses MAE (L1 Loss) or Huber Loss by default to reduce the impact of early prediction noise and outliers.
    """

    def __init__(self, top_k: int = 20,
                 model_params: Optional[Dict[str, Any]] = None,
                 estimator_type: Literal['lightgbm', 'xgboost'] = 'lightgbm',
                 robust_mode: bool = True,
                 random_state: int = 42):
        """
        Args:
            top_k: Number of features to retain.
            model_params: Dictionary of model hyperparameters.
            estimator_type: 'lightgbm' or 'xgboost'.
            robust_mode: If True, forces the use of outlier-insensitive Loss (MAE/Huber).
            random_state: Random seed.
        """
        super().__init__(top_k, model_params, random_state)
        self.estimator_type = estimator_type.lower()
        self.robust_mode = robust_mode

    def select(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], pd.DataFrame]:
        self.logger.info(f"Applying SHAP ({self.estimator_type}, Robust={self.robust_mode}) filtering...")
        X_filled = self._validate_and_prepare(X)

        try:
            model = self._fit_model(X_filled, y)

            # Compute SHAP values using TreeExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_filled, check_additivity=False)

            # Handle multi-dimensional output or list (behavior in some LightGBM versions)
            if isinstance(shap_values, list):
                shap_values = np.abs(shap_values).mean(axis=0)

            # Compute mean absolute SHAP values
            if isinstance(shap_values, np.ndarray):
                 mean_abs_shap = np.abs(shap_values).mean(axis=0)
            else:
                 # Fallback specific for older shap versions
                 mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': mean_abs_shap
            })

            return self._finalize_selection(importance_df)

        except Exception as e:
            self.logger.error(f"SHAP process failed: {str(e)}")
            raise

    def _fit_model(self, X, y):
        """Configure and train the model, optimizing Objective for battery data."""

        if self.estimator_type == 'lightgbm':
            params = {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'verbose': -1,
                'random_state': self.random_state
            }

            if self.robust_mode:
                params.update({'objective': 'regression_l1', 'metric': 'mae'})
                self.logger.info("LightGBM Robust Mode enabled (using L1 Loss/MAE).")
            else:
                params.update({'objective': 'regression', 'metric': 'rmse'})

            params.update(self.model_params)
            return lgb.LGBMRegressor(**params).fit(X, y)

        elif self.estimator_type == 'xgboost':
            params = {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'random_state': self.random_state
            }

            if self.robust_mode:
                params.update({'objective': 'reg:absoluteerror', 'eval_metric': 'mae'})
                self.logger.info("XGBoost Robust Mode enabled (using Absolute Error).")
            else:
                params.update({'objective': 'reg:squarederror', 'eval_metric': 'rmse'})

            params.update(self.model_params)
            return xgb.XGBRegressor(**params).fit(X, y)
        else:
            raise ValueError(f"Unsupported estimator_type: {self.estimator_type}")


class RFESelector(BaseSelector):
    """RFE-based feature selector."""

    def __init__(self, top_k: int = 20,
                 estimator: Optional[BaseEstimator] = None,
                 step: Union[int, float] = 0.1,
                 use_permutation_importance: bool = True,
                 model_params: Optional[Dict[str, Any]] = None,
                 robust_criterion: bool = True,
                 random_state: int = 42):
        super().__init__(top_k, model_params, random_state)
        self.step = step
        self.use_permutation_importance = use_permutation_importance

        if estimator is not None:
            self.estimator = estimator
        else:
            dt_params = {'random_state': self.random_state}
            if robust_criterion:
                try:
                    dt_params['criterion'] = 'absolute_error'
                except Exception:
                    self.logger.warning("sklearn in current environment does not support criterion='absolute_error', falling back to default.")
            self.estimator = DecisionTreeRegressor(**dt_params)

    def select(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], pd.DataFrame]:
        self.logger.info("Applying RFE (Base Model: %s) filtering...", self.estimator.__class__.__name__)
        X_filled = self._validate_and_prepare(X)

        if self.model_params and hasattr(self.estimator, 'set_params'):
            self.estimator.set_params(**self.model_params)

        try:
            # RFE Phase
            selector = RFE(estimator=self.estimator, n_features_to_select=self.top_k, step=self.step)
            selector.fit(X_filled, y)

            selected_cols = X_filled.columns[selector.support_].tolist()

            # Importance Phase
            if self.use_permutation_importance:
                self.logger.info("Running permutation importance (Metric: MAE)...")
                final_model = clone(self.estimator)
                X_subset = X_filled[selected_cols]
                final_model.fit(X_subset, y)

                result = permutation_importance(
                    final_model, X_subset, y,
                    scoring='neg_mean_absolute_error',
                    n_repeats=5, random_state=self.random_state, n_jobs=-1
                )
                importances = result.importances_mean
            else:
                if hasattr(selector.estimator_, 'feature_importances_'):
                    importances = selector.estimator_.feature_importances_
                elif hasattr(selector.estimator_, 'coef_'):
                    importances = np.abs(selector.estimator_.coef_)
                    if importances.ndim > 1: importances = importances.mean(axis=0)
                else:
                    importances = np.ones(len(selected_cols))

            importance_df = pd.DataFrame({'feature': selected_cols, 'importance': importances})
            return self._finalize_selection(importance_df)

        except Exception as e:
            self.logger.error("Error occurred during RFE selection: %s", str(e))
            raise
