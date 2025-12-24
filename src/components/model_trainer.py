import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifact', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbors': KNeighborsRegressor(),
                'XGBoost': XGBRegressor(),
                'CatBoost': CatBoostRegressor(verbose=False, allow_writing_files=False),
                'AdaBoost': AdaBoostRegressor()
            }

            params = {
                'Random Forest': {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'Decision Tree': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                'Gradient Boosting': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'Linear Regression': {},
                'K-Neighbors': {
                    'n_neighbors': [5, 7, 9, 11]
                },
                'XGBoost': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'CatBoost': {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                'AdaBoost': {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            # Get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # Get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found')

            logging.info(f'Best model found on both training and testing dataset: {best_model_name}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    from src.components.data_ingestion import DataIngestion, DataIngestionConfig
    from src.components.data_transformation import DataTransformation

    # Run data ingestion
    ingestion_config = DataIngestionConfig()
    data_ingestion = DataIngestion(ingestion_config)
    train_path, test_path = data_ingestion.initiate_data_ingestion()

    # Run data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

    # Run model training
    model_trainer = ModelTrainer()
    r2_score_result = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Model R2 Score: {r2_score_result}")
