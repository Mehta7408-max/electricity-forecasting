.PHONY: help train-all train-hetero train-homo train-xgboost compare serve mlflow-ui docker-up lint

help:
	@echo "Electricity Price Forecasting — Developer Workflow"
	@echo ""
	@echo "Training:"
	@echo "  make train-all        Train all three models"
	@echo "  make train-hetero     Train HeteroSAGE (quick_retrain.py)"
	@echo "  make train-homo       Train HomoGNN GraphSAGE (homo_retrain.py)"
	@echo "  make train-xgboost    Train XGBoost baseline (xgboost_baseline.py)"
	@echo ""
	@echo "Evaluation:"
	@echo "  make compare          Print model comparison report"
	@echo ""
	@echo "Serving:"
	@echo "  make serve            Start FastAPI server on :8000"
	@echo "  make mlflow-ui        Start MLflow UI on :5000"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up        Start MLflow + API containers"
	@echo ""
	@echo "Quality:"
	@echo "  make lint             Syntax-check all src/*.py files"

train-all: train-xgboost train-homo train-hetero compare

train-hetero:
	PYTHONPATH=src python src/quick_retrain.py

train-homo:
	PYTHONPATH=src python src/homo_retrain.py

train-xgboost:
	PYTHONPATH=src python src/xgboost_baseline.py

compare:
	PYTHONPATH=src python src/compare_models.py

serve:
	python -m uvicorn src.model_api:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlruns.db --host 0.0.0.0 --port 5000

docker-up:
	docker compose up mlflow api

lint:
	python -m py_compile src/*.py && echo All OK
