Application Elements

Client Information


Change sliders in the sidebar to enter boundaries 
like 'Cost per Unit' and 'Units Sold'.


Model Forecast
Select a classifier from the sidebar (Random Forest, K-Nearest Neighbors, L
ogistic Regression, Support Vector Machine).


Anticipated deals classification is shown in light of client input.


MLOps Coordination
MLflow is utilized for model following.
Prepared models are logged with boundaries and forecasts.
Model Assessment
Change to the "Model Assessment" page to see exactness, grouping
report, and disarray framework.
Perception
Change to the "Perception" page to investigate representations of the Adidas dataset.

MLflow Following
The MLflow UI shows explore subtleties and permits following of model antiques.
DAGsHub Vault for cooperative ML advancement.
Contributing
Fork the vault, make changes, and present a force demand.
For significant changes, if it's not too much trouble, open an issue to talk about the proposed adjustments.



MLFLOW_TRACKING_URI=https://dagshub.com/Mayankvlog/Adidas_mlops.mlflow \
MLFLOW_TRACKING_USERNAME=Mayankvlog \
MLFLOW_TRACKING_PASSWORD=a163dc11889cf8831384598f795e7c53cd54b1d2 \
streamlit run adidas.py
