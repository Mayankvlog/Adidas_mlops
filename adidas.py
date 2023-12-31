import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import Image
import mlflow
import mlflow.sklearn

# Load Adidas dataset
adidas_data = pd.read_excel('data/Adidas US Sales Datasets.xlsx')

# Convert 'Total Sales' to categories
sales_bins = [0, 100, 500, float('inf')]
sales_labels = ['Low', 'Medium', 'High']
adidas_data['Sales Category'] = pd.cut(adidas_data['Total Sales'], bins=sales_bins, labels=sales_labels, right=False)

# Sidebar for user input parameters
st.sidebar.header('User Input Parameters')

def user_input_features():
    price_per_unit = st.sidebar.slider('Price per Unit', float(adidas_data['Price per Unit'].min()), float(adidas_data['Price per Unit'].max()), float(adidas_data['Price per Unit'].mean()))
    units_sold = st.sidebar.slider('Units Sold', int(adidas_data['Units Sold'].min()), int(adidas_data['Units Sold'].max()), int(adidas_data['Units Sold'].mean()))
    data = {'Price per Unit': price_per_unit, 'Units Sold': units_sold}
    features = pd.DataFrame(data, index=[0])
    return features

# Main app content
st.write("""
# Adidas Sales Prediction App
This app predicts Adidas sales category based on user input!
""")

# Add logo
logo_path = Image.open("adidas_logo.jpg")
st.image(logo_path, width=150)

# Sidebar - User input features
df = user_input_features()

# Display user input parameters
st.subheader('User Input Parameters')
st.write(df)

# Classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine (SVM)': SVC()
}

# User selects a classifier
selected_classifier = st.sidebar.selectbox('Select Classifier', list(classifiers.keys()))

# Initialize the selected classifier
clf = classifiers[selected_classifier]

# Train the classifier (assuming 'Sales Category' is the target column)
X = adidas_data[['Price per Unit', 'Units Sold']]
y = adidas_data['Sales Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

# Log parameters to MLflow
with mlflow.start_run():
    mlflow.log_param("Classifier", selected_classifier)
    mlflow.log_param("Price per Unit", df['Price per Unit'].values[0])
    mlflow.log_param("Units Sold", df['Units Sold'].values[0])

    # Make predictions
    prediction = clf.predict(df)

    # Log parameters (since prediction is a category)
    mlflow.log_param("prediction", prediction[0])

    # Display prediction
    st.subheader('Prediction')
    st.write(f"Predicted Sales Category: {prediction[0]}")

# Page navigation
pages = ["Home", "Model Evaluation", "Visualization"]
page = st.sidebar.selectbox("Select Page:", pages)

if page == "Home":
    st.title("Home Page")
    st.write("Welcome to the Adidas Sales Prediction App. Use the sidebar to input parameters and make predictions.")

elif page == "Model Evaluation":
    st.title("Model Evaluation Page")
    st.write(f"Evaluate the performance of {selected_classifier}.")

    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Display evaluation results
    st.subheader("Model Evaluation Results:")
    st.write(f"Accuracy: {accuracy}")
    st.write("Classification Report:")
    st.write(class_rep)
    st.write("Confusion Matrix:")
    st.write(conf_matrix)

elif page == "Visualization":
    st.title("Visualization Page")
    st.write("Explore visualizations of the Adidas dataset.")

    # Scatter plot
    st.subheader("Scatter Plot:")
    scatter_fig = px.scatter(adidas_data, x='Units Sold', y='Total Sales', color='Sales Category', size='Price per Unit')
    st.plotly_chart(scatter_fig)

    # Violin Plot
    st.subheader("Violin Plot:")
    violin_fig = px.violin(adidas_data, y='Total Sales', box=True, points='all', color='Sales Category')
    st.plotly_chart(violin_fig)

    # Bar Plot
    st.subheader("Bar Plot:")
    bar_fig = px.bar(adidas_data, x='Sales Category', y='Total Sales', color='Sales Category')
    st.plotly_chart(bar_fig)

    # Line plot
    st.subheader("Line Plot:")
    line_fig = px.line(adidas_data, x='State', y='Total Sales', color='Sales Category')
    st.plotly_chart(line_fig)

    # Pie plot
    st.subheader("Pie Plot:")
    pie_fig = px.pie(adidas_data, names='Sales Category')
    st.plotly_chart(pie_fig)

# Set MLflow Tracking URI
st.markdown("[DAGsHub Repository](https://dagshub.com/Mayankvlog/Adidas_mlops.mlflow)")
