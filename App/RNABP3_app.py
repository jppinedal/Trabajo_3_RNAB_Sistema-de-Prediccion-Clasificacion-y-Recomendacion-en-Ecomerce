import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
print(tf.__version__)
from PIL import Image
import tempfile
import os
from sklearn.preprocessing import RobustScaler

# Función para cargar imagen
def cargar_img(uploaded_file, img_size, color_mode):
    image = Image.open(uploaded_file)
    image = image.resize(img_size)
    image = np.array(image)
    return image

# Función para clasificación de imágenes
def plot_imagen_classification(modelo, imagen, clases):
    if len(imagen.shape) == 3:
        imagen = np.expand_dims(imagen, axis=0)
    predictions = modelo.predict(imagen)
    predicted_class = np.argmax(predictions)
    plt.figure(figsize=(10, 3))
    plt.bar(clases, predictions[0], color="blue")
    plt.bar(clases[predicted_class], predictions[0, predicted_class], color="red")
    st.pyplot(plt)

# Función para crear secuencias de datos
def create_sequences(data, seq_length=7):
    X = [data[i:(i + seq_length)] for i in range(len(data) - seq_length)]
    return np.array(X)

# Función para preparar datos y hacer predicción
def prepare_data(model, data, days):
    scaler_dict = {}
    scaled_data = data.copy()
    numeric_features = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']
    
    for feature in numeric_features:
        scaler = RobustScaler()
        scaled_data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))
        scaler_dict[feature] = scaler
    
    features = ['Store', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Dept', 'Weekly_Sales', 'IsHoliday', 'Type', 'Size']
    X = scaled_data[features].values
    seq_length = 7
    X_seq = create_sequences(X, seq_length)
    last_sequence = X_seq[-1]
    predictions = []
    
    for _ in range(days):
        next_pred = model.predict(last_sequence.reshape(1, seq_length, X.shape[1]))
        predictions.append(next_pred[0])
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1] = next_pred
    
    sales_scaler = scaler_dict['Weekly_Sales']
    predictions = sales_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Función para graficar predicciones
def plot_predictions(predictions, data, days, hist):
    last_date = pd.to_datetime(data['Date'].max())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
    
    sales_data = data[['Date', 'Weekly_Sales']].copy()
    sales_data['Date'] = pd.to_datetime(sales_data['Date'])
    sales_data = sales_data.set_index('Date')
    sales_data = sales_data.resample('W')['Weekly_Sales'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(sales_data['Date'].tail(hist), sales_data['Weekly_Sales'].tail(hist), label='Historical Sales', color='blue')
    plt.legend()
    st.pyplot(plt)
    
    plt.figure(figsize=(12, 6))
    plt.plot(future_dates, predictions, label='Predicted Sales', color='red', linestyle='--')
    plt.legend()
    st.pyplot(plt)

# ----- Comienzo de la app ----- #
st.title("Redes Neuronales y algoritmos Bioinspirados")
st.subheader("2024-2S - Grupo 6 \n")
file_path = os.path.abspath("/mount/src/trabajo_3_rnab_sistema-de-prediccion-clasificacion-y-recomendacion-en-ecomerce/img/logo.png")

if os.path.exists(file_path):
    st.success(f"Archivo encontrado en: {file_path}")
    st.image(file_path, width=350)
else:
    st.error(f"Archivo NO encontrado. Directorio actual: {os.getcwd()}")


st.subheader("Sistema Inteligente Integrado para Predicción, Clasificación y Recomendación en Comercio Electrónico \n")


# Pestañas para segmentar los módulos
tabs = st.tabs(["Predicción con LSTM", "Clasificación de Imágenes", "Recomendación de Productos"])

# Sección de predicción con LSTM
with tabs[0]:
    st.header("Predicción de Series Temporales")
    st.text("Esta sección te ayuda a estimar las ventas futuras a partir de datos históricos. \n")
    st.text("1️⃣ Sube el modelo LSTM entrenado (.keras) \n")
    st.text("2️⃣ Elige cuántas semanas históricas deseas visualizar 📊 \n")
    st.text("3️⃣ Selecciona cuántos días deseas predecir 📅 \n")
    st.text("4️⃣ Haz clic en (Predecir) 🚀 \n")
    st.text("La aplicación analizará los datos y generará dos gráficos: \n")
    st.text("- Historial de ventas 📉 \n - Predicción de ventas futuras 🔮")
    lstm_model_file = st.file_uploader("Sube el modelo LSTM", type=["keras"])
    data = pd.read_csv('data/data_lstm.csv')
    
    if lstm_model_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as temp_model:
            temp_model.write(lstm_model_file.read())
            lstm_model = load_model(temp_model.name)
        

        st.success("Modelo de predicción y datos cargados correctamente")
        
        hist = st.number_input("Ingrese la cantidad de semanas históricas", min_value=1, step=1)
        days = st.number_input("Ingrese la cantidad de días a predecir", min_value=1, step=1)
        
        if st.button("Predecir"):
            predictions = prepare_data(lstm_model, data, days)
            plot_predictions(predictions, data, days, hist)

# Sección de clasificación de imágenes
with tabs[1]:
    st.header("Clasificación de Imágenes")
    st.text("Esta sección te permite subir una imagen para ser clasificada en una de las siguientes categorías: \n \n 👕 Camiseta (tshirt) | 🛋️ Sofá (sofa) | 👖 Jeans (jeans) | 📺 Televisor (tv)")
    img_model_file = st.file_uploader("Sube el modelo de clasificación (.keras)", type=["keras"])
    
    if img_model_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as temp_model:
            temp_model.write(img_model_file.read())
            img_model = load_model(temp_model.name)
        
        st.success("Modelo de clasificación cargado correctamente")
        st.text("1️⃣ Sube una imagen")
        st.text("🖼️Haz clic en el botón (Sube una imagen) y elige un archivo en formato JPG, PNG o JPEG desde tu dispositivo. \n")
        st.text("2️⃣ Visualiza la imagen original y procesada 🔍")
        st.text("3️⃣ Espera el análisis del modelo 🤖 \n")
        st.text("El sistema analizará la imagen y determinará a qué categoría pertenece entre: Camiseta (tshirt), Sofá (sofa), Jeans (jeans) o Televisor (tv).")

        uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            clases = ['tshirt', 'sofa', 'jeans', 'tv']
            image = Image.open(uploaded_file)
            image_classify = cargar_img(uploaded_file, (128, 128), "rgb")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Imagen subida", width=200)
            
            with col2:
                st.image(image_classify, caption="Imagen procesada", width=250)
            
            plot_imagen_classification(img_model, image_classify, clases)

with tabs[2]:
    st.header("Sistema de recomendación de Productos")
    st.text("Esta sección te genera recomendaciones de acuerdo a tu producto seleccionado. \n")
    st.text("1️⃣ Carga la matriz de similitud (npy) \n")
    st.text("2️⃣ Selecciona la categoría \n")
    st.text("3️⃣ Selecciona la sub categoría \n")
    st.text("4️⃣ Selecciona el producto de tu preferencia 🚀 \n")


    # Cargar datos
    def load_data():
        data = pd.read_csv("data/Amazon-Products-Filtered.csv")
        return data

    # Cargar matriz de similitud
    uploaded_file = st.file_uploader("Cargar archivo de matriz de similitud (.npy)", type="npy")

    def load_similarity_matrix(uploaded_file):
        if uploaded_file is not None:
            return np.load(uploaded_file)
        else:
            st.error("Por favor, carga un archivo de matriz de similitud.")
            return None
    
    st.title("Sistema de Recomendación de productos de Amazon")
    data = load_data()
    similarity_matrix = load_similarity_matrix(uploaded_file)

    if similarity_matrix is not None:
        # datos
        st.subheader("Vista de Datos")
        st.dataframe(data.head(10))

        # Resumen
        st.subheader("Resumen Estadístico de los Datos")
        st.write(data[['ratings', 'no_of_ratings', 'main_category', 'sub_category']].describe(include='all'))

        # Distribución por categoría
        st.subheader("Distribución por Categoría Principal")
        fig, ax = plt.subplots(figsize=(8, 6))
        data["main_category"].value_counts().plot(kind='bar', ax=ax, color='skyblue')
        plt.title("Distribución de Productos por main_category")
        plt.xlabel("Categoría")
        plt.ylabel("Cantidad de Productos")
        st.pyplot(fig)

        # Filtros de recomendación
        st.subheader("Filtros de Recomendación")
        category = st.selectbox("Selecciona la Categoría", data["main_category"].unique())
        subcategory = st.selectbox("Selecciona la Subcategoría", data[data["main_category"] == category]["sub_category"].unique())
        name = st.selectbox("Selecciona el Nombre", data[(data["main_category"] == category) & (data["sub_category"] == subcategory)]["name"].unique())
    
        # Obtener recomendaciones
        def recomendarProducto(data, similarity_matrix, product_name):
            #producto seleccionado
            product_index = data[data['name'] == product_name].index[0]

            # Indices de los productos más similares
            similarity_scores = similarity_matrix[product_index]

            top_indices = np.argsort(similarity_scores)[::-1][1:6]

            return data.iloc[top_indices], data.iloc[product_index]
        
        # Mostrar detalles del artículo seleccionado y recomendaciones
        if st.button("Obtener Recomendaciones"):
            recommendations, selected_product = recomendarProducto(data, similarity_matrix, name)

            st.subheader("Detalles del Producto Seleccionado")
            st.image(selected_product['image'], width=300)
            st.write(f"**Nombre:** {selected_product['name']}")
            st.write(f"**Categoría Principal:** {selected_product['main_category']}")
            st.write(f"**Subcategoría:** {selected_product['sub_category']}")
            st.write(f"**Rating:** {selected_product['ratings']:.2f}")
            st.write(f"**No. de Calificaciones:** {int(selected_product['no_of_ratings'])}")
            st.write(f"**Precio Descuento:** ₹{selected_product['discount_price']:.2f}")
            st.write(f"**Precio Actual:** ₹{selected_product['actual_price']:.2f}")

            # Mostrar recomendaciones
            st.subheader("Recomendaciones")
            for _, row in recommendations.iterrows():
                st.image(row['image'], width=300) 
                st.write(f"**Nombre:** {row['name']}")
                st.write(f"**Categoría Principal:** {row['main_category']}")
                st.write(f"**Subcategoría:** {row['sub_category']}")
                st.write(f"**Rating:** {row['ratings']:.2f}")
                st.write(f"**No. de Calificaciones:** {int(row['no_of_ratings'])}")
                st.write(f"**Precio Descuento:** ₹{row['discount_price']:.2f}")
                st.write(f"**Precio Actual:** ₹{row['actual_price']:.2f}")
                st.write("---")

# Espaciado para alejar el pie de página del contenido
st.write("\n\n\n")

# Pie de página estilizado
footer_placeholder = st.empty()
footer_html = """
    <style>
        .footer {
            text-align: center;
            margin-top: 50px;
            color: rgba(100, 100, 100, 0.7); /* Gris con opacidad */
            font-size: 14px;
        }
    </style>
    <div class="footer">
        2025 Universidad Nacional de Colombia Sede Medellín. <br>
        Valentina Ospina Narváez, Juan Pablo Pineda Lopera, Juan Camilo Torres Arboleda
    </div>
"""
footer_placeholder.markdown(footer_html, unsafe_allow_html=True)



