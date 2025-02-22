# Librerías
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# Interfaz 
st.image("img/logo.png", width=150)
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
