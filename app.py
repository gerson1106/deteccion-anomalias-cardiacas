import streamlit as st
from utils import *
autoencoder = Autoencoder()
#autoencoder = torch.load('C:/Users/gerso/OneDrive/Escritorio/ls/prod/autoencoder.pth')
autoencoder = torch.load('autoencoder.pth')
autoencoder.eval()
#autoencoder = cargar_modelo_preentrenado(r"/autoencoder.pth")
# Características básicas de la página
st.set_page_config(page_icon="🌳", page_title="Detección de vegetación Santurban", layout="wide")
st.image("https://web.udi.edu.co/aviacion/wp-content/uploads/2020/09/sRecurso-1.png", width=200)
st.title("Detección de vegetación Paramo de Santurban")

c29, c30, c31 = st.columns([1, 9, 1]) # 3 columnas: 10%, 60%, 10%

UMBRAL = 0.089

with c30:
    uploaded_file = st.file_uploader(
        "", type = 'pkl',
        key="1",
    )


    if uploaded_file is not None:
        file_container = st.expander("Verifique el archivo .pkl que acaba de subir")

        info_box_wait = st.info(
            f"""
                Realizando la clasificación...
                """)

        # Acá viene la predicción con el modelo
        dato = leer_dato(uploaded_file)
        #autoencoder = Autoencoder()
        #autoencoder = cargar_modelo_preentrenado('autoencoder.pth')
        prediccion = predecir(autoencoder, dato, UMBRAL)
        categoria = obtener_categoria(prediccion)


        # Y mostrar el resultado
        info_box_result = st.info(f"""
        	El dato analizado corresponde a un sujeto: {categoria}
        	""")

    else:
        st.info(
            f"""
                👆 Debe cargar primero un dato con extensión .pkl
                """
        )

        st.stop()
        

