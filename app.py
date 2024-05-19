from utils import *
import streamlit as st

#      python -m streamlit run app.py

st.set_page_config(
    page_title="Mundo de Wumpus",
)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.header("Detector de Parkinson por voz :brain:")

st.write('La enfermedad de Parkinson es un desorden neurodegenerativo progresivo caracteriza por una gran cantidad de características motrices.')

st.write('Uno de los sintomas iniciales que presenta en el 90% de los pacientes son problemas vocales. Esto lo hace una característica importante para el telediagnóstico de la enfermedad.')

st.write('**Instrucciones**:')
st.write('- Grábate diciendo la vocal /a/ durante 5 segundos.')
st.write('- Después, sube tu archivo (.wav) aquí y una vez subido has click en Procesar.')

files_uploaded = st.file_uploader(
    "Carga tu archivo",
    accept_multiple_files=False
    )

if st.button('Procesar'):

    datos = to_diagnose(files_uploaded)

    st.write(datos)

    decision = voting(datos)

    if decision:
        st.write('Tienes Parkinson')
    else:
        st.write('NO Tienes Parkinson :D')

InvestUrl = "https://doi.org/10.1016/j.asoc.2018.10.022"
KaggleUrl = "https://www.kaggle.com/datasets/dipayanbiswas/parkinsons-disease-speech-signal-features/data"

with st.sidebar:
    with st.expander("Sobre la investigación y la base de datos"):
        st.write('''
        Este proyecto está motivado por la investigación de Sakar, Serbes et al. 
        *A comparative analysis of speech signal processing algorithms for Parkinson’s 
        disease classification and the use of the tunable Q-factor wavelet transform*.
        ''')
        st.write('''
        Para su consulta, ingrese a [este link](%s).''' % InvestUrl)
        st.write('''
        El crédito de la base de datos corresponde a los mismos autores.
        ''')
        st.write('''
        Esta disponible en [Kaggle](%s).''' % KaggleUrl)



    with st.expander("Sobre la metodología"):
        st.write('''
        Se utilizó la base de datos para realizar entrenamiento a modelos de clasificación. Esto con la intención de
        hacer un diagnóstico basado en  los *Mel-frequency cepstral coefficients* obtenidos
        a partir de un archivo de audio.
        ''')

    with st.expander("Disclaimer"):
        st.write('''
        El propósito de este programa es proponer una opción de telediagnóstico que sea auxiliar a la detección temprana del Parkinson. 
        Para un diagnóstico correcto y el seguimiento adecuado, consulte a su médico.
        ''')

    

        
        
        




