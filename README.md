# guadalahacks_mundodewumpus
El programa del equipo El Mundo de Wumpus para el hachathon 2024 de guadalahacks.

Autores: 

Santiago Mora

Rene Calzadilla

Guillermo Villegas

Gabriel Reynoso

## Inspiration
Parkinson's disease is the second most common neurodegenerative disorder in adults over sixty. Increasing life expectancy has driven the development of new diagnostic systems, aiming for early detection and a reduction in medical visits required for monitoring.

## What It Does
The user records an audio clip of approximately 3 seconds directly into the software, sustaining a steady and uniform /a/ sound. Using artificial intelligence, the system estimates whether the user may have Parkinson's disease.

## How We Built It
We used a database containing acoustic voice signal data from people with and without Parkinson’s. First, we selected a subset of features based on Mel-frequency cepstral coefficients (MFCC). We trained linear, nonlinear, and logistic classification models to predict Parkinson’s presence using these features. Once the models were ready, we developed a program that converts an audio recording into numerical MFCC coefficients for classification. Finally, we built a front-end interface where users can record audio, process it, classify it, and receive information about the disease.

## Challenges We Faced
The challenges we encountered included:

Finding a suitable database, understanding its structure, and the mathematical transformations needed to derive the coefficients.

Adapting voice recordings to ensure compatibility with the database's format.

Difficulties in developing the audio pre-processing program before classification.

Selecting classification models that best fit the data and provided the highest accuracy.

## Accomplishments We’re Proud Of
We succeeded in:

Leveraging doctoral-level theoretical research to create a practical tool accessible worldwide.

Developing a program that converts audio into coefficients usable by classification models.

What We Learned
Exporting models to optimize program runtime.

The MFCC technique for audio feature extraction and signal pre-processing.

Technical terminology related to Parkinson’s disease and signal processing.

Audio data extraction and manipulation.

## What’s Next for AI-Powered Parkinson’s Detection Through Voice
Exploring other transformations and feature extraction methods for audio.

Incorporating additional user attributes (e.g., age) that may influence diagnosis.

Integrating EEG-based classification for more comprehensive diagnostics.

Acquiring professional audio recording equipment to ensure data quality.

Implementing more advanced classification models with robust hardware.

Developing a more accessible front-end and deploying it for broader public use.

## Requerimientos
Para correr este programa, es necesario tener instaladas las librerías de python (con pip o conda):
- streamlit
- pandas
- joblib
- pydub
- sklearn
- numpy
- librosa
- scipy
- wave
- streamlit-audiorec

Además, se debe instalar ffmpeg y tenerlo en %PATH%

## Ejecución

Una vez satisfechos los requerimientos, ubicar la terminal en la dirección en la que se tienen los archivos y correr el comando:

python -m streamlit run app.py
