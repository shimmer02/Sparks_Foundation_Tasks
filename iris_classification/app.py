import numpy as np
import pickle
import streamlit as st
from PIL import Image

loaded_model = pickle.load(open('iris.sav', 'rb'))

def predict(input):
    array = np.asarray(input)
    rs = array.reshape(1, -1)
    prediction = loaded_model.predict(rs)

    if prediction == 'Iris-setosa':
        setosa = Image.open('setosa.jpg')
        st.image(setosa, caption= 'Iris-setosa')
    elif prediction == 'Iris-virginica':
        virginica = Image.open('virginica.jpg')
        st.image(virginica, caption= 'Iris-virginica')
    elif prediction == 'Iris-versicolor':
        versicolor = Image.open('versicolor.jpg')
        st.image(versicolor, caption= 'Iris-versicolor')
def main():

    st.title('Iris Classifier')

    slength = st.number_input('Enter Sepal Length')
    swidth = st.number_input('Enter Sepal Width')
    plength = st.number_input('Enter Petal Length')
    pwidth = st.number_input('Enter Petal Width')
    result = ''

    if st.button('Classify'):
        result = predict([slength, swidth, plength, pwidth])
if __name__ == '__main__':
    main()