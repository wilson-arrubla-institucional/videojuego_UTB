import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
#Configuracion de la pagina  Predicción inversion tienda de video juego

st.set_page_config(page_title="Predicción inversion tienda de video juego", page_icon="🕹️", layout="wide")

# Crear punto de entrada def main
def main():

    #Cargar imagen 
    st.image("videojuego.jpg", width=900)
    

    #Cargar el modelo
    filename = 'modelo-reg-tree-RF.pkl'
    model_Tree,model_RF,variables = pickle.load(open(filename, 'rb')) #Cargar los modelos Tree,  RF y el objeto variebles

    #crear el sidbar de variables
    st.sidebar.title("Parámetros del usuario")    

    # Crear los campos de entrada para las variables
    def user_input_features():
        #Entrada Edad
        edad = st.sidebar.number_input("Edad", min_value=14, max_value=52)

        #Entrada videojuego
        option_videojuego = ["'Mass Effect'","'Sim City'","'Crysis'","'Dead Space'","'Battlefield'","'KOA: Reckoning'","'F1'","'Fifa'"]
        videojuego = st.sidebar.selectbox("Videojuego", option_videojuego, index=0)

        #Entrada Plataformas
        option_plataforma = ["'Play Station'","'Xbox'","PC","Otros"]
        plataforma = st.sidebar.selectbox("Plataforma", option_plataforma, index=0)

        #Entradas Sexo
        option_sex =["Mujer","Hombre"]
        sexo = st.sidebar.selectbox("Sexo",option_sex, index=0)

        #Entrada Consumidor Habitual
        Consumidor_habitual = st.sidebar.checkbox("Consumidor Habitual", value=False)

        #Crear diccionario de variables
        data = {
            'Edad': edad,
            'videojuego': videojuego,
            'Plataforma': plataforma,
            'Sexo': sexo,
            'Consumidor_habitual': Consumidor_habitual
        }
        
        data_imput = pd.DataFrame(data, index=[0])
        

        return data_imput
    data_imp = user_input_features()

    data_preparada = data_imp.copy()
    #st.write(data_preparada)
    
    #Transformar las variables categóricas en variables dummy
    data_preparada = pd.get_dummies(data_preparada, columns=['videojuego', 'Plataforma'], drop_first= False)
    data_preparada = pd.get_dummies(data_preparada, columns=['Sexo'], drop_first=False)
                                    
    #st.write(data_preparada)
    #Ajustar el dataframe a la forma del modelo Reidexación de columnas faltantes
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0) # rellenar con 0 las columnas que faltan

    #st.write(data_preparada)

    #Predicción del modelo Tree
    #Crear boton para la predicción
    if st.sidebar.button("Predecir"):
        #Realizar predicción con el modelo Tree
        y_pred_Tree = model_Tree.predict(data_preparada) #Predicción del modelo Tree
        
        st.success(f"🎮 El cliente invertirá: {y_pred_Tree[0]:.1f} dolares") #Mostrar la predicción del modelo Tree
        st.write("Precisión del modelo: 96%")

 


         
if __name__ == "__main__":
    main()
