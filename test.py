import streamlit as st
import pandas as pd
from pandasai import Agent
import os


os.environ["PANDASAI_API_KEY"] = "<API KEY HERE>"
# Função para exibir gráfico solicitado pela IA


# Carregar dataset
candidates= pd.read_csv(r'/home/henrique/Desktop/Linkedin_Content/pessoas.csv')
agent = Agent(candidates)

# Interface Streamlit
st.title("Talktative Dataset")

# Exibir preview do dataset
st.write("Dataset head:")
st.dataframe(candidates.head())

# Caixa de diálogo para interações
user_input = st.text_input("What would you like to know?:", "Give me a boxplot of the years of experience column")

if user_input:
    response = agent.chat(user_input)
    st.write(f"Response: {response}")

