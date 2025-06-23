import streamlit as st
import pandas as pd
import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from adapters.kaggle_downloader_adapter import KaggleDownloaderAdapter
from adapters.ydata_profiling_adapter import YDataProfilingAdapter
from adapters.dtale_adapter import DtaleAdapter
from adapters.pycaret_adapter import PyCaretAdapter


from application.use_cases import MLUseCases


from pycaret.classification import (
    setup as class_setup, compare_models as class_compare, pull as class_pull,
    save_model as class_save_model, load_model as class_load_model,
    plot_model as class_plot_model, predict_model as class_predict_model
)
from pycaret.regression import (
    setup as reg_setup, compare_models as reg_compare, pull as reg_pull,
    save_model as reg_save_model, load_model as reg_load_model,
    plot_model as reg_plot_model, predict_model as reg_predict_model
)
from pycaret.clustering import (
    setup as clus_setup, create_model as clus_create, assign_model as clus_assign,
    pull as clus_pull, save_model as clus_save_model, load_model as clus_load_model,
    plot_model as clus_plot_model
)


st.set_page_config(layout="wide")
st.title("Aplicação de Machine Learning com Streamlit e PyCaret (Integrado)")


kaggle_adapter = KaggleDownloaderAdapter()
profiler_adapter = YDataProfilingAdapter()
dtale_adapter = DtaleAdapter()
training_adapter = PyCaretAdapter()

ml_use_cases = MLUseCases(
    dataset_adapter=kaggle_adapter,
    profiler_adapter=profiler_adapter,
    dtale_adapter=dtale_adapter,
    training_adapter=training_adapter
)


@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
        else:
            st.error("Formato não suportado. Use CSV ou Excel.")
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
    return None


st.sidebar.header("Upload de Dados")
uploaded_file = st.sidebar.file_uploader("Escolha um arquivo CSV ou Excel", type=["csv", "xls", "xlsx"])

if uploaded_file:
    data = load_data(uploaded_file)
    if data is not None:
        st.success("Dados carregados com sucesso!")
        st.write("Prévia dos dados:", data.head())

        # === EDA ===
        st.sidebar.header("Análise Exploratória de Dados (EDA)")
        if st.sidebar.checkbox("Mostrar estatísticas descritivas"):
            st.subheader("Estatísticas Descritivas")
            st.write(data.describe())

        if st.sidebar.checkbox("Mostrar tipos de dados"):
            st.subheader("Tipos de Dados")
            st.write(data.dtypes)

        if st.sidebar.checkbox("Mostrar valores ausentes"):
            st.subheader("Valores Ausentes")
            st.write(data.isnull().sum())

    
        st.sidebar.header("Configuração do Modelo")
        all_columns = data.columns.tolist()
        target_column = st.sidebar.selectbox("Variável alvo", all_columns)
        problem_type = st.sidebar.radio("Tipo de problema", ("Classificação", "Regressão", "Clusterização"))
        model_path = "best_model"


        if st.sidebar.button("Executar PyCaret"):
            if problem_type == "Classificação" and target_column:
                st.subheader("Classificação com PyCaret")
                with st.spinner("Configurando..."):
                    class_setup(data, target=target_column, session_id=123, silent=True, verbose=False)
                best_model = class_compare()
                st.write(class_pull())
                class_save_model(best_model, model_path)
                st.success("Modelo treinado e salvo.")

                st.subheader("Análise do Modelo")
                plot_type = st.selectbox("Plot", ["auc", "confusion_matrix", "precision_recall", "error", "boundary"])
                try:
                    class_plot_model(best_model, plot=plot_type, save=True)
                    st.image(f"{plot_type}.png")
                except Exception as e:
                    st.error(f"Erro ao gerar plot: {e}")

            elif problem_type == "Regressão" and target_column:
                st.subheader("Regressão com PyCaret")
                with st.spinner("Configurando..."):
                    reg_setup(data, target=target_column, session_id=123, silent=True, verbose=False)
                best_model = reg_compare()
                st.write(reg_pull())
                reg_save_model(best_model, model_path)
                st.success("Modelo treinado e salvo.")

                st.subheader("Análise do Modelo")
                plot_type = st.selectbox("Plot", ["residuals", "error", "cooks", "learning", "vc", "manifold", "feature", "rfe", "tree"])
                try:
                    reg_plot_model(best_model, plot=plot_type, save=True)
                    st.image(f"{plot_type}.png")
                except Exception as e:
                    st.error(f"Erro ao gerar plot: {e}")

            elif problem_type == "Clusterização":
                st.subheader("Clusterização com PyCaret")
                with st.spinner("Configurando..."):
                    clus_setup(data, session_id=123, silent=True, verbose=False)
                num_clusters = st.number_input("Número de Clusters", min_value=2, value=3)
                if st.button("Criar Modelo"):
                    with st.spinner("Criando modelo..."):
                        kmeans = clus_create("kmeans", num_clusters=num_clusters)
                        kmeans_results = clus_assign(kmeans)
                        st.write(kmeans_results.head())
                        clus_save_model(kmeans, model_path)
                        st.success("Modelo de clusterização salvo.")

                        st.subheader("Análise do Modelo")
                        plot_type = st.selectbox("Plot", ["elbow", "silhouette", "distance", "distribution"])
                        try:
                            clus_plot_model(kmeans, plot=plot_type, save=True)
                            st.image(f"{plot_type}.png")
                        except Exception as e:
                            st.error(f"Erro ao gerar plot: {e}")
            else:
                st.warning("Por favor, selecione a variável alvo.")

    
        st.sidebar.header("Previsão com Novos Dados")
        new_data_file = st.sidebar.file_uploader("Novos dados (CSV ou Excel)", type=["csv", "xls", "xlsx"])

        if new_data_file:
            new_data = load_data(new_data_file)
            if new_data is not None:
                st.success("Novos dados carregados!")
                st.write("Prévia:", new_data.head())

                if st.sidebar.button("Fazer Previsão"):
                    if os.path.exists(f"{model_path}.pkl"):
                        if problem_type == "Classificação":
                            model = class_load_model(model_path)
                            predictions = class_predict_model(model, data=new_data)
                            st.subheader("Previsões")
                            st.write(predictions.head())
                        elif problem_type == "Regressão":
                            model = reg_load_model(model_path)
                            predictions = reg_predict_model(model, data=new_data)
                            st.subheader("Previsões")
                            st.write(predictions.head())
                        elif problem_type == "Clusterização":
                            st.warning("Clusterização atribui grupos, não faz previsão direta.")
                    else:
                        st.warning("Modelo não encontrado. Execute o treinamento primeiro.")
else:
    st.info("Faça upload de um conjunto de dados para começar.")
