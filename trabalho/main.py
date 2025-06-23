import argparse
import logging
import sys


from adapters.kaggle_downloader_adapter import KaggleDownloaderAdapter
from adapters.ydata_profiling_adapter import YDataProfilingAdapter
from adapters.dtale_adapter import DtaleAdapter
from adapters.pycaret_adapter import PyCaretAdapter


from application.use_cases import MLUseCases


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def setup_adapters():
    """Instancia os adapters e cria o orquestrador de casos de uso."""
    return MLUseCases(
        dataset_adapter=KaggleDownloaderAdapter(),
        profiler_adapter=YDataProfilingAdapter(),
        dtale_adapter=DtaleAdapter(),
        training_adapter=PyCaretAdapter()
    )

def download_dataset(ml_use_cases, dataset_name):
    logger.info("Autenticando com o Kaggle...")
    ml_use_cases.dataset_adapter.authenticate()
    logger.info("Autenticado com sucesso. Iniciando download...")
    ml_use_cases.download_dataset(dataset_name, "data")
    logger.info("Download finalizado com sucesso.")

def generate_profile(ml_use_cases, csv_filename):
    logger.info(f"Gerando relatório de perfil para: {csv_filename}")
    ml_use_cases.profile_data(csv_filename)
    logger.info("Relatório gerado. Verifique o arquivo HTML na pasta atual.")

def edit_dataset(ml_use_cases, csv_filename):
    logger.info(f"Abrindo o Dtale para edição: {csv_filename}")
    ml_use_cases.edit_data(csv_filename)
    logger.info("Dtale iniciado. Verifique o console para o link de acesso.")

def train_model(ml_use_cases, csv_filename, target_col, task_type):
    logger.info(f"Iniciando treinamento | Arquivo: {csv_filename} | Alvo: {target_col} | Tarefa: {task_type}")
    ml_use_cases.train_model(csv_filename, target_col, task_type)
    logger.info("Treinamento finalizado com sucesso.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hexagonal ML Pipeline - CLI")
    subparsers = parser.add_subparsers(dest="command")

    
    dl_parser = subparsers.add_parser("download", help="Faz download de um dataset do Kaggle")
    dl_parser.add_argument("kaggle_name", help="Nome do dataset no Kaggle (ex: usuario/dataset)")

   
    profile_parser = subparsers.add_parser("profile", help="Gera relatório de perfil com ydata-profiling")
    profile_parser.add_argument("csv_filename", help="Nome do arquivo CSV na pasta data/")

    edit_parser = subparsers.add_parser("edit", help="Edita um CSV com Dtale")
    edit_parser.add_argument("csv_filename", help="Nome do arquivo CSV na pasta data/")


    train_parser = subparsers.add_parser("train", help="Treina um modelo com PyCaret")
    train_parser.add_argument("csv_filename", help="Nome do arquivo CSV na pasta data/")
    train_parser.add_argument("target_col", help="Nome da coluna alvo (target)")
    train_parser.add_argument("task_type", choices=["classification", "regression", "clustering"], help="Tipo de tarefa")

    return parser.parse_args(), parser

def main():
    args, parser = parse_arguments()
    ml_use_cases = setup_adapters()

    if args.command == "download":
        download_dataset(ml_use_cases, args.kaggle_name)
    elif args.command == "profile":
        generate_profile(ml_use_cases, args.csv_filename)
    elif args.command == "edit":
        edit_dataset(ml_use_cases, args.csv_filename)
    elif args.command == "train":
        train_model(ml_use_cases, args.csv_filename, args.target_col, args.task_type)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()