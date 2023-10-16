import pkg_resources
import os

import torch


def download_spiga_model(dataset_name):
    MODEL_FILE_NAME = f"spiga_{dataset_name}.pt"
    WEIGHTS_PATH = pkg_resources.resource_filename(
        'spiga', os.path.join("models", "weights"))
    MODELS_URL = {'wflw': 'https://drive.google.com/uc?export=download&confirm=yes&id=1h0qA5ysKorpeDNRXe9oYkVcVe8UYyzP7',
                  '300wpublic': 'https://drive.google.com/uc?export=download&confirm=yes&id=1YrbScfMzrAAWMJQYgxdLZ9l57nmTdpQC',
                  '300wprivate': 'https://drive.google.com/uc?export=download&confirm=yes&id=1fYv-Ie7n14eTD0ROxJYcn6SXZY5QU9SM',
                  'merlrav': 'https://drive.google.com/uc?export=download&confirm=yes&id=1GKS1x0tpsTVivPZUk_yrSiMhwEAcAkg6',
                  'cofw68': 'https://drive.google.com/uc?export=download&confirm=yes&id=1fYv-Ie7n14eTD0ROxJYcn6SXZY5QU9SM'}

    torch.hub.load_state_dict_from_url(MODELS_URL[dataset_name],
                                       model_dir=WEIGHTS_PATH,
                                       map_location='cpu',
                                       file_name=MODEL_FILE_NAME)


def download_retinaface_model():
    MODEL_URL = "https://drive.google.com/uc?export=download&confirm=yes&id=1nxhtpdVLbmheUTwyIb733MrL53X4SQgQ"
    MODEL_FILE_NAME = "retinaface_mobile025.pth"
    torch.hub.load_state_dict_from_url(MODEL_URL,
                                       map_location='cpu',
                                       file_name=MODEL_FILE_NAME)


if __name__ == '__main__':
    DATASET_NAME = 'merlrav'
    download_retinaface_model()
    download_spiga_model(DATASET_NAME)
