"""
Module qui prépare les données de test, lance les tests sur le modèle et génère le csv avec les résultats.

Ce module utilise la fonction main pour :
- Créer le jeu de test du modèle
- Lancer les tests sur le modèle entraîné
- Compiler les résultats obtenus dans un CSV nommé resultats.csv

Auteurs :
    Gautheron Antoine <antoine.gautheron@etu.unistra.fr>
    Berton Ugo <ugo.berton@etu.unistra.fr>

Université : 
    Université de Strasbourg - Année 2025-2026
"""


# RAPPEL : 257*257 pixels est la dimension de nos images de test


# Importation des librairies nécessaires aux différentes fonctionnalités du projet : torch, csv, os, pandas, ...
from datetime import datetime
import argparse
import os
import torch
import math
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from medecinGPT import *


def main():
    # Gestion des différents arguments possibles
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="")
    parser.add_argument("-mx", "--matrix", type=bool, default=False)
    parser.add_argument("-v", "--version", type=int, default=1) # Argument qui gère la version du modèle utilisé MedecinGPT v1 ou v2
    args = parser.parse_args()
     
    mean = [0.6525]     # Moyenne
    std = [0.4582]      # Ecart-type
    
    # Transformations que l'on va appliquer à chaque image avant de les tester
    # Redimensionnement, conversion en tenseur et normalisation
    transform_test = transforms.Compose([
        transforms.Resize((257, 257)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])

    path = "test" if not args.matrix else "train"
    
    # On crée la liste des identifiants des photos sans leur extension pour ensuite la passer à la fonction de création de dataset
    image_ids = [f.split(".")[0] for f in os.listdir(path)]

    # Crée le dataset pour le jeu de test en appliquant les transformations de transform_test sur toutes les images dans images_ids
    if not args.matrix :
        test_dataset = KaggleTestLoader("test", image_ids, transform_test, for_resnet=(args.version == 4))
    else :
        test_dataset = make_kaggle_loader(
            img_dir="train",
            csv_file="train.csv",
            transform=transform_test,
            for_resnet=(args.version == 4),
        )

    # Charger les données de test
    test_load = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    # Choix du modèle utilisé pour l'évaluation Version 1 ou Version 2
    model = MedecinGPT(5) if args.version == 1 else MedecinGPT2(5) if args.version == 2 else MedecinGPT3(5) if args.version == 3 else make_resnet152(num_classes = 5, mode = 'full', pretrained = True)
    name = "MedecinGPT" if args.version == 1 else "MedecinGPT2" if args.version == 2 else "MedecinGPT3" if args.version == 3 else "Resnet152 & Custom Head"

    # Charger les bons poids dans mon modèle depuis un dictionnaire pour ensuite le tester sur le jeu de test
    model.load_state_dict(torch.load(f"model/{args.model}.pt"))

    # Création de l'objet qui va gérer mon test avec les paramètres nécessaires
    trainer = MedecinTrainer(name, model, test_load, test_load, 0, 1, test_load)
    
    # Lancer la construction du csv avec les résultats
    trainer.eval("resulats.csv", mode=("csv" if not args.matrix else "matrix"))


if __name__ == "__main__":
    main()


# Commande pour lancer évaluation
# python3 medecinGPT_eval.py -m model_20260104_201451_76
# python3 medecinGPT_eval.py -m model_20260114_171839_106 -v 3