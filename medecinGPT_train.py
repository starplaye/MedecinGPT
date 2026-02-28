"""
Module qui prépare les données et lance l'entraînement du modèle spécifié.

Ce module utilise la fonction main pour :
- Créer les jeux de d'entraînement et de validation
- Lancer l'entraînement du modèles avec les arguments passés

Auteurs :
    Gautheron Antoine <antoine.gautheron@etu.unistra.fr>
    Berton Ugo <ugo.berton@etu.unistra.fr>

Université : 
    Université de Strasbourg - Année 2025-2026
"""


# RAPPEL : 257*257 pixels est la dimension de nos images d'entraînement


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
    parser.add_argument("-fv", "--frac_val", type=int, default=20) # 20% du dataset à la validation par défaut
    parser.add_argument("-s", "--seed", type=int, default=-1) # -1 indique "pas de seed"
    parser.add_argument("-bs", "--batch_size", type=int, default=32) # batch size de 32 par défaut
    parser.add_argument("-lr", "--lrate", type=float, default=0.001) # learning rate de 0.001 par défaut
    parser.add_argument("-e", "--epochs", type=int, default=1000) # 1000 epochs par défaut
    parser.add_argument("-m", "--model", type=int, default=1) 
    parser.add_argument("-o", "--opti", type=int, default=1) 
    args = parser.parse_args()
    
    if args.seed != -1 :
        torch.manual_seed(args.seed)
    
    mean = [0.6525]     # Moyenne
    std = [0.4582]      # Ecart-type

    # Transformations que l'on va appliquer à chaque image du jeu d'entraînement
    # Redimensionnement, conversion en tenseur, normalisation, rotation, ...
    transform_train = transforms.Compose([
        transforms.Resize((257, 257)),
        transforms.ToTensor(),
        transforms.RandomRotation(360, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])
    
    # Transformations que l'on va appliquer à chaque image du jeu de validation
    # Redimensionnement, conversion en tenseur et normalisation
    transform_val = transforms.Compose([
        transforms.Resize((257, 257)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])

    # Appelle la fonction pour constituer les dataset d'entraînement et de validation
    train_dataset, val_dataset = split_kaggle_loader(
        img_dir="train",
        csv_file="train.csv",
        transform_train=transform_train,
        transform_val=transform_val,
        frac_val=args.frac_val,
        seed=args.seed,
        for_resnet=(args.model == 4),
    )

    # Charger les données d'entraînement et de validation
    train_load = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_load = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Choix du modèle utilisé pour l'entraînement Version 1 ou Version 2
    model = MedecinGPT(5) if args.model == 1 else MedecinGPT2(5) if args.model == 2 else MedecinGPT3(5) if args.model == 3 else make_resnet152(num_classes = 5, mode = 'full', pretrained = True)
    name = "MedecinGPT" if args.model == 1 else "MedecinGPT2" if args.model == 2 else "MedecinGPT3" if args.model == 3 else "Resnet152 & Custom Head"

    # Lancer initialisation des poids
    model.apply(init_weights_full if args.model < 4 else init_weights_linear)

    # Création de l'objet qui va gérer mon entraînement et ma validation
    trainer = MedecinTrainer(name, model, train_load, val_load, args.lrate, args.opti, None)
    
    # Lancer l'entraînement
    trainer.train(args.epochs)

if __name__ == "__main__":
    main()


# Commandes pour lancer entraînement  
#  python3 medecinGPT_train.py -s 484 -lr 0.0001 -e 100 -m 1
#  python3 medecinGPT_train.py -s 484 -lr 0.0001 -e 200 -m 2 
#  python3 medecinGPT_train.py -s 16844849 -lr 0.0001 -e 200 -m 3 -o 1
#  python3 medecinGPT_train.py -s 1684849 -lr 0.0002 -e 200 -m 3 -o 1 -bs 64
#  python3 medecinGPT_train.py -s 8995 -lr 0.0002 -e 200 -m 3 -o 1 -bs 16     => 96%
#  python3 medecinGPT_train.py -s 42668 -lr 0.0003 -e 200 -m 4 -o 1 -bs 32 -fv 1 => 98 %