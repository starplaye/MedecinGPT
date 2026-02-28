"""
Module de chargement de données, modèles et entraînement pour le projet de classification d’images médicales.

Ce fichier définit trois classes majeures :
- La classe 'KaggleLoader' qui est un Dataset PyTorch personnalisé pour la lecture des données Kaggle
- La classe 'KaggleTestLoader' qui est un Dataset PyTorch personnalisé pour la lecture des données tests de Kaggle
- La classe 'MedecinGPT' qui construit le réseau de neurones principal
- La classe 'MedecinGPT2' qui construit le réseau de neurones secondaire (de test)

Auteurs :
    Gautheron Antoine <antoine.gautheron@etu.unistra.fr>
    Berton Ugo <ugo.berton@etu.unistra.fr>

Université : 
    Université de Strasbourg - Année 2025-2026
"""


# RAPPEL : 257*257 pixels est la dimension de nos images dans notre dataset


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
from torchvision.models import resnet152, ResNet152_Weights, resnet101, ResNet101_Weights, resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import csv

import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================
#        DATASETS
# =========================

class KaggleLoader(Dataset):

    """
    Dataset PyTorch pour les données d'entraînement et de validation

    Charge des images à partir d'un dossier et associe chaque image à un label.
    """
       
    def __init__(self, img_dir, images, labels, transform=None, for_resnet=False, ext=".png"):

        """
        Docstring for __init__
        
        :param self: Description
        :param img_dir: Dossier contenant les images
        :param images: Liste des identifiants d'images (les images ont des noms avec des N°, c'est ici une liste des numéros pour reconstruire les chemins plus tard)
        :param labels: Tenseur des labels associés
        :param transform: Transformation Torch optionnelle à appliquer aux images (ToTensor(), Normalize(), ...)
        :param ext: Extension des fichiers (ici des images)
        """

        self.img_dir = img_dir
        self.images = images
        self.labels = labels
        self.transform = transform
        self.ext = ext
        self.for_resnet = for_resnet

    def __len__(self):
        """
        Retourne le nombre d'échantillons du dataset
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        :param idx: Indice de l'image que l'on veut obtenir
        Retourne une image et son label associé dans le jeu d'entraînement
        """
        img_name =  self.images[idx]
        label = self.labels[idx]

        img_path = os.path.join(self.img_dir, (img_name + self.ext))
        image = Image.open(img_path).convert('L') if not self.for_resnet else Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, label
  
    
class KaggleTestLoader(Dataset):
    """
    Dataset Pytorch pour les données de test (sans label => on doit les prédire)
    """
    def __init__(self, img_dir, images, transform=None, for_resnet=False, ext=".png"):
        """
        :param img_dir: Répertoire pour les images de test
        :param images: Description
        :param ext: Extension des fichiers (ici des images)
        """
        self.img_dir = img_dir
        self.images = images
        self.transform = transform
        self.ext = ext
        self.for_resnet = for_resnet

    def __len__(self):
        """
        Retourne le nombre d'échantillons du dataset
        """
        return len(self.images)

    def __getitem__(self, idx):
        """        
        :param idx: Indice de l'image du jeu de test que l'on veut obtenir

        Retourne le nom de l'image voulue et l'image elle-même
        """
        img_name =  self.images[idx]
        
        # Reconstruction du chemin relatif vers l'image sélectionné
        img_path = os.path.join(self.img_dir, (img_name + self.ext))
        image = Image.open(img_path).convert('L') if not self.for_resnet else Image.open(img_path).convert('RGB')

        # Si on a passé une transformation Torch, on l'applique à l'image obtenue
        if self.transform:
            image = self.transform(image)
        
        return img_name, image
    



# =========================
#        MODELES
# =========================



class MedecinGPT(nn.Module):
    
    """
    Réseau de neurones convolutif MedecinGPT version 1
    """

    def __init__(self, num_classes):
        """
        Docstring for __init__
        
        :param num_classes: Nombre de classes possibles à la sorties (dans notre contexte c'est 5)
        """
        super().__init__()

        # Bloc initial de deux couches de convolution pour extraire les features des images
        # Chaque couche a une fonction d'activation ReLU
        self.down = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Deuxième bloc de convolution et d'upsampling
        self.up = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        
        # Bloc final de convolution et de réduction spatiale
        self.f_block = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, kernel_size=3, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Partie finale avec les 6 couches Fully-connected qui aboutissent au vecteur de num_classes probabilités 
        self.final = nn.Sequential(
            nn.Linear(9216, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """
        Réalise la passe avant du réseau pour un batch d'images.
        
        :param x: Tensor d'entrée de forme (batch_size, 1, H, W) représentant un batch d'images en grayscale.

        :return: Tensor de sortie de forme (batch_size, num_classes)
             contenant les logits pour chaque classe pour chaque image.
             Chaque ligne correspond à une image et chaque colonne à une classe.

        """
        d = self.down(x)
        
        u = self.up(d)
        
        if u.shape[2:] != d.shape[2:]:
            d = d[:, :, :u.shape[2], :u.shape[3]]
        cc = torch.cat([u, d], dim=1)
        
        f = self.f_block(cc)
 
        r = self.final(f.flatten(1))          

        return r
   
    
class MedecinGPT2(nn.Module):

    """
    Réseau de neurones convolutif MedecinGPT version 2
    => On a le choix entre lui et le premier pour comparer les performances
    """
    def __init__(self, num_classes):
        """      
        :param num_classes: Nombre de classes possibles à la sorties (dans notre contexte c'est 5)
        """
        super().__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=0),
            nn.ReLU(),
        )
        
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, padding=0),
            nn.ReLU(),
        )
        
        self.up1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        
        self.f_block = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.final = nn.Sequential(
            nn.Linear(7056, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )


    def forward(self, x):
        """
        Réalise la passe avant du réseau pour un batch d'images.
        
        :param x: Tensor d'entrée de forme (batch_size, 1, H, W) représentant un batch d'images en grayscale.

        :return: Tensor de sortie de forme (batch_size, num_classes)
             contenant les logits pour chaque classe pour chaque image.
             Chaque ligne correspond à une image et chaque colonne à une classe.

        """
        d1 = self.down1(x)
        
        d2 = self.down2(d1)
        
        u1 = self.up1(d2)
        
        if u1.shape[2:] != d2.shape[2:]:
            d2 = d2[:, :, :u1.shape[2], :u1.shape[3]]
        cc1 = torch.cat([u1, d2], dim=1)
        
        u2 = self.up2(cc1)
        
        if u2.shape[2:] != d1.shape[2:]:
            d1 = d1[:, :, :u2.shape[2], :u2.shape[3]]
        cc2 = torch.cat([u2, d1], dim=1)
        
        f = self.f_block(cc2)
 
        r = self.final(f.flatten(1))          

        return r
    
class MedecinGPT3(nn.Module):
    
    """
    Réseau de neurones convolutif MedecinGPT version 1
    """

    def __init__(self, num_classes):
        """
        Docstring for __init__
        
        :param num_classes: Nombre de classes possibles à la sorties (dans notre contexte c'est 5)
        """
        super().__init__()

        """
        
        => 8 Convolutions succésives couplé d'un Batch-Norm et d'un MaxPool toutes les deux convolutions 

        """
        self.down = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2)
        )
        
        """
        
        => 8 couches linéaires pour une meilleure précision et des dropouts pour une meilleur généralisation
        
        """
        self.final = nn.Sequential(
            nn.Linear(73728, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(4096, 2048),
            nn.Dropout(0.3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(1024, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(512, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """
        Réalise la passe avant du réseau pour un batch d'images.
        
        :param x: Tensor d'entrée de forme (batch_size, 1, H, W) représentant un batch d'images en grayscale.

        :return: Tensor de sortie de forme (batch_size, num_classes)
             contenant les logits pour chaque classe pour chaque image.
             Chaque ligne correspond à une image et chaque colonne à une classe.

        """
        d = self.down(x)
 
        r = self.final(d.flatten(1))          

        return r
    

def make_resnet152(num_classes, mode='full', pretrained=True):
    """
    Créé un réseau de neurone basé sur Resnet152
        
    :param num_classes: Nombre de classes possibles à la sorties (dans notre contexte c'est 5)
    :param mode: Quelle partie du réseau ne sera pas freeze pour le train
    :param pretrained: Si l'on prend juste la structure du Resnet152 ou si l'on prend les poids pré-entrainés avec.

    :return: Tensor de sortie de forme (batch_size, num_classes)
            contenant les logits pour chaque classe pour chaque image.
            Chaque ligne correspond à une image et chaque colonne à une classe.
    """

    if not pretrained:
        m = resnet152()  # no pre-trained weights (random weights)
    else: 
        m = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1) # pre-trained weights on ImageNet-1K
    in_feats = m.fc.in_features

    print("[resnet 152] nombre de feats :",in_feats)

    # m.fc = nn.Sequential(
    #         nn.Linear(in_feats, 4096),
    #         nn.Dropout(0.5),
    #         nn.ReLU(),
    #         nn.Linear(4096, 2048),
    #         nn.Dropout(0.3),
    #         nn.ReLU(),
    #         nn.Linear(2048, 1024),
    #         nn.ReLU(),
    #         nn.Linear(1024, 512),
    #         nn.Dropout(0.4),
    #         nn.ReLU(),
    #         nn.Linear(512, 128),
    #         nn.Dropout(0.5),
    #         nn.ReLU(),
    #         nn.Linear(128, 64),
    #         nn.ReLU(),
    #         nn.Linear(64, 32),
    #         nn.ReLU(),
    #         nn.Linear(32, num_classes)
    #     )

    m.fc = nn.Sequential(
            nn.Linear(in_feats, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    # Freeze all parameters initially
    for p in m.parameters(): p.requires_grad=False

    # Unfreeze the final layer
    for p in m.fc.parameters(): p.requires_grad=True

    # Unfreeze additional layers based on the specified mode
    if mode=='last_block':
        for p in m.layer4.parameters(): p.requires_grad=True
    if mode=='full':
        for p in m.parameters(): p.requires_grad=True

    return m

def make_resnet101(num_classes, mode='full', pretrained=True):
    """
    Créé un réseau de neurone basé sur Resnet152
        
    :param num_classes: Nombre de classes possibles à la sorties (dans notre contexte c'est 5)
    :param mode: Quelle partie du réseau ne sera pas freeze pour le train
    :param pretrained: Si l'on prend juste la structure du Resnet152 ou si l'on prend les poids pré-entrainés avec.

    :return: Tensor de sortie de forme (batch_size, num_classes)
            contenant les logits pour chaque classe pour chaque image.
            Chaque ligne correspond à une image et chaque colonne à une classe.
    """

    if not pretrained:
        m = resnet101()  # no pre-trained weights (random weights)
    else: 
        m = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1) # pre-trained weights on ImageNet-1K
    in_feats = m.fc.in_features

    print("[resnet 101] nombre de feats :",in_feats)

    m.fc = nn.Sequential(
            nn.Linear(in_feats, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    # Freeze all parameters initially
    for p in m.parameters(): p.requires_grad=False

    # Unfreeze the final layer
    for p in m.fc.parameters(): p.requires_grad=True

    # Unfreeze additional layers based on the specified mode
    if mode=='last_block':
        for p in m.layer4.parameters(): p.requires_grad=True
    if mode=='full':
        for p in m.parameters(): p.requires_grad=True

    return m
    
# ==========================
#    MATRICE DE CONFUSION
# ==========================

def confusion_matrix_sklearn(
    y_true,
    y_pred,
    labels=None,
    normalize=None,   # None, 'true', 'pred', 'all'
    title="Matrice de confusion",
    cmap="Blues",
    save_path=None,
    show=True
):
    """
    Affiche et sauvegarde une matrice de confusion avec sklearn.

    Parameters
    ----------
    y_true : array-like
        Vraies étiquettes
    y_pred : array-like
        Étiquettes prédites
    labels : list, optionnel
        Ordre des classes
    normalize : {'true', 'pred', 'all', None}
        Normalisation sklearn
    title : str
        Titre de la figure
    cmap : str
        Colormap matplotlib
    save_path : str, optionnel
        Chemin pour sauvegarder l'image
    show : bool
        Afficher la figure
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize=normalize
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(
        ax=ax,
        cmap=cmap,
        colorbar=True,
        values_format=".4f" if normalize else "d"
    )

    ax.set_title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return cm


# =========================
#        ENTRAINEMENT
# =========================

  
def init_weights_full(m):
    '''
    Initialisation des poids du modèle 
    '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)

def init_weights_linear(m):
    '''
    Initialisation des poids du modèle 
    '''
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    
class MedecinTrainer :
    """
    Gère l'entraînement, la validation et l'inférence des modèles PyTorch.

    Cette classe encapsule :
    - l'entraînement sur un dataset d'entraînement
    - la validation sur un dataset de validation
    - l'enregistrement du meilleur modèle
    - l'évaluation sur un dataset de test et l'export des prédictions
    """
   
    def __init__(self, name, model, data_train, data_val, lr, optimizer, data_test, device="cuda"):
        self.name = name
        self.model = model.to(device=device)
        self.data_train = data_train
        self.data_val = data_val
        self.lr = lr
        self.data_test = data_test
        self.device = device
        
        # Fonction de perte Cross-Entropy : adaptée à la classification multi-classes
        # L = - Σ t_i * log(p_i) 
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4) #if optimizer == 1 else optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        self.val_loss_array = np.array([])
        self.train_loss_array = np.array([])
        self.val_acc_array = np.array([])
        self.train_acc_array = np.array([])
        
    
    def _train_one_epoch(self, epoch_index):
        """
        Entraîne le modèle pour une époque sur le dataset d'entraînement.

        :param epoch_index: numéro de l'époque (non utilisé dans le calcul, mais utile pour debug/log)
        :return: tuple (loss moyenne, accuracy moyenne) sur l'ensemble du dataset
        :rtype: (float, float)
        """

        self.model.train()
        running_loss = 0.
        running_acc = 0.
        total = 0

        for i, (inputs, labels) in enumerate(self.data_train):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.loss_fn(outputs, labels)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item() 
            
            preds = torch.argmax(outputs, dim=1)
            running_acc += (preds == labels).sum().item()
            total += labels.size(0)

        return (running_loss/len(self.data_train)), (running_acc/total)
    
    
    def train(self, epochs) :
        """
        Entraîne le modèle sur plusieurs époques et valide après chaque époque.

        Enregistre le modèle ayant obtenu la meilleure accuracy de validation.

        :param epochs: nombre d'époques à entraîner
        """

         # Timestamp pour nommer le modèle sauvegardé
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')    # Format YYYYMMDD_HHMMSS

        best_vacc = 0.955 # pour limiter l'enregistrement des models

        for epoch in range(epochs):

            start_time = datetime.now()
            
            # Entraînement sur une époque
            avg_loss, avg_acc = self._train_one_epoch(epoch)
            self.train_acc_array = np.append(self.train_acc_array, avg_acc)
            self.train_loss_array = np.append(self.train_loss_array, avg_loss)

            # Validation
            running_vloss = 0.0
            running_vacc = 0.0
            vtotal = 0

            self.model.eval()

            with torch.no_grad():
                for i, (vinputs, vlabels) in enumerate(self.data_val):
                    vinputs, vlabels = vinputs.to(self.device), vlabels.to(self.device)
                    voutputs = self.model(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss.item()
                    
                    vpreds = torch.argmax(voutputs, dim=1)
                    running_vacc += (vpreds == vlabels).sum().item()
                    vtotal += vlabels.size(0)
                    

            avg_vloss = running_vloss / (i + 1)
            avg_vacc = running_vacc / vtotal
            self.val_acc_array = np.append(self.val_acc_array, avg_vacc)
            self.val_loss_array = np.append(self.val_loss_array, avg_vloss)
            print('EPOCH {}: LOSS train {} valid {} / ACCURACY train {} valid {} / time {} '.format(epoch, avg_loss, avg_vloss, avg_acc, avg_vacc, (datetime.now()-start_time).total_seconds()))

            # Sauvergarder le meilleur modèle
            if avg_vacc > best_vacc:
                best_vacc = avg_vacc
                model_path = 'model/model_{}_{}.pt'.format(timestamp, epoch)
                torch.save(self.model.state_dict(), model_path)

        # Génération des graphiques
        plt.title('Accuracy {}'.format(self.name))
        plt.plot(np.array(range(epochs)), self.train_acc_array)
        plt.plot(np.array(range(epochs)), self.val_acc_array)
        plt.savefig('Accuracy{}_{}.png'.format(timestamp,self.name))
        plt.show()

        plt.title('Loss {}'.format(self.name))
        plt.plot(np.array(range(epochs)), self.train_loss_array)
        plt.plot(np.array(range(epochs)), self.val_loss_array)
        plt.savefig('Accuracy{}_{}.png'.format(timestamp,self.name))
        plt.show()
                
    
    def eval(self, file, mode="csv"):
        """
        Évalue le modèle sur le dataset de test et écrit les prédictions dans un CSV.

        :param file: chemin du fichier CSV de sortie
        """
        self.model.eval()

        with torch.no_grad():
                
            if mode == "csv" :
                # On ouvre le CSV en mode écriture
                with open(file, 'w', newline='') as csvfile:
                    
                    writer = csv.writer(csvfile, delimiter=',')
                    # On écrit l'entête de notre csv
                    writer.writerow(["ID", "Category"])
                    
                    # On écrit les lignes une par une pour les résultats du test
                    for i, (img_name, tinputs) in enumerate(self.data_test):
                        tinputs = tinputs.to(self.device)
                        toutputs = self.model(tinputs)
                        tpreds = torch.argmax(toutputs, dim=1)
                        
                        writer.writerow([img_name[0], tpreds.item()])
            else :
                true_label = np.array([])
                pred_label = np.array([])
                for i, (tinputs, tlabels) in enumerate(self.data_train):
                    tinputs = tinputs.to(self.device)
                    toutputs = self.model(tinputs)
                    tpreds = torch.argmax(toutputs, dim=1)
                    
                    true_label = np.append(true_label, tlabels[0].item())
                    pred_label = np.append(pred_label, tpreds.cpu()[0].item())

                confusion_matrix_sklearn(true_label,pred_label,title=f"Matrice de confusion de {self.name}", normalize=None, save_path=f"matrix_{self.name}.png")
                    
                    
                    

            
def split_kaggle_loader(img_dir, csv_file, transform_train, transform_val, frac_val, seed, for_resnet) :
    """
    Fonction qui va construire mes jeux d'entraînement et de validation
    
    :param img_dir: Répertoire où sont les images de test et de validation
    :param csv_file: Chemin vers le csv avec les identifiants des images et les classes associées
    :param transform_train: Transformation Pytorch à appliquer aux images d'entraînement
    :param transform_val: Transformation Pytorch à appliquer aux images de validation
    :param frac_val: Fraction du dataset pour le jeu de validation (généralement 20%)
    :param seed: Seed choisie pour la séparation à des fins de reproductibilité 

    :return: deux objets qui mes dataset d'entraînement et de validation
    """
    csv_data = pd.read_csv(csv_file)
        
    image_ids = csv_data.iloc[:, 0].astype(str).tolist()
    labels = torch.tensor(csv_data.iloc[:, 1].values, dtype=torch.long)
        
    generator = torch.Generator().manual_seed(seed) if seed != -1 else torch.Generator()
    indices = torch.randperm(len(csv_data), generator=generator)
    
    val_size = math.floor(len(indices) * frac_val / 100)
    
    train_idx = indices[val_size:].tolist()
    val_idx = indices[:val_size].tolist()

    image_train = [image_ids[i] for i in train_idx]
    label_train = labels[train_idx]

    image_val = [image_ids[i] for i in val_idx]
    label_val = labels[val_idx]
    
    loader_train = KaggleLoader(img_dir, image_train, label_train, transform=transform_train, for_resnet=for_resnet)
    loader_val = KaggleLoader(img_dir, image_val, label_val, transform=transform_val, for_resnet=for_resnet)
    
    return loader_train, loader_val

def make_kaggle_loader(img_dir, csv_file, transform, for_resnet) :
    csv_data = pd.read_csv(csv_file)
        
    image_ids = csv_data.iloc[:, 0].astype(str).tolist()
    labels = torch.tensor(csv_data.iloc[:, 1].values, dtype=torch.long)
        
    loader = KaggleLoader(img_dir, image_ids, labels, transform=transform, for_resnet=for_resnet)
    
    return loader
    

