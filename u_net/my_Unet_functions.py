import numpy as np
import torch
from torchvision import transforms
import random
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import jaccard_score
import torch.optim as optim
import torch.nn as nn
from codes.u_net.my_Scratch_dataset import ScratchDataset
from codes.u_net.my_Unet import UNet

def get_transforms():
    """
    Vytvorí a vráti transformácie pre spracovanie obrázkov pred vstupom do neurónovej siete.

    Transformácie:
    - Prevod obrázka na tenzor (ToTensor), ktorý prevedie obraz z formátu (H x W x C) s hodnotami 0–255
      na formát (C x H x W) s hodnotami 0.0–1.0.
    - Normalizácia tenzora pomocou priemerov (mean) a smerodajných odchýlok (std), ktoré sú bežne používané
      pre modely trénované na ImageNet datasete.

    Výstup:
        torchvision.transforms.Compose: Sekvencia transformácií aplikovateľná na vstupné obrázky.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def prepare_datasets(image_dir, mask_dir, transform, divide_dataset=0.1):
    """
    Pripraví a rozdelí dataset na trénovaciu, validačnú a testovaciu množinu.

    Načítava obrázky a ich príslušné masky pomocou triedy ScratchDataset, aplikuje zadané transformácie
    a rozdelí celý dataset v pomere určenom parametrom `divide_dataset`.

    Parametre:
        image_dir (str): Cesta k priečinku s obrázkami.
        mask_dir (str): Cesta k priečinku s maskami.
        transform (callable): Transformácie, ktoré sa majú aplikovať na obrázky a masky.
        divide_dataset (float, voliteľné): Percento dát, ktoré sa má použiť na validačný a testovací
                                           dataset (napr. 0.1 znamená 10 % pre každý z nich).
                                           Zvyšok (napr. 80 %) sa použije na trénovanie.

    Výstup:
        tuple: Trojica datasetov (train_dataset, val_dataset, test_dataset) typu torch.utils.data.Dataset,
               pripravených na použitie v DataLoaderi.
    """
    # Create the full dataset using the custom ScratchDataset class
    full_dataset = ScratchDataset(image_dir, mask_dir, transform=transform)

    # Calculate the total number of samples in the dataset
    number_total = len(full_dataset)
    number_val = int(divide_dataset * number_total)
    number_test = int(divide_dataset * number_total)
    number_train = number_total - number_val - number_test

    # Use clearly named size variables
    train_size = number_train
    val_size = number_val
    test_size = number_test
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size]
    )
    print(f"Rozdelenie datasetu na casti: {number_train} trenovacia, {number_val} validacna, {number_test} testovacia")
    return train_dataset, val_dataset, test_dataset

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """
    Vytvorí a vráti DataLoader objekty pre trénovaciu, validačnú a testovaciu množinu.

    Funkcia vytvorí tri DataLoader objekty, ktoré umožňujú efektívne načítanie dát
    počas trénovania a testovania modelu. Trénovací DataLoader bude mať zapnuté
    náhodné miešanie dát (shuffle=True), zatiaľ čo validačný a testovací DataLoader
    budú načítavať dáta bez miešania (shuffle=False).

    Parametre:
        train_dataset (torch.utils.data.Dataset): Trénovacia množina dát.
        val_dataset (torch.utils.data.Dataset): Validačná množina dát.
        test_dataset (torch.utils.data.Dataset): Testovacia množina dát.
        batch_size (int, voliteľné): Veľkosť dávky (batchu), ktorá sa načíta v každej iterácii.
                                      Predvolená hodnota je 32.

    Výstup:
        tuple: Trojica DataLoader objektov (train_dataloader, val_dataloader, test_dataloader),
               ktoré sú pripravené na použitie v trénovaní, validácii a testovaní modelu.
    """

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader

def dice_loss(pred, target, smooth=1.):
    """
    Vypočíta Diceovu stratu medzi predikciou a cieľom.

    Parametre:
        pred (Tensor): Predikcie modelu (výstup sigmoid funkcie).
        target (Tensor): Skutočné hodnoty cieľovej masky (ground truth).
        smooth (float, voliteľné): Hodnota pre hladké vylepšenie výpočtu (default = 1.0).

    Výstup:
        float: Diceova strata, ktorá meria podobnosť medzi predikciou a cieľom. Výsledok je 1 - Diceov index.

    Tento výpočet sa používa na vyhodnotenie presnosti segmentácie v úlohách, ako je segmentácia obrazov.
    """
    pred = pred.sigmoid()
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

def combined_loss(pred, target):
    """
    Vypočíta kombinovanú stratu skladajúcu sa z binárnej krížovej entropie a Diceovej straty.

    Parametre:
        pred (Tensor): Predikcie modelu (predikcia logits).
        target (Tensor): Skutočné hodnoty cieľovej masky (ground truth).

    Výstup:
        float: Kombinovaná strata, ktorá je súčtom binárnej krížovej entropie (BCE) a Diceovej straty.

    Táto funkcia sa používa na optimalizáciu modelov v úlohách segmentácie, kde sa využívajú obe metódy straty na zlepšenie výkonnosti modelu.
    """
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice

def initialize_model(n_classes=1, lr=1e-5):
    """
    Inicializuje model UNet, stratový funkciu, optimalizátor, a plánovač učenia.

    Parametre:
        n_classes (int, voliteľné): Počet tried na detekciu (default = 1).
        lr (float, voliteľné): Počiatočná hodnota učebnej rýchlosti (default = 1e-5).

    Výstup:
        tuple: Obsahuje model UNet, stratovú funkciu, optimalizátor, zariadenie (CPU alebo GPU) a plánovač učenia.

    Táto funkcia vytvára model UNet s daným počtom tried, nastavuje kombinovanú stratu, Adam optimalizátor, a plánovač učenia
    pomocou `ReduceLROnPlateau`, ktorý dynamicky upravuje učebnú rýchlosť na základe výkonu modelu.
    """

    # Use GPU if it is available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(n_classes)
    model.to(device)
    # Define the loss function
    criterion = combined_loss

    # Initialize Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Set up learning rate scheduler that reduces the learning rate
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
        min_lr=1e-6
    )
    return model, criterion, optimizer, device, lr_scheduler


def train_one_epoch(model, train_dataloader, criterion, optimizer, device):
    """
    Trénuje model počas jednej epochy.

    Táto funkcia vykonáva jeden krok trénovania modelu počas jednej epochy. Pre každý batch
    z trénovacej množiny sa vypočíta predpoveď, stratová funkcia, spätné šírenie (backpropagation),
    a optimalizátor sa aktualizuje na základe vypočítanej chyby.

    Parametre:
        model (torch.nn.Module): Model, ktorý sa trénuje.
        train_dataloader (torch.utils.data.DataLoader): DataLoader pre trénovaciu množinu,
                                                     ktorý poskytuje dávky obrázkov a štítkov.
        criterion (torch.nn.Module): Stratová funkcia, ktorá sa používa na výpočet chyby medzi
                                     predpoveďami modelu a skutočnými štítkami.
        optimizer (torch.optim.Optimizer): Optimalizátor, ktorý vykoná krok optimalizácie na
                                           základe gradientov.
        device (torch.device): Zariadenie (GPU alebo CPU), na ktorom bude model a dáta spracovávané.

    Výstup:
        float: Priemerná strata (loss) za jednu epochu.
    """
    model.train()
    model.to(device)
    total_loss = 0.0

    # Loop through batches of images and labels from the training dataloader
    for images, labels in train_dataloader:
        device = next(model.parameters()).device

        # Move images and labels to the device
        images = images.to(device)

        # Remove single-dimensional entries and convert labels to float
        labels = labels.to(device).squeeze(1).float()
        optimizer.zero_grad()
        outputs = model(images)

        # # Compute the loss between predictions and ground truth
        loss = criterion(outputs.squeeze(1), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_dataloader)

def calculate_accuracy(model, dataloader, device):
    """
    Vypočíta presnosť modelu na základe predikcií a skutočných hodnôt.

    Táto funkcia vypočíta presnosť modelu na základe porovnania predpovedí s reálnymi maskami.
    Predpovede sú získané pomocou sigmoidovej funkcie a sú zaokrúhlené na hodnoty 0 alebo 1.
    Presnosť je definovaná ako pomer správne predikovaných pixelov k celkovému počtu pixelov.

    Parametre:
        model (torch.nn.Module): Model, ktorého presnosť sa počíta.
        dataloader (torch.utils.data.DataLoader): DataLoader pre dataset, ktorý obsahuje obrázky a masky.
        device (torch.device): Zariadenie (GPU alebo CPU), na ktorom sa model a dáta spracovávajú.

    Výstup:
        float: Presnosť modelu, ktorá je vyjadrená ako pomer správne predikovaných pixelov k celkovému počtu pixelov.
    """
    model.eval()
    total_correct = 0
    total_pixels = 0

    # Disable gradient calculation to save memory and computation during evaluation
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            predicted_values = torch.sigmoid(outputs)

            # Threshold probabilities at 0.5 to get binary predictions (0 or 1)
            predicted_values = (predicted_values > 0.5).float()

            masks = masks.float()

            # Compare predictions with ground truth masks pixel-wise
            correct = (predicted_values == masks).float()
            total_correct += correct.sum().item()
            total_pixels += correct.numel()

    accuracy = total_correct / total_pixels
    return accuracy

def validate(model, dataloader, criterion, device):
    """
    Vyhodnotí výkon modelu na validačnej množine.

    Táto funkcia vykoná hodnotenie modelu počas jednej iterácie na validačných dátach. Pre každý batch
    v validačnej množine sa vypočíta predpoveď a strata, ale žiadne gradienty nie sú počítané (vďaka `torch.no_grad()`),
    čím sa šetrí výpočtový čas a pamäť.

    Parametre:
        model (torch.nn.Module): Model, ktorý sa vyhodnocuje.
        dataloader (torch.utils.data.DataLoader): DataLoader pre validačnú množinu, ktorý poskytuje dávky obrázkov a štítkov.
        criterion (torch.nn.Module): Stratová funkcia, ktorá sa používa na výpočet chyby medzi predpoveďami modelu a skutočnými štítkami.
        device (torch.device): Zariadenie (GPU alebo CPU), na ktorom bude model a dáta spracovávané.

    Výstup:
        float: Priemerná strata (loss) na validačnej množine.
    """
    # Set the model to evaluation mode
    model.eval()
    model.to(device)
    total_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            device = next(model.parameters()).device
            images = images.to(device)
            labels = labels.to(device).squeeze(1).float()

            outputs = model(images)

            # Calculate loss between model outputs and true labels
            loss = criterion(outputs.squeeze(1), labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_model(model, train_dataloader, val_dataloader, test_dataloader,
                criterion, optimizer, num_epochs, new_model_name, lr_scheduler,
                patience=5, min_delta=0.001, device=None):
    """
        Trénuje model pomocou tréningových a validačných dát, vykonáva
        early stopping na základe validačnej straty a na záver vyhodnotí
        model na testovacej množine.

        Parametre:
        - model: neurónová sieť, ktorá sa trénuje
        - train_dataloader: DataLoader pre tréningové dáta
        - val_dataloader: DataLoader pre validačné dáta
        - test_dataloader: DataLoader pre testovacie dáta
        - criterion: funkcia straty (loss function)
        - optimizer: optimalizátor (napr. Adam, SGD)
        - num_epochs: maximálny počet epoch trénovania
        - new_model_name: názov súboru na uloženie naučených parametrov modelu
        - lr_scheduler: plánovač učenia, ktorý mení learning rate podľa výkonu
        - patience: počet epoch bez zlepšenia pred ukončením (early stopping)
        - min_delta: minimálna zmena validacnej straty považovaná za zlepšenie
        - device: zariadenie (CPU alebo GPU) na vykonávanie výpočtov

        Výstup:
        - train_losses: zoznam tréningových strát pre každú epochu
        - val_losses: zoznam validačných strát pre každú epochu
        - train_accuracies: zoznam tréningových presností pre každú epochu
        - val_accuracies: zoznam validačných presností pre každú epochu
        - test_loss: strata na testovacej množine po skončení trénovania
        - test_accuracy: presnosť na testovacej množine po skončení trénovania
        """

    # All your original lists
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Variables for early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        train_losses.append(train_loss)

        train_accuracy = calculate_accuracy(model, train_dataloader, device)
        train_accuracies.append(train_accuracy)

        val_loss = validate(model, val_dataloader, criterion, device)
        val_losses.append(val_loss)

        val_accuracy = calculate_accuracy(model, val_dataloader, device)
        val_accuracies.append(val_accuracy)

        lr_scheduler.step(val_loss)
        print(f'Epocha [{epoch + 1}/{num_epochs}], '
              f'Trenovacia strata: {train_loss:.4f}, '
              f'Validacna strata: {val_loss:.4f}, '
              f'Trenovacia presnost: {train_accuracy:.4f}, '
              f'Validacna presnost: {val_accuracy:.4f}')

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f'EarlyStopping na epoche {epoch + 1}.')
            break

    # Evaluate on test set after training
    test_loss = validate(model, test_dataloader, criterion, device)
    test_accuracy = calculate_accuracy(model, test_dataloader, device)

    print(f'Testovacia strata: {test_loss:.4f}, Testovacia presnost: {test_accuracy:.4f}')

    # Save model
    torch.save(model.state_dict(), new_model_name)

    return train_losses, val_losses, train_accuracies, val_accuracies, test_loss, test_accuracy


def plot_and_save_confusion_matrix(cm, save_path="confusion_matrix.png"):
    """
    Zobrazí a uloží konfúznu maticu ako obrázok.

    Táto funkcia vykreslí konfúznu maticu vo forme tepelnej mapy pomocou knižnice Seaborn, kde hodnoty
    sú zobrazené v matici. Následne ukladá výsledný obrázok na zadanú cestu.

    Parametre:
        cm (numpy.ndarray alebo torch.Tensor): Konfúzna matica, ktorá bude zobrazená. Očakáva sa, že
                                              matica obsahuje počet správnych a nesprávnych predpovedí
                                              pre každú triedu.
        save_path (str, voliteľné): Cesta k súboru, kde bude konfúzna matica uložená. Predvolená hodnota
                                     je "confusion_matrix.png".

    Výstup:
        Žiadny. Funkcia uloží obrázok konfúznej matice na zadanú cestu.
    """

    # Vykreslenie matice
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Konfuzna matica')
    plt.xlabel('Predpovedane oznacenie')
    plt.ylabel('Pravdive oznacenie')
    plt.savefig(save_path)
    plt.close()

def compute_metrics(model, dataloader, device):
    """
    Vypočíta metriky pre hodnotenie modelu vrátane konfúznej matice, presnosti, odvolania, F1 skóre a IoU.

    Táto funkcia vyhodnocuje výkon modelu na zadanom DataLoaderi. Model je spustený v evaluačnom režime,
    predpovede sú získané pomocou sigmoid funkcie, a následne sa vypočítajú metriky ako presnosť, odvolanie,
    F1 skóre a IoU (Intersection over Union). Tieto metriky sa vypočítajú na celej sade dát a výstupom
    je konfúzna matica a hodnoty všetkých metrík.

    Parametre:
        model (torch.nn.Module): Model, ktorý sa používa na generovanie predpovedí.
        dataloader (torch.utils.data.DataLoader): DataLoader, ktorý poskytuje obrázky a skutočné masky pre hodnotenie.

    Výstup:
        tuple: Tupla obsahujúca:
            - cm (numpy.ndarray): Konfúzna matica z výsledkov predpovedí.
            - precision (float): Presnosť modelu na testovacích dátach.
            - recall (float): Odvolanie modelu na testovacích dátach.
            - f1 (float): F1 skóre modelu na testovacích dátach.
            - iou (float): Intersection over Union (IoU) modelu na testovacích dátach.
    """
    model.eval()
    all_preds = []
    all_labels = []
    model.to(device)
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(next(model.parameters()).device)
            masks = masks.to(next(model.parameters()).device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).long()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(masks.cpu().numpy())

    all_preds = np.concatenate([pred.flatten() for pred in all_preds])
    all_labels = np.concatenate([label.flatten() for label in all_labels])

    # Calculate values
    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    iou = jaccard_score(all_labels, all_preds, average='binary')

    # Print the final values
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"IoU: {iou:.4f}")
    return cm, precision, recall, f1, iou

def save_training_summary(model, num_epochs, train_losses, val_losses, train_accuracies, val_accuracies, accuracy, precision, recall, f1, iou,
                           save_path="training_summary.txt"):
    """
    Uloží súhrn tréningových výsledkov do textového súboru.

    Táto funkcia vytvorí textový súbor, ktorý obsahuje zhrnutie tréningu modelu vrátane počtu epoch,
    konečných hodnôt trénovacej a validačnej straty, trénovacej a validačnej presnosti, a metrik ako accuracy,
    precision, recall, F1-skóre a IoU.

    Parametre:
        num_epochs (int): Počet tréningových epoch.
        train_losses (list): Zoznam strat počas tréningu pre každú epóchu.
        val_losses (list): Zoznam strat počas validácie pre každú epóchu.
        train_accuracies (list): Zoznam presností na trénovacej sade pre každú epóchu.
        val_accuracies (list): Zoznam presností na validačnej sade pre každú epóchu.
        accuracy (float): Konečná presnosť modelu na testovacej sade.
        precision (float): Konečná presnosť modelu na testovacej sade.
        recall (float): Konečné odvolanie modelu na testovacej sade.
        f1 (float): Konečné F1 skóre modelu na testovacej sade.
        save_path (str, voliteľné): Cesta, kde bude uložený súbor so súhrnom. Predvolená hodnota je "training_summary.txt".

    Výstup:
        None: Funkcia ukladá súhrn tréningu do súboru na zadanú cestu.
    """
    model.eval()

    # Final values output
    summary = (
        f"Suhrn trenovania:\n"
        f"Pocet epoch: {num_epochs}\n"
        f"Finalna trenovacia strata: {train_losses[-1]:.4f}\n"
        f"Finalna validacna strata: {val_losses[-1]:.4f}\n"
        f"Finalna trenovacia presnost: {train_accuracies[-1]:.4f}\n"
        f"Finalna validacna presnost: {val_accuracies[-1]:.4f}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1-Score: {f1:.4f}\n"
        f"IoU: {iou:.4f}\n"
    )

    # Write into file
    with open(save_path, "w") as file:
        file.write(summary)

def plot_run_over_epochs(model, train_values, train_label, val_values, val_label, save_path, ylabel, title, test_values, test_label):
    """
    Vykreslí graf zobrazujúci hodnoty počas tréningových a validačných epoch.

    Táto funkcia vykreslí graf, ktorý zobrazuje hodnoty (napr. strata, presnosť) pre tréningovú a validačnú množinu
    počas tréningových epoch a uloží ho ako obrázok do určeného súboru.

    Parametre:
        train_values (list): Zoznam hodnôt pre tréningovú množinu (napr. strata alebo presnosť) počas epoch.
        train_label (str): Popis pre tréningovú krivku, ktorý sa zobrazuje v legendy.
        val_values (list): Zoznam hodnôt pre validačnú množinu (napr. strata alebo presnosť) počas epoch.
        val_label (str): Popis pre validačnú krivku, ktorý sa zobrazuje v legendy.
        save_path (str): Cesta, kde sa uloží vygenerovaný graf.
        ylabel (str): Názov osi y (napr. "Strata" alebo "Presnosť").
        title (str): Názov grafu (napr. "Tréningová a validačná strata počas epoch").

    Výstup:
        None: Funkcia uloží vygenerovaný graf ako obrázok na zadanú cestu.
    """

    # Plot training
    model.eval()
    epochs = range(1, len(train_values) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_values, label=train_label)
    plt.plot(epochs, val_values, label=val_label)
    plt.axhline(y=test_values, color='g', linestyle='--', label=test_label)
    plt.xlabel('Epochy')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    plt.savefig(save_path)
    plt.close()


def plot_bar_chart(train_label, train_losses, val_label, val_losses, test_label, test_loss, title, ylabel, save_path):
    """
    Vykreslí stĺpcový graf so stratami pre trénovaciu, validačnú a testovaciu množinu.

    Parametre:
    - train_label (str): Popis pre trénovaciu množinu.
    - train_losses (list): Zoznam hodnôt straty počas trénovania (použije sa posledná hodnota).
    - val_label (str): Popis pre validačnú množinu.
    - val_losses (list): Zoznam hodnôt straty počas validácie (použije sa posledná hodnota).
    - test_label (str): Popis pre testovaciu množinu.
    - test_loss (float): Hodnota straty na testovacej množine.
    - title (str): Názov grafu.
    - ylabel (str): Popis osi y (napr. „Strata“).
    - save_path (str): Cesta k súboru, kde sa má graf uložiť (napr. "graf.png").

    Graf sa zobrazí a následne uloží na zadané miesto.
    """
    labels = [train_label, val_label, test_label]
    values = [train_losses[-1], val_losses[-1], test_loss]

    # Plot bar chart
    plt.bar(labels, values, color=['blue', 'orange', 'green'])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.show()
    plt.close()



