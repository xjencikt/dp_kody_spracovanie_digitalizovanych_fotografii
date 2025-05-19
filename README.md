# Detekcia defektov na digitalizovaných fotografiách

Tento projekt sa zaoberá **detekciou poškodených častí** na digitalizovaných fotografiách pomocou vlastného modelu postaveného na **U-Net architektúre**.  
Cieľom je automaticky **označiť defekty** a vytvoriť z nich **masky**, čím sa eliminujú potreby manuálneho označovania poškodení. Projekt beží na verzii python 3.10.
---

## 🛠️ Inštalácia

Na inštaláciu všetkých potrebných knižníc použite príkaz:

```bash
pip install -r requirements.txt
```

## 📁 Štruktúra kódu
| Súbor                          | Popis                                                               |
|--------------------------------|---------------------------------------------------------------------|
| `coco_create_mask.py`          | Vygeneruje masky anotácií zo `.json` súboru vo formáte COCO.        |
| `creating_augmented_copies.py` | Augmentuje pôvodné fotografie.                                      |
| `dilatation_mask.py`           | Rozširuje masky pomocou morfologickej dilatácie.                    |
| `extract_defects.py`           | Extrahuje poškodené časti z obrázkov.                               |
| `first_enhance_defect.py`      | Skúšobné úpravy obrázkov, augmentácie a detekcia častí tváre.       |
| `my_functions.py`              | Obsahuje pomocné funkcie na spracovanie súborov.                    |
| `my_Scratch_dataset.py`        | Definícia vlastnej `ScratchDataset` triedy.                         |
| `my_Unet.py`                   | Implementácia U-Net architektúry.                                   |
| `my_Unet_functions.py`         | Funkcie na trénovanie a vyhodnotenie U-Net modelu.                  |
| `old_photos_new_defect.py`     | Spája pôvodné fotografie s defektmi do nových syntetických vzoriek. |
| `predict_defect.py`            | Predikcia defektov pomocou natrénovaného modelu.                    |
| `remove_not_labeled_images.py` | Odstraňuje obrázky bez masiek z datasetu.                           |
| `train_my_Unet.py`             | Spustenie trénovania vlastnej U-Net architektúry.                   |

V prílohe je možné nájsť najlepší natrénovaný model final_scratch_model.pth.