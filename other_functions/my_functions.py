import os
import cv2
import random
import numpy as np
import albumentations as A
import json
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


def get_files(file_list, path_to_file):
    """
    Pridá obrázkové súbory s príponami '.png', '.jpg' alebo '.JPG' zo zadaného adresára
    do poskytnutého zoznamu.

    Parametre:
        file_list (list): Zoznam, do ktorého budú pridané obrázkové súbory.
        path_to_file (str): Cesta k adresáru, kde sa nachádzajú obrázkové súbory.

    Výstup:
        list: Aktualizovaný zoznam obsahujúci obrázkové súbory z adresára.
    """
    for file in os.listdir(path_to_file):
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.JPG'):
            file_list.append(file)
    return file_list

def shuffle_files(file_list, path_to_file):
    """
        Načíta všetky obrazové súbory (.png, .jpg, .JPG) zo zadaného priečinka,
        pridá ich do zoznamu a následne tento zoznam náhodne premieša.

        Parametre:
            file_list (list): Zoznam, do ktorého sa pridajú názvy nájdených súborov.
            path_to_file (str): Cesta k priečinku, kde sa majú hľadať obrazové súbory.

        Výstup:
            list: Premiešaný zoznam názvov obrazových súborov.
        """
    for file in os.listdir(path_to_file):
        if file.endswith('.png') or file.endswith('.jpg')or file.endswith('.JPG'):
            file_list.append(file)
    random.shuffle(file_list)
    return file_list

def resize_file(old_file, new_image_shape):
    """
        Zmení veľkosť vstupného obrázka tak, aby zodpovedal zadaným rozmerom.

        Parametre:
            old_file (numpy.ndarray): Pôvodný obrázok, ktorý sa má zmeniť.
            new_image_shape (tuple): Cieľové rozmery v tvare (výška, šírka, [kanály]).

        Výstup:
            numpy.ndarray: Obrázok so zmenenou veľkosťou podľa zadaných rozmerov.
    """
    height = new_image_shape[0]
    width = new_image_shape[1]

    old_file_resized = cv2.resize(old_file, (width, height))

    return old_file_resized


def blend_images(source_image, target_image, mask):
    """
       Spojí (prekryje) dva obrázky pomocou masky. Pixely zo zdrojového obrázka
       sa použijú tam, kde je maska nenulová, a z cieľového obrázka tam, kde je maska nulová.

       Parametre:
           source_image (numpy.ndarray): Zdrojový obrázok, z ktorého sa preberajú pixely podľa masky.
           target_image (numpy.ndarray): Cieľový obrázok, ktorý sa použije tam, kde maska nemá hodnotu.
           mask (numpy.ndarray): Maska (čiernobiely obrázok), ktorá určuje, ktoré časti použiť zo zdrojového obrázka.

       Výstup:
           numpy.ndarray: Výsledný obrázok získaný kombináciou zdrojového a cieľového obrázka podľa masky.
    """
    masked_source = cv2.bitwise_and(source_image, source_image, mask=mask)
    blended_result = masked_source.copy()
    blended_result[mask == 0] = target_image[mask == 0]

    return blended_result

def rotate_image_and_mask(image, mask, angle):
    """
        Funkcia, ktorá otočí obrázok a masku o zadaný uhol okolo stredu obrázku.

        Parametre:
        image (numpy.ndarray): Obrázok, ktorý sa má otočiť.
        mask (numpy.ndarray): Maska, ktorá sa má otočiť.
        angle (float): Uhol rotácie v stupňoch (kladné hodnoty otáčajú obrázok proti smeru hodinových ručičiek).

        Návratové hodnoty:
        rotated_image (numpy.ndarray): Otočený obrázok.
        rotated_mask (numpy.ndarray): Otočená maska.
    """
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]))
    return rotated_image, rotated_mask


def augment_image(transform, image_path, save_dir):
    """
    Aplikuje transformáciu na obrázok a uloží výsledný obrázok do určeného adresára.

    Parametre:
        transform (callable): Funkcia alebo transformácia, ktorá sa aplikuje na obrázok.
        image_path (str): Cesta k pôvodnému obrázku, ktorý sa má upravit.
        save_dir (str): Cesta k adresáru, kam sa má uložiť upravený obrázok.

    Uloží upravený obrázok v novom názve súboru, ktorý obsahuje predponu "aug5_" a pôvodný názov obrázku.
    """
    image = cv2.imread(image_path)

    # Augment image
    augmented = transform(image=image)
    augmented_image = augmented['image']

    filename = os.path.basename(image_path)
    new_filename = os.path.join(save_dir, "aug5_" + filename)
    cv2.imwrite(new_filename, augmented_image)
    print(f"Uloz novy obrazok ako: {new_filename}")

def remove_images_without_masks(images_folder, masks_folder):
    """
        Odstráni obrázky z priečinka `images_folder`, ktoré nemajú zodpovedajúcu masku
        v priečinku `masks_folder`.

        Predpokladá sa, že masky aj obrázky majú rovnaký názov súboru (bez prípony).
        Pre každý obrázok sa skontroluje, či existuje maska s rovnakým názvom.
        Ak sa nenájde, obrázok sa odstráni.

        Args:
            images_folder (str): Cesta k priečinku s obrázkami.
            masks_folder (str): Cesta k priečinku s maskami.
    """
    masks = set()
    # Go through files in mask folder
    for filename in os.listdir(masks_folder):
        name, _ = os.path.splitext(filename)
        masks.add(name)

    # Go through files in image folder
    for image_file in os.listdir(images_folder):
        image_name, _ = os.path.splitext(image_file)
        if image_name not in masks:
            os.remove(os.path.join(images_folder, image_file))
            print(f"Vymazane: {image_file}")


def find_image_path(folder, filename, extensions):
    """
        Nájde cestu k obrázku so zadaným názvom súboru bez ohľadu na príponu.

        Parametre:
        - folder (str): Cesta k priečinku, kde sa hľadá obrázok.
        - filename (str): Názov súboru, vrátane alebo bez prípony.
        - extensions (list): Zoznam prípon (napr. ['.jpg', '.png']) na kontrolu.

        Výstup:
        - str alebo None: Cesta k nájdenému obrázku alebo None, ak sa nenašiel.
    """
    name_without_extension = os.path.splitext(filename)[0]
    for extension in extensions:
        path = os.path.join(folder, f"{name_without_extension}{extension}")
        if os.path.isfile(path):
            return path
    return None

def create_mask_from_annotations(annotations, height, width):
    """
        Vytvorí masku z COCO anotácií.

        Parametre:
        - annotations (list): Zoznam anotácií z COCO formátu.
        - height (int): Výška výstupnej masky.
        - width (int): Šírka výstupnej masky.

        Výstup:
        - np.ndarray: Binárna maska (0 a 1), kde objekty sú označené hodnotou 1.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # Go through each annotation
    for ann in annotations:
        segmentation = ann.get('segmentation')
        if not segmentation:
            continue

        if isinstance(segmentation, list):
            for poly in segmentation:
                poly_array = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [poly_array], 1)
        else:
            rle = maskUtils.frPyObjects(segmentation, height, width)
            rle_mask = maskUtils.decode(rle)
            mask = np.maximum(mask, rle_mask)

    return mask

def process_all_images(coco, image_dir, extensions):
    """
        Spracuje všetky obrázky z COCO anotácií: načíta ich, vytvorí masky a uloží ich.

        Parametre:
        - coco: Objekt COCO anotácií (napr. z pycocotools).
        - image_dir (str): Cesta k priečinku s obrázkami.
        - extensions (list): Zoznam podporovaných prípon obrázkov.

        Výstup:
        - Uloží masky do aktuálneho priečinka so zodpovedajúcim názvom súboru.
        - Vypíše upozornenie, ak sa niektorý obrázok nenašiel.
    """
    image_ids = coco.getImgIds()

    # Loop over each image ID
    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        img_path = find_image_path(image_dir, img_filename, extensions)
        if img_path is None:
            print(f"Obrazok {img_filename} nie je podporovany.")
            continue

        # Get all annotation IDs for the current image and load them
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        mask = create_mask_from_annotations(anns, img_info['height'], img_info['width'])
        mask_filename = img_filename.replace('.jpg', '.jpg').replace('.png', '.png').replace('.jfif', '.png')
        cv2.imwrite(mask_filename, mask * 255)
