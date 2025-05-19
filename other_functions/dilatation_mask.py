import cv2

def expand_mask(input_path, output_path, threshold = 127, kernel_size = (23, 23), iterations = 1):
    """
    Rozšíri binárnu masku pomocou morfologickej operácie dilatácie.

    Parametre:
        input_path (str): Cesta k vstupnému obrázku (v odtieňoch sivej).
        output_path (str): Cesta, kam sa uloží rozšírená maska.
        threshold (int): Prahová hodnota pre binarizáciu obrázka (defaultne 127).
        kernel_size (tuple): Veľkosť obdĺžnikového štruktúrovacieho prvku (defaultne (23, 23)).
        iterations (int): Počet iterácií dilatácie (defaultne 1).

    Výstup:
        FileNotFoundError: Ak sa nepodarí načítať vstupný obrázok.
    """


    mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=iterations)

    cv2.imwrite(output_path, dilated_mask)

def main():
    expand_mask('predicted_defect_img.png', 'expanded_mask.png')

if __name__ == '__main__':
    main()