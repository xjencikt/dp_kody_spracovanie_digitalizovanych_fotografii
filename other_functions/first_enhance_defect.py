from codes.other_functions.my_functions import *

#***
#Toto boli prvé pokusy o augmentácie fotiek a defektov
#***

def first_augmentation_image(image, contrast=0.5, brightness=50):
    """
        Aplikuje prvotnú augmentáciu na obrázok, ktorá zahŕňa prevod na odtiene šedej, zlepšenie kontrastu,
        úpravu jasu a rozmazanie.

        Parametre:
        - image: vstupný obrázok vo formáte numpy array.
        - contrast: koeficient kontrastu (predvolená hodnota: 0.5).
        - brightness: hodnota jasu, ktorá sa použije pri úprave (predvolená hodnota: 50).

        Výstup:
        - Rozmazaný obrázok po aplikácii augmentácie.
        """

    # Convert to grayscale if the image has 3 color channels
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast, adjust brightness and apply Gaussian blur
    contrast_enhanced = cv2.equalizeHist(image)
    adjusted = cv2.convertScaleAbs(contrast_enhanced, alpha=contrast, beta=brightness)
    blurred_image = cv2.GaussianBlur(adjusted, (5, 5), sigmaX=10)

    return blurred_image

def eye_mouth_locator(img_eye_mouth):
    """
        Detekuje oči a ústa na obrázku tváre pomocou Haarovských kaskád a vykreslí obvodové štvorce
        okolo detekovaných oblastí.

        Parametre:
        - img_eye_mouth: vstupný obrázok, na ktorom sa vykoná detekcia očí a úst.

        Výstup:
        - Zoznam obdĺžnikov obsahujúcich detekované oči v obrázku.
    """
    eye_rectangles = []
    gray = cv2.cvtColor(img_eye_mouth, cv2.COLOR_BGR2GRAY)

    # Haarcascade classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    print('Pocet detekovanych tvari:', len(faces))

    # Detect multiple faces
    for face in faces:
        face_x = face[0]
        face_y = face[1]
        face_width = face[2]
        face_height = face[3]

        face_region = gray[face_y:face_y + face_height, face_x:face_x + face_width]
        detected_eyes = eye_cascade.detectMultiScale(face_region)
        detected_mouths = mouth_cascade.detectMultiScale(face_region, scaleFactor=1.5, minNeighbors=11)

        # Detect eyes
        for (eye_x, eye_y, eye_width, eye_height) in detected_eyes:
            adjusted_eye = (
                face_x + eye_x,
                face_y + eye_y,
                eye_width,
                eye_height
            )
            eye_rectangles.append(adjusted_eye)

        # Detect mouths
        for (mouth_x, mouth_y, mouth_width, mouth_height) in detected_mouths:
            top_left = (face_x + mouth_x, face_y + mouth_y)
            bottom_right = (face_x + mouth_x + mouth_width, face_y + mouth_y + mouth_height)
            cv2.rectangle(img_eye_mouth, top_left, bottom_right, (255, 0, 0), 2)

    cv2.imshow('Eyes Detection', img_eye_mouth)

    return eye_rectangles

def damaged_parts_mask(img, thresh_val, kernel_val, radius, rectangles):
    """
        Vytvorí masku poškodených častí obrázka a obnoví tieto oblasti pomocou inpaintingu.

        Parametre:
        - img: vstupný obrázok, ktorý sa spracováva.
        - thresh_val: hodnota prahu pre binárne prahovanie.
        - kernel_val: veľkosť jadra pre dilatáciu.
        - radius: polomer pre obnovu poškodených oblastí.
        - rectangles: zoznam obdĺžnikov, ktoré sa majú ignorovať pri tvorbe masky.

        Výstup:
        - Maska poškodených častí a obnovený obrázok.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_img, thresh_val, 255, cv2.THRESH_BINARY)
    kernel = np.ones(kernel_val, np.uint8)

    # Iterate over the list of rectangles to exclude specific regions
    for (x, y, w, h) in rectangles:
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        thresh[y1:y2, x1:x2] = 0

    # mask dilatation
    mask = cv2.dilate(thresh, kernel, iterations=2)
    restored_image = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
    return thresh, restored_image

def pixel_whitener(image, threshold):
    """
        Zmení všetky pixely obrázka, ktoré majú hodnotu väčšiu alebo rovnakú ako prahovú hodnotu, na bielu farbu.

        Parametre:
        - image: vstupný obrázok.
        - threshold: prahová hodnota pre výber pixelov na bielenie.

        Výstup:
        - Upravený obrázok s bielymi pixelmi.
    """

    # Check if the image is grayscale or color
    if len(image.shape) == 2:
        mask = cv2.inRange(image, threshold, 255)
        # Loop over each pixel in the image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if mask[i, j] == 255:
                    image[i, j] = [255, 255, 255]
    else:
        lower_bound = np.array([threshold] * 3)
        upper_bound = np.array([255] * 3)
        mask = cv2.inRange(image, lower_bound, upper_bound)
        # Loop over each pixel in the image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if mask[i, j] == 255:
                    image[i, j] = [255, 255, 255]
    return image

def pixel_whitener_all_files(folder_path, save_folder, threshold):
    """
        Aplikuje bielenie pixelov na všetky obrázky v zadanom priečinku a uloží výsledky do určeného priečinka.

        Parametre:
        - folder_path: cesta k priečinku obsahujúcemu obrázky.
        - save_folder: cesta k priečinku, kde sa uloží výsledný obrázok.
        - threshold: prahová hodnota pre bielenie pixelov.

        Výstup:
        - Žiadna. Funkcia uloží obrázky s bielenými pixelmi do zadaného priečinka.
    """
    # Loop through all files
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check supported extensions
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.JPG')):
            image = cv2.imread(file_path)

            # Apply whitening function
            whitened_image = pixel_whitener(image, threshold)
            save_path = os.path.join(save_folder, filename)
            cv2.imwrite(save_path, whitened_image)
            print(f"Ulozene: {save_path}")


def main():

    # Paths
    img = cv2.imread('../DP2/generated_images_and_masks_2.1/input/img_1.png')
    input_folder = "../DP2/generated_images_and_masks_2.1/input"
    output_folder = "C:/Users/tomas/PycharmProjects/DP-photo_restoration/DP2/test"


    rectangles = eye_mouth_locator(img)
    restored_img = damaged_parts_mask(img, thresh_val=140, kernel_val=(9,9), radius=5, rectangles=rectangles)
    pixel_whitener_all_files(input_folder, output_folder, threshold=200)


    # Show photos
    cv2.imshow('IMG', img)
    cv2.imshow("Upravena fotka", first_augmentation_image(img))
    cv2.imshow("Threshold", restored_img[0])
    cv2.imshow("Opravena fotka", restored_img[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()