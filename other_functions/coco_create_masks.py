from codes.other_functions.my_functions import *

def main():
    coco_json_path = '../scratches/scratches.json'
    image_dir = '../scratches/dataset/images/'
    image_extensions = ['.jpg', '.png', '.jfif']

    coco = COCO(coco_json_path)
    process_all_images(coco, image_dir, image_extensions)

if __name__ == "__main__":
    main()
