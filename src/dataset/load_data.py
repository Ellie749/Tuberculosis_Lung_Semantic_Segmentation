import glob
import cv2


def load_chest_images(img_path: str):
    print("[INFO] READING CHEST IMAGES...")
    images = []
    chest_images = glob.glob(img_path+"\*.png")
    for address in chest_images:
        images.append(cv2.cvtColor(cv2.imread(address), cv2.COLOR_BGR2RGB))

    # for i in images: 
    #     cv2.imshow("Sample Image", i)
    #     cv2.waitKey(0)
    
    return images


def load_mask_images(img_path):
    print("[INFO] READING CHEST MASKS...")
    masks = []
    mask_images = glob.glob(img_path+"\*.png")
    for address in mask_images:
        masks.append(cv2.cvtColor(cv2.imread(address), cv2.COLOR_BGR2RGB))
    
    return masks
