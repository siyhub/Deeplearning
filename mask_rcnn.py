# -*- coding: utf-8 -*-

 
'''
Koden under er tatt utgangspunkt i en offentlig kilde kode fra Kaggle. Det er relativt få endringer, kun noen ekstra
implementeringer vi så som nødvendig, eksempelvis logging av resultater fra treningen.  
Utgangspunkt for kode : https://www.kaggle.com/samlin001/mask-r-cnn-ship-detection-minimum-viable-model-1


Koden oppsummert: konfiguering av datasett, konfiguering av Mask-RCNN, augmentering, trening- og treningsresultater og validering. 


'''



import numpy as np # linær algebra
import pandas as pd # data prosessering
import matplotlib.pyplot as plt # plotting og bilde prosessering
from skimage.morphology import label
from skimage.data import imread

import os
import time
import sys



'''Prosjekt konfiguering'''

TRAINING_VALIDATION_RATIO = 0.2 # Stor andel av treningssettet som skal brukes i valideringssettet
WORKING_DIR = '/kaggle/working'
INPUT_DIR = '/kaggle/input'
OUTPUT_DIR = '/kaggle/output'
LOGS_DIR = os.path.join(WORKING_DIR, "logs")
TRAIN_DATA_PATH = os.path.join(INPUT_DIR, 'train_v2')
TEST_DATA_PATH = os.path.join(INPUT_DIR, 'test_v2')
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, 'sample_submission_v2.csv')
TRAIN_SHIP_SEGMENTATIONS_PATH = os.path.join(INPUT_DIR, 'train_ship_segmentations_v2.csv')
MASK_RCNN_PATH = os.path.join(WORKING_DIR, 'Mask_RCNN-master')
COCO_WEIGHTS_PATH = os.path.join(WORKING_DIR, "mask_rcnn_coco.h5")
SHIP_CLASS_NAME = 'ship'
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 768
SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT)

test_ds = os.listdir(TEST_DATA_PATH)
train_ds = os.listdir(TRAIN_DATA_PATH)

print('Working Dir:', WORKING_DIR, os.listdir(WORKING_DIR))
print('Input Dir:', INPUT_DIR, os.listdir(INPUT_DIR))
print('train dataset from: {}, {}'.format(TRAIN_DATA_PATH, len(train_ds)))
print('test dataset from: {}, {}'.format(TRAIN_DATA_PATH, len(test_ds)))
print(TRAIN_SHIP_SEGMENTATIONS_PATH)




'''Preparering av datasett'''

# Leser maske koordninater fra CSV filen
masks = pd.read_csv(TRAIN_SHIP_SEGMENTATIONS_PATH)
masks.head()



# ref: https://www.kaggle.com/kmader/baseline-u-net-model-part-1
def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    bilde: numpy array, 1 - mask, 0 - background
    Returner lengden som et string format: [start0] [length0] [start1] [length1]... 1d array
    '''
    # omforming til 1d array
    pixels = img.T.flatten() 
    
    pixels = np.concatenate([[0], pixels, [0]])
    
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
  
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=SHAPE):
    '''
    mask_rle: lengden som et string format: [start0] [length0] [start1] [length1]... in 1d array
    form: (height,width) på arrayen som returneres
    Returner numpy array som formen, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    # Start og lengde i 1d array
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    # 1d array
    ends = starts + lengths
    #Oppretter en tom maske
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    # setter piksel
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    # omformer til et 2d array
    return img.reshape(shape).T  

def masks_as_image(in_mask_list, shape=SHAPE):
    
    '''Tar hver individuelle skips maske og lager en singel maske array for alle skipene'''
    
    all_masks = np.zeros(shape, dtype = np.int16)
    # Hvis det er et eksempel(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

def shows_decode_encode(image_id, path=TRAIN_DATA_PATH):
    
    '''Viser bilde, maske og koded/dekodet resultat'''
    
    fig, axarr = plt.subplots(1, 3, figsize = (10, 5))
    # bilde
    img_0 = imread(os.path.join(path, image_id))
    axarr[0].imshow(img_0)
    axarr[0].set_title(image_id)
    
    # input maske
    rle_1 = masks.query('ImageId=="{}"'.format(image_id))['EncodedPixels']
    img_1 = masks_as_image(rle_1)
    # 2d array (shape.h, sahpe.w)
    axarr[1].imshow(img_1[:, :, 0])
    axarr[1].set_title('Ship Mask')
    
    # kodet & dekodet maske
    rle_2 = multi_rle_encode(img_1)
    img_2 = masks_as_image(rle_2)
    axarr[2].imshow(img_0)
    axarr[2].imshow(img_2[:, :, 0], alpha=0.3)
    axarr[2].set_title('Encoded & Decoded Mask')
    plt.show()
    print(image_id , ' Check Decoding->Encoding',
          'RLE_0:', len(rle_1), '->',
          'RLE_1:', len(rle_2))

# Sjekker noen eksempler
shows_decode_encode('000155de5.jpg')
shows_decode_encode('00003e153.jpg')
print('It could be different when there is no mask.')
shows_decode_encode('00021ddc3.jpg')
print('It could be different when there are masks overlapped.')






'''Splitte datasett inn i test- og valideringssett'''

# Sjekker om en maske har et skip
masks['ships'] = masks['EncodedPixels'].map(lambda encoded_pixels: 1 if isinstance(encoded_pixels, str) else 0)
# summerer skipene i ImageId og skaper en unique image id/maske liste
start_time = time.time()
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'})
unique_img_ids['RleMaskList'] = masks.groupby('ImageId')['EncodedPixels'].apply(list)
unique_img_ids = unique_img_ids.reset_index()
end_time = time.time() - start_time
print("unique_img_ids groupby took: {}".format(end_time))


# Bare bilder med skip
unique_img_ids = unique_img_ids[unique_img_ids['ships'] > 0]
unique_img_ids['ships'].hist()
unique_img_ids.sample(3)


# del inn i trenings- og valideringsset
from sklearn.model_selection import train_test_split
train_ids, val_ids = train_test_split(unique_img_ids, 
                 test_size = TRAINING_VALIDATION_RATIO, 
                 stratify = unique_img_ids['ships'])
print(train_ids.shape[0], 'training masks')
print(val_ids.shape[0], 'validation masks')
train_ids['ships'].hist()
val_ids['ships'].hist()





'''Mask-RCNN model'''

#Importering av Mask-RCNN

# Hvis Mask_RCNN allerede eksisterer
UPDATE_MASK_RCNN = False

os.chdir(WORKING_DIR)
if UPDATE_MASK_RCNN:
    !rm -rf {MASK_RCNN_PATH}

# Last ned MASK-RCNN script til lokal fil
if not os.path.exists(MASK_RCNN_PATH):
    ! wget https://github.com/samlin001/Mask_RCNN/archive/master.zip -O Mask_RCNN-master.zip
    ! unzip Mask_RCNN-master.zip 'Mask_RCNN-master/mrcnn/*'
    ! rm Mask_RCNN-master.zip

# Importer Mask RCNN
sys.path.append(MASK_RCNN_PATH)  # Finner versjonen i bibliotekt
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log    


#Transformering av datasett

class AirbusShipDetectionChallengeDataset(utils.Dataset):
    """Airbus Ship Detection Challenge Dataset
    """
    def __init__(self, image_file_dir, ids, masks, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT):
        super().__init__(self)
        self.image_file_dir = image_file_dir
        self.ids = ids
        self.masks = masks
        self.image_width = image_width
        self.image_height = image_height
        
        # legge til klasser
        self.add_class(SHIP_CLASS_NAME, 1, SHIP_CLASS_NAME)
        self.load_dataset()
        
    def load_dataset(self):
        """Load dataset from the path
        """
        # legge til bilder
        for index, row in self.ids.iterrows():
            image_id = row['ImageId']
            image_path = os.path.join(self.image_file_dir, image_id)
            rle_mask_list = row['RleMaskList']
            #print(rle_mask_list)
            self.add_image(
                SHIP_CLASS_NAME,
                image_id=image_id,
                path=image_path,
                width=self.image_width, height=self.image_height,
                rle_mask_list=rle_mask_list)

    def load_mask(self, image_id):
        """Genererer masker for former til gitt bilde ID
        """
        info = self.image_info[image_id]
        rle_mask_list = info['rle_mask_list']
        mask_count = len(rle_mask_list)
        mask = np.zeros([info['height'], info['width'], mask_count],
                        dtype=np.uint8)
        i = 0
        for rel in rle_mask_list:
            if isinstance(rel, str):
                np.copyto(mask[:,:,i], rle_decode(rel))
            i += 1
        
        # Returnerer masker, og en array av ider for hvert eksempel. Siden vi bare har
        # en klasse IDer, returner vi en array av 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
    
    def image_reference(self, image_id):
        """Returner filbanen til bilde."""
        info = self.image_info[image_id]
        if info['source'] == SHIP_CLASS_NAME:
            return info['path']
        else:
            super(self.__class__, self).image_reference(image_id)
            

'''Model konfigurering. Vi setter kun de viktiste forhåndsbestemte parameterne, her er det mye prøving og feiling for å finne de optimale.
Verdiene satt her oversskriver de satte parameterne i config.py'''

class AirbusShipDetectionChallengeGPUConfig(Config):
   
    # https://www.kaggle.com/docs/kernels#technical-specifications
    NAME = 'ASDC_GPU'
    # Antall GPUer.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    
    NUM_CLASSES = 2  # ship eller background
    IMAGE_MIN_DIM = IMAGE_WIDTH
    IMAGE_MAX_DIM = IMAGE_WIDTH
    STEPS_PER_EPOCH = 300
    VALIDATION_STEPS = 50
    SAVE_BEST_ONLY = True
    
    # Minstekrav for deteksjon
    # Alle deteksjoner under grensen blir droppet
    DETECTION_MIN_CONFIDENCE = 0.90

    #Grense for overlappende deteksjoner
    DETECTION_NMS_THRESHOLD = 0.05
    #Start verdier for vekter før trening. Disse tilpasses underveis
    LOSS_WEIGHTS = {
        "rpn_class_loss": 30.0,
        "rpn_bbox_loss": 0.8,
        "mrcnn_class_loss": 6.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 1.2
    }

    
config = AirbusShipDetectionChallengeGPUConfig()
config.display()




'''Laster inn datasettene '''

start_time = time.time()
# Treningsdatasett
dataset_train = AirbusShipDetectionChallengeDataset(image_file_dir=TRAIN_DATA_PATH, ids=train_ids, masks=masks)
dataset_train.prepare()

# Valideringssdataset
dataset_val = AirbusShipDetectionChallengeDataset(image_file_dir=TRAIN_DATA_PATH, ids=val_ids, masks=masks)
dataset_val.prepare()

# Vis tilfeldige eksempler
image_ids = np.random.choice(dataset_train.image_ids, 3)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names, limit=1)

end_time = time.time() - start_time
print("dataset prepare: {}".format(end_time))





'''Overført læring. Laster ned vekter som er trent på datasettet MS COCO '''

start_time = time.time()
model = modellib.MaskRCNN(mode="training", config=config, model_dir=WORKING_DIR)

import errno
try:
    weights_path = model.find_last()
    load_weights = True
except FileNotFoundError:
    # Hvis det ikke er noen tidligere trente vekter, last ned MS COCO
    load_weights = True
    weights_path = COCO_WEIGHTS_PATH
    utils.download_trained_weights(weights_path)
    
if load_weights:
    print("Loading weights: ", weights_path)
    model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])

end_time = time.time() - start_time
print("loading weights: {}".format(end_time))





'''Kode for augmentering av datasettet'''

#Forhåndskonfigurering for augmentering
from skimage.io import imread
from matplotlib.cm import get_cmap
from skimage.segmentation import mark_boundaries
from skimage.util import montage
from skimage.morphology import binary_opening, disk, label
import gc; gc.enable() 
from imgaug import augmenters as iaa

# Bilde augmentering
augmentation = iaa.Sequential([
    iaa.OneOf([ ## rotering
        iaa.Affine(rotate=0),
        iaa.Affine(rotate=90),
        iaa.Affine(rotate=180),
        iaa.Affine(rotate=270),
    ]),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([ ## lysstyrke eller kontrast
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
    ]),
    iaa.OneOf([ ## uskarpe og skarpe
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.Sharpen(alpha=(0.0, 0.1)),
    ]),
])

# Test på det samme bilde som over
imggrid = augmentation.draw_grid(image, cols=5, rows=2)
plt.figure(figsize=(30, 12))
_ = plt.imshow(imggrid.astype(int))






"""Trening av modellen"""
start_time = time.time()   

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE * 1.5,
            epochs=40,
            layers='all',
            augmentation=augmentation)
end_time = time.time() - start_time
print("Train model: {}".format(end_time))

#Lagrer viktig data om treningsresultatene
history = model.keras_model.history.history



#Skriver ut en tabell over treningsresultatene
epochs = range(1, len(history['loss'])+1)
pd.DataFrame(history, index=epochs)


#Skriver ut grafer over treningsresultatene
plt.figure(figsize=(21,11))

plt.subplot(231)
plt.plot(epochs, history["loss"], label="Train loss")
plt.plot(epochs, history["val_loss"], label="Valid loss")
plt.legend()
plt.subplot(232)
plt.plot(epochs, history["rpn_class_loss"], label="Train RPN class ce")
plt.plot(epochs, history["val_rpn_class_loss"], label="Valid RPN class ce")
plt.legend()
plt.subplot(233)
plt.plot(epochs, history["rpn_bbox_loss"], label="Train RPN box loss")
plt.plot(epochs, history["val_rpn_bbox_loss"], label="Valid RPN box loss")
plt.legend()
plt.subplot(234)
plt.plot(epochs, history["mrcnn_class_loss"], label="Train MRCNN class ce")
plt.plot(epochs, history["val_mrcnn_class_loss"], label="Valid MRCNN class ce")
plt.legend()
plt.subplot(235)
plt.plot(epochs, history["mrcnn_bbox_loss"], label="Train MRCNN box loss")
plt.plot(epochs, history["val_mrcnn_bbox_loss"], label="Valid MRCNN box loss")
plt.legend()
plt.subplot(236)
plt.plot(epochs, history["mrcnn_mask_loss"], label="Train Mask loss")
plt.plot(epochs, history["val_mrcnn_mask_loss"], label="Valid Mask loss")
plt.legend()

plt.show()






'''Validering av modellen. Her skrvies det ut bilder med deteksjoner. Antall bilder man vil teste på modellen kan endres.'''

class InferenceConfig(AirbusShipDetectionChallengeGPUConfig):
    GPU_COUNT = 1
    # Setter til en når modellen skal valideres
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Setter modellen i validerings mode
infer_model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=WORKING_DIR)

model_path = infer_model.find_last()

# Laster trente vekter
print("Loading weights from ", model_path)
infer_model.load_weights(model_path, by_name=True)


# Test et tilfeldig bilde
image_id = np.random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))

results = infer_model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'])


# kode for validering av flere bilder. Bruk flere bilder for høyere nøyaktighet
image_ids = np.random.choice(dataset_val.image_ids, 20) #Antall tilfeldige bilder fra valideringssettet
APs = []
inference_start = time.time()
for image_id in image_ids:
    # Last ned bilde og de ekte koordinatene
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Objekt detektering
    results = infer_model.detect([image], verbose=1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'])

    #Utrekning av mAP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

inference_end = time.time()
print('Inference Time: %0.2f Minutes'%((inference_end - inference_start)/60))
print("mAP: ", np.mean(APs))




