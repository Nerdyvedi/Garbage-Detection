import numpy as np
from PIL import Image,ImageDraw,ImageFont
from skimage.transform import rescale

def log_images(images,targets,predictions,score_th = 0.02):
    images_out = []
    for i in range(len(images)):
         images_np = np.squeeze(images[i].detach().cpu().numpy())
         images_np = np.stack((image_np,)*3,axis=-1)
         boxes_true = targets[i]["boxes"].detach().cpu().numpy().astype(np.int)
         green     = [0, np.max(image_np),0]
         for b in range(boxes_true.shape[0]):
               box = boxes_true[0]
               image_np = draw_box(image_np,box[0],box[1],box[2],box[3],c=green)
         boxes_pred = predictions[i]["boxes"].detach().cpu().numpy().astype(np.int)
         scores     = predictions[i]["scores"].detach().cpu().numpy()

