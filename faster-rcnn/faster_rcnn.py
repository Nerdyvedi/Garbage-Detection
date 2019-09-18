from torchvision.models.detection import FasterRCNN 
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from resnet_fpn import resnet_fpn_backbone

def faster_rcnn():
    backbone       = resnet_fpn_backbone()
    model_config   = { 
        "num_classes" : 2
        "min_size": 1536,
        "max_size": 2048,
        "image_mean": (0.0,),
        "image_std": (1.0,),
        "rpn_pre_nms_top_n_train": 1024,
        "rpn_pre_nms_top_n_test": 512,
        "rpn_post_nms_top_n_train": 1024,
        "rpn_post_nms_top_n_test": 512,
        "rpn_nms_thresh": 0.6,
        "rpn_fg_iou_thresh": 0.6,
        "rpn_bg_iou_thresh": 0.3,
        "rpn_batch_size_per_image": 256,
        "rpn_positive_fraction": 0.5,
        "box_score_thresh": 0.01,
        "box_nms_thresh": 0.4,
        "box_detections_per_img": 100,
        "box_fg_iou_thresh": 0.5,
        "box_bg_iou_thresh": 0.2,
    }
    anchor_sizes   = (128,256)
    aspect_ratio   = (1.0,)
    rpn_anchor_generator  = AnchorGenerator(anchor_sizes,aspect_ratio)
    box_roi_pool   = MultiScaleRoIAlign(featmap_names = [3],output_size = 9,sampling_ratio = 1)
    return FasterRCNN(backbone = backbone, rpn_anchor_generator = rpn_anchor_generator, box_roi_pool = box_roi_pool,**model_config)
 
