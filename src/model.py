import torchvision
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights, FastRCNNPredictor)


def dfs_freeze(model):
    for _, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def get_model(num_classes: int) -> FasterRCNN:
    # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    #     weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    #     )

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        )
    dfs_freeze(model)

    in_featured = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_featured, num_classes)
    return model
