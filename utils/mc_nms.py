import time
import torch
import torchvision

from utils.general import xywh2xyxy

#confidence는 높지만 box variance가 큰 후보는 score를 낮춤

def mc_non_max_suppression(
    prediction,
    xyxy_var=None,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    max_det=300,
    variance_weight=0.0,
    variance_thres=None,
    img_size=640,
):
    """
    YOLOv5 NMS + MC variance 반영 버전.

    prediction:
        [B, N, 5 + nc]
        mean prediction from MC samples

    xyxy_var:
        [B, N, 4]
        각 raw prediction 위치의 xyxy 좌표 분산

    variance_weight:
        NMS score에 localization variance penalty를 얼마나 줄지.
        0이면 기존 NMS와 거의 동일.

    variance_thres:
        normalized localization std가 이 값보다 크면 제거.
        None이면 제거하지 않음.

    output:
        list of detections per image
        each det row:
        [x1, y1, x2, y2, score, cls, loc_std_norm, var_x1, var_y1, var_x2, var_y2]
    """
    assert 0 <= conf_thres <= 1
    assert 0 <= iou_thres <= 1

    device = prediction.device
    bs = prediction.shape[0]
    nc = prediction.shape[2] - 5

    xc = prediction[..., 4] > conf_thres

    max_wh = 7680
    max_nms = 30000
    time_limit = 0.5 + 0.05 * bs
    t = time.time()

    output = [torch.zeros((0, 11), device=device)] * bs

    for xi, x in enumerate(prediction):
        candidate_mask = xc[xi]
        x = x[candidate_mask]

        if xyxy_var is not None:
            v = xyxy_var[xi][candidate_mask]
        else:
            v = None

        if not x.shape[0]:
            continue

        # class confidence = objectness * class probability
        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])

        if v is not None:
            # loc_std_norm: 평균 좌표 표준편차를 이미지 크기로 나눈 값
            # 값이 클수록 위치 예측이 불안정함
            loc_std_norm = torch.sqrt(v.mean(dim=1, keepdim=True).clamp_min(0.0)) / float(img_size)
        else:
            loc_std_norm = torch.zeros((x.shape[0], 1), device=device)
            v = torch.zeros((x.shape[0], 4), device=device)

        if multi_label and nc > 1:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            conf = x[i, 5 + j, None]
            box_i = box[i]
            v_i = v[i]
            loc_i = loc_std_norm[i]
            cls_i = j[:, None].float()
        else:
            conf, j = x[:, 5:].max(dim=1, keepdim=True)
            keep = conf.view(-1) > conf_thres

            box_i = box[keep]
            conf = conf[keep]
            cls_i = j[keep].float()
            v_i = v[keep]
            loc_i = loc_std_norm[keep]

        if not box_i.shape[0]:
            continue

        # variance threshold로 아예 제거
        if variance_thres is not None:
            keep = loc_i.view(-1) <= variance_thres
            box_i = box_i[keep]
            conf = conf[keep]
            cls_i = cls_i[keep]
            v_i = v_i[keep]
            loc_i = loc_i[keep]

        if not box_i.shape[0]:
            continue

        # variance-aware score
        # 분산이 크면 confidence를 낮춤
        if variance_weight > 0:
            score = conf / (1.0 + variance_weight * loc_i)
        else:
            score = conf

        det = torch.cat((box_i, score, cls_i, loc_i, v_i), dim=1)

        # class filter
        if classes is not None:
            det = det[(det[:, 5:6] == torch.tensor(classes, device=device)).any(1)]

        if not det.shape[0]:
            continue

        # 너무 많으면 score 기준 상위만
        det = det[det[:, 4].argsort(descending=True)[:max_nms]]

        # class-aware NMS
        c = det[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = det[:, :4] + c, det[:, 4]

        keep_idx = torchvision.ops.nms(boxes, scores, iou_thres)
        keep_idx = keep_idx[:max_det]

        output[xi] = det[keep_idx]

        if time.time() - t > time_limit:
            break

    return output