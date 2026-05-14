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
    labels=(),
    max_det=300,
    variance_weight=0.0,
    variance_thres=None,
    img_size=640,
    return_extra=False,
):
    """
    YOLOv5 NMS + MC variance 반영 버전.

    prediction:
        [B, N, 5 + nc]
        MC sample 평균 prediction

    xyxy_var:
        [B, N, 4]
        같은 anchor/grid 위치에서 나온 xyxy 좌표 variance

    return_extra=False:
        val.py용. 기존 YOLOv5 형식 [x1,y1,x2,y2,conf,cls] 반환.

    return_extra=True:
        detect.py/pred.py용. [x1,y1,x2,y2,conf,cls,loc_std,var_x1,var_y1,var_x2,var_y2] 반환.
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

    out_cols = 11 if return_extra else 6
    output = [torch.zeros((0, out_cols), device=device)] * bs

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if xyxy_var is not None:
            v = xyxy_var[xi][xc[xi]]
        else:
            v = None

        # autolabelling용 labels 처리
        # val.py 기본 non_max_suppression과 완전히 같은 목적은 아니지만,
        # labels가 들어오면 candidate에 추가할 수 있도록 최소 지원
        if labels and len(labels[xi]):
            lb = labels[xi]
            v_lb = torch.zeros((len(lb), 4), device=device)
            label_box = lb[:, 1:5]
            label_cls = lb[:, 0].long()

            label_x = torch.zeros((len(lb), nc + 5), device=device)
            label_x[:, :4] = label_box
            label_x[:, 4] = 1.0
            label_x[range(len(lb)), label_cls + 5] = 1.0

            x = torch.cat((x, label_x), 0)
            if v is not None:
                v = torch.cat((v, v_lb), 0)

        if not x.shape[0]:
            continue

        # class confidence = objectness * class probability
        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])

        if v is not None:
            loc_std_norm = torch.sqrt(v.mean(dim=1, keepdim=True).clamp_min(0.0)) / float(img_size)
        else:
            v = torch.zeros((x.shape[0], 4), device=device)
            loc_std_norm = torch.zeros((x.shape[0], 1), device=device)

        if multi_label and nc > 1:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            box_i = box[i]
            conf_i = x[i, j + 5, None]
            cls_i = j[:, None].float()
            v_i = v[i]
            loc_i = loc_std_norm[i]
        else:
            conf_i, j = x[:, 5:].max(dim=1, keepdim=True)
            keep = conf_i.view(-1) > conf_thres

            box_i = box[keep]
            conf_i = conf_i[keep]
            cls_i = j[keep].float()
            v_i = v[keep]
            loc_i = loc_std_norm[keep]

        if not box_i.shape[0]:
            continue

        # variance threshold로 제거
        if variance_thres is not None:
            keep = loc_i.view(-1) <= variance_thres
            box_i = box_i[keep]
            conf_i = conf_i[keep]
            cls_i = cls_i[keep]
            v_i = v_i[keep]
            loc_i = loc_i[keep]

        if not box_i.shape[0]:
            continue

        # variance-aware score
        if variance_weight > 0:
            score_i = conf_i / (1.0 + variance_weight * loc_i)
        else:
            score_i = conf_i

        if return_extra:
            det = torch.cat((box_i, score_i, cls_i, loc_i, v_i), dim=1)
        else:
            det = torch.cat((box_i, score_i, cls_i), dim=1)

        # class filter
        if classes is not None:
            det = det[(det[:, 5:6] == torch.tensor(classes, device=device)).any(1)]

        if not det.shape[0]:
            continue

        det = det[det[:, 4].argsort(descending=True)[:max_nms]]

        c = det[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = det[:, :4] + c, det[:, 4]

        keep_idx = torchvision.ops.nms(boxes, scores, iou_thres)
        keep_idx = keep_idx[:max_det]

        output[xi] = det[keep_idx]

        if time.time() - t > time_limit:
            break

    return output