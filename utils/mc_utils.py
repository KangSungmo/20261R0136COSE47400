import torch

'''
같은 이미지 입력
→ stochastic model을 10번 forward
→ 같은 anchor/grid prediction끼리 평균
→ 좌표 분산 계산
→ 평균 prediction을 NMS로 보냄
'''

def xywh2xyxy_tensor(x):
    """
    x: [..., 4] = [cx, cy, w, h]
    return: [..., 4] = [x1, y1, x2, y2]
    """
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


@torch.no_grad()
def mc_forward_raw(model, im, samples=10, augment=False, visualize=False):
    """
    논문식 Pre-NMS MC sampling.

    model(im)의 raw prediction을 samples번 얻고,
    같은 anchor/grid 위치끼리 평균과 분산을 계산한다.

    반환:
        mean_pred: [B, N, 5 + nc]
        mc_info:
            xyxy_var: [B, N, 4]
            xyxy_std: [B, N, 4]
            obj_var:  [B, N, 1]
            cls_var:  [B, N, nc]
            pred_stack: [T, B, N, 5 + nc]
    """
    model.eval()

    preds = []

    for _ in range(samples):
        y = model(im, augment=augment, visualize=visualize)

        # YOLOv5 Detect는 보통 inference에서 (pred, train_out) 형태를 줄 수 있음
        if isinstance(y, (tuple, list)):
            y = y[0]

        preds.append(y.float())

    pred_stack = torch.stack(preds, dim=0)  # [T, B, N, 5 + nc]

    mean_pred = pred_stack.mean(dim=0)

    # 좌표 분산은 xywh보다 xyxy 기준이 NMS/PDQ 쪽에서 쓰기 편함
    xyxy_stack = xywh2xyxy_tensor(pred_stack[..., :4])  # [T, B, N, 4]
    xyxy_var = xyxy_stack.var(dim=0, unbiased=False)    # [B, N, 4]
    xyxy_std = torch.sqrt(xyxy_var.clamp_min(0.0))

    obj_var = pred_stack[..., 4:5].var(dim=0, unbiased=False)

    if pred_stack.shape[-1] > 5:
        cls_var = pred_stack[..., 5:].var(dim=0, unbiased=False)
    else:
        cls_var = None

    mc_info = {
        "xyxy_var": xyxy_var,
        "xyxy_std": xyxy_std,
        "obj_var": obj_var,
        "cls_var": cls_var,
        "pred_stack": pred_stack,
    }

    return mean_pred, mc_info