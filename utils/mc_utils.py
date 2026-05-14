import torch

'''
같은 이미지 입력
→ stochastic model을 10번 forward
→ 같은 anchor/grid prediction끼리 평균
→ 좌표 분산 계산
→ 평균 prediction을 NMS로 보냄
'''



def set_mc_dropout(model, enabled=True):
    """
    MCEdgeDropBlock2d처럼 always_on 속성을 가진 모듈을 켜거나 끈다.
    enabled=True  -> eval 모드에서도 stochastic DropBlock 켜짐
    enabled=False -> eval 모드에서는 꺼짐
    """
    for m in model.modules():
        if hasattr(m, "always_on"):
            m.always_on = enabled


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
    같은 입력 im을 samples번 forward.
    YOLO raw prediction의 평균과 bbox 좌표 variance를 계산한다.

    반환:
        mean_pred: [B, N, 5 + nc]
        mc_info:
            xyxy_var: [B, N, 4]
            xyxy_std: [B, N, 4]
            obj_var: [B, N, 1]
            cls_var: [B, N, nc]
            pred_stack: [T, B, N, 5 + nc]
    """
    model.eval()
    set_mc_dropout(model, True)

    preds = []

    for _ in range(samples):
        y = model(im, augment=augment, visualize=visualize)

        # YOLOv5 Detect 출력은 보통 (pred, train_out) 형태일 수 있음
        if isinstance(y, (tuple, list)):
            y = y[0]

        preds.append(y.float())

    pred_stack = torch.stack(preds, dim=0)  # [T, B, N, 5 + nc]
    mean_pred = pred_stack.mean(dim=0)

    xyxy_stack = xywh2xyxy_tensor(pred_stack[..., :4])
    xyxy_var = xyxy_stack.var(dim=0, unbiased=False)
    xyxy_std = torch.sqrt(xyxy_var.clamp_min(0.0))

    obj_var = pred_stack[..., 4:5].var(dim=0, unbiased=False)

    if pred_stack.shape[-1] > 5:
        cls_var = pred_stack[..., 5:].var(dim=0, unbiased=False)
    else:
        cls_var = None

    return mean_pred, {
        "xyxy_var": xyxy_var,
        "xyxy_std": xyxy_std,
        "obj_var": obj_var,
        "cls_var": cls_var,
        "pred_stack": pred_stack,
    }