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



# MC
def _get_raw_model(model):
    """
    DetectMultiBackend로 감싸진 모델이면 내부 PyTorch 모델을 꺼낸다.
    """
    if hasattr(model, "model") and hasattr(model.model, "forward_to_detect_features"):
        return model.model

    if hasattr(model, "forward_to_detect_features"):
        return model

    return None


def _set_mc_modules_temporarily(model, enabled=True):
    """
    always_on 속성을 가진 MC DropBlock 모듈 상태를 임시 변경하기 위해
    기존 상태를 저장하고 변경한다.
    """
    states = []

    for m in model.modules():
        if hasattr(m, "always_on"):
            states.append((m, m.always_on))
            m.always_on = enabled

    return states


def _restore_mc_modules(states):
    for m, old_state in states:
        m.always_on = old_state


@torch.no_grad()
def mc_forward_cached_detect(model, im, samples=10, augment=False, visualize=False):
    """
    backbone + neck은 1번만 실행하고,
    Detect head만 samples번 반복 실행하는 MC inference.

    반환:
        mean_pred: [B, N, 5 + nc]
        mc_info:
            xyxy_var
            xyxy_std
            obj_var
            cls_var
            pred_stack
    """
    # augment inference는 구조가 복잡하므로 기존 전체 forward 방식 사용 권장
    if augment:
        return mc_forward_raw(model, im, samples=samples, augment=augment, visualize=visualize)

    raw_model = _get_raw_model(model)

    # raw YOLO model을 못 찾으면 기존 전체 forward 방식으로 fallback
    if raw_model is None:
        return mc_forward_raw(model, im, samples=samples, augment=augment, visualize=visualize)

    raw_model.eval()

    states = _set_mc_modules_temporarily(raw_model, enabled=True)

    try:
        # 1) backbone + neck 1회 실행
        det_features, detect_layer = raw_model.forward_to_detect_features(
            im,
            profile=False,
            visualize=visualize,
        )

        preds = []

        # 2) Detect head만 samples번 반복
        for _ in range(samples):
            # Detect.forward는 x[i]를 덮어쓰므로 list는 매번 새로 만든다.
            # tensor 자체는 in-place 수정하지 않으므로 clone은 보통 필요 없다.
            if isinstance(det_features, (list, tuple)):
                det_in = [f for f in det_features]
            else:
                det_in = det_features

            y = detect_layer(det_in)

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

        mc_info = {
            "xyxy_var": xyxy_var,
            "xyxy_std": xyxy_std,
            "obj_var": obj_var,
            "cls_var": cls_var,
            "pred_stack": pred_stack,
        }

        return mean_pred, mc_info

    finally:
        _restore_mc_modules(states)

# MC
# EdgeDrop
def enable_edge_dropblock(model, enabled=True):
    for m in model.modules():
        if hasattr(m, "use_mc_dropblock"):
            m.use_mc_dropblock = enabled
