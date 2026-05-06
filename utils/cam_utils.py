import torch
import torch.nn.functional as F


def get_cam(model, x, target_class=None):
    was_training = model.training
    model.eval()

    with torch.no_grad():
        logits, feat = model(x)
        fc_weights = model.fc.weight

        if target_class is None:
            class_idx = logits.argmax(dim=1)
        elif isinstance(target_class, int):
            class_idx = torch.full(
                (x.size(0),),
                target_class,
                dtype=torch.long,
                device=x.device,
            )
        else:
            class_idx = target_class.to(x.device)

        weights = fc_weights[class_idx].unsqueeze(-1).unsqueeze(-1)
        cam = (feat * weights).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        b = cam.size(0)
        cam_flat = cam.view(b, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        cam_max = cam_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-5)

    model.train(was_training)
    return cam

