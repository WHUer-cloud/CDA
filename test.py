import torch
from torch.utils.data import DataLoader
from data_loader import TestDataset
from model import Encoder, GCMAndDecoder
from metric import anomaly_score, compute_auroc
from config import get_args, OBJ_NAMES
import numpy as np

def test(obj_name, ckp_path, data_root, img_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder().to(device).eval()
    dec = GCMAndDecoder()
    dec.load_state_dict(torch.load(ckp_path, map_location="cpu"))
    dec.to(device).eval()

    ds = TestDataset(root_dir=data_root, obj_name=obj_name, resize_shape=img_size)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    scores, labels, gts = [], [], []
    with torch.no_grad():
        for batch in dl:
            img, label, gt = batch["image"], batch["label"], batch["gt_mask"]
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            e1, e2, e3 = encoder(img.to(device))
            d_list = dec(e1, e2, e3)
            score_map = anomaly_score([e1, e2, e3], d_list, img_size)
            scores.append(score_map[0])
            labels.append(int(label.squeeze().numpy()))
            gts.append(gt.squeeze().numpy())
    return compute_auroc(scores, labels, gts)

if __name__ == "__main__":
    args = get_args()
    obj = OBJ_NAMES[args.obj_id]
    ckp = f"{args.checkpoint_path}WRes50/{obj}_best.pth"
    img_roc, pix_roc = test(obj, ckp, args.data_path, args.img_size)
    print(f"{obj}  auroc_img={img_roc}  auroc_pix={pix_roc}")