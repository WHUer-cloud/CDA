import os
import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_loader import TrainDataset
from model import Encoder, GCMAndDecoder
from metric import cos_dist
from config import get_args, OBJ_NAMES
from thop import profile, clever_format
from tqdm import tqdm
from pathlib import Path

def train(obj_name, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_path, exist_ok=True)
    run_name = f"{obj_name}_lr{args.lr}_bs{args.bs}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    writer = SummaryWriter(f"./logs/WRes50/{run_name}")
    ck_dir = Path(args.checkpoint_path) / "WRes50" / run_name
    ck_dir.mkdir(exist_ok=True)

    # ----- 模型 -----
    encoder = Encoder().to(device).eval()
    dec = GCMAndDecoder().to(device)
  
    # ----- 参数量 & FLOPs -----
    # 1. 教师
    # teacher_params = sum(p.numel() for p in encoder.parameters())

    # # # 2. 学生（全部）
    # d1 = torch.randn(1, 256, 64, 64).to(device)
    # d2 = torch.randn(1, 512, 32, 32).to(device)
    # d3 = torch.randn(1, 1024, 16, 16).to(device)
    # flops, params = profile(dec, inputs=(d1, d2, d3), verbose=False)

    # print("Teacher params :", clever_format([teacher_params]))
    # print("Student params :", clever_format([params]))
    # print("Student FLOPs  :", clever_format([flops]))
    # exit()
    # ----- 数据 -----
    ds = TrainDataset(root_dir=args.data_path, obj_name=obj_name, resize_shape=args.img_size)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, drop_last=True, num_workers=8, pin_memory=True)

    # ----- 优化器 & 损失 -----
    opt = torch.optim.Adam(dec.parameters(), lr=args.lr, betas=(0.5, 0.999))

    best_img, best_pix = 0, 0
    for epoch in tqdm(range(args.epochs), ascii=True):
        dec.train()
        total_loss = 0.
        for batch in dl:
            img = batch["image"].to(device)
            e1, e2, e3 = encoder(img)
            d1, d2, d3, l1m, l1n, l2m, l2n, l3m, l3n = dec(e1, e2, e3)

            loss1 = cos_dist(e1, d1).mean() + cos_dist(e2, d2).mean() + cos_dist(e3, d3).mean()
            loss2 = (0.4 * cos_dist(e1, l1n) + cos_dist(e1, l1m)).mean() + \
                    (0.2 * cos_dist(e2, l2n) + cos_dist(e2, l2m)).mean() + \
                    (0.4 * cos_dist(e3, l3n) + cos_dist(e3, l3m)).mean()
            loss = 2.0 * loss1 + 0.1 * loss2

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        writer.add_scalar("train_loss", total_loss, epoch)

        if args.test_interval > 0 and epoch % args.test_interval == 0:
            ck_path = ck_dir / f"epoch{epoch}.pth"
            torch.save(dec.state_dict(), ck_path)
            img_roc, pix_roc = test(obj_name, ck_path, args.data_path, args.img_size)
            writer.add_scalar("auroc_img", img_roc, epoch)
            writer.add_scalar("auroc_pix", pix_roc, epoch)
            # 自动删除无提升的 checkpoint
            if img_roc <= best_img and pix_roc <= best_pix:
                os.remove(ck_path)
            if img_roc > best_img:
                best_img = img_roc
                torch.save(dec.state_dict(), ck_dir / "best_img.pth")
            if pix_roc > best_pix:
                best_pix = pix_roc
                torch.save(dec.state_dict(), ck_dir / "best_pix.pth")
    return run_name, best_img, best_pix

def test(obj_name, ckp_path, data_root, img_size):
    # 保持与 test.py 一致，避免循环 import
    from test import test as _test
    return _test(obj_name, ckp_path, data_root, img_size)

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    # 初始化汇总文件
    summary_path = "./log_txt/summary.txt"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:  # 清空旧文件
        f.write("log title\n")
    
    # 逐类别训练
    for obj in OBJ_NAMES:
        # 训练当前类别
        run_name, best_img, best_pix = train(obj, args)
        
        # 实时打印结果（严格按你要求的格式）
        result_line = f"{run_name} || auroc_img: {best_img:.3f} || auroc_pix: {best_pix:.3f}"
        print(result_line)  # 控制台立即输出
        
        # 实时追加到文件（无需等待所有类别完成）
        with open(summary_path, "a") as f:
            f.write(result_line + "\n")