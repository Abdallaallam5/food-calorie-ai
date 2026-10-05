"""تدريب موديل التصنيف.

التشغيل:
    python -m src.train
لو التدريب وقف في النص، شغّله تاني وهيكمل من آخر epoch اتحفظ.
"""
import json
import os
import time

import torch
import torch.nn as nn

from . import config as C
from .data import get_loaders
from .model import build_model


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, epoch):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        seen += x.size(0)
        if i % 200 == 0:
            print(f"  epoch {epoch + 1} | step {i}/{len(loader)} | "
                  f"loss {total_loss / seen:.3f} | acc {correct / seen:.2%} | {time.time() - t0:.0f}s")
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    top1, top5, seen = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            out = model(x)
        top1 += (out.argmax(1) == y).sum().item()
        top5 += (out.topk(5, 1).indices == y[:, None]).any(1).sum().item()
        seen += x.size(0)
    return top1 / seen, top5 / seen


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    os.makedirs(C.SAVE_DIR, exist_ok=True)

    train_loader, test_loader, classes = get_loaders()
    with open(os.path.join(C.SAVE_DIR, "classes.json"), "w") as f:
        json.dump(classes, f)

    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=C.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=C.LR, epochs=C.EPOCHS, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    # نكمل من checkpoint لو موجود
    start_epoch, best_acc = 0, 0.0
    ckpt_path = os.path.join(C.SAVE_DIR, "checkpoint.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best_acc = ckpt["epoch"] + 1, ckpt["best_acc"]
        print(f"كملنا من epoch {start_epoch}, best acc = {best_acc:.2%}")

    for epoch in range(start_epoch, C.EPOCHS):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                    scheduler, scaler, device, epoch)
        top1, top5 = evaluate(model, test_loader, device)
        print(f"✅ Epoch {epoch + 1}/{C.EPOCHS} | train loss {loss:.3f} | train acc {acc:.2%} | "
              f"test top-1 {top1:.2%} | test top-5 {top5:.2%}")

        if top1 > best_acc:
            best_acc = top1
            torch.save(model.state_dict(), os.path.join(C.SAVE_DIR, "best_model.pth"))
            print(f"   💾 أحسن موديل اتحفظ ({best_acc:.2%})")

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "best_acc": best_acc}, ckpt_path)

    print(f"\n🎉 خلصنا! أحسن دقة: {best_acc:.2%}")


if __name__ == "__main__":
    main()
