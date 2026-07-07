# =============================================================
# training_mel_v1.py  —  Training + Validation Loop
# =============================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from encoding import MelEncoder, MFCCEncoder, CQTEncoder, CWTEncoder, ChirpletEncoder
from dataset  import PCGSpikeDataset
from model    import ConvSNN1D


def run_epoch(model, loader, device, T_SIM, loss_fn, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for spikes_batch, labels in loader:
            spikes_batch = spikes_batch.to(device)
            labels       = labels.to(device)

            model.lif1.reset_hidden()
            model.lif2.reset_hidden()

            acc_logits = torch.zeros(spikes_batch.size(0), 2, device=device)

            for t in range(T_SIM):
                x_t    = spikes_batch[:, t, :, :]
                logits = model(x_t)
                acc_logits += logits

            avg_logits = acc_logits / T_SIM
            loss = loss_fn(avg_logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct    += (avg_logits.argmax(dim=1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


def evaluate_detailed(model, loader, device, T_SIM, n_runs=10):
    """
    Full evaluation with sensitivity, specificity, confusion matrix.
    Uses n_runs averaged forward passes for stable predictions
    (rate encoding is stochastic, so averaging reduces variance).
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for spikes_batch, labels in loader:
            spikes_batch = spikes_batch.to(device)

            avg_over_runs = torch.zeros(
                spikes_batch.size(0), 2, device=device
            )
            for _ in range(n_runs):
                model.lif1.reset_hidden()
                model.lif2.reset_hidden()

                acc = torch.zeros(spikes_batch.size(0), 2, device=device)
                for t in range(T_SIM):
                    acc += model(spikes_batch[:, t, :, :])
                avg_over_runs += acc / T_SIM

            preds = (avg_over_runs / n_runs).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n" + "="*55)
    print("DETAILED EVALUATION (best checkpoint, 10-run average)")
    print("="*55)
    print(classification_report(
        all_labels, all_preds,
        target_names=['Normal', 'Abnormal'],
        digits=4
    ))
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print(f"Confusion matrix:\n{cm}")
    print(f"\nSensitivity (recall abnormal) : {sensitivity*100:.2f}%")
    print(f"Specificity (recall normal)   : {specificity*100:.2f}%")
    print(f"Balanced accuracy             : {(sensitivity+specificity)/2*100:.2f}%")


# ── Everything under this guard — required on Windows ─────────
if __name__ == '__main__':

    # ── Directories ───────────────────────────────────────────
    TRAIN_DIR = r"D:\ADHISH\ACADEMIC_PROJECT\PCG\PERFECT\TRAINING"
    VAL_DIR   = r"D:\ADHISH\ACADEMIC_PROJECT\PCG\PERFECT\VALID"

    # ── Swap encoder here — one line to try a different feature ─
    # encoder = MelEncoder(sr=2000, n_mels=24, n_fft=512, hop_len=128, fmin=20, fmax=500)                #done
    # encoder = MFCCEncoder(sr=2000, n_mfcc=24, n_fft=512, hop_len=128)                                  #done
    # encoder = CQTEncoder(sr=2000, n_bins=24, hop_len=128, fmin=32.7)                                   #done
    # encoder = CWTEncoder(sr=2000, n_scales=24, wavelet='morl', fmin=20, fmax=500, decimation=128)      #done
    encoder = ChirpletEncoder(sr=2000, n_scales=24, fmin=20, fmax=500, decimation=128)                   #done

    # ── Training config ───────────────────────────────────────
    T_SIM  = 50
    BATCH  = 16
    EPOCHS = 100
    LR     = 5e-3
    BETA   = 0.9

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Encoder : {encoder}")

    # ── Datasets — encoder injected, n_mels drives model shape ─
    train_ds = PCGSpikeDataset(TRAIN_DIR, encoder=encoder, T_sim=T_SIM)
    val_ds   = PCGSpikeDataset(VAL_DIR,   encoder=encoder, T_sim=T_SIM)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

    # n_mels comes directly from the encoder — no manual sync needed
    model     = ConvSNN1D(n_mels=encoder.n_channels, beta=BETA).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn   = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, T_SIM, loss_fn, optimizer)
        vl_loss, vl_acc = run_epoch(model, val_loader,   device, T_SIM, loss_fn)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train loss {tr_loss:.4f}  acc {tr_acc*100:.1f}% | "
              f"Val loss {vl_loss:.4f}  acc {vl_acc*100:.1f}%")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), "best_pcg_snn_chirp.pth")
            print(f"  → saved best model (val acc {vl_acc*100:.1f}%)")

    print(f"\nBest validation accuracy: {best_val_acc*100:.1f}%")

    # ── Detailed evaluation on best checkpoint ─────────────────
    print("\nLoading best checkpoint for detailed evaluation...")
    model.load_state_dict(torch.load("best_pcg_snn_chirp.pth", map_location=device))
    evaluate_detailed(model, val_loader, device, T_SIM, n_runs=10)