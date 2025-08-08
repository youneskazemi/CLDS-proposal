import time
from os.path import join

import torch
from tqdm import trange, tqdm

import Procedure
import register
import utils
import world
from register import dataset

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)

# Device info / speed tweaks
d = world.device
print("Using device:", d)
if d.type == "cuda":
    try:
        print("GPU:", torch.cuda.get_device_name(0))
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True
# ==============================

torch.autograd.set_detect_anomaly(False)

# Build model & loss
Recmodel = register.MODELS[world.model_name](world.config, dataset).to(d)
bpr = utils.BPRLoss(Recmodel, world.config)

# Load weights if requested
weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        state = torch.load(weight_file, map_location=d)
        Recmodel.load_state_dict(state)
        print(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

# Best metrics trackers
best0_ndcg, best0_recall, best0_pre = 0, 0, 0
best1_ndcg, best1_recall, best1_pre = 0, 0, 0
low_count = 0
start = time.time()

# Eval / save cadence
EVAL_START = 200  # start evaluating after this epoch
EVAL_EVERY = 10  # evaluate every N epochs
SAVE_EVERY = 100  # save checkpoint every N epochs

try:
    pbar = trange(world.TRAIN_epochs + 1, desc="Training", ncols=100)
    for epoch in pbar:
        # ---- Train one epoch ----
        loss = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch)

        # Show lightweight stats in the bar
        postfix = {"loss": f"{loss:.3f}"}
        if d.type == "cuda":
            mem = torch.cuda.memory_allocated() / (1024**2)
            postfix["gpu_mem"] = f"{mem:.0f}MB"
        pbar.set_postfix(postfix)

        # ---- Save checkpoints periodically ----
        if (epoch % SAVE_EVERY == 0 and epoch > 0) or epoch == world.TRAIN_epochs:
            try:
                torch.save(Recmodel.state_dict(), weight_file)
                tqdm.write(f"[save] epoch {epoch} -> {weight_file}")
            except Exception as e:
                tqdm.write(f"[save failed] {e}")

        # ---- Evaluate periodically (earlier than original 2000) ----
        if (epoch >= EVAL_START) and (
            epoch % EVAL_EVERY == 0 or epoch == world.TRAIN_epochs
        ):
            tqdm.write("[TEST] running evaluation...")
            results = Procedure.Test(dataset, Recmodel, epoch, False)
            # compact summary
            pre10, pre20 = results["precision"][0], results["precision"][1]
            rec10, rec20 = results["recall"][0], results["recall"][1]
            ndcg10, ndcg20 = results["ndcg"][0], results["ndcg"][1]
            tqdm.write(
                f"[TEST] P@10 {pre10:.4f} | R@10 {rec10:.4f} | NDCG@10 {ndcg10:.4f} "
                f"|| P@20 {pre20:.4f} | R@20 {rec20:.4f} | NDCG@20 {ndcg20:.4f}"
            )

            # early-stop style tracking (same logic as before, but quieter)
            if ndcg10 < best0_ndcg:
                low_count += 1
                if low_count == 30:
                    if epoch > EVAL_START * 2:
                        tqdm.write("[early stop] patience reached.")
                        break
                    else:
                        low_count = 0
            else:
                best0_recall = rec10
                best0_ndcg = ndcg10
                best0_pre = pre10
                low_count = 0

            if ndcg20 >= best1_ndcg:
                best1_recall = rec20
                best1_ndcg = ndcg20
                best1_pre = pre20

    end = time.time()
    tqdm.write(f"Total time: {(end - start) / 60:.2f} min")

finally:
    tqdm.write(f"best precision at 10: {best0_pre:.6f}")
    tqdm.write(f"best precision at 20: {best1_pre:.6f}")
    tqdm.write(f"best recall    at 10: {best0_recall:.6f}")
    tqdm.write(f"best recall    at 20: {best1_recall:.6f}")
    tqdm.write(f"best ndcg      at 10: {best0_ndcg:.6f}")
    tqdm.write(f"best ndcg      at 20: {best1_ndcg:.6f}")
