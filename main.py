import time
from os.path import join

import torch

import Procedure
import register
import utils
import world
from register import dataset

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)

# ---- Device info / speed tweaks ----
d = world.device
print("Using device:", d)
if d.type == "cuda":
    try:
        print("GPU:", torch.cuda.get_device_name(0))
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True  # speed up conv-like ops if shapes are stable
# ==============================

# Turn this off unless you are debugging exploding gradients
torch.autograd.set_detect_anomaly(False)

# Build model on the selected device
Recmodel = register.MODELS[world.model_name](world.config, dataset).to(d)

# Create optimizer-dependent objects AFTER moving model to device
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        # <<< ensure loading to correct device >>>
        state = torch.load(weight_file, map_location=d)
        Recmodel.load_state_dict(state)
        print(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

best0_ndcg, best0_recall, best0_pre = 0, 0, 0
best1_ndcg, best1_recall, best1_pre = 0, 0, 0
best0_ndcg_cold, best0_recall_cold, best0_pre_cold = 0, 0, 0
best1_ndcg_cold, best1_recall_cold, best1_pre_cold = 0, 0, 0
low_count, low_count_cold = 0, 0
start = time.time()

tip = "pre"
try:
    for epoch in range(world.TRAIN_epochs + 1):
        print("======================")
        print(f"EPOCH[{epoch}/{world.TRAIN_epochs}]")

        # Quick GPU mem peek (useful on Kaggle to verify)
        if d.type == "cuda":
            print(
                f"[gpu] mem_alloc={torch.cuda.memory_allocated()/1024**2:.1f}MB | mem_reserved={torch.cuda.memory_reserved()/1024**2:.1f}MB"
            )

        if epoch > 2000 and (epoch % 10 == 1 or epoch == world.TRAIN_epochs):
            print("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, False)
            # results_cold = Procedure.Test(dataset, Recmodel, epoch, True)
            if results["ndcg"][0] < best0_ndcg:
                low_count += 1
                if low_count == 30:
                    if epoch > 1000:
                        break
                    else:
                        low_count = 0
            else:
                best0_recall = results["recall"][0]
                best0_ndcg = results["ndcg"][0]
                best0_pre = results["precision"][0]
                low_count = 0

            if results["ndcg"][1] >= best1_ndcg:
                best1_recall = results["recall"][1]
                best1_ndcg = results["ndcg"][1]
                best1_pre = results["precision"][1]

        # >>> Training step; Procedure should push tensors to world.device <<<
        loss = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch)
        print(f"[saved][BPR aver loss{loss:.3e}]")

    end = time.time()
    print("The total time:", (end - start) / 60)
    # torch.save(Recmodel.state_dict(), weight_file)
finally:
    print(f"best precision at 10:{best0_pre}")
    print(f"best precision at 20:{best1_pre}")
    print(f"best recall at 10:{best0_recall}")
    print(f"best recall at 20:{best1_recall}")
    print(f"best ndcg at 10:{best0_ndcg}")
    print(f"best ndcg at 20:{best1_ndcg}")
