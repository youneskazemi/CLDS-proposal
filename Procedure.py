import numpy as np
import torch

import utils
import world


def BPR_train_original(dataset, recommend_model, loss_class, epoch):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    allusers = list(range(dataset.n_users))
    S, sam_time = utils.UniformSample_original(
        allusers, dataset
    )  # [user,pos,neg], [times list]

    users = torch.tensor(S[:, 0], dtype=torch.long, device=world.device)
    posItems = torch.tensor(S[:, 1], dtype=torch.long, device=world.device)
    negItems = torch.tensor(S[:, 2], dtype=torch.long, device=world.device)

    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config["bpr_batch_size"] + 1
    aver_loss = 0.0

    for batch_i, (batch_users, batch_pos, batch_neg) in enumerate(
        utils.minibatch(
            users, posItems, negItems, batch_size=world.config["bpr_batch_size"]
        )
    ):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg, epoch)
        aver_loss += cri

    aver_loss = aver_loss / total_batch
    return aver_loss


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:  # [10, 20]
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret["precision"])
        recall.append(ret["recall"])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {
        "recall": np.array(recall),
        "precision": np.array(pre),
        "ndcg": np.array(ndcg),
    }


def Test(dataset, Recmodel, epoch, cold=False, w=None):
    u_batch_size = world.config["test_u_batch_size"]
    testDict: dict = dataset.coldTestDict if cold else dataset.testDict
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    results = {
        "precision": np.zeros(len(world.topks)),
        "recall": np.zeros(len(world.topks)),
        "ndcg": np.zeros(len(world.topks)),
    }
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) // 10
        except AssertionError:
            print(
                f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}"
            )
        users_list, rating_list, groundTrue_list = [], [], []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]

            batch_users_gpu = torch.tensor(
                batch_users, dtype=torch.long, device=world.device
            )
            rating = Recmodel.getUsersRating(batch_users_gpu)

            exclude_index, exclude_items = [], []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            del rating

            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)

        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        pre_results = [test_one_batch(x) for x in X]
        for result in pre_results:
            results["recall"] += result["recall"]
            results["precision"] += result["precision"]
            results["ndcg"] += result["ndcg"]
        results["recall"] /= float(len(users))
        results["precision"] /= float(len(users))
        results["ndcg"] /= float(len(users))
        print(results)
        return results
