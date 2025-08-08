import gc
import json
import os
import re
from time import time
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import torch
import copy
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split, GroupShuffleSplit

import world


def _convert_sp_mat_to_sp_tensor(X):
    """
    Convert a scipy sparse matrix to a torch sparse COO tensor on world.device.
    """
    coo = X.tocoo().astype(np.float32)
    indices = torch.tensor(
        np.vstack((coo.row, coo.col)), dtype=torch.long, device=world.device
    )
    values = torch.tensor(coo.data, dtype=torch.float32, device=world.device)
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=torch.Size(coo.shape),
        device=world.device,
        dtype=torch.float32,
    ).coalesce()


class PairDataset:
    def __init__(self, src="lastfm"):
        self.src = src

        self.train_set = pd.read_csv(f"./data/preprocessed/{src}/train_set.txt")
        self.test_set = pd.read_csv(f"./data/preprocessed/{src}/test_set.txt")

        self.n_user = pd.concat([self.train_set, self.test_set])["user"].nunique()
        self.m_item = pd.concat([self.train_set, self.test_set])["item"].nunique()
        self.trainUser = np.array(self.train_set["user"])
        self.trainUniqueUser = np.unique(self.train_set["user"])
        self.trainItem = np.array(self.train_set["item"])
        self._trainDataSize = len(self.train_set)
        self._testDataSize = len(self.test_set)
        print(f"{self._trainDataSize} interactions for training")
        print(f"{self._testDataSize} interactions for testing")
        print(f"Number of users: {self.n_user}\n Number of items: {self.m_item}")
        print(f"Number of Ratings: {self._trainDataSize + self._testDataSize}")
        print(
            f"{world.dataset} Rating Density: {(self._trainDataSize + self._testDataSize) / self.n_user / self.m_item}"
        )

        # build (users,items) bipartite graph
        self.interactionGraph = None
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.train_set)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item),
        )
        # users' history interacted items
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        # test dictionaries
        self._testDic = self.__build_test()
        self._coldTestDic = self.__build_cold_test()
        self._userDic, self._itemDic = self._getInteractionDic()

    @property
    def userDic(self):
        return self._userDic

    @property
    def itemDic(self):
        return self._itemDic

    @property
    def testDict(self):
        return self._testDic

    @property
    def coldTestDict(self):
        return self._coldTestDic

    @property
    def allPos(self):
        return self._allPos

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self._trainDataSize

    def getUserPosItems(self, users):
        """
        Get all positive items for each user.
        Returns: list of np.ndarray, one per user.
        """
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def __build_test(self):
        test_data = {}
        for i in range(len(self.test_set)):
            user = self.test_set["user"][i]
            item = self.test_set["item"][i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def __build_cold_test(self):
        test_data = {}
        for i in range(len(self.test_set)):
            user = self.test_set["user"][i]
            item = self.test_set["item"][i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        for i in list(test_data.keys()):
            if self.train_set["user"].value_counts()[i] > 20:
                del test_data[i]
        return test_data

    def _getInteractionDic(self):
        user_interaction = {}
        item_interaction = {}

        def getDict(_set):
            for i in range(len(_set)):
                user = _set["user"][i]
                item = _set["item"][i]
                if user_interaction.get(user):
                    user_interaction[user].append(item)
                else:
                    user_interaction[user] = [item]
                if item_interaction.get(item):
                    item_interaction[item].append(user)
                else:
                    item_interaction[item] = [user]

        getDict(self.train_set)
        getDict(self.test_set)
        return user_interaction, item_interaction


class GraphDataset:
    def __init__(self, src="lastfm"):
        self.src = src
        self.train_set = pd.read_csv(f"./data/preprocessed/{src}/train_set.txt")
        self.test_set = pd.read_csv(f"./data/preprocessed/{src}/test_set.txt")
        self.friendNet = pd.read_csv(f"./data/preprocessed/{src}/trust.txt")

        self.n_user = pd.concat([self.train_set, self.test_set])["user"].nunique()
        self.m_item = pd.concat([self.train_set, self.test_set])["item"].nunique()
        self.trainUser = np.array(self.train_set["user"])
        self.trainUniqueUser = np.unique(self.train_set["user"])
        self.trainItem = np.array(self.train_set["item"])
        self._trainDataSize = len(self.train_set)
        self._testDataSize = len(self.test_set)
        print(f"{self._trainDataSize} interactions for training")
        print(f"{self._testDataSize} interactions for testing")
        print(f"Number of users: {self.n_user}\n Number of items: {self.m_item}")
        print(f"Number of Ratings: {self._trainDataSize + self._testDataSize}")
        print(
            f"{world.dataset} Rating Density: {(self._trainDataSize + self._testDataSize) / self.n_user / self.m_item}"
        )

        self.interactionGraph = None
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.train_set)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item),
        )
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self._testDic = self.__build_test()
        self._coldTestDic = self.__build_cold_test()
        self._userDic, self._itemDic = self._getInteractionDic()

    @property
    def userDic(self):
        return self._userDic

    @property
    def itemDic(self):
        return self._itemDic

    @property
    def testDict(self):
        return self._testDic

    @property
    def coldTestDict(self):
        return self._coldTestDic

    @property
    def allPos(self):
        return self._allPos

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self._trainDataSize

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def __build_test(self):
        test_data = {}
        for i in range(len(self.test_set)):
            user = self.test_set["user"][i]
            item = self.test_set["item"][i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def __build_cold_test(self):
        test_data = {}
        for i in range(len(self.test_set)):
            user = self.test_set["user"][i]
            item = self.test_set["item"][i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        for i in list(test_data.keys()):
            if self.train_set["user"].value_counts()[i] > 20:
                del test_data[i]
        return test_data

    def getInteractionGraph(self):
        print("loading adjacency matrix")
        if self.interactionGraph is None:
            try:
                pre_adj_mat = sp.load_npz(
                    f"./data/preprocessed/{self.src}/interaction_adj1_mat.npz"
                )
                pre_adj_mat2 = sp.load_npz(
                    f"./data/preprocessed/{self.src}/interaction_adj2_mat.npz"
                )
                print("successfully loaded...")
                norm_adj = pre_adj_mat
                norm2_adj = pre_adj_mat2  # <-- fixed (was pre_adj_mat)
            except IOError:
                print("generating adjacency matrix")
                start = time()

                adj_mat = sp.dok_matrix(
                    (self.n_user + self.m_item, self.n_user + self.m_item),
                    dtype=np.float32,
                ).tolil()
                adj2_mat = sp.dok_matrix(
                    (self.n_user + self.m_item, self.n_user + self.m_item),
                    dtype=np.float32,
                ).tolil()
                R = self.UserItemNet.tolil()
                U_U = R.dot(R.T)
                I_I = R.T.dot(R)

                adj_mat[: self.n_user, self.n_user :] = R
                adj_mat[self.n_user :, : self.n_user] = R.T
                adj_mat = adj_mat.todok()

                adj2_mat[: self.n_user, : self.n_user] = U_U
                adj2_mat[self.n_user :, self.n_user :] = I_I
                adj2_mat = adj2_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.0
                d_mat = sp.diags(d_inv)

                rowsum2 = np.array(adj2_mat.sum(axis=1))
                d2_inv = np.power(rowsum2, -0.5).flatten()
                d2_inv[np.isinf(d2_inv)] = 0.0
                d2_mat = sp.diags(d2_inv)

                # D^-0.5 * A * D^-0.5
                norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocsr()
                norm2_adj = d_mat.dot(adj2_mat).dot(d2_mat).tocsr()

                print(f"costing {time() - start}s, saved norm_mat...")
                sp.save_npz(
                    f"./data/preprocessed/{self.src}/interaction_adj1_mat.npz", norm_adj
                )
                sp.save_npz(
                    f"./data/preprocessed/{self.src}/interaction_adj2_mat.npz",
                    norm2_adj,
                )

            self.interactionGraph = _convert_sp_mat_to_sp_tensor(norm_adj)
            self.interactionGraph2 = _convert_sp_mat_to_sp_tensor(norm2_adj)
        return self.interactionGraph, self.interactionGraph2

    def _getInteractionDic(self):
        user_interaction = {}
        item_interaction = {}

        def getDict(_set):
            for i in range(len(_set)):
                user = _set["user"][i]
                item = _set["item"][i]
                if user_interaction.get(user):
                    user_interaction[user].append(item)
                else:
                    user_interaction[user] = [item]
                if item_interaction.get(item):
                    item_interaction[item].append(user)
                else:
                    item_interaction[item] = [user]

        getDict(self.train_set)
        getDict(self.test_set)
        return user_interaction, item_interaction


class SocialGraphDataset(GraphDataset):
    def __init__(self, src):
        super(SocialGraphDataset, self).__init__(src)

        self.friendNet = pd.read_csv(f"./data/preprocessed/{src}/trust.txt")
        self.socialNet = csr_matrix(
            (
                np.ones(len(self.friendNet)),
                (self.friendNet["user"], self.friendNet["friend"]),
            ),
            shape=(self.n_user, self.n_user),
        )

        self.interactionGraph = None
        self.socialGraph1 = None
        self.socialGraph2 = None
        self.socialGraph3 = None

        print(f"Number of Links: {len(self.friendNet)}")
        print(
            f"{world.dataset} Link Density: {len(self.friendNet) / self.n_user / self.n_user}"
        )

    def getSocialGraph1(self):
        if self.socialGraph1 is None:
            try:
                pre_adj_mat = sp.load_npz(
                    f"./data/preprocessed/{self.src}/social1_adj_mat.npz"
                )
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except IOError:
                print("generating adjacency matrix1")
                start = time()
                R = self.UserItemNet.tolil()

                rowsum = np.array(R.sum(axis=1))
                print(
                    "The number of user's average preference items:",
                    sum(rowsum) / len(rowsum),
                )
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.0
                d_mat = sp.diags(d_inv)

                adj_mat = self.socialNet.tolil()
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)

                value = []
                kk = 0
                edge = 0
                tmp_R = np.array(R.todense())
                for i, j in zip(self.friendNet["user"], self.friendNet["friend"]):
                    tt = tmp_R[i] + tmp_R[j]
                    tmp_degree = sum(tt == 2)
                    kk += tmp_degree
                    edge += 1
                    value.append(tmp_degree)
                avg_kk = kk / edge
                print("avg_kk:", avg_kk)

                simNet = csr_matrix(
                    (
                        np.array(value),
                        (self.friendNet["user"], self.friendNet["friend"]),
                    ),
                    shape=(self.n_user, self.n_user),
                ).tolil()

                norm_adj = norm_adj.multiply(simNet)
                norm_adj = norm_adj + sp.eye(norm_adj.shape[0])
                norm_adj = norm_adj.tocsr()

                print(f"costing {time() - start}s, saved norm_mat...")
                sp.save_npz(
                    f"./data/preprocessed/{self.src}/social1_adj_mat.npz", norm_adj
                )

            self.socialGraph1 = _convert_sp_mat_to_sp_tensor(norm_adj)
        return self.socialGraph1

    def getSocialGraph2(self):
        if self.socialGraph2 is None:
            try:
                pre_adj_mat = sp.load_npz(
                    f"./data/preprocessed/{self.src}/social2_adj_mat.npz"
                )
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except IOError:
                print("generating adjacency matrix2")
                start = time()
                adj_mat = self.socialNet.tolil()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.0
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                print(f"costing {time() - start}s, saved norm_mat...")
                sp.save_npz(
                    f"./data/preprocessed/{self.src}/social2_adj_mat.npz", norm_adj
                )

            self.socialGraph2 = _convert_sp_mat_to_sp_tensor(norm_adj)
        return self.socialGraph2

    def getDenseSocialGraph(self):
        if self.socialGraph1 is None:
            self.socialGraph1 = self.getSocialGraph1().to_dense()
            self.socialGraph2 = self.getSocialGraph2().to_dense()
        return self.socialGraph1, self.socialGraph2
