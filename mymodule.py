from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, precision_score, \
                            recall_score, f1_score


class PipeLine(object):
    """
    __init__ :
    アトリビュートで訓練データ、正解データを管理する
    （引数）
    df: callメソッドでオリジナルのデータが格納される
    df_num: callで指定した数値データが格納される。クラスメソッドで上書きされる
    df_cat: callで指定したカテゴリデータが格納される
    viewer: bool, viewer_row :int 更新後のデータを表示する

    __call__ :
    インスタンスの生成時に引数でオリジナルデータを渡す
    （引数）
    data: 使用するオリジナルデータ  (pd.DataFrame
    numerical: 数値データのカラム名  (list[str])
    categorical: カテゴリデータのカラム名  (list[str])
    target: 

    standard_scaler:
    アトリビュートのdf_numを標準化する
    （引数）
    view: 標準化したdf_numを確認できる

    one_hot: 
    指定したカラムのdf_catをワンホット化してdf_numにconcatする
    （引数）
    columns: ワンホット化したいカラム名_ 
    concat: Trueの場合はアトリビュートのself.df_numに連結し更新する
    view: ワンホットされたデータがdf_numにconcatされているのを確認できる
    return: x_train, x_test, y_train, y_test
    """

    def __init__(self):
        self.df: pd.DataFrame = None
        self.df_num: pd.DataFrame = None
        self.df_cat: pd.DataFrame = None
        self.df_target: pd.DataFrame = None
        self.viewer = True  # 更新したカラムの表示を切り替え
        self.viewer_row = 3  # 表示カラムの行数
        self.random_seed = 42  # 乱数シード値

    def __call__(self,
                 data: pd.DataFrame,
                 numerical=['Age', 'Sex', 'RestingBP', 'Cholesterol',\
                 'FastingBS', 'MaxHR', 'ExerciseAngina', 'Oldpeak'],
                 categorical=['ChestPainType', 'RestingECG', 'ST_Slope'],
                 target=['HeartDisease'],
                 train_flg=True
                 ) -> pd.DataFrame:

        self.df = data
        self.df_num = data[numerical]
        self.df_cat = data[categorical]
        # 正解ラベルが与えられない本番環境では引数からFalseにすること
        if train_flg:
            self.df_target = data[target]
        return self.df_num

    def standard_scaler(self):
        columns = self.df_num.columns
        scaler = StandardScaler()
        scaler.fit(self.df_num)
        self.df_num = scaler.transform(self.df_num)
        self.df_num = pd.DataFrame(self.df_num, columns=columns)
        if self.viewer:
            print('-'*20, '標準化されたdf_num', '-'*20)
            display(self.df_num.head(self.viewer_row))
        return None

    def one_hot(self, columns: list[str], concat=True) -> pd.DataFrame:
        one_hotted = pd.get_dummies(self.df_cat[columns])
        self.df_num = pd.concat((self.df_num, one_hotted), axis=1)
        if self.viewer:
            print('-'*20, f'ワンホットされたカラム{columns}', '-'*20)
            display(self.df_num.head(self.viewer_row))
        return None

    def fold_out_split(self, test_size=0.3, to_array=True) -> np.ndarray:
        pack = train_test_split(self.df_num,  self.df_target,
                                test_size=test_size,
                                random_state=self.random_seed)
        x_tr, x_te, y_tr, y_te = pack
        if to_array:
            x_tr, x_te, y_tr, y_te = [i.values for i in pack]
            y_tr, y_te = y_tr.reshape(-1), y_te.reshape(-1)
        if self.viewer:
            print('-'*20, '分割されたデータShape', '-'*20)
            print(f'x_train: {x_tr.shape} x_test: {x_te.shape}')
            print(f'y_train: {y_tr.shape} y_test: {y_te.shape}')
        return x_tr, x_te, y_tr, y_te

    def k_fold(self, n_splits=5, to_array=True) -> list[list[np.ndarray]]:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        datasets = []
        for train_index, test_index in kf.split(self.df_num):
            x_tr, x_te = self.df_num.iloc[train_index], self.df_num.iloc[test_index]
            y_tr, y_te = self.df_target.iloc[train_index], self.df_target.iloc[test_index]
            pack = (x_tr, x_te,  y_tr, y_te)
            if to_array:
                pack = [unpack.values for unpack in pack]
            datasets.append(pack)
        if self.viewer: 
            print(kf.get_n_splits)
        return datasets

    def training(self, valid, Model, valid_args={}, params={}):
        if valid == 'fold_out_split':
            pack = self.fold_out_split(**valid_args)
            model = Model(**params)
            model.fit(pack[0], pack[2])
            evaluations(model, *pack)
            return model

        if valid == 'k_fold':
            packs = self.k_fold(**valid_args)
            models = []
            for i, pack in enumerate(packs):
                model = Model(**params)
                model.fit(pack[0], pack[2].reshape(-1))
                print('-'*20, f'model{i} predict', '-'*20)
                evaluations(model, *pack)
                models.append(model)
            return models


# evaluation
def evaluations(model, x_train, x_test, y_train, y_test):
    evaluate = [accuracy_score, precision_score, recall_score, f1_score]
    # 訓練データの評価
    train_pred = model.predict(x_train)
    train_val = {func.__name__: func(y_train, train_pred) for func in evaluate}
    # 検証データの評価
    test_pred = model.predict(x_test)
    test_val = {func.__name__: func(y_test, test_pred) for func in evaluate}
    evals = pd.DataFrame((train_val, test_val), index=['train', 'test'])
    display(evals)
    return evals
