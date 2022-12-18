import pickle
from typing import Union, List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, \
                            recall_score, f1_score


class PipeLine(object):

    def __init__(self, train_flg=True):
        """
        アトリビュートで訓練データ、正解データを管理する
        （引数）
        train_flg: 訓練時と推論時を識別するフラグ
        """
        self.df: Optional[pd.DataFrame] = None  # callからオリジナルデータが格納される
        self.df_num: Optional[pd.DataFrame] = None  # callから数値データが格納される
        self.df_cat: Optional[pd.DataFrame] = None  # callからカテゴリデータが格納される
        self.df_target: Optional[pd.DataFrame] = None  # callで正解データを指定する
        self.train_flg: bool = train_flg  # 正解ラベルのないテストデータはFalseを設定
        self.viewer = False  # 更新したカラムの表示を切り替え
        self.viewer_row = 5  # 表示カラムの行数
        self.random_seed = 42  # 乱数シード値

    def __call__(
                self,
                data: pd.DataFrame,
                target='HeartDisease'
                ) -> pd.DataFrame:
        """
        objectタイプを検出して数値データとカテゴリデータに分別しアトリビュートに格納する
        （引数）
        data: 使用するオリジナルデータ
        target: 正解データのカラム名
        """
        if self.train_flg:
            df = data.copy().drop(target, axis=1)
        else:
            df = data.copy()
        num, cat = [], []
        for column in df.columns:
            if df.dtypes[column] == 'O':
                cat.append(column)
            else:
                num.append(column)

        self.df = data
        self.df_num = data[num].reset_index(drop=True)
        self.df_cat = data[cat].reset_index(drop=True)
        # 正解ラベルが与えられない本番環境では引数からFalseにすること
        if self.train_flg:
            self.df_target = data[target].reset_index(drop=True)
        return self.df_num

    def standard_scaler(self) -> None:
        """
        アトリビュートに格納された数値データを標準化する
        """
        columns = self.df_num.columns
        scaler = StandardScaler()
        scaler.fit(self.df_num)
        self.df_num = scaler.transform(self.df_num)
        self.df_num = pd.DataFrame(self.df_num, columns=columns)
        if self.viewer:
            print('-'*20, '標準化されたdf_num', '-'*20)
            display(self.df_num.head(self.viewer_row))
        return None

    def one_hot(self, columns: List[str]) -> None:
        """
        指定されたからカラム名をワンホットエンコードする
        （引数）
        columns: ワンホットするカラム名
        """
        one_hotted = pd.get_dummies(self.df_cat[columns]).reset_index(drop=True)
        self.df_num = pd.concat((self.df_num, one_hotted), axis=1)
        if self.viewer:
            print('-'*20, f'ワンホットされたカラム{columns}', '-'*20)
            display(self.df_num.head(self.viewer_row))
        return None

    def fold_out_split(
                    self,
                    test_size=0.3,
                    to_array=False
                    ) -> Tuple[np.ndarray]:
        """
        アトリビュートに格納されている数値データをホールドアウト法で分割
        （引数）
        test_size: 分割後のテストデータの割合
        to_array: DataFrameかndarrayで出力するかを変更
        """
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

    def k_fold(
                self,
                n_splits=5,
                to_array=True
                ) -> List[List[np.ndarray, np.ndarray]]:
        """
        交差検証用のデータセットを出力。[[訓練データ, 正解データ]*分割数]
        （引数）
        n_split: 検証データの分割数
        to_array: DataFrameかndarrayで出力するかを変更
        """
        kf = KFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.random_seed
                    )
        packs = []
        for train_index, test_index in kf.split(self.df_num):
            x_tr = self.df_num.iloc[train_index]
            x_te = self.df_num.iloc[test_index]
            y_tr = self.df_target.iloc[train_index]
            y_te = self.df_target.iloc[test_index]
            pack = (x_tr, x_te,  y_tr, y_te)
            if to_array:
                pack = [unpack.values for unpack in pack]
            packs.append(pack)
        if self.viewer:
            print(kf.get_n_splits)
        return packs

    def training(
                self,
                valid: str,
                model: object,
                valid_args={},
                params={},
                view=True
                ) -> object:
        """
        pipelineでのモデル訓練
        （引数）
        valid: fold_out_splitもしくはk_foldを指定
        model: fitメソッドに対応したモデルクラス
        valid_args: pipelineの検証メソッドの引数
        params: モデルパラメータを指定
        """
        if view:
            print('-'*20, '使用された特徴量', '-'*20)
            display(self.df_num.head(self.viewer_row))

        if valid == 'fold_out_split':
            packs = self.fold_out_split(**valid_args)
            train_model = model(**params)
            train_model.fit(packs[0], packs[2])
            evaluations(train_model, *packs)
            return train_model

        if valid == 'k_fold':
            packs = self.k_fold(**valid_args)
            models = []
            for i, pack in enumerate(packs):
                train_model = model(**params)
                train_model.fit(pack[0], pack[2].reshape(-1))
                print('-'*20, f'model{i} predict', '-'*20)
                evaluations(train_model, *pack)
                models.append(train_model)
            return models


# グリッドサーチの関数
def grid_search_cv(
                    pack: List[np.ndarray],
                    param_grid: Dict[str, List[Union[str, int, float]]],
                    model: object,
                    model_arg={},
                    score='accuracy'
                    ) -> object:
    """
    グリッドサーチ関数
    （引数）
    pack: 検証用データセット (x_train, x_test, y_train, y_test) = pack
    param_grid: グリッドサーチの探索対象パラメータ
    model: モデルクラス
    model_arg: 探索対象以外のパラメータ
    """
    gs_model = GridSearchCV(estimator=model(**model_arg),
                            param_grid=param_grid,  # 設定した候補を代入
                            scoring=score,  # デフォルトではaccuracyを基準に探してくれる
                            refit=True,
                            cv=3,
                            n_jobs=-1)
    # 訓練データで最適なパラメータを交差検証する
    if model.__name__ == 'XGBClassifier':  # early_stoppingのためにeval_setを用意
        eval_set = [(pack[1], pack[3])]  # x_train, x_test, y_train, y_test = pack
        gs_model.fit(pack[0], pack[2], eval_set=eval_set, verbose=False)
    else:
        gs_model.fit(pack[0], pack[2])
    evaluations(gs_model, *pack)
    return gs_model


# 評価指標の関数
def evaluations(
                model: object,
                x_train: np.ndarray,
                x_test: np.ndarray,
                y_train: np.ndarray,
                y_test: np.ndarray
                ) -> pd.DataFrame:
    """
    正解率、適合率、再現率、F1スコアをDataFrameで表示
    （引数）
    mdoel: 訓練済みモデル
    x_train, x_test, y_train, y_test: モデル訓練に使用した前処理済みデータセット
    """
    evaluate = [accuracy_score, precision_score, recall_score, f1_score]
    # 訓練データの評価
    train_pred = model.predict(x_train)
    train_val = {func.__name__: func(y_train, train_pred) for func in evaluate}
    # 検証データの評価
    test_pred = model.predict(x_test)
    test_val = {func.__name__: func(y_test, test_pred) for func in evaluate}
    evals = pd.DataFrame((train_val, test_val), index=['train', 'test'])
    print('-'*20, '評価結果', '-'*20)
    display(evals)
    return evals


def k_fold_prediction(models: object, x: np.ndarray) -> np.ndarray:
    """
    交差検証によるアンサンブル
    （引数）
    model: 訓練済みモデル
    x: 前処理済み推論データ
    """
    try:  # probaによる出力が可能である場合
        predict = [model.predict_proba(x) for model in models]
        predict_sum = np.sum(predict, axis=0)
        ensemble_prediction = np.array(
            [np.where(pre[0] < pre[1], 1, 0) for pre in predict_sum]
            )
    except AttributeError:  # probaによる出力が可能でない場合
        print('確率で出力するようパラメータもしくはモデルを選択することを推奨'.center(100))
        predict = [model.predict(x) for model in models]
        predict_sum = np.sum(predict, axis=0)
        ensemble_prediction = np.array(
                [np.where(len(models)//2 <= pre, 1, 0) for pre in predict_sum]
                )
    return ensemble_prediction


def ensemble_evals(
                    ensemble_pred: np.ndarray,
                    target: np.ndarray
                    ) -> pd.DataFrame:
    """
    予測値を入力して評価する関数
    （引数）
    ensemble_pred: アンサンブルによる出力結果
    target: 正解データ
    """
    evaluate = [accuracy_score, precision_score, recall_score, f1_score]
    # 訓練データの評価
    ensemble = {func.__name__: func(target, ensemble_pred) for func in evaluate}
    evals = pd.DataFrame((ensemble), index=['ensemble'])
    print('-'*20, 'ensemble', '-'*20)
    display(evals)
    return None


# 最適パラメータでの再訓練用関数
def best_parameters(
                    train_models: Dict[str, object],
                    pipe_lines: List[object]
                    ) -> Dict[str, Dict[str, str]]:
    """
    訓練済みモデルから最適なパラメータを抽出
    （引数）
    train_models: 訓練済みモデルを格納した辞書
    pipe_lines: 訓練に使用したデータセットオブジェクトを格納したリスト
    """
    parameters = {}
    for key in train_models.keys():
        model = train_models[key]
        best_params = {}
        for pipe in pipe_lines:
            try:
                best = model[pipe.__name__].best_params_
                best_params[pipe.__name__] = best
            except KeyError:
                pass
        parameters[key] = best_params
    return parameters


def retrained(
            retrain: List[object],
            pipe_lines: List[object],
            data_set: Dict[str, List[np.ndarray]],
            best_param: Dict[str, Dict[str, Union[str, int, float]]],
            file_name: str
            ) -> Dict[str, object]:
    """
    テストデータの推論前に最適なパラメータでモデルの再訓練
    （引数）
    retrain: 再訓練対象のモデルオブジェクト
    pipe_lines: 訓練に使用したデータセットオブジェクトを格納したリスト
    data_set: 訓練医に使用したデータセット
    best_param: 各モデルの最適なパラメータを格納した辞書
    file_name: 再訓練したモデルを保存するファイル名
    """
    retrained = {}
    for re_tr in retrain:
        retrained[re_tr.__name__] = {}
        for pipe in pipe_lines:
            x, y = data_set[pipe.__name__]
            try:
                param = best_param[re_tr.__name__][pipe.__name__]
                model = re_tr(**param)
                model.fit(x.values, y.values.reshape(-1))
                retrained[re_tr.__name__][pipe.__name__] = model
                with open(f'./data/retrained_{file_name}.pkl', 'wb') as f:
                    pickle.dump(retrained, f)
            except KeyError:
                pass
    return retrained


# サブミット用の評価関数
def test_eval(
            train_models: Dict[str, object],
            pipe_lines: List[object],
            data_set: Dict[str, List[np.ndarray]],
            y: np.ndarray
            ) -> pd.DataFrame:
    """
    訓練済みモデルで正解率を評価
    （引数）
    train_model: 訓練済みモデルを格納した辞書
    pipe_lines: 訓練に使用したデータセットオブジェクトを格納したリスト
    data_set: 訓練医に使用したデータセット
    y: 正解データ
    """
    predicts = {}
    for key in train_models.keys():
        if train_models[key] != {}:
            model = train_models[key]
            scores = []
            index = []
            for pipe in pipe_lines:
                try:
                    x = data_set[pipe.__name__]
                    pred = model[pipe.__name__].predict(x)
                    score = accuracy_score(y.values, pred)
                    scores.append(score)
                    index.append(pipe.__name__)
                except KeyError:
                    pass
            predicts[key] = scores
        else:
            pass
    return pd.DataFrame(predicts, index=index)
