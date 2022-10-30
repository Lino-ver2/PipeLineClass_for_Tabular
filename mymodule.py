import pandas as pd


def one_hot_split(
        path='./data/train.csv',
        one_cat=[],
        display_columns=True,
        test_size=0.3):
    from sklearn.model_selection import train_test_split

    # データを読み込む関数
    df = pd.read_csv(path)
    cats = ['ChestPainType', 'RestingECG', 'ST_Slope']

    # 指定されたものをワンホット化
    if one_cat != []:
        df_cat = df[one_cat]
        oned_cat = pd.get_dummies(df_cat)
        df = pd.concat((df, oned_cat), axis=1)

    # 説明変数と目的変数を分割
    x = df.drop(cats + ['HeartDisease'], axis=1)
    y = df.HeartDisease
    # ワンホット化した説明位変数を表示
    if display_columns:
        print('-'*15, f' {one_cat}をエンコード', '-'*15)
        display(x)
    # 訓練データと検証データをホールドアウトで分割
    x_train, x_test, y_train, y_test = train_test_split(
                                                        x, y,
                                                        test_size=test_size,
                                                        shuffle=True,
                                                        random_state=42
                                                        )
    return x_train, x_test, y_train, y_test