import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main():
    # slow expensive generate data script
    data = [
        ['kmeans', 0.3, 'seed1', 0],
        ['kmeans', 1.3, 'seed1', 0],
        ['kmeans', 2.3, 'seed1', 0],
        ['kmeans', 2.4, 'seed1', 0],
        ['kmeans', 2.5, 'seed1', 0],
        ['kmeans', 3.3, 'seed1', 0],
        ['kmeans', 0.1, 'easy', 1],
        ['kmeans', 0.2, 'easy', 2],
        ['kmeans', 0.31, 'easy', 3],
        ['kmeans', 0.33, 'easy', 4],
        ['method2', 0.9, 'easy', 0],
        ['method2', 0.8, 'easy', 1],
        ['method2', 1.2, 'easy', 2],
        ['method2', 0.61, 'easy', 3],
        ['method2', 0.73, 'easy', 4],
        ['kmeans', 1.1, 'hard', 1],
        ['kmeans', 1.2, 'hard', 2],
        ['kmeans', 1.31, 'hard', 3],
        ['kmeans', 1.33, 'hard', 4],
        ['method2', 1.9, 'hard', 0],
        ['method2', 1.8, 'hard', 1],
        ['method2', 2.2, 'hard', 2],
        ['method2', 1.61, 'hard', 3],
        ['method2', 1.73, 'hard', 4],
    ]
    df = pd.DataFrame(data, columns=['method_name', 'distance', 'dataset', 'example_idx'])

    # import pickle
    # with open("results.pkl", 'wb') as f:
    #     pickle.dump(df, f)
    #
    # # analysis script
    # with open("results.pkl", 'rb') as f:
    #     df = pickle.load(f)

    # process df to have a column that is success!
    df['success'] = df['distance'] < 0.25

    sns.boxplot(data=df, x='method_name', y='distance')

    df_1 = df.loc[df['dataset'] == 'easy']

    def my_median(series):
        pass

    sns.lineplot(data=df, x='example_idx', y='distance', hue='method_name', estimator=my_median)
    plt.show()


if __name__ == '__main__':
    main()
