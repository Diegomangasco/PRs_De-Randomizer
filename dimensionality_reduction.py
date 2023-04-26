#  Copyright (c) 2023 Diego Gasco (diego.gasco99@gmail.com), Diegomangasco on GitHub

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

OUTPUT_DIRECTORY = "./features_files/"


def read_principal_component_analysis_file(file: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIRECTORY + file + ".csv", sep=",")


def explainable_principal_component_analysis(reduction: PCA, feature_names: list) -> None:
    # number of components
    n_pcs = reduction.components_.shape[0]

    # get the index of the most important feature on each component
    most_important = [np.abs(reduction.components_[i]).argmax() for i in range(n_pcs)]

    # get the names
    most_important_names = [feature_names[most_important[i]] for i in range(n_pcs)]
    names = [most_important_names[i] for i in range(n_pcs)]
    df = pd.DataFrame(names)

    print("Most relevant features for PCA")
    print(df)
    print("Features variance ratio (%)")
    print(100 * reduction.explained_variance_ratio_)
    print("Features singular values")
    print(reduction.singular_values_)


def principal_component_analysis(features: pd.DataFrame, components: int, explainable: bool) -> pd.DataFrame:
    pca = PCA(n_components=components)
    # Apply Principal Component Analysis
    out = pca.fit_transform(features)
    features_names = list(features.keys())
    if explainable:
        explainable_principal_component_analysis(pca, features_names)
    df = pd.DataFrame(out)
    df.to_csv(OUTPUT_DIRECTORY + "after_PCA_" + str(components) + ".csv", sep=",", float_format="%.4f", index=False)
    return df
