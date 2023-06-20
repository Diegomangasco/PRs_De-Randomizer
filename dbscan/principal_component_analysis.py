from sklearn.decomposition import PCA
from read_data import *

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


def principal_component_analysis(pre_proc: PreProcessing, components: int, explainable: bool):
    pca = PCA(n_components=components)
    features = pre_proc.get_features()
    out = pca.fit_transform(features)
    features_names = list(features.keys())
    if explainable:
        explainable_principal_component_analysis(pca, features_names)
    pre_proc._features = out