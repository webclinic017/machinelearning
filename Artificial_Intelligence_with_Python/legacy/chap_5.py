import sys
from sklearn.datasets import make_blobs
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor


def recommender_sys(option=1):
    if option == 1:
        ''' Creating a training pipeline '''
        print("Example: I say Hello world")
        X, y = make_blobs(n_samples=150,
                          n_features=25, n_classes=3, n_informative=6,
                          n_redundant=0, random_state=7)

        # Select top K features
        k_best_selector = SelectKBest(f_regression, k=9)

        # Initialize Extremely Random Forests classifier
        classifier = ExtraTreesClassifier(n_estimators=60, max_depth=4)

        # Construct the pipeline
        processor_pipeline = Pipeline([('selector', k_best_selector), ('erf',
                                                                       classifier)])
    elif option == 2:
        ''' Extracting the nearest neighbors '''
        pass
    elif option == 3:
        ''' Building a K-Nearest Neighbors classifier '''
        pass
    elif option == 4:
        ''' Computing similarity scores '''
        pass
    elif option == 5:
        ''' Finding similar users using collaborative filtering '''
        pass
    else:
        ''' Building a movie recommendation system '''
        pass
    pass


def main():
    recommender_sys(int(sys.argv[1]))


if __name__ == "__main__":
    main()
