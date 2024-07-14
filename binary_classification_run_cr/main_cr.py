import os
import numpy as np
import pandas as pd
import logging
import warnings

from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("ML_Development_cr.log", mode='w'),
                        logging.StreamHandler()
                    ]
                    )


warnings.filterwarnings('ignore')

# Configure the random state
random_state = 1000

def create_directory(path):
    # Check if the directory already exists
    if not os.path.exists(path):
        # Create the directory
        os.makedirs(path)
        print(f"Directory '{path}' created")
        return path
    else:
        print(f"Directory '{path}' already exists")
        return path

def get_model_acronym(model_index):
    # Dictionary mapping model indices to their acronyms
    model_acronyms = {
        'xgboost': 'XGB',
        'gbc': 'GBC',
        'lightgbm': 'LGBM',
        'catboost': 'CB',
        'et': 'ET',
        'rf': 'RF',
        'lr': 'LR',
        'lda': 'LDA',
        'qda': 'QDA',
        'dt': 'DT',
        'knn': 'KNN',
        'ada': 'ADA',
        'nb': 'NB',
        'ridge': 'RIDGE',
        'svm': 'SVM'
    }

    # Return the acronym or a default message if the model index is not found
    return model_acronyms.get(model_index, "Unknown Model")

def load_engineered_datasets(path, sample=None, n=1000, random_state=1000):

    logging.info(f'Loading the engineered dataset from {path}')
    dataset = pd.read_csv(f"{path}/engineered_dataset.csv")
    logging.info(f'Dataset columns name: {dataset.columns}')

    # Determine unique classes and their counts
    class_counts = dataset['Class'].value_counts()
    min_samples = class_counts.min()
    logging.info(f'Class count: {class_counts}')

    if sample:
        if n > min_samples:
            logging.error(
                f'The requested n ({n}) samples from each Class is bigger than '
                f'the minimum sample of the Classes in the given data. '
                f'So, n is updated to {min_samples}'
            )
            n = min_samples

        # Sample an equal number of samples from each class
        sampled_dfs = []
        for cls in class_counts.index:
            cls_df = dataset[dataset['Class'] == cls].sample(n=(n if cls!='Noise' else n*(len(class_counts.index)-1)), random_state=random_state)
            sampled_dfs.append(cls_df)

        # Concatenate sampled dataframes
        dataset = pd.concat(sampled_dfs)
        dataset = dataset.sample(frac=1, random_state=random_state, ignore_index=True)

    logging.info(f'Dataset shape: {dataset.shape}')

    return dataset

def export_tables(df:pd.DataFrame, csv_path:str, tex_path:str, label:str, caption:str):
    df=df.copy()
    df.index.rename('Model', inplace=True)
    df.reset_index(drop=False, inplace=True)
    df['Model'] = df['Model'].apply(lambda x: get_model_acronym(x))
    df.to_csv(f'{csv_path}/{label}.csv', index=False)
    df.to_latex(f'{tex_path}/{label}.tex', index=False, float_format="%.4f",
                label=f'tbl:{label}',
                caption=caption)
    pass


def plot_model(classification_type, X_train, y_train, X_test, y_test, model, figs_path, model_index, title):
    # To copy the original model
    from sklearn.base import clone
    _model = clone(model)

    # Set up the default plotting settings
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    sns.set_theme(context="paper", style='whitegrid', palette='deep', font_scale=1.6)
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman']
    # plt.rcParams['font.size'] = 10  # You can change the base font size here
    mpl.use('Agg')
    # mpl.rcParams['savefig.dpi'] = 300

    try:
        from yellowbrick.classifier import ClassificationReport
        visualizer = ClassificationReport(_model, support=True, title=title)
        visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
        visualizer.show(outpath=f'{figs_path}/{classification_type}_ClassificationReport_{model_index}.png',
                        dpi=300, clear_figure=True,
                        bbox_inches="tight")
    except:
        print(f"ERROR: ClassificationReport for {model_index}")

    try:
        from yellowbrick.classifier import ROCAUC
        visualizer = ROCAUC(_model, title=title, binary=True if classification_type=='binary' else False)
        visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
        visualizer.show(outpath=f'{figs_path}/{classification_type}_ROCAUC_{model_index}.png',
                        dpi=300, clear_figure=True,
                        bbox_inches="tight")
    except:
        print(f"ERROR: ROCAUC for {model_index}")

    try:
        from yellowbrick.classifier import PrecisionRecallCurve
        visualizer = PrecisionRecallCurve(_model, title=title, per_class=False if classification_type=='binary' else True)
        visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
        visualizer.show(outpath=f'{figs_path}/{classification_type}_PrecisionRecallCurve_{model_index}.png',
                        dpi=300, clear_figure=True,
                        bbox_inches="tight")
    except:
        print(f"ERROR: PrecisionRecallCurve for {model_index}")

    try:
        from yellowbrick.classifier import ConfusionMatrix
        visualizer = ConfusionMatrix(_model, title=title)
        visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
        visualizer.show(outpath=f'{figs_path}/{classification_type}_ConfusionMatrix_{model_index}.png',
                        dpi=300, clear_figure=True,
                        bbox_inches="tight")
    except:
        print(f"ERROR: ConfusionMatrix for {model_index}")

    try:
        from yellowbrick.classifier import ClassPredictionError
        visualizer = ClassPredictionError(_model, title=title)
        visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
        visualizer.show(outpath=f'{figs_path}/{classification_type}_ClassPredictionError_{model_index}.png',
                        dpi=300, clear_figure=True,
                        bbox_inches="tight")
    except:
        print(f"ERROR: ClassPredictionError for {model_index}")

    # try:
    #     from yellowbrick.classifier import DiscriminationThreshold
    #     visualizer = DiscriminationThreshold(_model, title=title)
    #     visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    #     visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    #     visualizer.show(outpath=f'{figs_path}/{classification_type}_DiscriminationThreshold_{model_index}.png',
    #                     dpi=300, clear_figure=True,
    #                     bbox_inches="tight")
    # except:
    #     print(f"ERROR: DiscriminationThreshold for {model_index}")

    pass


def run_classification_pipeline(dataset, experiment_name=None, m_select=10, random_state=1000):
    logging.info(f'Classification pipeline is started ({experiment_name})')
    # Experiment Short Name (ESN)
    esn = ''.join([w[0] for w in experiment_name.split()])

    from pycaret.classification import ClassificationExperiment

    cla = ClassificationExperiment()
    cla.setup(data=dataset, target='Class', index=False,
              train_size=0.7, data_split_shuffle=True, data_split_stratify=True,
              fold_strategy="stratifiedkfold", fold=5, fold_shuffle=False,
              normalize=True, normalize_method='zscore',  # minmax, zscore
              transformation=True, transformation_method="yeo-johnson",
              low_variance_threshold=0.10,

              fix_imbalance=False, fix_imbalance_method='RandomUnderSampler', #"SMOTE",
              remove_multicollinearity=False, multicollinearity_threshold=0.9,
              pca=False, pca_method="linear", pca_components=0.99,  # 'mle' only for pca_method='linear'

              system_log=True, experiment_name=experiment_name,
              session_id=random_state, n_jobs=-1,
              memory=f"./tmp_{esn}",
              html=False
              )
    classification_type = 'binary' if not cla.get_config('is_multiclass') else 'multiclass'
    # Removes Kappa and MCC metrics from the experiment.
    cla.remove_metric('Kappa')
    cla.remove_metric('MCC')

    logging.info(f'Class distribution after the data preparation stage:'
                 f'\nTrain: \n{cla.y_train_transformed.value_counts()}'
                 f'\nTest: \n{cla.y_test_transformed.value_counts()}'
                 )


    logging.info(f"Compare a baseline models using CV, and select the {m_select} top performing ones, sorted by F1")
    candidate_models = ['xgboost', 'gbc', 'lightgbm', 'catboost', 'et', 'rf', 'lr', 'lda', 'qda', 'dt', 'knn', 'ada', 'nb', 'ridge', "svm"]
    # candidate_models = ['lr', 'dt']
    m_select = len(candidate_models)
    top_models = cla.compare_models(n_select=m_select, sort= "f1", include=candidate_models)

    #Comparison results of the base estimators using the CV data, sorted by F1
    cv_res_compare_models_df = cla.pull()
    cv_res_compare_models_df.drop(['Model', 'TT (Sec)'], axis=1, inplace=True)
    # print(cv_res_compare_models_df)

    print("\nExamine using hold out data and Calibrate the selected models, and determine the best performing one.")
    figs_path = create_directory(path=f'./{esn}/figs')
    ho_res_top_models = []
    calibrated_models = []
    ho_res_calibrated_models = []
    cv_res_calibrated_models = []
    top_calibrated_model = None
    for estimator in top_models:
        print('.'*25)
        print(cla._get_model_name(estimator))
        print("Params:\n", estimator.get_params(True))
        print("Holdout Data:")
        cla.predict_model(estimator=estimator)
        ho_res_e = cla.pull()
        ho_res_top_models.append(ho_res_e)

        # Calibrate the estimator
        print(f"Calibrate the {cla._get_model_name(estimator)} estimator using CV")
        calibrated_estimator = cla.calibrate_model(estimator=estimator, method="sigmoid") #  method = "isotonic" OR "sigmoid"
        calibrated_models.append(calibrated_estimator)
        cv_res_calibrated_models.append(cla.pull().loc['Mean'])
        if top_calibrated_model:
            if cla.pull().loc['Mean']['F1'] > top_calibrated_model['F1']:
                top_calibrated_model = {'F1': cla.pull().loc['Mean']['F1'],
                                        'calibrated_estimator': calibrated_estimator}
        else:
            top_calibrated_model = {'F1': cla.pull().loc['Mean']['F1'], 'calibrated_estimator': calibrated_estimator}
        print("Params:\n", calibrated_estimator.get_params(True))
        print("Holdout Data:")
        cla.predict_model(estimator=calibrated_estimator)
        ho_res_calibrated_models.append(cla.pull())

        model_acronym = get_model_acronym(cla._get_model_id(calibrated_estimator._get_estimator()))
        plot_model(classification_type, cla.X_train_transformed, cla.y_train_transformed, cla.X_test_transformed,
                   cla.y_test_transformed, calibrated_estimator, figs_path,
                   model_acronym,
                   cla._get_model_name(calibrated_estimator._get_estimator()))

        print('\n')

    # ---------------------------------------
    print('*'*50)
    print('Export the results of the base estimators')
    csv_path = create_directory(path=f'./{esn}/tables/csv')
    tex_path = create_directory(path=f'./{esn}/tables/tex')


    print('*' * 50)
    print(f"Comparative performance of {classification_type} estimators using cross-validation data, sorted by F1-score.")
    print(cv_res_compare_models_df)
    export_tables(cv_res_compare_models_df, csv_path, tex_path,
                  label=f"{classification_type}_cv_res_compare_models_df",
                  caption=f"Comparative performance of {classification_type} estimators using cross-validation data, sorted by F1-score."
                  )

    print(f"Comparative performance of {classification_type} estimators on holdout data, ordered by F1-score.")
    ho_res_top_models_df = pd.concat(ho_res_top_models, axis=0, ignore_index=True)
    ho_res_top_models_df.index = cv_res_compare_models_df.index[:m_select]
    ho_res_top_models_df.drop(['Model'], axis=1, inplace=True)
    ho_res_top_models_df.sort_values(by='F1', ascending=False, inplace=True)
    print(ho_res_top_models_df)
    export_tables(ho_res_top_models_df, csv_path, tex_path,
                  label=f"{classification_type}_ho_res_top_models_df",
                  caption=f"Comparative performance of {classification_type} estimators on holdout data, ordered by F1-score."
                  )

    print(f"\nComparative performance of calibrated {classification_type} estimators using cross-validation data, sorted by F1-score.")
    cv_res_calibrated_models_df = pd.concat(cv_res_calibrated_models, axis=1, ignore_index=True).T
    cv_res_calibrated_models_df.index = cv_res_compare_models_df.index[:m_select]
    cv_res_calibrated_models_df.sort_values(by='F1', ascending=False, inplace=True)
    print(cv_res_calibrated_models_df)
    export_tables(cv_res_calibrated_models_df, csv_path, tex_path,
                  label=f"{classification_type}_cv_res_calibrated_models_df",
                  caption=f"Comparative performance of calibrated {classification_type} estimators using cross-validation data, sorted by F1-score."
                  )

    print(f"Comparative performance of calibrated {classification_type} estimators on holdout data, ranked by F1-score.")
    ho_res_calibrated_models_df = pd.concat(ho_res_calibrated_models, axis=0, ignore_index=True)
    ho_res_calibrated_models_df.index = cv_res_compare_models_df.index[:m_select]
    ho_res_calibrated_models_df.drop(['Model'], axis=1, inplace=True)
    ho_res_calibrated_models_df.sort_values(by='F1', ascending=False, inplace=True)
    print(ho_res_calibrated_models_df)
    export_tables(ho_res_calibrated_models_df, csv_path, tex_path,
                  label=f"{classification_type}_ho_res_calibrated_models_df",
                  caption=f"Comparative performance of calibrated {classification_type} estimators on holdout data, ranked by F1-score."
                  )

    print('*' * 50)

    logging.info(f'Classification pipeline is Completed ({experiment_name})\n')
    pass

def main():
    logging.info('ML development is started.')
    dataset_directory = '../ds/'

    # load the engineered dataset
    dataset = load_engineered_datasets(path=dataset_directory, sample=False, n=1000, random_state=random_state)

    logging.info(f'Prepare the binary classification dataset (DroneRF or NoiseRF)')
    logging.info(f'Map the RF Class into (DroneRF or NoiseRF)')
    binary_class_df = dataset.copy()
    binary_class_df['Class'] = binary_class_df['Class'].apply(lambda x: 'NoiseRF' if x == 'Noise' else 'DroneRF')
    # Drop the Duty_Cycle due to its perfect correlation with the target Class to avoid data leakage
    logging.info(f'Drop the Duty_Cycle due to its perfect correlation with the target Class')
    binary_class_df.drop(['Duty_Cycle'], axis=1, inplace=True)
    # Run the classification pipeline in the binary dataset
    run_classification_pipeline(dataset=binary_class_df,
                                      experiment_name='Binary - DroneRF NoiseRF',
                                      m_select=10,
                                      random_state=random_state)

    logging.info('ML development pipeline is completed.')

if __name__ == '__main__':
    main()

