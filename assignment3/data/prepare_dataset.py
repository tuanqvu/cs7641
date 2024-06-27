import pandas as pd
import numpy as np

def prepare_uciml_adult():
    """
    Prepare the Adult dataset from UC Irvine ML archive
    """
    inputs = ['data/adult/adult.test', 'data/adult/adult.data']
    outputs = ['data/uciml_adult_test.pkl', 'data/uciml_adult_train.pkl']

    for input_file, output_file in zip(inputs, outputs):
        data = pd.read_csv(input_file, header=None)

        data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        uniques = [('?', -1)]
        for col in data.columns[[1, 3, 5, 6, 7, 8, 9, 13, 14]]:
            temp = data.loc[:, col].unique()
            temp = np.delete(temp, np.argwhere(temp == '?'))
            uniques += [(val, idx + 1) for idx, val in enumerate(temp)]
        data = data.replace(dict(uniques))

        data.to_pickle(output_file)


def prepare_uciml_drybean():
    """
    Prepare the Dry Bean dataset from UC Irvine ML archive
    """
    inputs = ['data/DryBeanDataset/Dry_Bean_Dataset.xlsx']
    outputs = ['data/uciml_drybean.pkl']

    for input_file, output_file in zip(inputs, outputs):
        data = pd.read_excel(input_file, skiprows=1, header=None)

        uniques = dict([(val, idx) for idx, val in enumerate(data.iloc[:, -1].unique())])
        data = data.replace(uniques)

        data.to_pickle(output_file)


if __name__ == '__main__':
    prepare_uciml_adult()
    prepare_uciml_drybean()
