# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A Random Forest classifier (150 trees, max depth 8, min 5 samples per leaf). The model, encoder and label binarizer are saved in `model/`.

## Intended Use
Predicts if a person earns more than 50K a year from census data. It's for this course project only and shouldn't be used for real decisions about people.

## Training Data
The 1994 US Census Income dataset (32,561 rows). The label is `salary`, which is 76% `<=50K` and 24% `>50K`. I stripped the leading spaces and kept `?` as its own category. 80% of the data was used for training.

## Evaluation Data
The other 20% (6,513 rows), processed with the encoder fitted on the training data.

## Metrics
I used precision, recall and F1 because the classes are imbalanced. On the test set the model gets precision 0.799, recall 0.514 and F1 0.626. Slice results are in `slice_output.txt`.

## Ethical Considerations
The model leans heavily on `relationship=Husband` and `marital-status=Married-civ-spouse`. Almost every husband in the data is male, so it acts like a hidden sex feature. Recall is 0.547 for men but only 0.348 for women.

## Caveats and Recommendations
The data is from 1994. Very small slices (like `workclass=Never-worked` with 1 row) have scores that don't mean anything. The model should be retrained on newer data and checked for fairness before any real use.
