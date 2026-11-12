# Face Recognition Using Higher-Order SVD

Face recognition on the Yale Faces dataset using the Higher-Order SVD (HOSVD). The idea is to keep the images in a 3D tensor (pixels x expressions x persons) instead of flattening everything into one big matrix, so the "expression" and "person" directions are treated separately.

The tensor operations (unfold, fold, mode-n product) are written by hand in NumPy, they are in `All.py`.

## Data

Yale Faces, 15 people. Each image is 243 x 320, so 77,760 pixels when flattened.

- Train: 135 images, 9 per person (centerlight, happy, leftlight, normal, rightlight, sad, sleepy, surprised, wink)
- Test: 30 images, 2 per person (glasses and noglasses). These two conditions are not in the training set at all.

The data is in `YALE faces.zip`. Unzip it next to the notebook before running.

## How it works

1. Put the training images in a tensor `A` of shape 77760 x 9 x 15. Column `(e, p)` is the image of person `p` with expression `e`.
2. Unfold `A` along each of the three modes and take the SVD of each unfolding. The left singular vectors give `F` (pixel mode), `G` (expression mode) and `H` (person mode).
3. Compute the core tensor `S = A x1 F^T x2 G^T x3 H^T`.
4. Form `B = S x2 G`. For a fixed expression `e`, the slice `B_e` is a basis for that expression, and person `p` has coordinates `h_p` (row `p` of `H`) in every one of these bases.
5. For a test image `z`, first project it: `z_hat = F^T z`. Then for each expression solve the least squares problem `B_e a_e = z_hat` using the QR factorization of `B_e` (computed once, before the test loop).
6. Compare `a_e` with every `h_p`. The person whose `h_p` is closest is the prediction.

Working with `z_hat` instead of `z` matters here, because the least squares problems become 135-dimensional instead of 77,760-dimensional.

## Two versions of the classifier

I tried two ways of doing step 6:

- **With tolerance** (`face_recognition`): go through the expressions and persons in order and return the first person with `||a_e - h_p|| < tol`. I used `tol = 0.75`. This is the version described in the book.
- **Without tolerance** (`face_recognition_without_tol`): compute all 9 x 15 distances and take the smallest one.

## Results

On the 30 test images:

| Version | Accuracy |
|---|---|
| With tolerance (0.75) | 80.00% |
| Without tolerance (nearest) | 93.33% |

The tolerance version depends a lot on the value of `tol` and on the order of the loop, because it stops at the first match even if a better one comes later. Taking the global minimum fixes that and only 2 of 30 images are misclassified. The notebook also prints precision and recall for each person for both versions.

## Files

- `Face_Recognition.ipynb` : the main notebook
- `All.py` : `unfold`, `fold`, `Mul_1`, `Mul_2`, `Mul_3` (mode-1, 2, 3 products)
- `YALE faces.zip` : train and test images

## Requirements

numpy, matplotlib, scikit-learn (only for `classification_report`)

## Reference

Lars Eldén, *Matrix Methods in Data Mining and Pattern Recognition*, SIAM. Chapter on tensor decomposition and face recognition using HOSVD.
