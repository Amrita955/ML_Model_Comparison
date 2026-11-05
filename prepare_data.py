import pandas as pd

columns = [
    "id", "diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

df = pd.read_csv("data/wdbc.data", header=None, names=columns)

df = df.drop(columns=["id"])

df.to_csv("data/breast_cancer.csv", index=False)

print("Cleaned dataset shape:", df.shape)
print("\nDiagnosis counts:")
print(df["diagnosis"].value_counts())
print("\nColumn names:")
print(df.columns.tolist())
