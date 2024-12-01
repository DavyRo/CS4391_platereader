import pandas as pd

# File paths
detected_results_file = "./results.csv"  # Detected license plate numbers
ground_truth_file = "./ground_truth.csv"  # Ground truth license plate numbers

# Load data
detected_results = pd.read_csv(detected_results_file)
ground_truth = pd.read_csv(ground_truth_file)

# Extract unique license numbers
detected_licenses = set(detected_results["license_number"].dropna().unique())
ground_truth_licenses = set(ground_truth["license_number"].unique())

# Calculate metrics
true_positives = detected_licenses.intersection(ground_truth_licenses)
false_positives = detected_licenses - ground_truth_licenses
false_negatives = ground_truth_licenses - detected_licenses

# Print summary
print("Accuracy Metrics:")
print(f"Total Ground Truth Licenses: {len(ground_truth_licenses)}")
print(f"Total Detected Licenses: {len(detected_licenses)}")
print(f"True Positives (Matched Licenses): {len(true_positives)}")
print(f"False Positives (Extra Licenses Detected): {len(false_positives)}")
print(f"False Negatives (Missed Licenses): {len(false_negatives)}")

# Print mismatched results for analysis
print("\nMismatched Results:")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")

# Save detailed comparison to CSV
comparison_data = {
    "True Positives": list(true_positives),
    "False Positives": list(false_positives),
    "False Negatives": list(false_negatives),
}
comparison_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in comparison_data.items()]))
comparison_df.to_csv("./detailed_license_comparison.csv", index=False)
print("\nComparison saved to 'detailed_license_comparison.csv'")