from matching_features import make_pair_features


row1 = {
    "business_name": "ABC Technologies Pvt. Ltd.",
    "address": "123 Main Street, Mumbai",
    "country": "India",
}

row2 = {
    "business_name": "ABC Technologies Private Limited",
    "address": "123 Main St Mumbai",
    "country": "India",
}


features = make_pair_features(row1, row2)


print("=" * 60)
print("FEATURE TEST")
print("=" * 60)

for key, value in features.items():
    print(f"{key:30s}: {value}")

print("=" * 60)
print(f"Total features: {len(features)}")