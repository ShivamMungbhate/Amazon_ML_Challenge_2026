from preprocessing import (
    normalize_name,
    normalize_address,
    normalize_country
)


examples = [
    "ARUNODAYA FOREIGN PVT LTD",
    "New Caney College, Inc.",
    "Brightic Data LLC",
    "डिजिटल डेवलपर्स प्राइवेट लिमिटेड",
]


print("NAME NORMALIZATION")
print("-" * 50)

for x in examples:
    print(f"Original : {x}")
    print(f"Normalized: {normalize_name(x)}")
    print()


print("ADDRESS NORMALIZATION")
print("-" * 50)

address = "21207 Main Street, Houston, TX."
print("Original :", address)
print("Normalized:", normalize_address(address))


print()
print("COUNTRY NORMALIZATION")
print("-" * 50)

for country in ["India", " INDIA ", "United States"]:
    print(f"Original : {country}")
    print(f"Normalized: {normalize_country(country)}")