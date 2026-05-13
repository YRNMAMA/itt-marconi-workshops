# === Scalar types — one value each ===
amount = 12.50          # float
category = "food"       # str
date = "2026-05-12"     # str
is_recurring = False    # bool
notes = None            # NoneType

print(amount, category, date, is_recurring, notes)
print(type(amount), type(category), type(is_recurring), type(notes))

# === Collection types — many values ===
# A list: ordered and mutable. The natural "array" of Python.
tags = ["lunch", "weekday", "card-payment"]
print("First tag:", tags[0])
tags.append("urgent")
print("After append:", tags)

# A tuple: ordered but immutable — use it for fixed records.
expense_record = (12.50, "food", "2026-05-12")
print("Tuple:", expense_record)
print("Amount from tuple:", expense_record[0])

# A set: unordered and unique — duplicates disappear automatically.
unique_categories = {"food", "transport", "food", "rent", "food"}
print("Unique categories:", unique_categories)
# === Scalar types — one value each ===
amount = 12.50
category = "food"
date = "2026-05-12"
is_recurring = False
notes = None

print(amount, category, date, is_recurring, notes)
print(type(amount), type(category), type(is_recurring), type(notes))


# === Collection types — many values ===

# LIST
tags = ["lunch", "weekday", "card-payment"]

print("First tag:", tags[0])

tags.append("urgent")

print("After append:", tags)

# Lists are mutable
tags[0] = "dinner"

print("Modified list:", tags)


# TUPLE
expense_record = (12.50, "food", "2026-05-12")

print("Tuple:", expense_record)
print("Amount from tuple:", expense_record[0])

# This will generate an error
# expense_record[0] = 99.99


# SET
unique_categories = {"food", "transport", "food", "rent", "food"}

print("Unique categories:", unique_categories)

unique_categories.add("food")

print("After adding duplicate:", unique_categories)


# LEN
print("Length of tags:", len(tags))
print("Length of tuple:", len(expense_record))
print("Length of set:", len(unique_categories))


# BONUS — single element tuple
x = (5)
y = (5,)

print(type(x))
print(type(y))