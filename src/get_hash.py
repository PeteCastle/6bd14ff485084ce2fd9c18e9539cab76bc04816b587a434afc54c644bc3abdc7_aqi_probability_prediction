import hashlib

student_number = "2024-10356"
salt = "MLOPS2025B"
hashed_value = hashlib.sha256((student_number + salt).encode()).hexdigest()
print(hashed_value)
