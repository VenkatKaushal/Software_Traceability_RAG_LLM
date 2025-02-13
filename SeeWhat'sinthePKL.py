import pickle

# Replace 'your_file.pkl' with the actual path to your .pkl file
pkl_file_path = 'indices/eANCI/eANCI/vector_index.pkl'

# Load the .pkl file
with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)

# Print the content to see what is inside
print(data)

# Optionally, you can use more advanced techniques to explore the structure, e.g., printing keys, checking types, etc.
