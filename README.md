
# Ragit
🚀 Smart, Fast, Scalable Search 🚀

**ragit** is a lightweight Python library that simplifies the management of vector databases. With ragit, you can easily create, update, query, and manage your vector database, all from CSV files containing text data.

## Features

- **Create a Vector Database:** Build your database from a CSV file with two required columns: `id` and `text`.
- **Add New Entries:** Insert additional entries from CSV files or add them individually.
- **Similarity Search:** Find nearby texts using various distance metrics (e.g., cosine, L2) with similarity scores.
- **Data Retrieval:** Fetch entries by IDs or exact text matches.
- **Deletion:** Remove single entries or entire collections when needed.

## CSV File Format
ragit expects your CSV file to have exactly two columns: `id` and `text`.

## Example CSV (`data.csv`):

```csv
id,text
1,The quick brown fox jumps over the lazy dog.
2,Another sample entry for testing.
```

## Usage
Below are some examples that demonstrate how to use `ragit`. The examples cover creating a database, adding entries, performing similarity searches, and more.

### 1. Importing and Initializing
First, import the `VectorDBManager` class from `ragit` and initialize it:

```python
from ragit import VectorDBManager

# Initialize the vector database manager with a custom persistence directory and model
db_manager = VectorDBManager(
    persist_directory="./my_vector_db", # Optional # default : "./vector_db"
    provider="sentence_transformer", # Optional # default : "sentence_transformer"
    model_name="all-mpnet-base-v2" # Optional # default : "all-mpnet-base-v2"
)
```

### 2. Creating a Database
Create a new collection (named `my_collection`) using your CSV file. In this example, the `distance_metric` is set to "cosine"(available options: l2, cosine, ip, l1) :

```python
db_manager.create_database(
    csv_path="data.csv", 
    collection_name="my_collection",
    distance_metric="cosine" # Optional # default : l2
)
```
### Reloading Your Database

After creating and populating your vector database, simply load it later by reinitializing with the same persistence directory:

```python
from ragit import VectorDBManager

db_manager = VectorDBManager(
    persist_directory="./my_vector_db",
)
```

### 3. Adding a Single Entry
Add an individual entry to the collection:

```python
db_manager.add_single_row(
    id_="101",
    text="This is a new test entry for the database.",
    collection_name="my_collection"
)
```

### 4. Adding Multiple Entries from CSV
You can also add multiple entries from a CSV file. This function skips any entries that already exist in the collection:

```python
stats = db_manager.add_values_from_csv(
    csv_path="data.csv",
    collection_name="my_collection"
)
print(f"Added {stats['new_entries_added']} new entries")
```

### 5. Retrieving Collection Information
Fetch and display information about your collection:

```python
info = db_manager.get_collection_info("my_collection")
print(f"Collection size: {info['count']} entries")
```

### 6. Performing a Similarity Search
Find texts that are similar to your query. In this example, the query text is "ai", and the search is filtered using the string "Artificial intelligence". The top 2 results are returned:

```python
results = db_manager.find_nearby_texts(
    text="ai",
    collection_name="my_collection",
    k=2,
    search_string="Artificial intelligence" # Optional
)

print("Results:")
for item in results:
    print(f"\nID: {item['id']}")
    print(f"Text: {item['text']}")
    print(f"Similarity: {item['similarity']}%")
    print(f"Distance ({item['metric']}): {item['raw_distance']}")
```

### 7. Deleting an Entry
Remove an entry from the collection by its ID:

```python
db_manager.delete_entry_by_id(
    id_="1",
    collection_name="my_collection"
)
```

### 8. Fetching Texts by IDs
Retrieve text entries for a list of IDs:

```python
ids_to_fetch = ["1", "2", "3"]
texts = db_manager.get_by_ids(ids_to_fetch, "my_collection")
print("Texts:", texts)
```

### 9. Fetching IDs by Texts
For an exact text match, get the corresponding IDs:

```python
texts_to_fetch = [
    "Plato was an ancient Greek philosopher of the Classical period who is considered a foundational thinker in Western philosophy"
]
ids = db_manager.get_by_texts(texts_to_fetch, "my_collection")
print("IDs:", ids)
```

### 10. Deleting a Collection
Delete an entire collection. **Note:** You must pass `confirmation="yes"` to proceed with deletion.

```python
db_manager.delete_collection(
    collection_name="my_collection",
    confirmation="yes"
)
```

## Contributing
Contributions are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on GitHub.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
