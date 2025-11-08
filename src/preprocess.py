"""
Convert raw BIOLINCC data dictionary files to JSONL format for RAG pipeline.
Each row becomes a self-contained JSON chunk for embedding and indexing.
"""

import pandas as pd
import json
import os
from pathlib import Path


def preprocess_biolincc_csv(csv_path, output_dir="data/processed"):
    """
    Convert BIOLINCC CSV data dictionary to JSONL format.
    
    Args:
        csv_path (str): Path to the input CSV file
        output_dir (str): Directory to save the processed JSONL file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file with encoding detection
    print(f"Reading CSV file: {csv_path}")
    
    # Try different encodings to handle the file
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None
    
    for encoding in encodings_to_try:
        try:
            print(f"Trying encoding: {encoding}")
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"Successfully read with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            print(f"Failed with encoding: {encoding}")
            continue
        except Exception as e:
            print(f"Error with encoding {encoding}: {e}")
            continue
    
    if df is None:
        raise ValueError("Could not read CSV file with any of the attempted encodings")
    
    print(f"Loaded {len(df)} rows from CSV")
    print(f"Columns: {list(df.columns)}")
    
    # Process each row into a JSON chunk
    chunks = []
    
    for idx, row in df.iterrows():
        # Create a comprehensive chunk for each variable
        chunk = {
            "id": f"cardia_var_{idx:06d}",
            "dataset": row["Dataset"],
            "variable_name": row["Variable_Name"],
            "label": row["Label"] if pd.notna(row["Label"]) else "",
            "type": row["Type"],
            "length": row["Length"] if pd.notna(row["Length"]) else None,
            "variable_number": row["Variable_Number"] if pd.notna(row["Variable_Number"]) else None,
            "format": row["Format"] if pd.notna(row["Format"]) else "",
            "format_length": row["Formatl"] if pd.notna(row["Formatl"]) else None,
            "format_decimal": row["Formatd"] if pd.notna(row["Formatd"]) else None,
            "informat": row["Informat"] if pd.notna(row["Informat"]) else "",
            "informat_length": row["Informl"] if pd.notna(row["Informl"]) else None,
            "informat_decimal": row["Informd"] if pd.notna(row["Informd"]) else None,
            "number_observations": row["Number_Obs_Dataset"] if pd.notna(row["Number_Obs_Dataset"]) else None,
            
            # Create searchable text content for embedding
            "content": create_searchable_content(row),
            
            # Add metadata for filtering and retrieval
            "metadata": {
                "source": "BIOLINCC_Main_Study_Data_Dictionary",
                "dataset": row["Dataset"],
                "variable_type": row["Type"],
                "has_label": pd.notna(row["Label"]) and row["Label"] != "",
                "observation_count": row["Number_Obs_Dataset"] if pd.notna(row["Number_Obs_Dataset"]) else None
            }
        }
        
        chunks.append(chunk)
    
    # Save as JSONL
    output_path = os.path.join(output_dir, "biolincc_data_dictionary.jsonl")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(chunks)} chunks to {output_path}")
    
    # Also save a summary
    summary_path = os.path.join(output_dir, "preprocessing_summary.json")
    summary = {
        "total_chunks": len(chunks),
        "datasets": df["Dataset"].nunique(),
        "variable_types": df["Type"].value_counts().to_dict(),
        "chunks_with_labels": sum(1 for chunk in chunks if chunk["metadata"]["has_label"]),
        "output_file": output_path
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Saved preprocessing summary to {summary_path}")
    return output_path


def create_searchable_content(row):
    """
    Create searchable text content from a CSV row for embedding.
    This combines all relevant information into a single searchable string.
    """
    content_parts = []
    
    # Add variable name
    if pd.notna(row["Variable_Name"]):
        content_parts.append(f"Variable: {row['Variable_Name']}")
    
    # Add label if available
    if pd.notna(row["Label"]) and row["Label"] != "":
        content_parts.append(f"Description: {row['Label']}")
    
    # Add dataset information
    if pd.notna(row["Dataset"]):
        content_parts.append(f"Dataset: {row['Dataset']}")
    
    # Add type information
    if pd.notna(row["Type"]):
        content_parts.append(f"Data type: {row['Type']}")
    
    # Add format information if available
    if pd.notna(row["Format"]) and row["Format"] != "":
        content_parts.append(f"Format: {row['Format']}")
    
    if pd.notna(row["Informat"]) and row["Informat"] != "":
        content_parts.append(f"Input format: {row['Informat']}")
    
    # Add length information
    if pd.notna(row["Length"]):
        content_parts.append(f"Length: {row['Length']}")
    
    # Add observation count
    if pd.notna(row["Number_Obs_Dataset"]):
        content_parts.append(f"Observations: {row['Number_Obs_Dataset']}")
    
    return " | ".join(content_parts)


if __name__ == "__main__":
    # Process the main BIOLINCC data dictionary
    csv_path = "data/raw/BIOLINCC_Main Study Data Dictionary.csv"
    
    if os.path.exists(csv_path):
        output_path = preprocess_biolincc_csv(csv_path)
        print(f"Preprocessing complete! Output saved to: {output_path}")
    else:
        print(f"CSV file not found at: {csv_path}")
        print("Please ensure the file exists in the correct location.")
