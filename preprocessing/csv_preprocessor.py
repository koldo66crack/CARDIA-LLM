"""
Convert raw BIOLINCC data dictionary CSV files to JSONL format for RAG pipeline.
Each row becomes a self-contained JSON chunk for embedding and indexing.
"""

import pandas as pd
import json
import os


def preprocess_biolincc_csv(csv_path, output_dir="data/processed", study_type="main"):
    """
    Convert BIOLINCC CSV data dictionary to JSONL format.
    
    Args:
        csv_path (str): Path to the input CSV file
        output_dir (str): Directory to save the processed JSONL file
        study_type (str): Type of study - "main" or "ancillary"
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading CSV file: {csv_path}")
    
    # Try different encodings to handle the file
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"Successfully read with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with encoding {encoding}: {e}")
            continue
    
    if df is None:
        raise ValueError("Could not read CSV file with any of the attempted encodings")
    
    print(f"Loaded {len(df)} rows from CSV")
    
    # Process each row into a JSON chunk
    chunks = []
    
    for idx, row in df.iterrows():
        chunk = {
            "id": f"cardia_var_{idx:06d}",
            "study": study_type,
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
            "content": _create_searchable_content(row),
            "metadata": {
                "source": "BIOLINCC_Main_Study_Data_Dictionary" if study_type == "main" else "BIOLINCC_Ancillary_Studies_Data_Dictionary",
                "study": study_type,
                "dataset": row["Dataset"],
                "variable_type": row["Type"],
                "has_label": pd.notna(row["Label"]) and row["Label"] != "",
                "observation_count": row["Number_Obs_Dataset"] if pd.notna(row["Number_Obs_Dataset"]) else None
            }
        }
        chunks.append(chunk)
    
    return chunks, df


def save_combined_jsonl(all_chunks, all_dataframes, output_dir="data/processed"):
    """
    Save combined chunks from both main and ancillary studies to JSONL with unified IDs.
    """
    # Reassign IDs to ensure they're sequential
    for idx, chunk in enumerate(all_chunks):
        chunk["id"] = f"cardia_var_{idx:06d}"
    
    output_path = os.path.join(output_dir, "biolincc_data_dictionary.jsonl")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(all_chunks)} chunks to {output_path}")
    
    # Create summary statistics
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    summary_path = os.path.join(output_dir, "preprocessing_summary.json")
    summary = {
        "total_chunks": len(all_chunks),
        "main_study_chunks": len([c for c in all_chunks if c["study"] == "main"]),
        "ancillary_study_chunks": len([c for c in all_chunks if c["study"] == "ancillary"]),
        "datasets": combined_df["Dataset"].nunique(),
        "variable_types": combined_df["Type"].value_counts().to_dict(),
        "chunks_with_labels": sum(1 for chunk in all_chunks if chunk["metadata"]["has_label"]),
        "output_file": output_path
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Saved preprocessing summary to {summary_path}")
    return output_path


def _create_searchable_content(row):
    """Create searchable text content from a CSV row for embedding."""
    content_parts = []
    
    if pd.notna(row["Variable_Name"]):
        content_parts.append(f"Variable: {row['Variable_Name']}")
    
    if pd.notna(row["Label"]) and row["Label"] != "":
        content_parts.append(f"Description: {row['Label']}")
    
    if pd.notna(row["Dataset"]):
        content_parts.append(f"Dataset: {row['Dataset']}")
    
    if pd.notna(row["Type"]):
        content_parts.append(f"Data type: {row['Type']}")
    
    if pd.notna(row["Format"]) and row["Format"] != "":
        content_parts.append(f"Format: {row['Format']}")
    
    if pd.notna(row["Informat"]) and row["Informat"] != "":
        content_parts.append(f"Input format: {row['Informat']}")
    
    if pd.notna(row["Length"]):
        content_parts.append(f"Length: {row['Length']}")
    
    if pd.notna(row["Number_Obs_Dataset"]):
        content_parts.append(f"Observations: {row['Number_Obs_Dataset']}")
    
    return " | ".join(content_parts)


if __name__ == "__main__":
    all_chunks = []
    all_dataframes = []
    
    main_study_csv = "data/raw/BIOLINCC_Main Study Data Dictionary.csv"
    if os.path.exists(main_study_csv):
        print("\n" + "=" * 60)
        print("PROCESSING MAIN STUDY")
        print("=" * 60)
        main_chunks, main_df = preprocess_biolincc_csv(main_study_csv, study_type="main")
        all_chunks.extend(main_chunks)
        all_dataframes.append(main_df)
        print(f"Added {len(main_chunks)} chunks from main study\n")
    
    ancillary_csv = "data/raw/Ancillary Studies Data Dictionary - cleaned.csv"
    if os.path.exists(ancillary_csv):
        print("=" * 60)
        print("PROCESSING ANCILLARY STUDIES")
        print("=" * 60)
        ancillary_chunks, ancillary_df = preprocess_biolincc_csv(ancillary_csv, study_type="ancillary")
        all_chunks.extend(ancillary_chunks)
        all_dataframes.append(ancillary_df)
        print(f"Added {len(ancillary_chunks)} chunks from ancillary studies\n")
    
    if all_chunks:
        print("=" * 60)
        print("SAVING COMBINED JSONL")
        print("=" * 60)
        output_path = save_combined_jsonl(all_chunks, all_dataframes)
        print(f"Preprocessing complete! Output saved to: {output_path}")
    else:
        print("No chunks to save. Please ensure CSV files exist in data/raw/")

