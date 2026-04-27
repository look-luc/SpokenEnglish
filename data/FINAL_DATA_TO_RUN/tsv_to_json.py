import csv
import json
import re

with open("OverlapProject_TRNOverlapClassifications.tsv", 'r', encoding='utf-8') as f:
    first_line = f.readline().strip()
    headers = first_line.split('\t')
    print("Found headers:", headers)

def parse_edge(edge_str):
    if not edge_str or edge_str.strip() == "":
        return None
    match = re.search(r'\[(\d+) to (\d+),\s*(.+?)\]', edge_str)
    if match:
        return {
            "from": int(match.group(1)),
            "to": int(match.group(2)),
            "label": match.group(3).strip()
        }
    return None

def tsv_to_json(tsv_filepath, json_filepath, include_edges=True):
    data = []
    
    with open(tsv_filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            record = {
                "transcript_name": row["Transcript Name"],
                "time_mark_ut1": row["TimeMark Ut_1"],
                "time_mark_ut2": row["TimeMark Ut_2"],
                "ut1_text": row["Ut_1"],
                "ut1_index": int(row["Index_1"]),
                "ut2_text": row["Ut_2"],
                "ut2_index": int(row["Index_2"]),
                "overlap_type": row["Overlap Type"]
            }
            
            if include_edges:
                edges = []
                edge1 = parse_edge(row["Edge(s)_1"])
                edge2 = parse_edge(row["Edge(s)_2"])
                if edge1:
                    edges.append(edge1)
                if edge2:
                    edges.append(edge2)
                record["dda_edge"] = edges
            else:
                record["dda_edge"] = []  # or omit this key
            
            data.append(record)
    
    with open(json_filepath, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=2, ensure_ascii=False)

tsv_to_json("OverlapProject_TRNOverlapClassifications.tsv", "data_with_edges.json", include_edges=True)
tsv_to_json("OverlapProject_TRNOverlapClassifications.tsv", "data_without_edges.json", include_edges=False)