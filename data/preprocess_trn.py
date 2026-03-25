import re
from pathlib import Path
from typing import List, Tuple

SCRIPT_DIR = Path(__file__).parent

class TRNPreprocessor:
    def __init__(self):
        self.timestamp_pattern = r'^\d+\.\d+\s+\d+\.\d+\s+' # match timestamps at the beginning of lines
    
    def parse_trn_file(self, filepath: str) -> List[str]:
        """
        Parse a .trn file and extract the text lines w/o timestamps
        @param filepath: path to the .trn file
        @return: list of utterances with no timestamps
        """
        utterances = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line_without_timestamps = re.sub(self.timestamp_pattern, '', line)

            if line_without_timestamps.strip():  # only keep non-empty lines
                utterances.append(line_without_timestamps.rstrip('\n')) # we can edit this if we don't want to keep the text exactly the same
        
        return utterances
    
    def create_training_pairs(self, utterances: List[str]) -> List[Tuple[str, str]]:
        """
        Create consecutive utterance pairs for training
        @param utterances: list of text lines 
        @return: list of tuples (utterance1, utterance2) to feed into tokenizer
        """
        pairs = []
        
        for i in range(len(utterances) - 1):
            text1 = utterances[i]
            text2 = utterances[i + 1]
            
            # only create pairs where both utterances have content
            if text1 and text2:
                pairs.append((text1, text2))
        
        return pairs
    
    def prepare_for_tokenizer(self, filepath: str) -> List[Tuple[str, str]]:
        """
        Main preprocessing pipeline in one place. 
        @param filepath: path to the .trn file
        @return: list of training pairs to feed into tokenizer 
        """
        utterances = self.parse_trn_file(filepath)
        training_pairs = self.create_training_pairs(utterances)
        return training_pairs
    
    def save_processed_data(self, training_pairs: List[Tuple[str, str]], output_file: str):
        """
        Save processed data in TSV format (CSV might be weird with commas in text)
        @param training_pairs: list of tuples (utterance1, utterance2)
        @param output_file: path to save the processed data 
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for text1, text2 in training_pairs:
                # save as TSV with original text preserved
                f.write(f"{text1}\t{text2}\n")

# using the processor
def process_trn(trn_filepath: str, output_filepath: str = None):
    """
    Process a .trn file and prepare it for BERT-style tokenization
    """
    preprocessor = TRNPreprocessor()
    training_pairs = preprocessor.prepare_for_tokenizer(trn_filepath)
    
    if output_filepath:
        preprocessor.save_processed_data(training_pairs, output_filepath)
        print(f"Saved processed data to {output_filepath}")
    
    return training_pairs

# run the preprocessing
if __name__ == "__main__":
    # Define paths
    trn_folder = SCRIPT_DIR / "TRN"
    tsv_folder = SCRIPT_DIR / "TSV"
    
    if not trn_folder.exists():
        print(f"Error: TRN folder not found at {trn_folder}")
        print(f"Make sure the TRN folder with all .trn files is in: {SCRIPT_DIR}")
        exit(1)
    
    tsv_folder.mkdir(exist_ok=True)
    print(f"Created/verified TSV folder: {tsv_folder}")
    print()
    
    all_pairs = []
    processed_count = 0
    failed_files = []
    
    for i in range(1, 61): 
        trn_filename = f"SBC{i:03d}.trn"  # SBC001.trn, SBC002.trn, etc.
        trn_filepath = trn_folder / trn_filename
        tsv_filename = f"SBC{i:03d}.tsv"  # SBC001.tsv, SBC002.tsv, etc.
        tsv_filepath = tsv_folder / tsv_filename
        
        if trn_filepath.exists():
            try:
                training_pairs = process_trn(str(trn_filepath), str(tsv_filepath))
                all_pairs.extend(training_pairs)
                processed_count += 1
            except Exception as e:
                print(f"  ERROR processing {trn_filename}: {e}")
                failed_files.append(trn_filename)
        else:
            print(f"Warning: {trn_filename} not found in {trn_folder}")
            failed_files.append(trn_filename)
    
    print()
    print("=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Successfully processed: {processed_count} files")
    print(f"Total training pairs created: {len(all_pairs)}")
    
    if failed_files:
        print(f"Failed/Missing files: {len(failed_files)}")
        print("Files not processed:")
        for f in failed_files:
            print(f"  - {f}")
    
    # save combined file with all pairs
    if all_pairs:
        combined_file = tsv_folder / "all_combined.tsv"
        print(f"\nSaving combined file with all pairs to: {combined_file}")
        with open(combined_file, 'w', encoding='utf-8') as f:
            for text1, text2 in all_pairs:
                f.write(f"{text1}\t{text2}\n")
        print(f"Saved {len(all_pairs)} total pairs to combined file")
    
    print("\nTSV files saved in:", tsv_folder)