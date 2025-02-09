import os
import pandas as pd
import requests
import logging
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

AUDIO_DATA_DIR = os.path.join(os.path.dirname(__file__), "data/cv-valid-dev-sample")
INPUT_FILE = os.path.join(os.path.dirname(__file__), "data/cv-valid-dev.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "cv-valid-dev.csv")
API_BASE_URL = "http://localhost:8001"
API_HEALTH_PATH = "/ping"
API_ASR_PATH = "/asr"
MAX_WORKERS: int = 4

class AudioTranscriber:
    def __init__(self):
        self.health_endpoint = f"{API_BASE_URL}{API_HEALTH_PATH}"
        self.asr_endpoint = f"{API_BASE_URL}{API_ASR_PATH}"
        
    def health_check(self) -> bool:
        try:
            response = requests.get(self.health_endpoint)
            return response.status_code == 200
        except requests.RequestException as e:
            logger.error(f"Health check failed: {e}")
            return False

    def transcribe_file(self, file_path: Path) -> Dict:
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'audio/mpeg')}
                response = requests.post(self.asr_endpoint, files=files)
                response.raise_for_status()
                return {
                    'filename': file_path.name,
                    'generated_text': response.json()['transcription'],
                    'duration': response.json()['duration'],
                }
        except requests.RequestException as e:
            logger.error(f"Failed to transcribe {file_path}: {e}")
            return {
                'filename': file_path.name,
                'generated_text': 'ERROR',
                'duration': '0'
            }

class AudioProcessor:
    def __init__(self):
        self.audio_data_dir = Path(AUDIO_DATA_DIR)
        self.input_file = Path(INPUT_FILE)
        self.output_file = Path(OUTPUT_FILE)
        self.max_workers = MAX_WORKERS
        self.transcriber = AudioTranscriber()
        
        self.audio_data_dir.mkdir(parents=True, exist_ok=True)
        if not self.audio_data_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.audio_data_dir}")
        
        if not self.input_file.is_file():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        mp3_files = list(self.audio_data_dir.glob("*.mp3"))
        logger.info(f"Found {len(mp3_files)} MP3 files.")
        

    def get_audio_files(self) -> List[Path]:
        return list(self.audio_data_dir.glob("*.mp3"))

    def update_csv(self, results: List[Dict]) -> None:
        df = pd.DataFrame(results)
        logger.info(f"Number of MP3 files processed: {len(df)}")
        
        if 'generated_text' in df.columns and df['generated_text'].str.contains('ERROR', na=False).any():
            logger.error("Processing error: An error occured while processing MP3 files.")
        
        try:
            existing_df = pd.read_csv(self.input_file)
            existing_df['filename_only'] = existing_df['filename'].str.replace('cv-valid-dev/', '', regex=False)
            logger.info(f"Number of rows in existing csv file: {len(existing_df)}")
            
            if len(df) != len(existing_df):
                logger.error("Processing error: Mismatch in the number of MP3 files processed.")
            
            df = df.set_index('filename')
            existing_df = existing_df.set_index('filename_only')
            existing_df['generated_text'] = df['generated_text']
            existing_df['duration'] = df['duration']
            existing_df = existing_df.reset_index()
            existing_df = existing_df.drop('filename_only', axis=1)
            
            existing_df.to_csv(self.output_file, index=False)
            logger.info(f"Updated {self.output_file} with new transcriptions")
            
            try:
                os.remove(self.input_file)
                logger.info(f"Successfully deleted input file: {self.input_file}")
            except OSError as e:
                logger.error(f"Error deleting input file {self.input_file}: {e}")
        except FileNotFoundError:
            logger.error("Input file not found. Aborting CSV file creation.")

    def process_files(self) -> None:
        if not self.transcriber.health_check():
            logger.error("API is not available")
            return

        audio_files = self.get_audio_files()
        if not audio_files:
            logger.warning(f"No MP3 files found in {self.audio_data_dir}")
            return

        logger.info(f"Starting to process {len(audio_files)} MP3 files...")
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.transcriber.transcribe_file, file): file 
                for file in audio_files
            }

            with tqdm(total=len(audio_files), desc="Processing audio files") as pbar:
                for future in as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing {file}: {e}")
                        results.append({
                            'filename': file.name,
                            'generated_text': 'ERROR',
                        })
                    pbar.update(1)

        self.update_csv(results)

if __name__ == "__main__":
    processor = AudioProcessor()
    processor.process_files()