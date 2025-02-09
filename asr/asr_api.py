import uvicorn
import logging
import io
from typing import Dict, Any, List
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter, Depends
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "facebook/wav2vec2-large-960h"
SAMPLE_RATE = 16000
ALLOWED_EXTENSIONS = {".mp3"}
BATCH_SIZE = 4
BATCH_TIMEOUT = 1.0

class ASRModel:    
    def __init__(self):
        self.device = None
        self.processor = None
        self.model = None
        self._setup_complete = False

    async def setup(self):
        if self._setup_complete:
            return

        def get_device():
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")

        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._load_model
        )
        self._setup_complete = True

    def _load_model(self):
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
            self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
            self.model.to(self.device).eval()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    async def cleanup(self):
        if self.model is not None:
            self.model.cpu()
            del self.model
            del self.processor
            torch.cuda.empty_cache()
            self._setup_complete = False

    @torch.no_grad()
    def predict(self, audio_batch: List[torch.Tensor]) -> List[str]:
        try:
            processed_batch = []
            for audio in audio_batch:
                if audio.dim() > 1:
                    audio = audio.squeeze()
                if audio.dim() == 0:
                    audio = audio.unsqueeze(0)
                processed_batch.append(audio)

            max_length = max(audio.size(0) for audio in processed_batch)
            padded_batch = torch.zeros((len(processed_batch), max_length))
            for i, audio in enumerate(processed_batch):
                padded_batch[i, :audio.size(0)] = audio

            input_values = self.processor(
                padded_batch.numpy(),
                return_tensors="pt",
                padding=True,
                sampling_rate=SAMPLE_RATE
            ).input_values.to(self.device)
            
            logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            return self.processor.batch_decode(predicted_ids)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise


class BatchProcessor:    
    def __init__(self, model: ASRModel):
        self.model = model
        self.request_queue = asyncio.Queue()
        self.shutdown_event = asyncio.Event()
        self.batch_task = None

    async def start(self):
        self.batch_task = asyncio.create_task(self._batch_worker())

    async def _batch_worker(self):
        while not self.shutdown_event.is_set():
            requests_batch = []
            try:
                first_request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=BATCH_TIMEOUT
                )
                requests_batch.append(first_request)

                while len(requests_batch) < BATCH_SIZE:
                    try:
                        request = self.request_queue.get_nowait()
                        requests_batch.append(request)
                    except asyncio.QueueEmpty:
                        break

                if requests_batch:
                    audio_batch = [r["audio"] for r in requests_batch]
                    durations = [len(audio) / SAMPLE_RATE for audio in audio_batch]
                    transcriptions = self.model.predict(audio_batch)
                    
                    for i, r in enumerate(requests_batch):
                        response = {
                            "transcription": transcriptions[i],
                            "duration": round(durations[i], 1)
                        }
                        r["response"].set_result(response)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                for request in requests_batch:
                    request["response"].set_exception(e)

    async def process_request(self, audio_data: torch.Tensor) -> str:
        response = asyncio.Future()
        await self.request_queue.put({"audio": audio_data, "response": response})
        return await response

    async def shutdown(self):
        self.shutdown_event.set()
        if self.batch_task:
            await self.batch_task

class ASRService:
    def __init__(self, batch_processor: BatchProcessor):
        self.batch_processor = batch_processor
        
    async def validate_audio_file(self, file: UploadFile) -> None:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed formats: {ALLOWED_EXTENSIONS}"
            )
        
    async def process_audio_stream(self, file: UploadFile) -> torch.Tensor:
        try:
            file_content = await file.read() 
            with io.BytesIO(file_content) as buffer:
                waveform, sample_rate = torchaudio.load(buffer)
                
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0)
                elif waveform.shape[0] == 1:
                    waveform = waveform.squeeze(0)
                
                if sample_rate != SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=SAMPLE_RATE
                    )
                    waveform = resampler(waveform)
                
                return waveform
        
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            raise HTTPException(status_code=400, detail="Invalid audio data")

    def _resample(self, data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate:
            return data
    
        resampler = torchaudio.transforms.Resample(orig_freq=src_rate, new_freq=dst_rate)
        return resampler(torch.tensor(data)).numpy() 
    
    async def validate_and_process(self, file: UploadFile) -> Dict[str, Any]:
        await self.validate_audio_file(file)

        try:
            audio_data = await self.process_audio_stream(file)
            transcription = await self.batch_processor.process_request(audio_data)
            logger.info(f"Successfully transcribed {file.filename}")
            return transcription

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = ASRModel()
    await model.setup()
    processor = BatchProcessor(model)
    await processor.start()
    
    app.state.model = model
    app.state.processor = processor
    
    yield
    
    await processor.shutdown()
    await model.cleanup()

app = FastAPI(
    title="Automatic Speech Recognition API",
    description="Microservice to deploy an Automatic Speech Recognition",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

router = APIRouter()

@router.get("/ping")
async def health_check():
    return "pong"

@router.post("/asr")
async def inference(
    file: UploadFile = File(...),
    service: ASRService = Depends(lambda: ASRService(app.state.processor))
):
    return await service.validate_and_process(file)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(
        "asr_api:app",
        host="0.0.0.0",
        port=8001,
        log_level="info",
        workers=1
    )