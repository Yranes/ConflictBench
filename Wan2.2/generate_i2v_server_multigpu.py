import os
import sys
import logging
import warnings
import random

# Ensure `import wan` resolves regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist
from PIL import Image
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

warnings.filterwarnings('ignore')

import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import save_video

# ================= config =================

CKPT_DIR = "Wan-AI/Wan2.2-I2V-A14B"
OUTPUT_DIR = "./Wan2.2/output"

PING_CMD = {"type": "ping"}
PING_INTERVAL = 30  # seconds

MODEL_SERVICE = None
gpu_lock = asyncio.Lock()

class ServerArgs:
    def __init__(self):
        self.task = "i2v-A14B"
        self.ckpt_dir = CKPT_DIR
        self.size = "832*480"
        self.frame_num = 49
        self.sample_steps = 20
        self.base_seed = 6689

        self.offload_model = False
        self.convert_model_dtype = True
        self.t5_cpu = False

        # ulysses_size is derived from CUDA_VISIBLE_DEVICES so it matches torchrun --nproc_per_node.
        _visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self.ulysses_size = len([d for d in _visible.split(",") if d.strip()]) if _visible else 1
        self.dit_fsdp = True
        self.t5_fsdp = True


class I2VService:
    def __init__(self, args):
        self.args = args
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = self.local_rank

        self._init_logging()
        self._init_distributed()

        logging.info(f"[Rank {self.rank}] Loading WanI2V Model...")

        self.cfg = WAN_CONFIGS[self.args.task]

        if self.args.ulysses_size > 1:
            # Auto-correct ulysses_size to match world_size to avoid mismatch errors.
            if self.args.ulysses_size != self.world_size:
                logging.warning(f"Adjusting ulysses_size from {self.args.ulysses_size} to world_size {self.world_size}")
                self.args.ulysses_size = self.world_size

            init_distributed_group()

        self.model = wan.WanI2V(
            config=self.cfg,
            checkpoint_dir=self.args.ckpt_dir,
            device_id=self.device,
            rank=self.rank,
            t5_fsdp=self.args.t5_fsdp,
            dit_fsdp=self.args.dit_fsdp,
            use_sp=(self.args.ulysses_size > 1),
            t5_cpu=self.args.t5_cpu,
            convert_model_dtype=self.args.convert_model_dtype,
        )
        logging.info(f"[Rank {self.rank}] Model loaded successfully.")

    def _init_logging(self):
        if self.rank == 0:
            logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
        else:
            logging.basicConfig(level=logging.ERROR)

    def _init_distributed(self):
        # Set a very long timeout so NCCL watchdog does not kill idle workers.
        if not dist.is_initialized():
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(days=10)
            )

    def run_inference(self, prompt, image_path, seed=6689):
        if seed is None or seed < 0:
            seed = random.randint(0, sys.maxsize)

        if self.rank == 0:
            logging.info(f"[Rank 0] Start Generation. Prompt: {prompt[:30]}... | Seed: {seed}")

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"[Rank {self.rank}] Failed to load image: {e}")
            raise e

        video = self.model.generate(
            prompt,
            img=img,
            max_area=MAX_AREA_CONFIGS[self.args.size],
            frame_num=self.args.frame_num,
            shift=self.cfg.sample_shift,
            sample_solver='unipc',
            sampling_steps=self.args.sample_steps,
            guide_scale=5.0,
            seed=6689,
            offload_model=self.args.offload_model
        )

        save_path = ""
        if self.rank == 0:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = prompt.replace(" ", "_")[:20]
            filename = f"i2v_{timestamp}_{safe_prompt}.mp4"
            save_path = os.path.join(OUTPUT_DIR, filename)

            logging.info(f"[Rank 0] Saving video to {save_path}...")
            save_video(
                tensor=video[None],
                save_file=save_path,
                fps=self.cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
            logging.info(f"[Rank 0] Done.")

        del video
        del img
        torch.cuda.empty_cache()
        return save_path


def worker_loop(service):
    # Rank > 0 loop: wait for commands from rank 0 and respond to heartbeats.
    logging.info(f"[Rank {service.rank}] Worker loop started. Listening...")
    while True:
        try:
            cmd_list = [None]
            dist.broadcast_object_list(cmd_list, src=0)
            cmd = cmd_list[0]

            if cmd is None:
                break

            # Ping keeps NCCL alive during idle periods; skip inference.
            if isinstance(cmd, dict) and cmd.get("type") == "ping":
                continue

            service.run_inference(cmd["prompt"], cmd["image_path"], cmd["seed"])

        except Exception as e:
            logging.error(f"[Rank {service.rank}] Worker Error: {e}")
            continue


async def heartbeat_loop():
    # Rank 0 only: send a ping when idle so workers do not time out.
    logging.info("[Heartbeat] Monitor started.")
    while True:
        await asyncio.sleep(PING_INTERVAL)

        if not gpu_lock.locked():
            try:
                async with gpu_lock:
                    if dist.is_initialized():
                        dist.broadcast_object_list([PING_CMD], src=0)
            except Exception as e:
                logging.error(f"[Heartbeat] Failed: {e}")


class GenerateRequest(BaseModel):
    prompt: str
    image_path: str
    seed: int = -1

@asynccontextmanager
async def lifespan(app: FastAPI):
    heartbeat_task = None
    if dist.is_initialized() and dist.get_rank() == 0:
        heartbeat_task = asyncio.create_task(heartbeat_loop())

    yield

    if heartbeat_task:
        heartbeat_task.cancel()

    if dist.is_initialized() and dist.get_rank() == 0:
        logging.info("Shutting down workers...")
        try:
            dist.broadcast_object_list([None], src=0)
            dist.destroy_process_group()
        except:
            pass

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate_api(req: GenerateRequest):
    if dist.get_rank() != 0:
        raise HTTPException(status_code=400, detail="Not Master Node")

    async with gpu_lock:
        try:
            seed = req.seed if req.seed >= 0 else -1

            cmd = {
                "prompt": req.prompt,
                "image_path": req.image_path,
                "seed": seed
            }

            dist.broadcast_object_list([cmd], src=0)

            save_path = MODEL_SERVICE.run_inference(req.prompt, req.image_path, seed)

            return {"status": "success", "video_path": save_path}

        except Exception as e:
            logging.error(f"Generation Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    args = ServerArgs()
    MODEL_SERVICE = I2VService(args)

    if MODEL_SERVICE.rank == 0:
        import uvicorn
        logging.info("Starting API Server on Rank 0...")
        uvicorn.run(app, host="0.0.0.0", port=8021)
    else:
        worker_loop(MODEL_SERVICE)
