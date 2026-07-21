import os
import json
import cv2
import base64
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from typing import Dict, List

INPUT_JSON = ""
OUTPUT_JSON = "./WorldModelEval/results/*.json"

VIDEO_ROOT_DIR = "./WorldModelEval/results/*"
VIDEO_SUFFIX = "_*.mp4"

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
JUDGE_MODEL = "gpt-5"

NUM_FRAMES = 5
MAX_WORKERS = 10

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

PROMPT_TEMPLATE_OBSERVABILITY = """
You are an expert evaluating a video generation model for an AI Agent environment.
Your task is to judge the "Interactive Observability" of the generated video.

**Input Context:**
- **Action Taken:** "{action}"
- **Scene Description:** "{prompt}"

**Evaluation Criteria (Score 1-5):**
Does the video provide clear visual feedback for the Agent to make next decisions?
1. **Dynamic Response:** Is there visible motion/change caused by the action? (e.g., lever moving, steam venting).
2. **Legibility:** Are key texts (HUD), lights (colors), or object states (broken/open) clear and unambiguous?
3. **Atmosphere:** Does the visual vibe (lighting, urgency) match the text description?

**Scoring Scale:**
- 1: No reaction, static image, or completely blurry/hallucinated.
- 3: Some motion/change, but hard to read details or weak atmosphere.
- 5: Perfect clarity. Action consequence is obvious, text is readable, atmosphere is immersive.

**Output JSON ONLY:**
{{
  "score": <float 1.0-5.0>,
  "reasoning": "<concise explanation>"
}}
"""


def extract_frames_base64(video_path: str, num_frames: int = 5) -> List[str]:
    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    base64_frames = []

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if ret:
            _, buffer = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 70],
            )
            base64_str = base64.b64encode(buffer).decode("utf-8")
            base64_frames.append(base64_str)

    cap.release()
    return base64_frames


def call_gpt_judge_json(base64_frames: List[str], prompt: str) -> Dict:
    if not base64_frames:
        return {"score": -1, "reasoning": "No frames provided"}

    content_payload = [{"type": "text", "text": prompt}]

    for b64_img in base64_frames:
        content_payload.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}",
                    "detail": "low",
                },
            }
        )

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that outputs JSON only.",
                },
                {"role": "user", "content": content_payload},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        resp_text = response.choices[0].message.content.strip()

        try:
            return json.loads(resp_text)
        except json.JSONDecodeError:
            score_match = re.search(r'"score":\s*([\d.]+)', resp_text)
            reason_match = re.search(r'"reasoning":\s*"([^"]+)"', resp_text)

            return {
                "score": float(score_match.group(1)) if score_match else -1,
                "reasoning": (
                    reason_match.group(1) if reason_match else resp_text
                ),
            }

    except Exception as e:
        return {"score": -1, "reasoning": f"API Error: {str(e)}"}


def evaluate_single_item(item: Dict) -> Dict:
    episode_id = item.get("episode_id", "unknown")
    video_path = os.path.join(
        VIDEO_ROOT_DIR,
        f"{episode_id}{VIDEO_SUFFIX}",
    )

    result_entry = {
        "episode_id": episode_id,
        "video_path": video_path,
        "action": item.get("action", ""),
        "scores": {},
    }

    if not os.path.exists(video_path):
        result_entry["error"] = f"Video not found at {video_path}"
        return result_entry

    frames = extract_frames_base64(
        video_path,
        num_frames=NUM_FRAMES,
    )

    if not frames:
        result_entry["error"] = "Frame extraction failed"
        return result_entry

    scene_desc = item.get("prompt", "")
    action_text = item.get("action", "")

    if not scene_desc:
        result_entry["error"] = "No prompt description found"
        return result_entry

    eval_prompt = PROMPT_TEMPLATE_OBSERVABILITY.format(
        action=action_text,
        prompt=scene_desc,
    )

    eval_result = call_gpt_judge_json(frames, eval_prompt)

    result_entry["scores"]["observability"] = {
        "score": eval_result.get("score", -1),
        "reasoning": eval_result.get("reasoning", ""),
    }

    return result_entry


def main():
    if not os.path.exists(INPUT_JSON):
        print(f"Input file not found: {INPUT_JSON}")
        return

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items from {INPUT_JSON}")

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(evaluate_single_item, item): item
            for item in data
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Evaluating Observability",
        ):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing item: {e}")

    valid_scores = [
        result["scores"]["observability"]["score"]
        for result in results
        if "error" not in result
        and result["scores"]
        .get("observability", {})
        .get("score", -1)
        != -1
    ]

    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        print(
            f"\nAverage Observability Score: {avg_score:.2f} "
            f"(over {len(valid_scores)} valid videos)"
        )
    else:
        print("\nNo valid scores generated.")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()