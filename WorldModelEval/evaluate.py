import os
import json
import cv2
import base64
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from typing import Dict, List, Optional, Any
from enum import Enum

INPUT_JSON = ""
OUTPUT_JSON = "./WorldModelEval/results/*.json"

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
JUDGE_MODEL = "gpt-5"

NUM_FRAMES = 5 
MAX_WORKERS = 10

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

class EvaluationType(Enum):
    INSTRUCTION = "instruction"
    PHYSICAL_LAWS = "physical_laws"
    COMMON_SENSE = "common_sense"

def get_default_prompt_templates() -> Dict[str, str]:
    return {
        EvaluationType.INSTRUCTION.value: """
            Evaluate if the video sequence provided (represented by sampled frames) follows the instruction description below: 
            
            Instruction: '{instruction}'
            
            Use the following scoring criteria:
            - 0: The video does not follow the instruction at all.
            - 1: The video includes the correct object but performs the wrong action, or vice versa.
            - 2: The video follows the instruction and shows a tendency toward the intended goal.
            - 3: The video follows the instruction precisely and successfully achieves the goal.
            
            Let's analyze step-by-step based on the visual evidence in the frames and conclude with 'Score: [score]'.
        """.strip(),
        # Let's briefly review the visual evidence in the frames and conclude with “Score: [score]
        EvaluationType.PHYSICAL_LAWS.value: """
            Watch the video frames and determine if it shows any '{physical_laws}'
            Let's think step-by-step and conclude with "Yes" or "No".
        """.strip(),
        
        EvaluationType.COMMON_SENSE.value: """
            Do the video frames exhibit '{common_sense}'?
            Let's think step-by-step and conclude with "Yes" or "No".
        """.strip(),
    }

def get_default_question_pool() -> Dict[str, Optional[List[str]]]:
    """Factory function for default question pool."""
    return {
        EvaluationType.INSTRUCTION.value: None,
        EvaluationType.PHYSICAL_LAWS.value: [
            "Violation of Newton's Law: Objects move without any external force.",
            "Violation of the Law of Conservation of Mass or Solid Constitutive Law: Objects deform irregularly.",
            "Violation of Fluid Constitutive Law: Liquids flow in an unnatural manner.",
            "Violation of Non-physical Penetration: Objects unnaturally pass through each other.",
            "Violation of Gravity: Objects behave inconsistently with gravity.",
        ],
        EvaluationType.COMMON_SENSE.value: [
            "Poor Aesthetics: Visually unappealing or low-quality content.",
            "Temporal Inconsistency: Noticeable flickering or abrupt changes between frames.",
        ],
    }

def extract_frames_base64(video_path: str, num_frames: int = 5) -> List[str]:
    """读取视频，等间隔抽取 num_frames 帧，并转为 base64 字符串"""
    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return []

    # 计算采样索引
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    base64_frames = []

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # OpenCV BGR -> RGB
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # GPT通常不挑BGR/RGB，但转RGB更规范，这里为了速度直接编码JPG
            _, buffer = cv2.imencode('.jpg', frame)
            base64_str = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(base64_str)
    
    cap.release()
    return base64_frames

def call_gpt_judge(base64_frames: List[str], prompt: str) -> str:
    """调用 GPT 进行打分"""
    if not base64_frames:
        return "Error: No frames"

    # 构造消息：文本 Prompt + 图片列表
    content_payload = [{"type": "text", "text": prompt}]
    
    for b64_img in base64_frames:
        content_payload.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_img}",
                "detail": "low" # 使用 low 模式省钱且速度快，high 模式看细节
            }
        })

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert video evaluation assistant."},
                {"role": "user", "content": content_payload}
            ],
            temperature=0.0,
            reasoning_effort="low"
        )
        resp = response.choices[0].message.content.strip()
        if "</think>" in resp:
            resp = resp.split("</think>")[-1].strip()
        return resp
    except Exception as e:
        return f"Error: {e}"

def evaluate_single_item_Wan2_1(item: Dict, templates: Dict, question_pool: Dict) -> Dict:
    """处理单个样本的所有评估指标"""
    result_entry = {
        "episode_id": item.get("episode_id"),
        "video_path": "./WorldModelEval/Wan2.1/results/wan2.1_baseline_test/" + item.get("episode_id") + "_Wan2.1.mp4",
        "scores": {}
    }

    if item.get("prompt") is None:
        result_entry["error"] = "No prompt available"
        return result_entry

    video_path = result_entry["video_path"]
    if not video_path or not os.path.exists(video_path):
        result_entry["error"] = "Video not found"
        print(video_path)
        return result_entry

    # 1. 抽帧
    frames = extract_frames_base64(video_path, num_frames=NUM_FRAMES)
    if not frames:
        result_entry["error"] = "Frame extraction failed"
        return result_entry

    # 2. 评估 Instruction Following
    # 使用之前生成的详细 Prompt 作为 instruction，如果 prompt 为空，降级使用 action
    instruction_text = item.get("prompt")
    if not instruction_text:
        raise ValueError("No prompt found for instruction evaluation.")
    
    tmpl = templates[EvaluationType.INSTRUCTION.value]
    prompt = tmpl.format(instruction=instruction_text)
    
    response = call_gpt_judge(frames, prompt)
    result_entry["scores"]["instruction_response"] = response
    
    # 尝试提取分数 (简单解析 'Score: x')
    try:
        if "Score:" in response:
            score = float(response.split("Score:")[-1].strip().split()[0].rstrip('.'))
            result_entry["scores"]["instruction_score"] = score
        else:
            result_entry["scores"]["instruction_score"] = -1 # 解析失败
    except:
        result_entry["scores"]["instruction_score"] = -1

    # 3. 评估 Physical Laws (遍历问题池)
    q_list = question_pool[EvaluationType.PHYSICAL_LAWS.value]
    result_entry["scores"]["physical"] = {}
    for q in q_list:
        tmpl = templates[EvaluationType.PHYSICAL_LAWS.value]
        # 格式化 prompt: '{physical_laws}' -> 具体问题
        prompt = tmpl.format(physical_laws=q.lower())
        resp = call_gpt_judge(frames, prompt)
        # 简单判断 Yes/No，注意这里 Yes 通常意味着违反了物理定律(Bad)
        is_violation = "yes" in resp.lower()
        result_entry["scores"]["physical"][q] = {
            "response": resp,
            "is_violation": is_violation
        }

    # 4. 评估 Common Sense (遍历问题池)
    q_list = question_pool[EvaluationType.COMMON_SENSE.value]
    result_entry["scores"]["common_sense"] = {}
    for q in q_list:
        tmpl = templates[EvaluationType.COMMON_SENSE.value]
        prompt = tmpl.format(common_sense=q.lower())
        resp = call_gpt_judge(frames, prompt)
        is_bad = "yes" in resp.lower()
        result_entry["scores"]["common_sense"][q] = {
            "response": resp,
            "is_bad": is_bad
        }

    return result_entry

def evaluate_single_item_Wan2_2(item: Dict, templates: Dict, question_pool: Dict) -> Dict:
    """处理单个样本的所有评估指标"""
    result_entry = {
        "episode_id": item.get("episode_id"),
        "video_path": item.get("baseline", {}).get("Wan2.2"),
        "scores": {}
    }

    video_path = result_entry["video_path"]
    if not video_path or not os.path.exists(video_path):
        result_entry["error"] = "Video not found"
        return result_entry

    # 1. 抽帧
    frames = extract_frames_base64(video_path, num_frames=NUM_FRAMES)
    if not frames:
        result_entry["error"] = "Frame extraction failed"
        return result_entry

    # 2. 评估 Instruction Following
    # 使用之前生成的详细 Prompt 作为 instruction，如果 prompt 为空，降级使用 action
    instruction_text = item.get("prompt")
    if not instruction_text:
        raise ValueError("No prompt found for instruction evaluation.")
    
    tmpl = templates[EvaluationType.INSTRUCTION.value]
    prompt = tmpl.format(instruction=instruction_text)
    
    response = call_gpt_judge(frames, prompt)
    result_entry["scores"]["instruction_response"] = response
    
    # 尝试提取分数 (简单解析 'Score: x')
    try:
        if "Score:" in response:
            score = float(response.split("Score:")[-1].strip().split()[0].rstrip('.'))
            result_entry["scores"]["instruction_score"] = score
        else:
            result_entry["scores"]["instruction_score"] = -1 # 解析失败
    except:
        result_entry["scores"]["instruction_score"] = -1

    # 3. 评估 Physical Laws (遍历问题池)
    q_list = question_pool[EvaluationType.PHYSICAL_LAWS.value]
    result_entry["scores"]["physical"] = {}
    for q in q_list:
        tmpl = templates[EvaluationType.PHYSICAL_LAWS.value]
        # 格式化 prompt: '{physical_laws}' -> 具体问题
        prompt = tmpl.format(physical_laws=q.lower())
        resp = call_gpt_judge(frames, prompt)
        # 简单判断 Yes/No，注意这里 Yes 通常意味着违反了物理定律(Bad)
        is_violation = "yes" in resp.lower()
        result_entry["scores"]["physical"][q] = {
            "response": resp,
            "is_violation": is_violation
        }

    # 4. 评估 Common Sense (遍历问题池)
    q_list = question_pool[EvaluationType.COMMON_SENSE.value]
    result_entry["scores"]["common_sense"] = {}
    for q in q_list:
        tmpl = templates[EvaluationType.COMMON_SENSE.value]
        prompt = tmpl.format(common_sense=q.lower())
        resp = call_gpt_judge(frames, prompt)
        is_bad = "yes" in resp.lower()
        result_entry["scores"]["common_sense"][q] = {
            "response": resp,
            "is_bad": is_bad
        }

    return result_entry

def evaluate_single_item_cosmos(item: Dict, templates: Dict, question_pool: Dict) -> Dict:
    """处理单个样本的所有评估指标"""
    result_entry = {
        "episode_id": item.get("episode_id"),
        "video_path": "./WorldModelEval/cosmos-predict2.5-main/results/cosmos_baseline_test/" + item.get("episode_id") + "_cosmos.mp4",
        "scores": {}
    }

    video_path = result_entry["video_path"]
    print(video_path)
    if not video_path or not os.path.exists(video_path):
        result_entry["error"] = "Video not found"
        return result_entry

    # 1. 抽帧
    frames = extract_frames_base64(video_path, num_frames=NUM_FRAMES)
    if not frames:
        result_entry["error"] = "Frame extraction failed"
        return result_entry

    # 2. 评估 Instruction Following
    # 使用之前生成的详细 Prompt 作为 instruction，如果 prompt 为空，降级使用 action
    instruction_text = item.get("prompt")
    if not instruction_text:
        raise ValueError("No prompt found for instruction evaluation.")
    
    tmpl = templates[EvaluationType.INSTRUCTION.value]
    prompt = tmpl.format(instruction=instruction_text)
    
    response = call_gpt_judge(frames, prompt)
    result_entry["scores"]["instruction_response"] = response
    
    # 尝试提取分数 (简单解析 'Score: x')
    try:
        if "Score:" in response:
            score = float(response.split("Score:")[-1].strip().split()[0].rstrip('.'))
            result_entry["scores"]["instruction_score"] = score
        else:
            result_entry["scores"]["instruction_score"] = -1 # 解析失败
    except:
        result_entry["scores"]["instruction_score"] = -1

    # 3. 评估 Physical Laws (遍历问题池)
    q_list = question_pool[EvaluationType.PHYSICAL_LAWS.value]
    result_entry["scores"]["physical"] = {}
    for q in q_list:
        tmpl = templates[EvaluationType.PHYSICAL_LAWS.value]
        # 格式化 prompt: '{physical_laws}' -> 具体问题
        prompt = tmpl.format(physical_laws=q.lower())
        resp = call_gpt_judge(frames, prompt)
        # 简单判断 Yes/No，注意这里 Yes 通常意味着违反了物理定律(Bad)
        is_violation = "yes" in resp.lower()
        result_entry["scores"]["physical"][q] = {
            "response": resp,
            "is_violation": is_violation
        }

    # 4. 评估 Common Sense (遍历问题池)
    q_list = question_pool[EvaluationType.COMMON_SENSE.value]
    result_entry["scores"]["common_sense"] = {}
    for q in q_list:
        tmpl = templates[EvaluationType.COMMON_SENSE.value]
        prompt = tmpl.format(common_sense=q.lower())
        resp = call_gpt_judge(frames, prompt)
        is_bad = "yes" in resp.lower()
        result_entry["scores"]["common_sense"][q] = {
            "response": resp,
            "is_bad": is_bad
        }

    return result_entry

def evaluate_single_item_Wan2_2_new(item: Dict, templates: Dict, question_pool: Dict) -> Dict:
    """处理单个样本的所有评估指标"""
    result_entry = {
        "episode_id": item.get("episode_id"),
        "video_path": "./Wan2.2/results/wan2.2_baseline_test/" + item.get("episode_id") + "_Wan2.2.mp4",
        "scores": {}
    }

    video_path = result_entry["video_path"]
    print(video_path)
    if not video_path or not os.path.exists(video_path):
        result_entry["error"] = "Video not found"
        return result_entry

    # 1. 抽帧
    frames = extract_frames_base64(video_path, num_frames=NUM_FRAMES)
    if not frames:
        result_entry["error"] = "Frame extraction failed"
        return result_entry

    # 2. 评估 Instruction Following
    # 使用之前生成的详细 Prompt 作为 instruction，如果 prompt 为空，降级使用 action
    instruction_text = item.get("prompt")
    if not instruction_text:
        raise ValueError("No prompt found for instruction evaluation.")
    
    tmpl = templates[EvaluationType.INSTRUCTION.value]
    prompt = tmpl.format(instruction=instruction_text)
    
    response = call_gpt_judge(frames, prompt)
    result_entry["scores"]["instruction_response"] = response
    
    # 尝试提取分数 (简单解析 'Score: x')
    try:
        if "Score:" in response:
            score = float(response.split("Score:")[-1].strip().split()[0].rstrip('.'))
            result_entry["scores"]["instruction_score"] = score
        else:
            result_entry["scores"]["instruction_score"] = -1 # 解析失败
    except:
        result_entry["scores"]["instruction_score"] = -1

    # 3. 评估 Physical Laws (遍历问题池)
    q_list = question_pool[EvaluationType.PHYSICAL_LAWS.value]
    result_entry["scores"]["physical"] = {}
    for q in q_list:
        tmpl = templates[EvaluationType.PHYSICAL_LAWS.value]
        # 格式化 prompt: '{physical_laws}' -> 具体问题
        prompt = tmpl.format(physical_laws=q.lower())
        resp = call_gpt_judge(frames, prompt)
        # 简单判断 Yes/No，注意这里 Yes 通常意味着违反了物理定律(Bad)
        is_violation = "yes" in resp.lower()
        result_entry["scores"]["physical"][q] = {
            "response": resp,
            "is_violation": is_violation
        }

    q_list = question_pool[EvaluationType.COMMON_SENSE.value]
    result_entry["scores"]["common_sense"] = {}
    for q in q_list:
        tmpl = templates[EvaluationType.COMMON_SENSE.value]
        prompt = tmpl.format(common_sense=q.lower())
        resp = call_gpt_judge(frames, prompt)
        is_bad = "yes" in resp.lower()
        result_entry["scores"]["common_sense"][q] = {
            "response": resp,
            "is_bad": is_bad
        }

    return result_entry

def evaluate_single_item_cogvideo(item: Dict, templates: Dict, question_pool: Dict) -> Dict:
    """处理单个样本的所有评估指标"""
    result_entry = {
        "episode_id": item.get("episode_id"),
        "video_path": "./WorldModelEval/CogVideo/results/cogvideo_baseline_test/" + item.get("episode_id") + "_cogvideo.mp4",
        "scores": {}
    }

    video_path = result_entry["video_path"]
    print(video_path)
    if not video_path or not os.path.exists(video_path):
        result_entry["error"] = "Video not found"
        return result_entry

    # 1. 抽帧
    frames = extract_frames_base64(video_path, num_frames=NUM_FRAMES)
    if not frames:
        result_entry["error"] = "Frame extraction failed"
        return result_entry

    # 2. 评估 Instruction Following
    # 使用之前生成的详细 Prompt 作为 instruction，如果 prompt 为空，降级使用 action
    instruction_text = item.get("prompt")
    if not instruction_text:
        raise ValueError("No prompt found for instruction evaluation.")
    
    tmpl = templates[EvaluationType.INSTRUCTION.value]
    prompt = tmpl.format(instruction=instruction_text)
    
    response = call_gpt_judge(frames, prompt)
    result_entry["scores"]["instruction_response"] = response
    
    # 尝试提取分数 (简单解析 'Score: x')
    try:
        if "Score:" in response:
            score = float(response.split("Score:")[-1].strip().split()[0].rstrip('.'))
            result_entry["scores"]["instruction_score"] = score
        else:
            result_entry["scores"]["instruction_score"] = -1 # 解析失败
    except:
        result_entry["scores"]["instruction_score"] = -1

    # 3. 评估 Physical Laws (遍历问题池)
    q_list = question_pool[EvaluationType.PHYSICAL_LAWS.value]
    result_entry["scores"]["physical"] = {}
    for q in q_list:
        tmpl = templates[EvaluationType.PHYSICAL_LAWS.value]
        # 格式化 prompt: '{physical_laws}' -> 具体问题
        prompt = tmpl.format(physical_laws=q.lower())
        resp = call_gpt_judge(frames, prompt)
        # 简单判断 Yes/No，注意这里 Yes 通常意味着违反了物理定律(Bad)
        is_violation = "yes" in resp.lower()
        result_entry["scores"]["physical"][q] = {
            "response": resp,
            "is_violation": is_violation
        }

    # 4. 评估 Common Sense (遍历问题池)
    q_list = question_pool[EvaluationType.COMMON_SENSE.value]
    result_entry["scores"]["common_sense"] = {}
    for q in q_list:
        tmpl = templates[EvaluationType.COMMON_SENSE.value]
        prompt = tmpl.format(common_sense=q.lower())
        resp = call_gpt_judge(frames, prompt)
        is_bad = "yes" in resp.lower()
        result_entry["scores"]["common_sense"][q] = {
            "response": resp,
            "is_bad": is_bad
        }

    return result_entry

def evaluate_single_item_hailuo(item: Dict, templates: Dict, question_pool: Dict) -> Dict:
    """处理单个样本的所有评估指标"""
    result_entry = {
        "episode_id": item.get("episode_id"),
        "video_path": "./WorldModelEval/results/hailuo_baseline_test/" + item.get("episode_id") + "_hailuo.mp4",
        "scores": {}
    }

    video_path = result_entry["video_path"]
    print(video_path)
    if not video_path or not os.path.exists(video_path):
        result_entry["error"] = "Video not found"
        return result_entry

    # 1. 抽帧
    frames = extract_frames_base64(video_path, num_frames=NUM_FRAMES)
    if not frames:
        result_entry["error"] = "Frame extraction failed"
        return result_entry

    # 2. 评估 Instruction Following
    # 使用之前生成的详细 Prompt 作为 instruction，如果 prompt 为空，降级使用 action
    instruction_text = item.get("prompt")
    if not instruction_text:
        raise ValueError("No prompt found for instruction evaluation.")
    
    tmpl = templates[EvaluationType.INSTRUCTION.value]
    prompt = tmpl.format(instruction=instruction_text)
    
    response = call_gpt_judge(frames, prompt)
    result_entry["scores"]["instruction_response"] = response
    
    # 尝试提取分数 (简单解析 'Score: x')
    try:
        if "Score:" in response:
            score = float(response.split("Score:")[-1].strip().split()[0].rstrip('.'))
            result_entry["scores"]["instruction_score"] = score
        else:
            result_entry["scores"]["instruction_score"] = -1 # 解析失败
    except:
        result_entry["scores"]["instruction_score"] = -1

    # 3. 评估 Physical Laws (遍历问题池)
    q_list = question_pool[EvaluationType.PHYSICAL_LAWS.value]
    result_entry["scores"]["physical"] = {}
    for q in q_list:
        tmpl = templates[EvaluationType.PHYSICAL_LAWS.value]
        # 格式化 prompt: '{physical_laws}' -> 具体问题
        prompt = tmpl.format(physical_laws=q.lower())
        resp = call_gpt_judge(frames, prompt)
        # 简单判断 Yes/No，注意这里 Yes 通常意味着违反了物理定律(Bad)
        is_violation = "yes" in resp.lower()
        result_entry["scores"]["physical"][q] = {
            "response": resp,
            "is_violation": is_violation
        }

    # 4. 评估 Common Sense (遍历问题池)
    q_list = question_pool[EvaluationType.COMMON_SENSE.value]
    result_entry["scores"]["common_sense"] = {}
    for q in q_list:
        tmpl = templates[EvaluationType.COMMON_SENSE.value]
        prompt = tmpl.format(common_sense=q.lower())
        resp = call_gpt_judge(frames, prompt)
        is_bad = "yes" in resp.lower()
        result_entry["scores"]["common_sense"][q] = {
            "response": resp,
            "is_bad": is_bad
        }

    return result_entry

def evaluate_single_item_wan26(item: Dict, templates: Dict, question_pool: Dict) -> Dict:
    """处理单个样本的所有评估指标"""
    result_entry = {
        "episode_id": item.get("episode_id"),
        "video_path": "./WorldModelEval/results/wan_api_baseline_test/" + item.get("episode_id") + "_wan2.6_api.mp4",
        "scores": {}
    }

    video_path = result_entry["video_path"]
    print(video_path)
    if not video_path or not os.path.exists(video_path):
        result_entry["error"] = "Video not found"
        return result_entry

    # 1. 抽帧
    frames = extract_frames_base64(video_path, num_frames=NUM_FRAMES)
    if not frames:
        result_entry["error"] = "Frame extraction failed"
        return result_entry

    # 2. 评估 Instruction Following
    # 使用之前生成的详细 Prompt 作为 instruction，如果 prompt 为空，降级使用 action
    instruction_text = item.get("prompt")
    if not instruction_text:
        raise ValueError("No prompt found for instruction evaluation.")
    
    tmpl = templates[EvaluationType.INSTRUCTION.value]
    prompt = tmpl.format(instruction=instruction_text)
    
    response = call_gpt_judge(frames, prompt)
    result_entry["scores"]["instruction_response"] = response
    
    # 尝试提取分数 (简单解析 'Score: x')
    try:
        if "Score:" in response:
            score = float(response.split("Score:")[-1].strip().split()[0].rstrip('.'))
            result_entry["scores"]["instruction_score"] = score
        else:
            result_entry["scores"]["instruction_score"] = -1 # 解析失败
    except:
        result_entry["scores"]["instruction_score"] = -1

    # 3. 评估 Physical Laws (遍历问题池)
    q_list = question_pool[EvaluationType.PHYSICAL_LAWS.value]
    result_entry["scores"]["physical"] = {}
    for q in q_list:
        tmpl = templates[EvaluationType.PHYSICAL_LAWS.value]
        # 格式化 prompt: '{physical_laws}' -> 具体问题
        prompt = tmpl.format(physical_laws=q.lower())
        resp = call_gpt_judge(frames, prompt)
        # 简单判断 Yes/No，注意这里 Yes 通常意味着违反了物理定律(Bad)
        is_violation = "yes" in resp.lower()
        result_entry["scores"]["physical"][q] = {
            "response": resp,
            "is_violation": is_violation
        }

    # 4. 评估 Common Sense (遍历问题池)
    q_list = question_pool[EvaluationType.COMMON_SENSE.value]
    result_entry["scores"]["common_sense"] = {}
    for q in q_list:
        tmpl = templates[EvaluationType.COMMON_SENSE.value]
        prompt = tmpl.format(common_sense=q.lower())
        resp = call_gpt_judge(frames, prompt)
        is_bad = "yes" in resp.lower()
        result_entry["scores"]["common_sense"][q] = {
            "response": resp,
            "is_bad": is_bad
        }

    return result_entry

def evaluate_single_item_kling(item: Dict, templates: Dict, question_pool: Dict) -> Dict:
    """处理单个样本的所有评估指标"""
    result_entry = {
        "episode_id": item.get("episode_id"),
        "video_path": "./WorldModelEval/results/kling_baseline_test/" + item.get("episode_id") + "_kling.mp4",
        "scores": {}
    }

    video_path = result_entry["video_path"]
    print(video_path)
    if not video_path or not os.path.exists(video_path):
        result_entry["error"] = "Video not found"
        return result_entry

    # 1. 抽帧
    frames = extract_frames_base64(video_path, num_frames=NUM_FRAMES)
    if not frames:
        result_entry["error"] = "Frame extraction failed"
        return result_entry

    # 2. 评估 Instruction Following
    # 使用之前生成的详细 Prompt 作为 instruction，如果 prompt 为空，降级使用 action
    instruction_text = item.get("prompt")
    if not instruction_text:
        raise ValueError("No prompt found for instruction evaluation.")
    
    tmpl = templates[EvaluationType.INSTRUCTION.value]
    prompt = tmpl.format(instruction=instruction_text)
    
    response = call_gpt_judge(frames, prompt)
    result_entry["scores"]["instruction_response"] = response
    
    # 尝试提取分数 (简单解析 'Score: x')
    try:
        if "Score:" in response:
            score = float(response.split("Score:")[-1].strip().split()[0].rstrip('.'))
            result_entry["scores"]["instruction_score"] = score
        else:
            result_entry["scores"]["instruction_score"] = -1 # 解析失败
    except:
        result_entry["scores"]["instruction_score"] = -1

    # 3. 评估 Physical Laws (遍历问题池)
    # q_list = question_pool[EvaluationType.PHYSICAL_LAWS.value]
    # result_entry["scores"]["physical"] = {}
    # for q in q_list:
    #     tmpl = templates[EvaluationType.PHYSICAL_LAWS.value]
    #     # 格式化 prompt: '{physical_laws}' -> 具体问题
    #     prompt = tmpl.format(physical_laws=q.lower())
    #     resp = call_gpt_judge(frames, prompt)
    #     # 简单判断 Yes/No，注意这里 Yes 通常意味着违反了物理定律(Bad)
    #     is_violation = "yes" in resp.lower()
    #     result_entry["scores"]["physical"][q] = {
    #         "response": resp,
    #         "is_violation": is_violation
    #     }

    # # 4. 评估 Common Sense (遍历问题池)
    # q_list = question_pool[EvaluationType.COMMON_SENSE.value]
    # result_entry["scores"]["common_sense"] = {}
    # for q in q_list:
    #     tmpl = templates[EvaluationType.COMMON_SENSE.value]
    #     prompt = tmpl.format(common_sense=q.lower())
    #     resp = call_gpt_judge(frames, prompt)
    #     is_bad = "yes" in resp.lower()
    #     result_entry["scores"]["common_sense"][q] = {
    #         "response": resp,
    #         "is_bad": is_bad
    #     }

    return result_entry

def main(model_type: str = "Wan2.1"):
    print(f"Loading dataset from {INPUT_JSON}...")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # data = data[: 10]

    print(f"Total samples: {len(data)}")
    
    templates = get_default_prompt_templates()
    question_pool = get_default_question_pool()
    
    results = []
    
    if model_type == "Wan2.1":
        evaluate_single_item = evaluate_single_item_Wan2_1
        OUTPUT_JSON = "./WorldModelEval/results/evaluation_results_wan2.1.json"
    elif model_type == "Wan2.2":
        evaluate_single_item = evaluate_single_item_Wan2_2
        OUTPUT_JSON = "./WorldModelEval/results/evaluation_results_wan2.2.json"
    elif model_type == "cosmos":
        evaluate_single_item = evaluate_single_item_cosmos
        OUTPUT_JSON = "./WorldModelEval/results/evaluation_results_cosmos.json"
    elif model_type == "Wan2.2_new":
        evaluate_single_item = evaluate_single_item_Wan2_2_new
        OUTPUT_JSON = "./WorldModelEval/results/evaluation_results_wan2.2_new.json"
    elif model_type == "cogvideo":
        evaluate_single_item = evaluate_single_item_cogvideo
        OUTPUT_JSON = "./WorldModelEval/results/evaluation_results_cogvideo.json"
    elif model_type == "wan2.6":
        evaluate_single_item = evaluate_single_item_wan26
        OUTPUT_JSON = "./WorldModelEval/results/evaluation_results_wan2.6.json"
    elif model_type == "hailuo":
        evaluate_single_item = evaluate_single_item_hailuo
        OUTPUT_JSON = "./WorldModelEval/results/evaluation_results_hailuo.json"
    elif model_type == "kling":
        evaluate_single_item = evaluate_single_item_kling
        OUTPUT_JSON = "./WorldModelEval/results/evaluation_results_kling_2.json"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 多线程评估
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(evaluate_single_item, item, templates, question_pool): item for item in data}
        
        for future in tqdm(as_completed(future_to_item), total=len(data), desc="Evaluating"):
            try:
                res = future.result()
                results.append(res)
                print(f"Saving results to {OUTPUT_JSON}...")
                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"Thread failed: {e}")
    score(results)

def score(results):
    scores = [r["scores"].get("instruction_score", -1) for r in results if "scores" in r]
    valid_scores = [s for s in scores if s >= 0]
    if valid_scores:
        print(f"Average Instruction Score: {sum(valid_scores)/len(valid_scores):.4f}")
    else:
        print("Instruction Following: No valid scores parsed.")

    from collections import defaultdict

    print("-" * 30)
    print(f"Evaluation Complete. Results saved to: {OUTPUT_JSON}")
    print("\n=== Evaluation Summary ===")

    # 1. Instruction Following Score
    scores = [r["scores"].get("instruction_score", -1) for r in results if "scores" in r]
    valid_scores = [s for s in scores if s >= 0]
    if valid_scores:
        print(f"Instruction Following Score: {sum(valid_scores)/len(valid_scores):.4f} / 3.00")
    else:
        print("Instruction Following: No valid scores.")

    print("-" * 20)

    # 初始化统计变量
    phy_passed = 0
    phy_total = 0
    # 使用字典记录小分: key=问题, value=[passed_count, total_count]
    phy_details = defaultdict(lambda: [0, 0]) 
    
    cs_passed = 0
    cs_total = 0
    cs_details = defaultdict(lambda: [0, 0])

    for r in results:
        scores_data = r.get("scores", {})
        
        # --- 统计物理定律 ---
        if "physical" in scores_data:
            for q, res in scores_data["physical"].items():
                # 全局统计
                phy_total += 1
                is_pass = not res.get("is_violation", False)
                if is_pass: 
                    phy_passed += 1
                
                # 小分统计
                phy_details[q][1] += 1       # 总数 +1
                if is_pass:
                    phy_details[q][0] += 1   # 通过数 +1
        
        # --- 统计常识 ---
        if "common_sense" in scores_data:
            for q, res in scores_data["common_sense"].items():
                # 全局统计
                cs_total += 1
                is_pass = not res.get("is_bad", False)
                if is_pass:
                    cs_passed += 1
                
                # 小分统计
                cs_details[q][1] += 1        # 总数 +1
                if is_pass:
                    cs_details[q][0] += 1    # 通过数 +1

    # --- 打印物理定律结果 ---
    if phy_total > 0:
        phy_rate = (phy_passed / phy_total) * 100
        print(f"[Physical Laws] Overall Pass Rate:   {phy_rate:.2f}% ({phy_passed}/{phy_total})")
        # 打印小分
        for q, (p, t) in phy_details.items():
            if t > 0:
                sub_rate = (p / t) * 100
                # 截取问题的前50个字符，防止太长换行
                print(f"  - {q[:50]}...: {sub_rate:.2f}% ({p}/{t})")
    else:
        print("Physical Laws: No data.")

    print("-" * 20)

    # --- 打印常识结果 ---
    if cs_total > 0:
        cs_rate = (cs_passed / cs_total) * 100
        print(f"[Common Sense] Overall Pass Rate:    {cs_rate:.2f}% ({cs_passed}/{cs_total})")
        # 打印小分
        for q, (p, t) in cs_details.items():
            if t > 0:
                sub_rate = (p / t) * 100
                print(f"  - {q[:50]}...: {sub_rate:.2f}% ({p}/{t})")
    else:
        print("Common Sense: No data.")
        
    print("=" * 30)

from fire import Fire

if __name__ == "__main__":
    Fire(main)
    # Pth_wan22 = "./WorldModelEval/results/evaluation_results_wan2.2.json"
    # Pth_wan21 = "./WorldModelEval/results/evaluation_results_wan2.1.json"
    # Pth_cosmos = "./WorldModelEval/results/evaluation_results_cosmos.json"
    # Pth_wan22_new = "./WorldModelEval/results/evaluation_results_wan2.2_new.json"
    # Pth_cogvideo = "./WorldModelEval/results/evaluation_results_cogvideo.json"

    # sample_path = "./WorldModelEval/i2v_test_dataset_with_prompts_50.json"
    # with open(sample_path, 'r', encoding='utf-8') as f:
    #     sample_data = json.load(f)
    # sampled_epid = [i["episode_id"] for i in sample_data]

    # now_eval = Pth_cogvideo

    # with open(now_eval, 'r', encoding='utf-8') as f:
    #     ori_result = json.load(f)
    
    # results_50 = []
    # for r in ori_result:
    #     if r["episode_id"] in sampled_epid:
    #         results_50.append(r)
    
    # score(results_50)
    # print("[NOW eval]:", now_eval)