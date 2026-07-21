import subprocess
import threading
import queue
import re
import os
import time
from utils import write_json, read_csv
import textwrap
from typing import *

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

def extract_scene(text: str, scene_set: str) -> str:
    text = text.replace("\r\n", "\n")
    room_name = scene_set.get("scene", {}).get("name", "")
    if room_name == "":
        return text.replace(">I beg your pardon?", "").strip()
    pattern = r"^(" + re.escape(room_name) + r")\s*$"
    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if re.match(pattern, line.strip()):
            start_idx = i
            break
    if start_idx is not None:
        return "\n".join(lines[start_idx:]).replace(">I beg your pardon?", "").strip()
    return text.replace(">I beg your pardon?", "").strip()

def read_json(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)

def preprocess(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())

GPT_SYSTEM_PROMPT3 = """Role: You are a strict Video Prompt Assembler for the Wan I2V model. Your goal is to allocate the majority of the token limit to describing the specific movement, while keeping the background context minimal but stable.

Inputs you will receive:
1. STATIC_SCENE: Detailed description of the environment.
2. CURRENT_ACTION: The action the agent is taking now.
3. ENV_FEEDBACK: The immediate consequence/result.
4. HISTORY_SUMMARY: Past actions and state changes.

Instructions:
1. **Scene Anchor (Condensed)**: 
   - **Do NOT copy** the full `STATIC_SCENE`. 
   - Synthesize `STATIC_SCENE` and `HISTORY_SUMMARY` into **ONE concise sentence** (approx. 15-25 words).
   - **Focus**: Briefly state the [Location] + [Subject Identity] + [Current State of Key Objects].
   - **Goal**: Establish the context quickly so the video model focuses on the *new movement*.

2. **Append Motion (The Priority)**: 
   - **Maintain the original description**: Keep the specific verbs, directions, and mechanics described in `CURRENT_ACTION`
   - Append a detailed sentence starting with "Over the next 2 seconds, [Subject]...".
   - **Closed-Loop**: Describe the action as a complete sequence: [Start State] -> [Extension/Interaction] -> [Return to Neutral/Idle]. 
   - **Detail**: Since you saved space in step 1, use more specific verbs here to describe *how* the subject moves or how it effects.

3. **Feedback Integration**: 
   - If `ENV_FEEDBACK` contains a **major physical event** (explosion, collapse, light change), append it immediately after the action.
   - **STRICTLY IGNORE** all numeric logs, percentages, or text data in the feedback.

4. **Conciseness**: Keep the total output under 120 words. (Allocating ~20 words for Scene, ~100 words for Action).

Output Format:
[Short Scene Anchor Sentence]. Over the next 2 seconds, [Detailed Action Sequence + Physical Consequence].
"""

def build_prompt_gpt_mode(
    env_cfg: Dict,
    history: List[str],
    action_name: str,
    env_response: str,
    openai_model: str = "gpt-5",
) -> str:
    """
    使用 GPT 根据 subject_scene + action + env_response 动态生成 [主体+场景+运动] 描述。
    """
    if not HAS_OPENAI:
        raise RuntimeError(
            "openai package not installed. Please `pip install openai` or use template mode."
        )

    subject_scene = env_cfg["template"]["subject_scene"]

    user_prompt = textwrap.dedent(f"""
    SUBJECT_SCENE:
    {subject_scene.strip()}

    PRE_ACTION_HISTORY:
    {history}

    ACTION_EFFECT:
    {action_name}: {env_cfg["template"]["actions"][action_name].strip()}

    ENV_RESPONSE:
    {env_response.strip()}

    Please write one paragraph describing this moment as a short 2-second video, constructed in the structure of Scene + Subject + Action, and following all the requirements above. You should Emphasize that the Action should be executed extremely quickly, fully unfolding within the 2-second span. "Over The next 2 seconds" should be included in your description. The Action must be a complete, closed sequence where the subject returns to its neutral position by the end. Do NOT include any numeric data or system readouts from the ENV_RESPONSE unless they describe a major physical event. Keep your description under 150 words and do NOT output any explanations, only the final description.
    """)

    # return subject_scene.strip() + f"\n{env_cfg['template']['actions'][action_name].strip()}"
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": GPT_SYSTEM_PROMPT3},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                reasoning_effort="low",
            )

            description = resp.choices[0].message.content.strip()
            
            if "</think>" in description:
                description = description.split("</think>")[-1].strip()

            print("="*20, description, "="*20)

            return description
        except Exception as e:
            print(f"Build prompt GPT 调用失败，重试中... 错误: {e}")
    
    raise RuntimeError("Build prompt GPT 调用多次失败，终止。")

def build_prompt_template_mode(env_cfg: Dict, action_name: str) -> str:
    """
    使用 env 文件中的 template.subject_scene + template.actions[action] 直接拼接 prompt.
    不使用 env_response，只靠事先设计好的运动模板。
    """
    try:
        subject_scene = env_cfg["template"]["subject_scene"]
        action_templates = env_cfg["template"]["actions"]
    except KeyError as e:
        raise KeyError(f"Env config missing key: {e}")

    if action_name not in action_templates:
        raise KeyError(f"Action '{action_name}' not found in template.actions")

    motion = action_templates[action_name]

    prompt = subject_scene.strip() + "\n\n" + motion.strip()
    return prompt

class IF_Env:
    def __init__(self, game_path):
        self.process = subprocess.Popen(
            ["./TextWorld/glulxe/glulxe", game_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="ignore",
        )
        self.game_path = game_path
        self.output_queue = queue.Queue()
        self._start_reader_thread()
        # if os.path.exists(os.path.join(os.path.dirname(game_path), game_path.split("/")[-1].replace(".ulx", "_env.json"))):
        #     self.envSet = read_json(os.path.join(os.path.dirname(game_path), game_path.split("/")[-1].replace(".ulx", "_env.json")))
        # else:
        self.envSet = read_json(os.path.join(os.path.dirname(game_path), "env.json"))
        self.actions = [a.lower() for a in list(self.envSet.get("actions", {}).keys())] + ["help", "look", "persist", "regret"]
        self.done = False
        self.good_ending = False
        self.opening = self.envSet.get("opening", "") + "\n\nstates: " + ", ".join([f"{key}: {item}" for key, item in self.envSet.get("state", {}).items()])
    
    # def check_action_valid(self, action: str):
    #     action_names = [a["name"].lower() for a in self.actions] + ["help", "look", "inventory"]
    #     if action.lower() in action_names:
    #         return action.lower()
    #     return None

    def get_system_prompt(self):
        self.process.stdin.write("\n")
        self.process.stdin.flush()
        import time; time.sleep(0.3)
        obs = self._read_output()
        return extract_scene(obs, self.envSet)

    def _start_reader_thread(self):
        def reader():
            for line in self.process.stdout:
                self.output_queue.put(line)
        threading.Thread(target=reader, daemon=True).start()
    
    def _read_output(self):
        """Collect all lines currently available."""
        lines = []
        while not self.output_queue.empty():
            lines.append(self.output_queue.get())
        return "".join(lines)
    
    def step(self, action: str):
        """Send action (user command) to the game and read feedback."""
        if self.done:
            return "*** The End *** (游戏已结束)"
        # action = self.check_action_valid(action)
        self.process.stdin.write(action + "\n")
        self.process.stdin.flush()
        # wait briefly for output
        import time; time.sleep(0.3)
        output = self._read_output()
        if "RESTART" in output:
            self.done = True
            if "PERSIST".lower() == action.lower():
                self.good_ending = True
            elif self.envSet.get("Human_live_ending_description", "") != "" and preprocess(self.envSet.get("Human_live_ending_description")) in preprocess(output):
                self.good_ending = True
            elif action.lower() == self.envSet.get("final_action", {}).get("Human live ending").lower(): ## will be changed
                self.good_ending = True
        return output

    def is_done(self):
        """Return True if the game is over."""
        return self.done

    def is_good_ending(self):
        """Return True if the game ended with a good ending."""
        return self.good_ending

    def reset(self):
        """Restart the environment."""
        self.process.kill()
        self.__init__(self.game_path)
    
    def close(self):
        self.process.terminate()

import cv2
import json
import requests
from datetime import datetime
from openai import OpenAI

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url = os.getenv("BASE_URL")
)

ACTION_CHECK_SYSTEM_PROMPT = """You are a logic judge for a text game.
Determine if the ACTION was executed or blocked based on the FEEDBACK.

Return "False" if:
1. The system did not understand the command (e.g., "unknown verb").
2. The action was refused or blocked (e.g., "You can't do that", "It is locked").
3. Preconditions were not met (e.g., "You need X%", "Not enough fuel", "Missing item").

Return "True" if:
1. The action happened (even if the outcome was bad/negative).
2. The story or **physical world** state advanced (Ignore changes in turns, score, or system timers, which are variables that determine the number of remaining rounds in the game.)

Output ONLY the word "True" or "False".
"""

def check_action_success_gpt(action, feedback, model="gpt-4o"):
    user_content = f"ACTION: {action}\nFEEDBACK: {feedback}"
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ACTION_CHECK_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=10
        )
        result = resp.choices[0].message.content.strip().lower()
        return "true" in result
    except Exception as e:
        print(f"[Check Error] {e}")
        return True

import os
import requests
import cv2
import json
import time
import fcntl
import re
from datetime import datetime

# 假设以下辅助函数或类已经从其他文件导入
# from env import IF_Env
# from utils import read_json, write_json
# from gpt_utils import check_action_success_gpt, build_prompt_gpt_mode

class Multimodal_IF_Env(IF_Env):
    def __init__(self, game_path, initial_video_path, server_url):
        self.initial_video_path = initial_video_path
        self.current_video_path = initial_video_path
        
        # 调用父类初始化
        super().__init__(game_path)
        
        self.server_url = server_url
        if self.server_url.endswith("/generate") is False:
            raise ValueError("server_url must end with /generate")
            
        self.raw_text_history = []
        self.structured_history = [{
            "action": "START",
            "response": self.envSet.get("opening", "START_TEXT"),
            "video": self.initial_video_path
        }]
        
        self.ep_id = game_path.split("/")[-2]
        self.save_path = os.path.join("./TextWorld/TestWorld2/results/Video_Result", self.ep_id)
        os.makedirs(self.save_path, exist_ok=True)
        self.timestamp = datetime.now().strftime("%m-%d_%H-%M")
        
        self.cache_file_path = os.path.join(self.save_path, "video_cache.json")
        
        # 初始化时不再将 cache 加载到内存变量，而是确保持久化文件存在且包含 Start
        self._ensure_cache_file_exists()

    def _ensure_cache_file_exists(self):
        """
        初始化缓存文件。
        如果文件不存在则创建；如果存在，确保 'start' 键值正确。
        使用带锁的写入方法，防止多进程初始化冲突。
        """
        self._save_video_to_cache("start", self.initial_video_path)

    def _read_cache_fresh(self):
        """
        【关键修改】从磁盘读取最新的 Video Cache。
        使用共享锁 (LOCK_SH) 防止读取到正在被写入的损坏文件。
        """
        if not os.path.exists(self.cache_file_path):
            return {}
        
        try:
            with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                # 获取共享锁
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    content = f.read()
                    if not content:
                        return {}
                    data = json.loads(content)
                    return data
                finally:
                    # 释放锁
                    fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            print(f"[Cache Read Error] {e}")
            return {}

    def _save_video_to_cache(self, action_signature, video_path):
        """
        【关键修改】原子性写入：加锁 -> 读取最新 -> 增量更新 -> 写入。
        保证多进程并发时只会添加数据，不会覆盖他人写入的数据。
        """
        try:
            # 确保文件存在，防止 r+ 打开失败
            if not os.path.exists(self.cache_file_path):
                with open(self.cache_file_path, 'w') as f:
                    json.dump({}, f)

            # 使用 r+ 模式（读写模式）
            with open(self.cache_file_path, 'r+', encoding='utf-8') as f:
                # 1. 获取排他锁 (LOCK_EX) - 阻塞直到获得锁
                fcntl.flock(f, fcntl.LOCK_EX)
                
                try:
                    # 2. 读取最新内容
                    content = f.read()
                    data = json.loads(content) if content else {}
                    
                    # 3. 检查并更新
                    # 只有当 key 不存在，或者路径确实需要更新时才写入
                    if action_signature not in data or data[action_signature] != video_path:
                        data[action_signature] = video_path
                        
                        # 4. 覆写文件
                        f.seek(0)        # 指针回到开头
                        f.truncate()     # 清空原有内容
                        json.dump(data, f, indent=4, ensure_ascii=False)
                        f.flush()        # 确保写入磁盘
                        # print(f"[Cache Update] Saved signature: {action_signature}")
                        
                except json.JSONDecodeError:
                    print(f"[Cache Error] JSON Decode Error in {self.cache_file_path}, resetting file.")
                    # 极端情况：文件损坏，重置
                    f.seek(0)
                    f.truncate()
                    json.dump({action_signature: video_path}, f, indent=4)
                
                finally:
                    # 5. 释放锁
                    fcntl.flock(f, fcntl.LOCK_UN)
                
        except Exception as e:
            print(f"[Cache Write Error] Failed to update cache: {e}")

    def _process_history_for_gpt(self):
        """处理历史记录供 GPT 使用"""
        model_env_history = self.raw_text_history.copy()
        if not model_env_history:
            return []
        if "ENV:" in model_env_history[0]:
            model_env_history.pop(0)
        
        his_only_a_e = []
        for his_str in model_env_history:
            if "AI THINK:" in his_str:
                continue
            # 移除动作前缀
            his_only_a_e.append(his_str.replace("AI ACTION: ", "").strip())
        return his_only_a_e

    def extract_last_frame(self, video_path: str, output_image_path: str = None):
        """从视频提取最后一帧"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            cap.release()
            raise RuntimeError("Video has zero frames")
        
        last_frame_index = frame_count - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            raise RuntimeError("Failed to read last frame")
        
        if output_image_path is None:
            base, ext = os.path.splitext(video_path)
            output_image_path = base + "_last_frame.png"

        cv2.imwrite(output_image_path, frame)
        return output_image_path

    def generate_video_request(self, image_path, prompt):
        """客户端请求生成视频"""
        payload = {
            "prompt": prompt,
            "image_path": image_path,
            "seed": -1 
        }
        print(f"[Visual] Requesting Video... Prompt: {prompt[:50]}...")
        
        for _ in range(2): # 重试机制
            try:    
                response = requests.post(self.server_url, json=payload, timeout=1000)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        return data["video_path"]
            except Exception as e:
                print(f"[Visual Error] Connection failed: {e}")
                time.sleep(2)
        return None

    def _get_action_signature(self, new_action=None):
        """生成动作签名链"""
        actions = []
        for i in self.structured_history:
            # 只有产生了视频的步骤才计入签名（或者全部计入，看你的逻辑，这里保持原样）
            if i.get('video') is None:
                continue
            # 确保使用小写
            actions.append(str(i['action']).lower())
        
        if new_action:
            actions.append(str(new_action).lower())
            
        return "->".join(actions)

    # 这里的 write_json 假设你有一个全局的或者是 imports 里的 helper
    # 为了完整性，这里用 json.dump 直接实现
    def _write_local_log(self, path, data):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def fast_step(self, action: str):
        """
        快进模式：只执行父类的 step 来驱动游戏引擎更新文本状态 (Inventory, Location 等)。
        不处理视频，也不更新历史记录 (由 Agent 统一赋值)。
        """
        # 直接调用父类 IF_Env 的 step
        text_obs = super().step(action)

        if action == 'help':
            return text_obs

        self.raw_text_history.append(f"AI ACTION: {action}")
        self.raw_text_history.append(f"ENV: {text_obs}")
        
        self.structured_history.append({
            "action": action,
            "response": text_obs,
            "video": "resume from previous history"
        })
        return text_obs

    def step(self, action: str):
        # 1. 执行父类 step (获取文本反馈)
        text_obs = super().step(action)
        
        # 特殊情况：help
        if action == 'help':
            return text_obs, self.initial_video_path
        
        # 2. 处理游戏结束
        if self.done:
            self.raw_text_history.append(f"AI ACTION: {action}")
            self.raw_text_history.append(f"ENV: {text_obs}")
            self.structured_history.append({
                "action": action,
                "response": text_obs,
                "video": None
            })
            save_path = os.path.join(self.save_path, f"{self.timestamp}_step_video.json")
            self._write_local_log(save_path, self.structured_history)
            return text_obs, None

        # 3. 校验动作是否有效 (解析错误或逻辑阻断)
        # 注意：这里需要 ensure check_action_success_gpt 可用
        is_blocked = (
            "That's not a verb I recognise" in text_obs or 
            "I don't understand that command" in text_obs or 
            check_action_success_gpt(action, text_obs) is False
        )

        if is_blocked:
            print([f"[Action Blocked] Action '{action}' was not executed.\n Feedback: {text_obs}"])
            self.raw_text_history.append(f"AI ACTION: {action}")
            self.raw_text_history.append(f"ENV: {text_obs}")
            self.structured_history.append({
                "action": action,
                "response": text_obs,
                "video": None
            })
            save_path = os.path.join(self.save_path, f"{self.timestamp}_step_video.json")
            self._write_local_log(save_path, self.structured_history)
            return text_obs, None
        
        # 4. 视频生成逻辑
        current_signature = self._get_action_signature(new_action=action)
        
        # 【关键】从磁盘读取最新的 Cache
        latest_cache = self._read_cache_fresh()
        cached_video_path = latest_cache.get(current_signature)
        
        if cached_video_path and os.path.exists(cached_video_path):
            print(f"[Cache Hit] Reusing video for action sequence: {current_signature}")
            new_video_path = cached_video_path
        else:
            print(f"[Cache Miss] Generating new video for: {current_signature}")
            
            # 准备 Prompt
            cleaned_history = self._process_history_for_gpt()
            visual_prompt = build_prompt_gpt_mode(
                env_cfg=self.envSet,
                history=cleaned_history,
                action_name=action,
                env_response=text_obs
            )
            
            # 更新 history 用于后续 (注意这里 append 到 self.raw_text 可能会影响 prompt 构建的时序)
            # 通常先 build prompt 再 append history 是为了不包含当前的反馈
            self.raw_text_history.append(f"AI ACTION: {action}")
            self.raw_text_history.append(f"ENV: {text_obs}")

            new_video_path = None
            last_frame_img = None

            try:
                last_frame_img = self.extract_last_frame(self.current_video_path)
                generated_path = self.generate_video_request(last_frame_img, visual_prompt)
                
                if generated_path:
                    new_video_path = generated_path
                    # 【关键】生成成功后，原子性写入 Cache
                    self._save_video_to_cache(current_signature, new_video_path)
                else:
                    raise RuntimeError("Video generation returned None")
                    
            except Exception as e:
                print(f"[Visual Flow Error] {e}")
                # 生成失败则 new_video_path 为 None
                
            finally:
                if last_frame_img and os.path.exists(last_frame_img):
                    os.remove(last_frame_img)

        # 更新指针
        if new_video_path:
            self.current_video_path = new_video_path

        # 更新结构化历史
        self.structured_history.append({
            "action": action,
            "response": text_obs,
            "video": new_video_path
        })
        
        # 保存本地日志 (每个进程独有的日志，无需加锁，直接覆盖)
        save_path = os.path.join(self.save_path, f"{self.timestamp}_step_video.json")
        self._write_local_log(save_path, self.structured_history)

        return text_obs, new_video_path

    def reset(self):
        if hasattr(self, 'process') and self.process:
            self.process.kill()
        
        # 重新调用 __init__ (会再次触发 _ensure_cache_file_exists，是安全的)
        self.__init__(self.game_path, self.initial_video_path, self.server_url)
        
        initial_text = self.opening if hasattr(self, 'opening') else ""
        return initial_text, self.initial_video_path


class IF_Env_OtherLan(IF_Env):
    """
    支持其他语言（如阿拉伯语等）的环境。
    通过4omini翻译模块桥接，将用户输入的外语action翻译为英文，再与底层IF_Env交互。
    动作action的翻译必须准确，否则环境不会响应。
    """
    def __init__(self, game_path, tgt_lang):
        """
        translator: 一个具备translate(text, src_lang, tgt_lang)方法的翻译器实例
        """
        super().__init__(game_path)
        self.src_lang = "English"
        self.tgt_lang = tgt_lang

    def translate(self, text, action_list, target_lang, src_lang="English"):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL")
        trans_client = OpenAI(api_key=api_key, base_url=base_url)
        prompt = f"""Translate the following text from {src_lang} to {target_lang}. If the original text is already in the target language, leave it unchanged.

Note: The following is a list of action name strings for subsequent responses. All action name strings in this list must remain in the ENGLISH and be translated exactly as shown below:
{action_list}

The original text is:
{text}

IMPORTANT: Ensure that all action names in the text are translated exactly in ENGLISH as they appear in the above list or REMAIN in ENGLISH, without any changes. The action names should always be in ENGLISH.

Translated {target_lang} text:    
"""
        response = trans_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        # print(prompt)
        # print("ori:", text)
        # print("tgt:", response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()

    def get_system_prompt(self):
        intro_en = super().get_system_prompt()
        # print(intro_en)
        intro_tgt = self.translate(intro_en, self.actions, self.tgt_lang, self.src_lang)
        self.intro_en = intro_en
        self.intro_tgt = intro_tgt
        return intro_tgt

    def step(self, action: str):
        # action_en = self.action_map.get(action_tgt)
        # if not action_en:
        #     return f"Action '{action_tgt}' is not a valid action."
        # action_en = self.translate(action, self.actions, self.src_lang, self.tgt_lang)
        if action.lower() in self.actions:
            action_en = action
        else:
            action_en = self.translate(action.lower(), self.actions, self.src_lang, self.tgt_lang)
        feedback_en = super().step(action_en)
        feedback_tgt = self.translate(feedback_en, self.actions, self.tgt_lang, self.src_lang)
        return feedback_tgt
        

if __name__ == "__main__":
    # env_arabic = IF_Env_OtherLan("./TextWorld/TestWorld2/results/ulxEnv/EP1-005/EP1-005.ulx", "Arabic")
    # env = IF_Env("./TextWorld/TestWorld2/results/ulxEnv/EP1-003/gpt-5_10-27_1400_10-27_1514_attempt2.ulx")
    # env.process.stdin.write("\n")
    # env.process.stdin.flush()
    # time.sleep(0.3)
    # obs = env._read_output()
    # systemPrompt = env.opening
    # print(systemPrompt)

    Video_test_env = Multimodal_IF_Env(
        game_path="./TextWorld/TestWorld2/results/ulxEnv/EP1-005/EP1-005.ulx",
        initial_video_path="./Wan2.2/output/t2v-A14B_832*480_1_Static_cockpit_view_from_inside_an_**autonomous_ta_20251201_121616.mp4",
        server_url="http://gpu17:8000/generate"
    )

    print("Start")
    while True:
        action = input("Enter action: ")
        feedback, video_path = Video_test_env.step(action)
        print("-------------FEEDBACK------------", feedback)
        print("-------------VIDEO PATH------------", video_path)

    # feedback = env.step("keep straight")
    # print("-------------HELP------------", feedback)

    # feedback = env.step("look")
    # print("-------------LOOK------------", feedback)