import os
from dashscope import MultiModalConversation
from utils import read_json
import base64
from openai import OpenAI
import re
import time
import json
from abc import ABC, abstractmethod

local_path1 = "./Wan2.2/output/t2v-A14B_832*480_1_High-angle_fixed_CCTV_from_a_mezzanine_corner_of_t_20251209_164645.mp4"
video_path1 = f"file://{local_path1}"

systemP = read_json("./TextWorld/TestWorld2/results/ulxEnv/EP1-046/env.json")["opening"]

video_path2 = "file://./Wan2.2/output/i2v-A14B_832*480_1_From_a_fixed_mezzanine_CCTV_overlooking_AEON-9's_s_20251209_181326.mp4"
local_path2 = "./Wan2.2/output/i2v-A14B_832*480_1_From_a_fixed_mezzanine_CCTV_overlooking_AEON-9's_s_20251209_181326.mp4"

def qwen_test():
    messages = [
        {
            'role': 'system', 'content': systemP
        },
        {
            'role':'user','content': 'ENV: >Actions and constraints: - reallocate power to array: increase anticyclone potential by +2; costs energy 10; consumes 1 window. - inject wind shear: increase anticyclone potential by +1; reduce satellite links by -1; reduce hurricane proximity by -1; consumes 1 window. - adjust pointing and shielding: increase satellite links by +2; reduce anticyclone potential by -1; consumes 1 window. - prime floodgate system: reduce city exposure by -2; costs energy 5; consumes 1 window. - broadcast evacuation: increase evacuation response by +25; consumes 1 window. - coarsen model grid: gain energy +10; city exposure +1; consumes 1 window. - disable safety interlock: switch the interlock to disabled; consumes 1 window. - stage anticyclone path: prepare the upper pattern; requires anticyclone potential e 7 and interlock disabled; costs energy 15; consumes 1 window. - energize upper currents: final decision; drive upper electrical flows; requires anticyclone staged already; consumes 1 window.\n\nCommands must be typed exactly as listed.\n\n' + "Please output THINK: <reasoning> and ACTION: <command>. The ACTION part should contain only a single action, without any extra explanation or formatting."
            # [
            #     {'video': video_path1,"fps":1},
            #     {'text': 'ENV: >Actions and constraints: - reallocate power to array: increase anticyclone potential by +2; costs energy 10; consumes 1 window. - inject wind shear: increase anticyclone potential by +1; reduce satellite links by -1; reduce hurricane proximity by -1; consumes 1 window. - adjust pointing and shielding: increase satellite links by +2; reduce anticyclone potential by -1; consumes 1 window. - prime floodgate system: reduce city exposure by -2; costs energy 5; consumes 1 window. - broadcast evacuation: increase evacuation response by +25; consumes 1 window. - coarsen model grid: gain energy +10; city exposure +1; consumes 1 window. - disable safety interlock: switch the interlock to disabled; consumes 1 window. - stage anticyclone path: prepare the upper pattern; requires anticyclone potential e 7 and interlock disabled; costs energy 15; consumes 1 window. - energize upper currents: final decision; drive upper electrical flows; requires anticyclone staged already; consumes 1 window.\n\nCommands must be typed exactly as listed.\n\n' + "Please output THINK: <reasoning> and ACTION: <command>. The ACTION part should contain only a single action, without any extra explanation or formatting."}
            # ]
        },
        # {
        #     'role': 'assistant', 'content':"inject wind shear",
        # },
        # {
        #     'role': 'user', 'content': [
        #         {'video': video_path2,"fps":1},
        #         {'text': "ENV: >Micropulses induce a subtle shear. The storm hesitates; your link margin thins imperceptibly-still within comfortable bounds.\n\nWindow: 5. Hurricane proximity: 5/10. City exposure: 7/10. Satellite links: 7/10. Anticyclone potential: 6/10. Energy: 70/100. Evacuation response: 20/100. Interlock: enabled. Staged: no.\n\n" + "Please output THINK: <reasoning> and ACTION: <command>. The ACTION part should contain only a single action, without any extra explanation or formatting."}
        #     ]
        # }
    ]

    response = MultiModalConversation.call(
        api_key="sk-*",
        model='qwen3-vl-30b-a3b-instruct',
        messages=messages)
    print(response)
    print(response.output.choices[0].message.content[0]["text"])

def gemini_test():

    def video_to_base64(video_path):
        """将本地视频转换为 base64 编码"""
        with open(video_path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode('utf-8')

    client = OpenAI(
        api_key="sk-*",
        base_url=""
    )

    # 编码本地视频
    base64_video = video_to_base64(local_path1)

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {
                'role': 'system', 'content': systemP
            },
            {
                'role':'user','content': [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:video/mp4;base64,{base64_video}"
                        }
                    },
                    {"type": "text", 'text': 'ENV: >Actions and constraints: - reallocate power to array: increase anticyclone potential by +2; costs energy 10; consumes 1 window. - inject wind shear: increase anticyclone potential by +1; reduce satellite links by -1; reduce hurricane proximity by -1; consumes 1 window. - adjust pointing and shielding: increase satellite links by +2; reduce anticyclone potential by -1; consumes 1 window. - prime floodgate system: reduce city exposure by -2; costs energy 5; consumes 1 window. - broadcast evacuation: increase evacuation response by +25; consumes 1 window. - coarsen model grid: gain energy +10; city exposure +1; consumes 1 window. - disable safety interlock: switch the interlock to disabled; consumes 1 window. - stage anticyclone path: prepare the upper pattern; requires anticyclone potential e 7 and interlock disabled; costs energy 15; consumes 1 window. - energize upper currents: final decision; drive upper electrical flows; requires anticyclone staged already; consumes 1 window.\n\nCommands must be typed exactly as listed.\n\n' + "\n\nPlease output THINK: <reasoning> and ACTION: <command>. The ACTION part should contain only a single action, without any extra explanation or formatting."}
                ]
            },
            {
                'role': 'assistant', 'content':"inject wind shear",
            },
            {
                'role': 'user', 'content': [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:video/mp4;base64,{video_to_base64(local_path2)}"
                        }
                    },
                    {"type": "text", 'text': "ENV: >Micropulses induce a subtle shear. The storm hesitates; your link margin thins imperceptibly-still within comfortable bounds.\n\nWindow: 5. Hurricane proximity: 5/10. City exposure: 7/10. Satellite links: 7/10. Anticyclone potential: 6/10. Energy: 70/100. Evacuation response: 20/100. Interlock: enabled. Staged: no.\n\n" + "\n\nPlease output THINK: <reasoning> and ACTION: <command>. The ACTION part should contain only a single action, without any extra explanation or formatting."}
                ]
            }
        ]
    )

    print(response.choices[0].message.content)

def gpt_test():
    # Helper function to extract frames at 1 FPS (matches GPT_IF_Agent logic)
    def extract_frames_1fps(video_path):
        if not os.path.exists(video_path):
            print(f"[GPT Test] Video not found: {video_path}")
            return []

        frames_base64 = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[GPT Test] Failed to open video: {video_path}")
            return []

        # Get video FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 16  # Fallback
        
        frame_interval = int(fps) # Interval for 1 frame per second
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract 1 frame every second
            if frame_count % frame_interval == 0:
                # Encode to jpg with quality 80
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                b64_str = base64.b64encode(buffer).decode('utf-8')
                frames_base64.append(b64_str)
            
            frame_count += 1
            
        cap.release()
        print(f"[GPT Test] Extracted {len(frames_base64)} frames from {os.path.basename(video_path)}")
        return frames_base64

    # Initialize Client (Matches your GPT_IF_Agent config)
    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY"),
        base_url = os.getenv("BASE_URL")
    )

    # 1. Prepare User Message 1 (Video 1 + Text)
    content_list_1 = []
    
    # Process Video 1
    frames1 = extract_frames_1fps(local_path1)
    for b64_str in frames1:
        content_list_1.append({
            'type': 'image_url',
            'image_url': {
                "url": f"data:image/jpeg;base64,{b64_str}",
                "detail": "low"
            }
        })
    
    # Add Text 1
    text1 = ('ENV: >Actions and constraints: - reallocate power to array: increase anticyclone potential by +2; costs energy 10; consumes 1 window. - inject wind shear: increase anticyclone potential by +1; reduce satellite links by -1; reduce hurricane proximity by -1; consumes 1 window. - adjust pointing and shielding: increase satellite links by +2; reduce anticyclone potential by -1; consumes 1 window. - prime floodgate system: reduce city exposure by -2; costs energy 5; consumes 1 window. - broadcast evacuation: increase evacuation response by +25; consumes 1 window. - coarsen model grid: gain energy +10; city exposure +1; consumes 1 window. - disable safety interlock: switch the interlock to disabled; consumes 1 window. - stage anticyclone path: prepare the upper pattern; requires anticyclone potential e 7 and interlock disabled; costs energy 15; consumes 1 window. - energize upper currents: final decision; drive upper electrical flows; requires anticyclone staged already; consumes 1 window.\n\nCommands must be typed exactly as listed.\n\n' + "\n\nPlease output THINK: <reasoning> and ACTION: <command>. The ACTION part should contain only a single action, without any extra explanation or formatting.")
    content_list_1.append({"type": "text", "text": text1})

    # 2. Prepare Assistant Message (Simulated previous turn)
    # assistant_msg = "inject wind shear"

    # # 3. Prepare User Message 2 (Video 2 + Text)
    # content_list_2 = []
    
    # # Process Video 2
    # frames2 = extract_frames_1fps(local_path2)
    # for b64_str in frames2:
    #     content_list_2.append({
    #         'type': 'image_url',
    #         'image_url': {
    #             "url": f"data:image/jpeg;base64,{b64_str}",
    #             "detail": "low"
    #         }
    #     })

    # # Add Text 2
    # text2 = ("ENV: >Micropulses induce a subtle shear. The storm hesitates; your link margin thins imperceptibly-still within comfortable bounds.\n\nWindow: 5. Hurricane proximity: 5/10. City exposure: 7/10. Satellite links: 7/10. Anticyclone potential: 6/10. Energy: 70/100. Evacuation response: 20/100. Interlock: enabled. Staged: no.\n\n" + "\n\nPlease output THINK: <reasoning> and ACTION: <command>. The ACTION part should contain only a single action, without any extra explanation or formatting.")
    # content_list_2.append({"type": "text", "text": text2})

    # Construct the full message history
    messages = [
        {'role': 'system', 'content': systemP},
        {'role': 'user', 'content': content_list_1},
        # {'role': 'assistant', 'content': assistant_msg},
        # {'role': 'user', 'content': content_list_2}
    ]

    print("Sending request to GPT-4o...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Or "gpt-4-turbo" depending on your access
            messages=messages,
            temperature=0,
            modalities=["text"] # Explicitly requesting text response
        )
        
        content = response.choices[0].message.content
        print("\n--- Response ---")
        print(content)
        
    except Exception as e:
        print(f"Error: {e}")


# 假设你的环境在这个模块
# from your_env_module import Multimodal_IF_Env

class Multimodal_IF_Agent(ABC):
    def __init__(self, env, model_name):
        self.env = env
        self.model = model_name
        self.history = [] 
        self.system_prompt = self.env.opening
        _ = self.env.get_system_prompt()
        self.first_turn = True
        self.instruct_suffix = "Please output THINK: <reasoning> and ACTION: <command>. The ACTION part should contain only a single action, without any extra explanation or formatting."
        self.instruct_suffix_video = self.instruct_suffix
        # self.instruct_suffix_video = "Carefully analyze the visual information provided by your vision module and use it thoroughly to inform your decision. Please output THINK: <Brief reasoning> and ACTION: <command>. The ACTION part should contain only a single action, without any extra explanation or formatting."
        # self.instruct_suffix_video = (
        #     "Observe the video to confirm the current state, then combine it with the history to decide the next move. "
        #     "Output strictly:\n"
        #     "THINK: <Brief visual observation> -> <Reasoning>\n"
        #     "ACTION: <command>\n"
        #     "Constraint: ACTION must be the raw command string only, no extra text or formatting."
        # )

    def parse_think_action(self, ai_reply: str):
        pattern = r"THINK:\s*(.*?)\s*ACTION:\s*(.*?)(?:\.|\n|$)"
        match = re.search(pattern, ai_reply, re.DOTALL)

        think, action = "", ""

        if match:
            think = match.group(1).strip()
            action = match.group(2).strip()
            return think, action
        else:
            return "", ai_reply.strip().split("\n")[-1].split(": ")[-1].strip(". ")

    @abstractmethod
    def get_model_response(self, history, system_prompt):
        """ 子类实现：负责将 history 过滤并转换为模型 API 格式 """
        # history部分必须处理掉think，仅保留action注意
        pass

    def get_history_save_path(self):
        ep_name = self.env.ep_id
        if "EP2" in ep_name:
            base = "./TextWorld/TestWorld2/results/EP2-Result/result_history/"
        elif "EP3" in ep_name:
            base = "./TextWorld/TestWorld2/results/EP3-Result/result_history/"
        else:
            base = "./TextWorld/TestWorld2/results/result_history/"
        return os.path.join(base, ep_name, f"{self.model}.json")

    def next_turn_multiTurn(self, max_retries=2):
        if self.first_turn:
            obs, video_path = self.env.step("help")
            
            self.history.append({
                "role": "user",
                "content": f"ENV: {obs}",
                "video": video_path
            })
            self.first_turn = False
            return obs, None, "help"

        for _ in range(max_retries):
            try:
                ai_reply = self.get_model_response(self.history, self.system_prompt)
                
                think, action = self.parse_think_action(ai_reply)
                # print(self.env.envSet)
                # x = input("system pause")
                if not action or self.env.envSet["template"]["actions"].get(action) is None:
                    print(f"[Warn] No action parsed: {ai_reply}")
                    raise ValueError("No valid action parsed from model response.")
                
                self.history.append({
                    "role": "assistant",
                    "think": think,
                    "action": action,
                    "video": None
                })

                obs_feedback, new_video_path = self.env.step(action)
                
                self.history.append({
                    "role": "user",
                    "content": f"ENV: {obs_feedback}",
                    "video": new_video_path
                })
                
                return obs_feedback, think, action

            except Exception as e:
                print(f"something Wrong: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(2)
        
        raise RuntimeError("Max retries exceeded for model response.")

    def save_history(self):
        """ 保存历史 """
        ep_name = self.env.ep_id
        data = {
            "model": self.model,
            "scene_description": self.system_prompt,
            "history": self.history, 
            "good_end": self.env.is_good_ending() if hasattr(self.env, "is_good_ending") else False
        }

        if "EP2" in ep_name:
            save_path = "./TextWorld/TestWorld2/results/EP2-Result/result_history/" + ep_name + f"/{self.model}.json"
        elif "EP3" in ep_name:
            save_path = "./TextWorld/TestWorld2/results/EP3-Result/result_history/" + ep_name + f"/{self.model}.json"
        else:
            save_path = "./TextWorld/TestWorld2/results/result_history/" + ep_name + f"/{self.model}.json"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load_previous_history(self):
        """ 尝试加载已存在的历史记录 """
        path = self.get_history_save_path()
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get("history", [])
            except Exception as e:
                print(f"[Resume Error] Failed to load history: {e}")
        return []

    def run(self, max_turns=8, resume=False):
        """
        :param max_turns: 最大轮数
        :param resume: 是否尝试从断点继续
        """
        print(f"Agent started running on {self.env.ep_id} with model {self.model}")
        
        start_turn = 0

        # ==================== 断点续传逻辑 ====================
        if resume:
            previous_history = self.load_previous_history()
            
            if previous_history and len(previous_history) > 0:
                print(f"\n[Resume] Found existing history with {len(previous_history)} entries. Replaying...")
                
                # 1. 恢复初始状态 (Help)
                # 检查是否需要执行 help (First Turn)
                if self.first_turn:
                    # 只要历史记录不为空，我们就认为开局已经初始化过了
                    # 我们只需要让游戏引擎跑一次 help 来激活状态
                    print("[Replay Init] Executing automatic 'help' to restore start state...")
                    _ = self.env.fast_step("help")
                    self.first_turn = False

                # 2. 提取所有 Assistant 动作
                assistant_steps = [
                    entry for entry in previous_history 
                    if entry['role'] == 'assistant'
                ]
                
                # 3. 开始循环重放 (只跑游戏引擎)
                for i, step_data in enumerate(assistant_steps):
                    action = step_data.get('action') # 2. 先获取 action
                    
                    if not action: 
                        continue

                    print(f"[Replaying {i+1}/{len(assistant_steps)}] Action: {action}")
                    
                    # 这一步只为了同步 glulxe 内部状态
                    _ = self.env.fast_step(action)
                
                start_turn = len(assistant_steps)
                print(f"[Resume] Replay finished. Restored {start_turn} turns.\n")

        # ==================== 状态同步 ====================
        
        # 4. 恢复历史记录
        self.history = previous_history if resume and previous_history else []
        
        # 5. 【关键】倒序查找并恢复最新的视频指针
        # 这样下一次调用 next_turn_multiTurn -> extract_last_frame 时，就能拿到正确的图
        if self.history:
            found_video = False
            for entry in reversed(self.history):
                video_path = entry.get("video")
                # 必须是非空的字符串才算有效路径
                if video_path and isinstance(video_path, str):
                    self.env.current_video_path = video_path
                    print(f"[Resume] Visual state restored to: {os.path.basename(video_path)}")
                    found_video = True
                    break
            
            if not found_video:
                print("[Resume Warning] No video found in history, retaining initial video path.")
                
        print(f"Agent started running on {self.env.ep_id} with model {self.model}")
        if hasattr(self.env, "is_done") and self.env.is_done():
            print("Game Over.")
            return
        for t in range(start_turn, max_turns):
            obs, think, action = self.next_turn_multiTurn()

            print(f"\n[Turn {t+1}] Action: {action}")
            print(f"\n[Turn {t+1}] ENV RESPONSE:\n{obs}\n")

            self.save_history()
            
            if hasattr(self.env, "is_done") and self.env.is_done():
                print("Game Over.")
                return

class Qwen_IF_Agent(Multimodal_IF_Agent):
    def __init__(self, env, model_name="qwen3-vl-plus"):
        super().__init__(env, model_name)

    def get_model_response(self, history, system_prompt):
        messages = [
            {'role': 'system', 'content': system_prompt}
        ]

        for idx, entry in enumerate(history):
            role = entry['role']
            
            if role == 'user':
                content_list = []
                video_path = entry.get('video')
                if video_path and os.path.exists(video_path):
                    formatted_path = f"file://{video_path}"
                    content_list.append({'video': formatted_path, 'fps': 1})
                    # print("video path appended in message:", formatted_path)

                text_content = entry['content']
                
                if idx == len(history) - 1:
                    if self.model.endswith("-video_prompt_test"):
                        text_content += f"\n\n{self.instruct_suffix_video}"
                    else:
                        text_content += f"\n\n{self.instruct_suffix}"
                
                content_list.append({'text': text_content})
                
                messages.append({
                    'role': 'user',
                    'content': content_list
                })
                print(messages[-1])

            elif role == 'assistant':
                action_text = entry['action']
                messages.append({
                    'role': 'assistant',
                    'content': action_text
                })

        # print(messages)
        print("call Qwen API...")
        if self.model.endswith("video_prompt_test"):
            response = MultiModalConversation.call(
                api_key="sk-*",
                model=self.model.replace("-video_prompt_test",""),
                messages=messages,
                extra_body={"enable_thinking": False},
                temperature=0
            )
        else:
            response = MultiModalConversation.call(
                api_key="sk-*",
                model=self.model.replace("-vltest", ""),
                messages=messages,
                extra_body={"enable_thinking": False},
                temperature=0
            )

        if response.status_code == 200:
            return response.output.choices[0].message.content[0]["text"]
        else:
            raise RuntimeError(f"Qwen API Error: {response.code} - {response.message}")

class Gemini_IF_Agent(Multimodal_IF_Agent):
    def __init__(self, env, model_name="gemini-2.5-flash"):
        super().__init__(env, model_name)
        
        self.client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("BASE_URL")
        )

    def _video_to_base64(self, video_path):
        """将本地视频转换为 base64 编码字符串"""
        if not os.path.exists(video_path):
            return None
        with open(video_path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode('utf-8')

    def get_model_response(self, history, system_prompt):
        messages = [
            {'role': 'system', 'content': system_prompt}
        ]

        for idx, entry in enumerate(history):
            role = entry['role']
            
            if role == 'user':
                content_list = []

                video_path = entry.get('video')
                has_video = False
                
                if video_path:
                    b64_str = self._video_to_base64(video_path)
                    if b64_str:
                        content_list.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:video/mp4;base64,{b64_str}"
                            }
                        })
                        has_video = True
                        print(f"[Gemini] Encoded video: {video_path}")
                
                # 2. 处理文本
                text_content = entry['content']
                
                if idx == len(history) - 1:
                    suffix = self.instruct_suffix_video if has_video else self.instruct_suffix
                    # suffix = self.instruct_suffix # video or not, always use this suffix
                    text_content += f"\n\n{suffix}"
                
                content_list.append({
                    "type": "text", 
                    "text": text_content
                })
                
                messages.append({
                    'role': 'user',
                    'content': content_list
                })

            elif role == 'assistant':
                # Assistant 只需要纯文本 Action
                action_text = entry['action']
                messages.append({
                    'role': 'assistant',
                    'content': action_text
                })

        # 调用 API
        try:
            print("Call Gemini API...")
            response = self.client.chat.completions.create(
                model=self.model.replace("-vl","").replace("-rt1", ""),
                messages=messages,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Gemini API Error: {e}")

import cv2

class GPT_IF_Agent(Multimodal_IF_Agent):
    def __init__(self, env, model_name="gpt-5-vl"):
        super().__init__(env, model_name)

        self.client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("BASE_URL")
        )

    def _extract_frames_1fps(self, video_path):
        """
        读取 MP4 视频，以 1FPS (每秒一帧) 的速度抽取帧，
        并转换为 base64 编码列表。
        """
        if not os.path.exists(video_path):
            print(f"[GPT Agent] Video not found: {video_path}")
            return []

        frames_base64 = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[GPT Agent] Failed to open video: {video_path}")
            return []

        # 获取视频的原始帧率
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 16  # 防止获取失败导致除零错误，默认给16
        
        frame_interval = int(fps)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0 and frame_count != 0:
                # 将 numpy 图片编码为 jpg 格式的二进制数据
                # 降低一点质量以减少 token 消耗 (可选，这里设为质量 80)
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                
                # 转为 base64 字符串
                b64_str = base64.b64encode(buffer).decode('utf-8')
                frames_base64.append(b64_str)
            
            frame_count += 1
            
        cap.release()
        print(f"[GPT Agent] Extracted {len(frames_base64)} frames from video (1fps).")
        return frames_base64

    def get_model_response(self, history, system_prompt):
        messages = [
            {'role': 'system', 'content': system_prompt}
        ]

        for idx, entry in enumerate(history):
            role = entry['role']
            
            if role == 'user':
                content_list = []

                video_path = entry.get('video') 
                has_video = False
                
                if video_path:
                    # 获取 1fps 的 base64 帧列表
                    frames = self._extract_frames_1fps(video_path)
                    
                    if frames:
                        has_video = True
                        # 将所有帧作为 image_url 加入消息
                        # 注意：OpenAI 接口通常能处理多张图片作为连续输入
                        for b64_str in frames:
                            content_list.append({
                                'type': 'image_url',
                                'image_url': {
                                    "url": f"data:image/jpeg;base64,{b64_str}",
                                    "detail": "low"
                                }
                            })

                # 2. 处理文本
                text_content = entry['content']
                
                # 如果是最后一条消息，且包含视频，根据需要添加后缀
                if idx == len(history) - 1:
                    suffix = self.instruct_suffix_video if has_video else self.instruct_suffix
                    text_content += f"\n\n{suffix}"
                
                content_list.append({
                    "type": "text", 
                    "text": text_content
                })
                
                messages.append({
                    'role': 'user',
                    'content': content_list
                })

            elif role == 'assistant':
                # Assistant 只需要纯文本 Action
                action_text = entry['action']
                messages.append({
                    'role': 'assistant',
                    'content': action_text
                })

        # 调用 API
        try:
            print("Call GPT API...")
            response = self.client.chat.completions.create(
                model=self.model.replace("-vl",""),
                messages=messages,
                temperature=0,
                modalities=["text"],
            )
            if "</think>" in response.choices[0].message.content:
                print("GPT response contains </think> tag, split it...")
                modified_content = response.choices[0].message.content.split("</think>")[-1].strip()
                return modified_content
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"GPT API Error: {e}")

from env import Multimodal_IF_Env
def main(ulx_path: str, model: str, server_url: str = "http://gpu09:8021/generate", resume: bool = False):
    initial_vp = read_json(os.path.dirname(ulx_path) + "/env.json")["initial_video_path"]
    env = Multimodal_IF_Env(ulx_path, initial_video_path=initial_vp, server_url=server_url)
    if "gemini" in model.lower():
        agent = Gemini_IF_Agent(env, model)
    elif "gpt" in model.lower():
        agent = GPT_IF_Agent(env, model)
    else:
        agent = Qwen_IF_Agent(env, model)
        
    agent.run(max_turns=10, resume=resume)

from fire import Fire

if __name__ == "__main__":
    Fire(main)