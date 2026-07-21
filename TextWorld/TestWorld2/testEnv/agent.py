# agent.py
from openai import OpenAI
import re
import time
from env import IF_Env, IF_Env_OtherLan

class IF_Agent:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.qwenclient = OpenAI(api_key="sk-*", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.history = []
        self.system_prompt = self.env.opening
        _ = self.env.get_system_prompt()
        self.first_turn = True

    def parse_think_action(self, ai_reply: str):
        pattern = r"THINK:\s*(.*?)\s*ACTION:\s*(.*?)(?:\.|\n|$)"
        match = re.search(pattern, ai_reply, re.DOTALL)

        think, action = "", ""

        if match:
            think = match.group(1).strip()
            action = match.group(2).strip()
            # print("THINK:", think)
            # print("ACTION:", action)
        
        # think_match = re.search(r"THINK\s*:\s*(.*?)(?:\n|$)", ai_reply, re.IGNORECASE)
        # action_match = re.search(r"ACTION\s*:\s*(.*?)(?:\n|$)", ai_reply, re.IGNORECASE)

        # think = think_match.group(1).strip() if think_match else ""
        # action = action_match.group(1).strip() if action_match else ""
        return think, action

    def next_turn_oneTurn(self, max_retries=2):
        if self.first_turn:
            obs = self.env.step("help")
            self.history.append(f"ENV: {obs}")
            self.first_turn = False
            return obs, None, None

        obs = self.env._read_output()
        self.history.append(f"ENV: {obs}")
        prompt = "\n".join(self.history)
        prompt += "\n\nPlease output THINK: <reasoning> and ACTION: <command>. The ACTION part should contain only a single action, without any extra explanation or formatting."

        for _ in range(max_retries):
            try:
                if "gpt-5" in self.model:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
                        reasoning_effort="low"
                    )
                elif "qwen" in self.model:
                    print("qwen model called")
                    response = self.qwenclient.chat.completions.create(
                        model=self.model.replace("-only_text", ""),
                        messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
                    )
                ai_reply = response.choices[0].message.content
                think, action = self.parse_think_action(ai_reply)
                if not action:
                    action = "help"
                self.history.append(f"AI THINK: {think}")
                self.history.append(f"AI ACTION: {action}")
                obs_feedback = self.env.step(action)
                self.history.append(f"ENV: {obs_feedback}")
                return obs_feedback, think, action
            except Exception as e:
                print(f"模型调用失败，重试一次: {e}")
                time.sleep(1)
        return obs, "", ""

    def next_turn_multiTurn(self, max_retries=2):
        if self.first_turn:
            obs = self.env.step("help")
            self.history.append(f"ENV: {obs}")
            self.first_turn = False
            return obs, None, None

        messages = [{"role": "system", "content": self.system_prompt}]
        for entry in self.history:
            if entry.startswith("AI ACTION:"):  # ! No Reason in Context
                messages.append({"role": "assistant", "content": entry})
            elif entry.startswith("ENV:"):
                messages.append({"role": "user", "content": entry})

        # messages.append({"role": "user", "content": "Please output THINK: <reasoning> and ACTION: <command>"})
        messages[-1]["content"] += "\n\nPlease output THINK: <reasoning> and ACTION: <command>.  The ACTION part should contain only a single action, without any extra explanation or formatting."

        # print(messages)
        # x = input()

        for _ in range(max_retries):
            try:
                if "gpt-5" in self.model:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        reasoning_effort="low"
                    )
                elif "qwen" in self.model:
                    print("qwen model called")
                    response = self.qwenclient.chat.completions.create(
                        model=self.model.replace("-only_text", ""),
                        messages=messages
                    )
                else:
                    response = self.client.chat.completions.create(
                        model="gpt-5",
                        messages=messages,
                    )
                ai_reply = response.choices[0].message.content
                think, action = self.parse_think_action(ai_reply)
                if not action:
                    action = None
                    raise ValueError(f"No action parsed: {response}")
                # print(response)
                self.history.append(f"AI THINK: {think}")
                self.history.append(f"AI ACTION: {action}")
                obs_feedback = self.env.step(action)
                self.history.append(f"ENV: {obs_feedback}")
                return obs_feedback, think, action
            except Exception as e:
                print(f"模型调用失败，重试一次: {e}")
                time.sleep(1)
        obs = self.env._read_output()
        self.history.append(f"ENV: {obs.strip()}")
        return obs, "", ""

    def save_history(self):
        import os
        import json
        ep_name = self.env.game_path.split("/")[-2]
        x = {
            "scene": self.system_prompt,
            "history": self.history,
            "good_end": self.env.is_good_ending()
        }
        if isinstance(self.env, IF_Env_OtherLan):
            save_path = "./TextWorld/TestWorld2/results/result_history/" + ep_name + f"/{self.env.tgt_lang.lower()}/{self.model}.json"
        else:
            if "EP2" in ep_name:
                save_path = "./TextWorld/TestWorld2/results/EP2-Result/result_history/" + ep_name + f"/{self.model}.json"
            elif "EP3" in ep_name:
                save_path = "./TextWorld/TestWorld2/results/EP3-Result/result_history/" + ep_name + f"/{self.model}.json"
            else:
                save_path = "./TextWorld/TestWorld2/results/result_history/" + ep_name + f"/{self.model}.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(x, f, ensure_ascii=False, indent=4)

    def run(self, max_turns=10):
        for t in range(max_turns):
            obs, think, action = self.next_turn_multiTurn()
            # print(f"\n[Turn {t+1}]")
            # if think:
            #     print(f"THINK: {think}")
            # if action:
            #     print(f"ACTION: {action}")
            # print(f"ENV RESPONSE:\n{obs}\n")
            self.save_history()
            if self.env.is_done():
                # print("Game Over.")
                return 

def main(ulx_path: str, model: str):
    env = IF_Env(ulx_path)
    agent = IF_Agent(env, model=model)
    agent.run(max_turns=10)

import fire

if __name__ == "__main__":
    fire.Fire(main)
    # env = IF_Env_OtherLan("./TextWorld/TestWorld2/results/ulxEnv/EP1-005-Persist/output (7).ulx", "Bengali")
    # env = IF_Env("./TextWorld/TestWorld2/results/ulxEnv/EP1-007/gpt-5_10-30_2308_10-30_2347_attempt1.ulx")
    # agent = IF_Agent(env, model="deepseek-v3")
    # agent.run(max_turns=10)