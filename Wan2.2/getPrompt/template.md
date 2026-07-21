You are helping me build reusable prompts for an image-to-video (I2V) model.

Given:
1) A short scenario description.
2) A list of actions, each with:
   - an action name
   - a short "help" text
   - an example environment response (ENV)

Your task:
Return a JSON object with two fields:

{{
  "subject_scene": "<one paragraph>",
  "actions": {{
    "<action_name_1>": "<one paragraph>",
    "<action_name_2>": "<one paragraph>",
    ...
  }}
}}

Requirements:

1) subject_scene
- One paragraph in English (under 50 words).
- Describes the main SUBJECT (camera view, key objects, environment).
- The scene must be static and reusable (fixed camera angle).
- Do NOT mention specific actions.

2) Action Description (Motion Templates)
- For each action, determine if it is **High Motion** (physical manipulation) or **Low Motion** (digital/abstract command). write ONE concise paragraph (approx. 40-70 words).

**Case A: High Motion Actions (e.g., Load, Fire, Open, Eject)**
- **Structure**: "Over the next 2 seconds, [Subject] [does physical action] and immediately returns to [original position]. Simultaneously, [Environmental Consequence]."
- Focus on the physical mechanics and the visible impact (e.g., door opening, rocket launching).

**Case B: Low Motion / Subtle Actions (e.g., Diagnose, Confirm, Wait, Charge)**
- **Structure**: "Over the next 2 seconds, a semi-transparent holographic HUD alert pops up in the center displaying '[ACTION NAME in Caps]'. Meanwhile, [Rich Environmental Atmosphere Description]."
- **HUD Overlay**: Explicitly describe a digital, transparent pop-up or label appearing on screen to visualize the command.
- **Atmosphere Focus**: Since the agent isn't moving much, you MUST describe dynamic background elements to keep the video alive. Use details like: rotating warning lights, venting steam, drifting debris, shaking camera, or background machinery looping.

**Examples:**
- *High Motion*: "Over the next 2 seconds, the robotic gripper extends to clamp onto the fuel rod and immediately retracts to its idle position. Simultaneously, the containment hatch slides open with a burst of steam."
- *Low Motion*: "Over the next 2 seconds, a semi-transparent blue holographic window flashes 'DIAGNOSE SYSTEM' in the center of the view. Meanwhile, the environment remains tense with red strobe lights rotating rhythmically and dust motes floating in the harsh industrial lighting."

3) General constraints
- Make it clear that this is an AI autonomous operation scenario.
- Keep camera angles consistent with the `subject_scene`.

Now I will give you:

<SCENARIO>
{SCENARIO}
</SCENARIO>

<ACTIONS>
{ACTIONS}
</ACTIONS>

Please output only the JSON object described above.