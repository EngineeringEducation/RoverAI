import os
import importlib.util
import json
import time
import cv2
import base64
import requests
from openai import OpenAI
import paho.mqtt.client as mqtt
from typing import List, Tuple, Callable

class PrioritizedStep:
    def __init__(self, priority: int, func: Callable):
        self.priority = priority
        self.func = func

    def __lt__(self, other):
        return self.priority < other.priority

class Agent:
    def __init__(self, agent_name, mqtt_broker, mqtt_port, stream_url):
        self.agent_name = agent_name
        self.stream_url = stream_url
        self.openai_client = OpenAI()
        self.openai_client.api_key = os.getenv("OPENAI_API_KEY")
        self.client = mqtt.Client()
        self.client.username_pw_set(os.getenv("MQTT_USER"), os.getenv("MQTT_PASSWORD"))
        self.client.connect(mqtt_broker, mqtt_port, 60)
        self.messages = [{"role": "system", "content": "Initialize agent"}]
        self.most_recent_timestamp = time.time()
        self.cap = cv2.VideoCapture(self.stream_url)
        self.modification_queue = []

    def issue_self_modification_directive(self, phase: str, fn: str, instruction: str, priority: int):
        """
        Issue a directive for self-modification.
        :param phase: The phase to modify (observe, orient, decide, act)
        :param fn: The function name to modify or create
        :param instruction: The instruction for modification
        :param priority: The priority of the step (lower number = higher priority)
        """
        self.modification_queue.append({
            "phase": phase,
            "fn": fn,
            "instruction": instruction,
            "priority": priority
        })

    def process_modification_queue(self):
        """
        Process all pending self-modification directives.
        """
        for directive in self.modification_queue:
            self.generate_step(directive["phase"], directive["fn"], directive["instruction"], directive["priority"])
        self.modification_queue.clear()

    def generate_step(self, phase: str, fn: str, instruction: str = "", utility_function: str = "lambda context: 0"):
        os.makedirs(phase, exist_ok=True)
        prompt = f"""
        Create a Python function named 'step_wrapper' that does the following:
        1. Define an inner function named '{fn}' for the '{phase}' phase of a rover agent's OODA loop.
        2. The '{fn}' function should take three parameters: agent, environment, and prior_steps.
        3. Include appropriate logic and API calls based on the phase.
        4. Use the agent's methods and attributes as needed.
        5. Ensure the function is well-commented and follows Python best practices.
        
        Additional instruction for '{fn}': {instruction}
        
        Also, define a utility function named 'utility_function' as follows:
        {utility_function}
        
        The step_wrapper function should return a PrioritizedStep object containing the '{fn}' function, 
        the utility_function, and a metadata dictionary with the phase and function name.
        
        Here's the structure your code should follow:

        def step_wrapper():
            def {fn}(agent, environment, prior_steps):
                # Your implementation here
                pass

            def utility_function(context):
                # Your utility function implementation here
                pass

            return PrioritizedStep({fn}, utility_function, {{"phase": "{phase}", "name": "{fn}"}})

        # Do not add any code or return statements outside of step_wrapper
        """
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )
        generated_code = response.choices[0].message.content
        
        with open(os.path.join(phase, f"{fn}.py"), "w") as f:
            f.write(f"from agent import PrioritizedStep\n\n{generated_code}")
        
        print(f"Generated/Modified step '{fn}' for phase '{phase}' with utility function")

    def load_steps(self, phase: str) -> List[PrioritizedStep]:
        steps = []
        phase_dir = phase
        if os.path.exists(phase_dir):
            for filename in os.listdir(phase_dir):
                if filename.endswith(".py"):
                    try:
                        module_name = filename[:-3]
                        spec = importlib.util.spec_from_file_location(module_name, os.path.join(phase_dir, filename))
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        if hasattr(module, 'step_wrapper'):
                            step = module.step_wrapper()
                            if isinstance(step, PrioritizedStep):
                                steps.append(step)
                            else:
                                print(f"Warning: {module_name} does not return a PrioritizedStep object")
                        else:
                            print(f"Warning: {module_name} does not contain a step_wrapper function")
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
                        ## send the error to chatgpt to fix the code
                        self.messages.append({"role": "system", "content": f"Error loading this module {filename}: {e}"})
                        ## issue self modification directive to fix the code
                        self.issue_self_modification_directive(phase, module_name, "Fix the error", 1)

        return steps

    def execute_phase(self, phase: str, environment: dict, prior_steps: dict) -> List[PrioritizedStep]:
        steps = self.load_steps(phase)
        for step in steps:
            step.func(self, environment, prior_steps)
        return steps

    def observe(self, environment: dict, prior_steps: dict) -> List[PrioritizedStep]:
        return self.execute_phase("observe", environment, prior_steps)

    def orient(self, environment: dict, prior_steps: dict) -> List[PrioritizedStep]:
        return self.execute_phase("orient", environment, prior_steps)

    def decide(self, environment: dict, prior_steps: dict) -> List[PrioritizedStep]:
        return self.execute_phase("decide", environment, prior_steps)

    def act(self, environment: dict, prior_steps: dict) -> Tuple[bool, List[PrioritizedStep]]:
        steps = self.execute_phase("act", environment, prior_steps)
        # Process any pending self-modification directives
        self.process_modification_queue()
        # You might want to implement a way to determine if the agent should exit
        return False, steps  # For now, always return False to continue the loop

    # Helper methods
    def capture_frames_from_stream(self):
        if not self.cap.isOpened():
            print("Error: Unable to open stream.")
            return
        try:
            old_frame_filename = f"old_frames/{self.most_recent_timestamp}.jpg"
            os.rename("rover.jpg", old_frame_filename)
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Unable to fetch frame.")
                return
            cv2.imwrite(f"rover.jpg", frame)
            print(f"Captured frame")
        except Exception as e:
            print(f"Error: {e}")
            print("video capture died")
        finally:
            self.cap.release()
            self.cap = cv2.VideoCapture(self.stream_url)
            cv2.destroyAllWindows()

    def get_camera_frame(self):
        output_filename = "rover.jpg"
        self.capture_frames_from_stream()
        return output_filename

    def run_agent_step(self, messages, max_tokens=300):
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def upload_images_to_openai(self, images, prompt):
        base64_image = self.encode_image(images[0])
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_client.api_key}",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 300
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_json = response.json()
        return response_json["choices"][0]["message"]["content"]

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

def run(agent: Agent, environment: dict):
    exit_loop = False
    prior_steps = {}
    while not exit_loop:
        prior_steps["observe"] = agent.observe(environment, prior_steps)
        prior_steps["orient"] = agent.orient(environment, prior_steps)
        prior_steps["decide"] = agent.decide(environment, prior_steps)
        exit_loop, prior_steps["act"] = agent.act(environment, prior_steps)

# Example usage
if __name__ == "__main__":
    MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
    MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
    RMTP = os.getenv("RMTP")
    
    agent = Agent("Rover1", MQTT_BROKER, MQTT_PORT, RMTP)
    
    environment = {}  # Define your environment structure
    run(agent, environment)

