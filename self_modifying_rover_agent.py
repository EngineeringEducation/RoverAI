"""
Self-Modifying Rover Agent

This module implements a self-modifying agent based on the OODA loop (Observe, Orient, Decide, Act)
for controlling a rover. The agent is capable of analyzing its environment through a video stream,
making decisions, and executing actions. It can also modify its own code during runtime through
the use of OpenAI's GPT models to generate new steps in its decision-making process.

The agent organizes its functions into different phases (observe, orient, decide, act) and can
dynamically load, execute, and modify these functions as needed. Each phase consists of one or
more steps that are executed in order of priority.
"""

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
    """
    A class representing a prioritized step in the agent's OODA loop.
    
    This class encapsulates a function (step) with its priority value, allowing steps
    to be sorted and executed in order of importance. Lower priority values indicate
    higher importance (will be executed first).
    
    Attributes:
        priority (int): The priority of the step (lower number = higher priority).
        func (Callable): The function to be executed as part of this step.
    """
    
    def __init__(self, priority: int, func: Callable):
        """
        Initialize a new PrioritizedStep.
        
        Args:
            priority (int): The priority of the step (lower number = higher priority).
            func (Callable): The function to be executed as part of this step.
        """
        self.priority = priority
        self.func = func

    def __lt__(self, other):
        """
        Compare this step with another based on priority.
        
        This allows steps to be sorted automatically by priority.
        
        Args:
            other (PrioritizedStep): Another step to compare with.
            
        Returns:
            bool: True if this step has higher priority (lower number) than the other.
        """
        return self.priority < other.priority

class Agent:
    """
    The main agent class that implements the self-modifying OODA loop.
    
    This class provides the core functionality for the agent, including communication
    with the MQTT broker, accessing the video stream, executing the OODA loop phases,
    and self-modification capabilities using OpenAI's GPT models.
    
    The agent can observe its environment through a video stream, interpret observations,
    make decisions, and execute actions. It can also modify its own code during runtime
    to adapt to new situations or fix issues.
    
    Attributes:
        agent_name (str): The name identifier for this agent.
        stream_url (str): The URL of the video stream to capture frames from.
        openai_client (OpenAI): Client for interacting with OpenAI API.
        client (mqtt.Client): MQTT client for communication.
        messages (list): List of message history for OpenAI interactions.
        most_recent_timestamp (float): Timestamp of the most recent frame capture.
        cap (cv2.VideoCapture): Video capture object for the stream.
        modification_queue (list): Queue of pending self-modification directives.
    """
    
    def __init__(self, agent_name, mqtt_broker, mqtt_port, stream_url):
        """
        Initialize a new Agent instance.
        
        Args:
            agent_name (str): The name identifier for this agent.
            mqtt_broker (str): The hostname or IP of the MQTT broker.
            mqtt_port (int): The port number of the MQTT broker.
            stream_url (str): The URL of the video stream to capture frames from.
        """
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
        """
        Generate a new step function for a specified phase of the OODA loop.
        
        This method uses OpenAI's GPT model to generate Python code for a new step
        function, then writes that code to a file in the appropriate phase directory.
        The generated code includes both the step function itself and a utility
        function that determines the priority of the step.
        
        Args:
            phase (str): The phase of the OODA loop to generate a step for ('observe', 'orient', 'decide', 'act').
            fn (str): The name of the function to generate.
            instruction (str, optional): Additional instructions for the GPT model. Defaults to "".
            utility_function (str, optional): Definition of the utility function. Defaults to "lambda context: 0".
            
        Returns:
            None
        """
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
            max_tokens=2000,
        )
        generated_code = response.choices[0].message.content
        
        with open(os.path.join(phase, f"{fn}.py"), "w") as f:
            f.write(f"from agent import PrioritizedStep\n\n{generated_code}")
        
        print(f"Generated/Modified step '{fn}' for phase '{phase}' with utility function")

    def load_steps(self, phase: str) -> List[PrioritizedStep]:
        """
        Load all step functions for a specified phase from the corresponding directory.
        
        This method dynamically imports all Python modules in the phase directory,
        each of which should contain a step_wrapper function that returns a PrioritizedStep
        object. If an error occurs during import, the agent will attempt to self-modify
        to fix the error.
        
        Args:
            phase (str): The phase to load steps for ('observe', 'orient', 'decide', 'act').
            
        Returns:
            List[PrioritizedStep]: A list of PrioritizedStep objects loaded from the phase directory.
        """
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
        """
        Execute all steps for a specified phase of the OODA loop.
        
        This method loads all steps for the specified phase, then executes each one
        in turn, passing in the agent itself, the environment, and any prior steps.
        
        Args:
            phase (str): The phase to execute ('observe', 'orient', 'decide', 'act').
            environment (dict): The current environment state.
            prior_steps (dict): Results from previous phases.
            
        Returns:
            List[PrioritizedStep]: The list of executed steps.
        """
        steps = self.load_steps(phase)
        for step in steps:
            step.func(self, environment, prior_steps)
        return steps

    def observe(self, environment: dict, prior_steps: dict) -> List[PrioritizedStep]:
        """
        Execute the observe phase of the OODA loop.
        
        This is the first phase of the OODA loop, responsible for gathering
        information about the environment, such as capturing frames from the
        video stream and analyzing them.
        
        Args:
            environment (dict): The current environment state.
            prior_steps (dict): Results from previous phases.
            
        Returns:
            List[PrioritizedStep]: The list of executed steps for this phase.
        """
        return self.execute_phase("observe", environment, prior_steps)

    def orient(self, environment: dict, prior_steps: dict) -> List[PrioritizedStep]:
        """
        Execute the orient phase of the OODA loop.
        
        This is the second phase of the OODA loop, responsible for interpreting
        the observations made in the observe phase and forming a mental model
        of the current situation.
        
        Args:
            environment (dict): The current environment state.
            prior_steps (dict): Results from previous phases.
            
        Returns:
            List[PrioritizedStep]: The list of executed steps for this phase.
        """
        return self.execute_phase("orient", environment, prior_steps)

    def decide(self, environment: dict, prior_steps: dict) -> List[PrioritizedStep]:
        """
        Execute the decide phase of the OODA loop.
        
        This is the third phase of the OODA loop, responsible for determining
        what action to take based on the current mental model of the situation.
        
        Args:
            environment (dict): The current environment state.
            prior_steps (dict): Results from previous phases.
            
        Returns:
            List[PrioritizedStep]: The list of executed steps for this phase.
        """
        return self.execute_phase("decide", environment, prior_steps)

    def act(self, environment: dict, prior_steps: dict) -> Tuple[bool, List[PrioritizedStep]]:
        """
        Execute the act phase of the OODA loop.
        
        This is the fourth and final phase of the OODA loop, responsible for
        executing the actions determined in the decide phase. After executing
        all actions, this method also processes any pending self-modification
        directives.
        
        Args:
            environment (dict): The current environment state.
            prior_steps (dict): Results from previous phases.
            
        Returns:
            Tuple[bool, List[PrioritizedStep]]: A tuple containing a boolean indicating
                whether to exit the loop, and the list of executed steps for this phase.
        """
        steps = self.execute_phase("act", environment, prior_steps)
        # Process any pending self-modification directives
        self.process_modification_queue()
        # You might want to implement a way to determine if the agent should exit
        return False, steps  # For now, always return False to continue the loop

    # Helper methods
    def capture_frames_from_stream(self):
        """
        Capture a frame from the video stream and save it to a file.
        
        This method captures a frame from the video stream, moves the previous frame
        to an archive directory, and saves the new frame as "rover.jpg". If any errors
        occur during capture, they are logged and the video capture is reinitialized.
        
        Returns:
            None
        """
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
        """
        Capture a frame from the video stream and return the filename.
        
        This is a wrapper method around capture_frames_from_stream that returns
        the filename of the captured frame.
        
        Returns:
            str: The filename of the captured frame.
        """
        output_filename = "rover.jpg"
        self.capture_frames_from_stream()
        return output_filename

    def run_agent_step(self, messages, max_tokens=300):
        """
        Run a single step using the OpenAI API to generate a response.
        
        This method sends a series of messages to the OpenAI API and returns
        the generated response content.
        
        Args:
            messages (list): A list of message dictionaries to send to the API.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 300.
            
        Returns:
            str: The generated response content.
        """
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def upload_images_to_openai(self, images, prompt):
        """
        Upload images to the OpenAI API for analysis.
        
        This method encodes an image as a base64 string and sends it to the
        OpenAI API along with a prompt for analysis.
        
        Args:
            images (list): A list of image filenames to upload. Currently only the first is used.
            prompt (str): The text prompt to send along with the image.
            
        Returns:
            str: The AI's generated response to the image and prompt.
        """
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
        """
        Encode an image file as a base64 string.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            str: Base64-encoded image data as a UTF-8 string.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

def run(agent: Agent, environment: dict):
    """
    Run the agent's OODA loop continuously until an exit condition is met.
    
    This function executes the four phases of the OODA loop (Observe, Orient, Decide, Act)
    in sequence, passing the results of each phase to the next. It continues running
    until the act phase signals that the loop should exit.
    
    Args:
        agent (Agent): The agent to run.
        environment (dict): The initial environment state.
        
    Returns:
        None
    """
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

