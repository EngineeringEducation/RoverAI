import paho.mqtt.client as mqtt
# from your_vision_library import VisionModel  # Placeholder for actual vision model library
import json
from openai import OpenAI
import os
import base64
import requests

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class Agent:
    def __init__(self, agent_name, mqtt_broker, mqtt_port, topic_prefix):
        self.openai_client = OpenAI()
        self.openai_client.api_key = os.getenv("OPENAI_API_KEY")
        self.agent_name = agent_name
        self.client = mqtt.Client()
        self.client.connect(mqtt_broker, mqtt_port, 60)
        self.topic_prefix = topic_prefix
        # self.vision_model = VisionModel()  # Initialize your vision model here
        self.messages = [{"role": "system", "content": "Initialize agent"}]

    def observe(self):
        # Capture a frame from the Twitch stream
        frame = self.get_twitch_frame()  # Placeholder function
        # Analyze the frame using the vision model by sending the image to openai and asking it to interpret what it sees and give directions to the next agent to generate the next move
        
        observation = self.upload_images_to_openai([frame], "What do you see? Your response should help the next agent to generate the next move for the rover you are riding on.")

        return json.dumps(observation)

    def orient(self, observation):
        # Process observation data using GPT-4
        self.messages.append({"role": "user", "content": observation})
        response = self.run_agent_step(self.messages)
        return response

    def decide(self, orientation):
        # Decide the action based on the orientation
        self.messages.append({"role": "user", "content": orientation})
        decision = self.run_agent_step(self.messages)
        return decision

    def act(self, decision):
        # Extract the command from decision
        action = json.loads(decision)['action']
        # Send command to the rover
        self.client.publish(f"{self.topic_prefix}/{action['command']}", action['value'])
        return action.get('should_exit', False)

    def run(self):
        exit_loop = False
        while not exit_loop:
            environment_data = self.observe()
            orientation = self.orient(environment_data)
            decision = self.decide(orientation)
            exit_loop = self.act(decision)

    def get_twitch_frame(self):
        # Interface with Twitch API or capture software to get current video frame
        pass

    def run_agent_step(self, messages, max_tokens=300):
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message
        

    def upload_images_to_openai(self,images, prompt):
        for image in images:
            # Getting the base64 string
            base64_image = encode_image(image)

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
                    {
                    "type": "text",
                    "text": prompt
                    },
                    {
                    "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
                }
            ],
            "max_tokens": 300
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            

# Example usage
agent = Agent("Rover1", "localhost", 1883, "rover/commands")
agent.run()

