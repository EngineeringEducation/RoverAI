import paho.mqtt.client as mqtt
import json
from openai import OpenAI
import os
import base64
import requests

MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
RMPT=os.getenv("RMPT")

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

import cv2
import time


class Agent:
    def __init__(self, agent_name, mqtt_broker, mqtt_port, stream_url):
        self.cap = cv2.VideoCapture(stream_url)
        self.openai_client = OpenAI()
        self.openai_client.api_key = os.getenv("OPENAI_API_KEY")
        self.agent_name = agent_name
        self.client = mqtt.Client()
        self.client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
        self.client.connect(mqtt_broker, mqtt_port, 60)
        # self.vision_model = VisionModel()  # Initialize your vision model here
        self.messages = [{"role": "system", "content": "Initialize agent"}]
    def capture_frames_from_stream(self):
        # Initialize video capture from the stream URL
        

        if not self.cap.isOpened():
            print("Error: Unable to open stream.")
            return
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Unable to fetch frame.")
                return

            # Save the frame as an image file
            cv2.imwrite(f"rover.jpg", frame)
            print(f"Captured frame")


        finally:
            # Release the video capture object
            self.cap.release()
            cv2.destroyAllWindows()

    def observe(self):
        ## here's where the "limbic system" of the agent will live
        ## which will dial up or down the number of times per minuite we capture an image and send it to openai
        ## we'll need to add some system prompting and rover logic here
        ## but also figure out which frame number we're on so we can either capture or not capture a frame
        # Capture a frame from the Camera stream
        frame = self.get_camera_frame()  
        # Analyze the frame using the vision model by sending the image to openai and asking it to interpret what it sees and give directions to the next agent to generate the next move
        
        observation = self.upload_images_to_openai([frame], "What do you see? Your response should help the next agent to generate the next move for the rover you are riding on.")

        return observation

    def orient(self, observation):
        ## #todo a lot of the business logic of deciding what the next move should be
        ## in relation to the goals of the agent will live here
        ## we'll need to add some system prompting and rover logic here
        # Process observation data using gpt4-o-mini
        self.messages.append({"role": "user", "content": "Please interpret what the vision model saw, and give a thoughtful description of the plan for what we can do next: " + observation})
        response = self.run_agent_step(self.messages)
        return response

    def decide(self, orientation):
        # Decide the action based on the orientation
        # #todo we'll need to add some system prompting and rover logic here
        self.messages.append({"role": "user", "content": orientation})
        self.messages.append({"role": "user", "content": """
            You have the following options for what to do next:
            "hold",
            "go",
            "forward",
            "backward",
            "left",
            "right",
            "superForward",
            "superRight",
            "superLeft",
            "superBackward",
            "aux",
            "shoot",
            "turnaround"
            Please reply with one of these strings only, exactly, with no other text.
                              """})
        decision = self.run_agent_step(self.messages)
        return decision

    def act(self, decision):
        # Extract the command from decision
        action = {
            "hold": "h",
            "go": "g",
            "forward": "f",
            "backward": "b",
            "left": "l",
            "right": "r",
            "superForward": "w",
            "superRight": "d",
            "superLeft": "a",
            "superBackward": "s",
            "aux": "x",
            "shoot": "z",
            "turnaround": "t"
        }

        # Send command to the rover
        self.client.publish(f"action", action.get(decision,"h"))
        return False

    def run(self):
        exit_loop = False
        while not exit_loop:
            environment_data = self.observe()
            orientation = self.orient(environment_data)
            decision = self.decide(orientation)
            exit_loop = self.act(decision)

    def get_camera_frame(self):

        ## read rover.png
        output_filename = "rover.jpg"
        self.capture_frames_from_stream()

        return output_filename

    def run_agent_step(self, messages, max_tokens=300):
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
        )
        print(response.choices[0].message)
        return response.choices[0].message.content
        

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
            response_json = response.json()
            ## return only the text
            return response_json["choices"][0]["message"]["content"]
            
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
# Example usage
agent = Agent("Rover1", MQTT_BROKER, MQTT_PORT, RMPT)
agent.run()
