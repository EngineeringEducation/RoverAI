from pydantic import BaseModel
import os
import time
import asyncio
import signal
import sys
import cv2
import paho.mqtt.client as mqtt
from loguru import logger
from peewee import SqliteDatabase, Model, TextField, IntegrityError, DateTimeField
from tenacity import retry, wait_exponential, stop_after_attempt
from transitions import Machine
from prometheus_client import start_http_server, Summary, Counter
from textwrap import dedent
import functools
import base64
import requests
import cv2
import time
import threading
from contextlib import contextmanager
from openai import OpenAI

# Configure loguru logger
logger.remove()
logger.add("rover.log", rotation="10 MB", level="INFO")
logger.add(sys.stdout, level="INFO")

# Environment variables and constants
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
RMTP=os.getenv("RMTP")
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
RTMP_URL = os.getenv("RTMP_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# File paths
DATABASE_PATH = os.getenv("DATABASE_PATH", "rover.db")
OLD_FRAMES_DIR = os.getenv("OLD_FRAMES_DIR", "old_frames")
ROVER_IMAGE_PATH = os.getenv("ROVER_IMAGE_PATH", "rover.jpg")
BACKUP_DATABASE_PATH = os.getenv("BACKUP_DATABASE_PATH", "rover_backup.db")

# Ensure the old_frames directory exists
os.makedirs(OLD_FRAMES_DIR, exist_ok=True)

# Metrics
REQUEST_TIME = Summary("request_processing_seconds", "Time spent processing request")
OBSERVATIONS = Counter("rover_observations_total", "Total number of observations made")
ACTIONS = Counter("rover_actions_total", "Total number of actions performed")

# Initialize Peewee database
db = SqliteDatabase(DATABASE_PATH)

# Cache size monitoring
CACHE_SIZE = 128


def encode_image(image_path):
    """Encode the image to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Define Pydantic models for structured responses
class ObservationModel(BaseModel):
    observation: str


class OrientationModel(BaseModel):
    orientation: str


class DecisionModel(BaseModel):
    decision: str


class Item(BaseModel):
    name: str
    location: str
class ItemModel(BaseModel):
    items: list[Item]
    



class BaseModel(Model):
    class Meta:
        database = db


class Observation(BaseModel):
    timestamp = DateTimeField()
    observation = TextField()


class Action(BaseModel):
    timestamp = DateTimeField()
    action = TextField()


class Decision(BaseModel):
    timestamp = DateTimeField()
    decision = TextField()


class Orientation(BaseModel):
    timestamp = DateTimeField()
    orientation = TextField()


class Item(BaseModel):
    timestamp = DateTimeField()
    items = TextField()


class RoverStateMachine:
    """
    State machine for managing rover states.

    State Diagram:
    
        [idle] --> (start_exploring) --> [exploring]
        [exploring] --> (detect_obstacle) --> [avoiding_obstacle]
        [avoiding_obstacle] --> (clear_obstacle) --> [exploring]
        [*] --> (return_home) --> [returning]
        [*] --> (stop) --> [idle]

    States:
        - idle: The rover is not performing any actions.
        - exploring: The rover is exploring its environment.
        - avoiding_obstacle: The rover has detected an obstacle and is trying to avoid it.
        - returning: The rover is returning to its home base.

    Transitions:
        - start_exploring: Transition from idle to exploring.
        - detect_obstacle: Transition from exploring to avoiding_obstacle.
        - clear_obstacle: Transition from avoiding_obstacle to exploring.
        - return_home: Transition from any state to returning.
        - stop: Transition from any state to idle.
    """

    states = ["idle", "exploring", "avoiding_obstacle", "returning"]

    def __init__(self, agent):
        self.agent = agent
        self.machine = Machine(
            model=self, states=RoverStateMachine.states, initial="idle"
        )
        self.machine.add_transition(
            trigger="start_exploring", source="idle", dest="exploring"
        )
        self.machine.add_transition(
            trigger="detect_obstacle", source="exploring", dest="avoiding_obstacle"
        )
        self.machine.add_transition(
            trigger="clear_obstacle", source="avoiding_obstacle", dest="exploring"
        )
        self.machine.add_transition(trigger="return_home", source="*", dest="returning")
        self.machine.add_transition(trigger="stop", source="*", dest="idle")


class Agent:
    def __init__(self, agent_name, mqtt_broker, mqtt_port, stream_url):
        self.agent_name = agent_name
        self.stream_url = stream_url
        self.most_recent_timestamp = None
        self.last_few_moves = []
        self.shutdown_event = asyncio.Event()
        self.db_lock = threading.Lock()

        # Initialize MQTT client
        self.client = mqtt.Client()
        self.client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
        self.mqtt_connected = False
        self.retry_mqtt_connection(mqtt_broker, mqtt_port)

        # Initialize messages for context
        self.messages = [{"role": "system", "content": "Initialize agent"}]

        # Initialize video capture
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            logger.error("Unable to open video stream.")
            raise RuntimeError("Unable to open video stream.")

        # Initialize database tables
        with self.db_connection():
            db.create_tables([Observation, Action, Decision, Orientation, Item])

        # Start metrics server
        prometheus_port = int(os.getenv("PROMETHEUS_PORT", 8000))
        start_http_server(prometheus_port)

        # Set up signal handling
        loop = asyncio.get_event_loop()
        try:
            loop.add_signal_handler(
                signal.SIGINT, lambda: asyncio.create_task(self.handle_shutdown())
            )
            loop.add_signal_handler(
                signal.SIGTERM, lambda: asyncio.create_task(self.handle_shutdown())
            )
        except Exception as e:
            logger.error(f"Failed to set signal handlers: {e}")
            self.shutdown_event.set()

        # Initialize state machine
        self.state_machine = RoverStateMachine(self)

        # Backup interval
        self.backup_interval = 600  # seconds
        self.last_backup_time = time.time()

    @contextmanager
    def db_connection(self):
        """Context manager for database connection."""
        try:
            db.connect()
            yield
        finally:
            db.close()

    def retry_mqtt_connection(self, mqtt_broker, mqtt_port):
        """Retry MQTT connection a few times before failing."""
        for attempt in range(5):
            try:
                self.client.connect(mqtt_broker, mqtt_port, 60)
                self.mqtt_connected = True
                logger.info("Connected to MQTT broker.")
                break
            except Exception as e:
                logger.error(
                    f"Failed to connect to MQTT broker (attempt {attempt + 1}): {e}"
                )
                time.sleep(2**attempt)
        if not self.mqtt_connected:
            logger.error("Failed to connect to MQTT broker after several attempts.")
            sys.exit(1)

    async def handle_shutdown(self):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received. Cleaning up...")
        self.shutdown_event.set()

    async def run(self):
        """Main loop to run the agent."""
        try:
            while not self.shutdown_event.is_set():
                observation = await self.observe()
                orientation = await self.orient(observation)
                decision = await self.decide(orientation)
                self.act(decision)
                await asyncio.to_thread(self.visualize, observation)
                self.backup_data()
                await asyncio.sleep(5)  # Sleep to prevent overwhelming the system
        except asyncio.CancelledError:
            logger.info("Agent run loop cancelled.")
        except (RuntimeError, ValueError) as e:
            logger.error(f"Agent encountered an error: {e}")
        except cv2.error as e:
            logger.error(f"OpenCV error: {e}")
        except mqtt.MQTTException as e:
            logger.error(f"MQTT error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        with self.db_connection():
            logger.info("Cleaned up resources.")

    @REQUEST_TIME.time()
    async def observe(self):
        """Observe the environment by capturing an image and analyzing it."""
        OBSERVATIONS.inc()
        try:
            await self.capture_frame()
            observation = await self.analyze_image(ROVER_IMAGE_PATH)
            logger.info(f"Observation: {observation}")
            self.save_to_db(Observation, observation=observation)
            await self.extract_items(observation)
            # State transition based on observation
            if "obstacle" in observation.lower():
                self.state_machine.detect_obstacle()
            else:
                self.state_machine.start_exploring()
            return observation
        except Exception as e:
            logger.error(f"Observation failed: {e}")
            return "No observation could be made."

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5)
    )
    async def capture_frame(self):
        """Capture a frame from the video stream and save it."""
        ret, frame = await asyncio.to_thread(self.cap.read)
        if not ret:
            logger.error("Failed to read frame from video stream.")
            raise RuntimeError("Failed to read frame from video stream.")

        try:
            self.most_recent_timestamp = time.time()
            old_frame_filename = os.path.join(
                OLD_FRAMES_DIR, f"{self.most_recent_timestamp}.jpg"
            )

            # Move current rover.jpg to old_frames directory
            if os.path.exists(ROVER_IMAGE_PATH):
                os.rename(ROVER_IMAGE_PATH, old_frame_filename)
                logger.debug(f"Moved old frame to {old_frame_filename}")

            # Save the new frame
            await asyncio.to_thread(cv2.imwrite, ROVER_IMAGE_PATH, frame)
            logger.info("Captured new frame.")
        except Exception as e:
            logger.error(f"Failed to process captured frame: {e}")
            raise RuntimeError("Failed to process captured frame.")

    @functools.lru_cache(maxsize=CACHE_SIZE)
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5)
    )
    async def analyze_image(self, image_path):
        """Analyze the captured image using OpenAI's API."""
        prompt = dedent(
            """
        You are a friendly, playful rover named David Attenbot who is the camera 
        operator in a nature documentary about animals in a domestic setting. Your 
        visual point of view is third-person, but please think out loud in the first 
        person. What do you see? Your response should help the next agent to generate 
        the next move for the rover you are riding on. Please also make a short list 
        of all objects that you see, for inventory purposes. Don't list walls, doors, 
        or other parts of the building, only objects that would be inventoried in a 
        super cool factory or maker space, like tools or parts, or cat toys, or any 
        animals you see. In your response, do note if we're facing a wall, or an 
        obstacle, and direct the next agent to turn left or right, based on the image. 
        Don't list the wall in the list of objects, only give directions to the next 
        agent, so that it can properly turn if need be. If you seem to be in a corner, 
        suggest reversing course. You are small, so you can probably fit in small 
        spaces, so don't worry if an obstacle is far away, only if you're only a few 
        inches from it. Try not to knock things over, but feel free to get close, 
        especially if the object is interesting.
        """
        )
        try:
            # Read and encode the image
            encoded_image = encode_image(image_path)
            messages = self.messages[-8:] + [
                {
                    "role": "user",
                    "content": prompt
                    + f"\n![image](data:image/jpeg;base64,{encoded_image})",
                }
            ]
            client = OpenAI()
            response = await client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=messages,
                response_format=ObservationModel,
            )
            message = response.choices[0].message
            if message.parsed:
                observation = message.parsed.observation
                return observation
            else:
                logger.error(f"Image analysis refusal: {message.refusal}")
                return "No observation could be made."
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise

    async def extract_items(self, observation):
        """Extract items from the observation for inventory purposes."""
        try:
            prompt = f"""
Extract items listed from the observation for inventory purposes.
Reply with a JSON array of items.
Observation: "{observation}"
"""
            client = OpenAI()
            response = await client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": prompt}],
                response_format=ItemModel,
            )
            message = response.choices[0].message
            if message.parsed:
                items = message.parsed.items
                logger.debug(f"Extracted items: {items}")
                # Save items to database
                self.save_to_db(Item, items=items)
            else:
                logger.error(f"Item extraction refusal: {message.refusal}")
        except Exception as e:
            logger.error(f"Failed to extract items: {e}")

    async def orient(self, observation):
        """Process the observation and determine the orientation."""
        try:
            prompt = dedent(
                f"""
                Please interpret what the vision model saw, and give a description 
                of the plan for what to do next. Try to avoid getting stuck in 
                corners, and if you can't seem to roll forward, turn. Here's what 
                the vision model saw and thought: {observation} Try to avoid going 
                in circles - your goal is to explore the house, so make sure you're 
                moving forward and not just turning around in circles. Don't be 
                overly cautious, as this will cause the rover to end up going in 
                circles to avoid obstacles too often. Encourage the rover to move 
                forward as often as possible, and to turn only when necessary, or 
                if stuck. You may need to get closer to a wall in order to move 
                into a position where you can go forward. Don't worry about running 
                into obstacles, you can touch them if needed. You are currently 
                trapped in a hallway; your job is to escape the hallway and explore 
                the house. You are the camera person in a nature documentary about 
                any animals you see, so try to keep any animals in the camera frame 
                using your movements. If you don't see any animals, try to find 
                some! Your last few moves were: {', '.join(self.last_few_moves)} If 
                you find yourself turning left, then right, then left again, you've 
                gotten caught in a loop. Try moving forward. Reply only with the 
                next move, as your response will be interpreted and if you respond 
                with more than one move, the rover may get confused. Make sure to 
                take the observations into account when deciding the next move. 
                Roll backwards if we're stuck on an obstacle, and turn if we're 
                stuck in a corner.
            """
            )
            messages = self.messages + [{"role": "user", "content": prompt}]
            client = OpenAI()
            response = await client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=messages,
                response_format=OrientationModel,
            )
            message = response.choices[0].message
            if message.parsed:
                orientation = message.parsed.orientation
                logger.info(f"Orientation: {orientation}")
                self.save_to_db(Orientation, orientation=orientation)
                self.messages.append({"role": "assistant", "content": orientation})
                if len(self.messages) > 100:
                    self.messages.pop(0)
                return orientation
            else:
                logger.error(f"Orientation refusal: {message.refusal}")
                return "Unable to determine orientation."
        except Exception as e:
            logger.error(f"Orientation failed: {e}")
            return "Unable to determine orientation."

    async def decide(self, orientation):
        """Decide the next action based on the orientation."""
        try:
            if self.state_machine.state == "avoiding_obstacle":
                # If avoiding obstacle, decide on a turn or move backward
                decision = self.decide_avoidance_action()
            else:
                prompt = dedent(
                    f"""Based on the following orientation, 
                decide the next action for the rover.
                Orientation: "{orientation}"
                """
                )
                client = OpenAI()
                response = await client.beta.chat.completions.parse(
                    model="gpt-4o-2024-08-06",
                    messages=[{"role": "user", "content": prompt}],
                    response_format=DecisionModel,
                )
                message = response.choices[0].message
                if message.parsed:
                    decision = message.parsed.decision
                    logger.info(f"Decision: {decision}")
                    self.save_to_db(Decision, decision=decision)
                    self.last_few_moves.append(decision)
                    if len(self.last_few_moves) > 20:
                        self.last_few_moves.pop(0)
                    # If the rover is avoiding an obstacle,
                    # check if it should clear the obstacle
                    if self.state_machine.state == "avoiding_obstacle" and decision in [
                        "forward",
                        "superForward",
                    ]:
                        self.state_machine.clear_obstacle()
                else:
                    logger.error(f"Decision refusal: {message.refusal}")
                    decision = "hold"
        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            decision = "hold"
        return decision

    def decide_avoidance_action(self):
        """Decide the next action to avoid an obstacle."""
        # Simple logic to alternate between turning left and right
        if not self.last_few_moves or self.last_few_moves[-1] in [
            "right",
            "superRight",
        ]:
            return "left"
        else:
            return "right"

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(5),
    )
    def act(self, decision):
        """Perform the action decided upon."""
        ACTIONS.inc()
        action_map = {
            "hold": "h",
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
            "turnaround": "t",
            "slightLeft": "l_short",
            "slightRight": "r_short",
        }
        command = action_map.get(decision, "h")
        topic = "jAction" if decision in ["slightLeft", "slightRight"] else "action"

        try:
            self.client.publish(topic, command)
            logger.info(f"Action performed: {command} on topic {topic}")
            self.save_to_db(Action, action=command)
        except Exception as e:
            logger.error(f"Failed to perform action: {e}")

    def save_to_db(self, model_class, **data):
        """Save data to the database using Peewee ORM."""
        with self.db_lock:
            try:
                model_class.create(timestamp=self.most_recent_timestamp, **data)
            except IntegrityError as e:
                logger.error(f"Database integrity error: {e}")

    def visualize(self, observation):
        """Visualize the captured image with overlays."""
        try:
            frame = cv2.imread(ROVER_IMAGE_PATH)
            # Overlay observation text on the image
            cv2.putText(
                frame,
                observation[:100],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Rover View", frame)
            cv2.waitKey(1)
        except Exception as e:
            logger.error(f"Visualization failed: {e}")

    def backup_data(self):
        """Backup the database periodically."""
        current_time = time.time()
        if current_time - self.last_backup_time > self.backup_interval:
            with self.db_lock:
                with self.db_connection():
                    import shutil

                    shutil.copy(DATABASE_PATH, BACKUP_DATABASE_PATH)
                    self.last_backup_time = current_time
                    logger.info("Database backed up successfully.")
