# File: act/execute_decision.py

from self_modifying_rover_agent import PrioritizedStep

def step_wrapper():
    def execute_decision(agent, environment, prior_steps):
        decision = environment.get('decision', 'hold')
        action_map = {
            "hold": "h", "go": "g", "forward": "f", "backward": "b", "left": "l", "right": "r",
            "superForward": "w", "superRight": "d", "superLeft": "a", "superBackward": "s",
            "aux": "x", "shoot": "z", "turnaround": "t"
        }
        
        action = action_map.get(decision, "h")
        agent.client.publish(f"action", action)
        print(f"Action: {action}")
        
        with open("actions.jsonl", "a") as f:
            f.write(f"{agent.most_recent_timestamp}|{action}\n")
        
        environment['last_action'] = action
        environment['last_decision_time'] = agent.most_recent_timestamp

    def utility_function(context):
        base_priority = 9
        action_importance = {
            "shoot": 5, "superForward": 4, "superBackward": 4, "superLeft": 4, "superRight": 4,
            "forward": 3, "backward": 3, "left": 3, "right": 3,
            "go": 2, "turnaround": 2, "aux": 1, "hold": 0
        }
        last_action = context.get('last_action', 'hold')
        return base_priority + action_importance.get(last_action, 0)

    return PrioritizedStep(utility_function, execute_decision)