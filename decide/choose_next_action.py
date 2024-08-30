# File: decide/choose_next_action.py

from self_modifying_rover_agent import PrioritizedStep

def step_wrapper():
    def choose_next_action(agent, environment, prior_steps):
        orientation = environment.get('orientation', '')
        agent.messages.append({"role": "user", "content": orientation})
        agent.messages.append({"role": "user", "content": """
            You have the following options for what to do next:
            "hold", "go", "forward", "backward", "left", "right", "superForward", "superRight", "superLeft", "superBackward", "aux", "shoot", "turnaround"
            Please reply with one of these strings only, exactly, with no other text.
        """})
        decision = agent.run_agent_step(agent.messages)
        print(f"Decision: {decision}")
        
        with open("decisions.jsonl", "a") as f:
            f.write(f"{agent.most_recent_timestamp}|{decision}\n")
        
        environment['decision'] = decision

        # Self-modification example
        if "obstacle" in orientation.lower():
            agent.issue_self_modification_directive(
                "observe", 
                "detect_obstacles",
                "Add a new function to handle this specific type of obstacle."
            )

    def utility_function(context):
        base_priority = 7
        if context.get('urgent_action_needed', False):
            base_priority += 5
        if 'obstacle' in context.get('last_observation', ''):
            base_priority += 3
        time_since_last_decision = context.get('timestamp', 0) - context.get('last_decision_time', 0)
        base_priority += min(time_since_last_decision, 5)  # Cap the time factor at 5
        return base_priority

    return PrioritizedStep(utility_function, choose_next_action)