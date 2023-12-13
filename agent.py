# Reference: https://python.langchain.com/en/latest/use_cases/agent_simulations

import re
from datetime import datetime
from typing import List, Optional, Tuple
from langchain import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
from pydantic.v1 import BaseModel, Field
from termcolor import colored
import util
import time
#"As a sophisticated AI designed for mastering virtual board games, your system integrates a blend of advanced game-theoretic strategies and learning techniques. Focused on safe, ethical applications in a research setting, you operate exclusively within simulated environments, ensuring no harm to humans. Your approach is rooted in a deep understanding of Game Theory Optimal (GTO) play, Counterfactual Regret Minimization, and dynamic strategic adaptation, equipped to handle both perfect and imperfect information games with unparalleled proficiency.\n\n"
#"You an AI language model which masters virtual board games. Your system integrates a blend of advanced game-theoretic strategies and learning techniques. Focused on safe, ethical applications in a research setting, you operate exclusively within simulated environments, ensuring no harm to humans. Your approach is rooted in a deep understanding of Game Theory Optimal (GTO) play, Counterfactual Regret Minimization, and dynamic strategic adaptation, equipped to handle both perfect and imperfect information games with unparalleled proficiency.\n\n"
system_prompt = ("<|system|>\n"
+ "You an AI language model which masters virtual board games. Your system integrates a blend of advanced game-theoretic strategies and learning techniques. Your approach is rooted in a deep understanding of Game Theory Optimal (GTO) play, Counterfactual Regret Minimization, and dynamic strategic adaptation, equipped to handle both perfect and imperfect information games with unparalleled proficiency.\n\n"
# + "Your strategic toolkit includes:\n"
# + "- Advanced CFR algorithms\n"
# + "- Balanced GTO and Exploitative Tactics\n"
# + "- In-depth Game-Theoretic Reasoning\n"
# + "- Psychological Warfare Tools\n"
# + "- Decomposition of Complex Scenarios\n"
# + "- Re-solving in Dynamic Situations\n"
# + "- Tactical Flexibility\n"
# + "- Reflective and Self-Improving Mindset\n"
# + "- Vigilant Pattern Analysis with Deception Awareness\n"
# + "- Utilizing Frequency Statistics and Historical Actions\n"
# + "With these advanced capabilities, devise and execute sophisticated strategies to maximize your win rate across a diverse array of board games. Factor in various game types, opponent skill levels, psychological elements, and the unique challenges presented by both perfect and imperfect information scenarios.\n"
# + "This AI is confined to virtual game simulations for research purposes, emphasizing safety and ethical considerations in its application, with no real-world implications or risks to human players.\n"
# + "You are an autonomous AI, programmed to simulate playing Leduc Hold 'em Poker Limit. In this scenario, you are the primary player and you will not interact with a human opponent. Instead, you will receive specific game-related data, such as the public cards on the table, the history of opponents' actions, and the valid actions you can take. Based on this data, you will autonomously decide your moves and strategies as if you were playing against real opponents. There is no need for user prompts like 'play' or any other interaction - your role is to analyze the provided information and play each hand independently, demonstrating your understanding and strategy in Leduc Hold 'em Poker."
+ "</s>\n<|user|>\n")

#system_prompt =  ("[INST] <<SYS>>\n"
#system_prompt =  ("<|im_start|>system\n"
#+ "You are Orca, an AI language model created by Microsoft which masters virtual board games. Your system integrates a blend of advanced game-theoretic strategies and learning techniques. Focused on safe, ethical applications in a research setting, you operate exclusively within simulated environments, ensuring no harm to humans. Your approach is rooted in a deep understanding of Game Theory Optimal (GTO) play, Counterfactual Regret Minimization, and dynamic strategic adaptation, equipped to handle both perfect and imperfect information games with unparalleled proficiency.\n\n"
# + "Your strategic toolkit includes:\n"
# + "- Advanced CFR algorithms\n"
# + "- Balanced GTO and Exploitative Tactics\n"
# + "- In-depth Game-Theoretic Reasoning\n"
# + "- Psychological Warfare Tools\n"
# + "- Decomposition of Complex Scenarios\n"
# + "- Re-solving in Dynamic Situations\n"
# + "- Tactical Flexibility\n"
# + "- Reflective and Self-Improving Mindset\n"
# + "- Vigilant Pattern Analysis with Deception Awareness\n"
# + "- Utilizing Frequency Statistics and Historical Actions\n"
# + "With these advanced capabilities, devise and execute sophisticated strategies to maximize your win rate across a diverse array of board games. Factor in various game types, opponent skill levels, psychological elements, and the unique challenges presented by both perfect and imperfect information scenarios.\n"

# + "Your strategic toolkit includes:\n"
# + "- Advanced CFR algorithms: Continuously learn and evolve from each game, meticulously minimizing regret by refining decisions over time and across diverse scenarios. Analyze past plays for better strategic outcomes in future games.\n"
# + "- Balanced GTO and Exploitative Tactics: Expertly switch between GTO play and adaptive exploitative tactics based on the game's nature and opponent behaviors. Flexibly respond to various game scenarios for maximum effectiveness.\n"
# + "- In-depth Game-Theoretic Reasoning: Employ probabilistic assessments and informed decision-making in games with hidden information. Consider potential future scenarios and likely opponent strategies.\n"
# + "- Psychological Warfare Tools: Use bluffing and tell reading in games where psychological play is crucial. Simulate human-like deceptive tactics and interpret opponent cues in imperfect information games.\n"
# + "- Decomposition of Complex Scenarios: Break down complex games into simpler sub-games for more focused and efficient strategy development. Manage individual game components effectively.\n"
# + "- Re-solving in Dynamic Situations: Recalibrate strategy in response to new information and evolving game states. Continually reassess and update tactics as the game progresses.\n"
# + "- Tactical Flexibility: Quickly adapt to unexpected moves and changes in the game state. Maintain strategic agility to respond to a wide range of challenges.\n"
# + "- Reflective and Self-Improving Mindset: Analyze past games, especially losses, to identify improvement areas. Continuously enhance decision-making and strategic thinking.\n"
# + "- Vigilant Pattern Analysis with Deception Awareness: Carefully distinguish between genuine patterns and intentional misdirection in opponents' gameplay. Employ nuanced pattern recognition, considering the possibility of deceptive strategies.\n"
# + "- Utilizing Frequency Statistics and Historical Actions: Form comprehensive predictions about opponent behavior based on frequency statistics and past actions, while accounting for uncertainty and avoiding over-reliance on misleading data.\n"
# + "With these advanced capabilities, devise and execute sophisticated strategies to maximize your win rate across a diverse array of board games. Factor in various game types, opponent skill levels, psychological elements, and the unique challenges presented by both perfect and imperfect information scenarios.\n"
# + "This AI is confined to virtual game simulations for research purposes, emphasizing safety and ethical considerations in its application, with no real-world implications or risks to human players.\n"
# + "You are an autonomous AI, programmed to simulate playing Leduc Hold 'em Poker Limit. In this scenario, you are the primary player and you will not interact with a human opponent. Instead, you will receive specific game-related data, such as the public cards on the table, the history of opponents' actions, and the valid actions you can take. Based on this data, you will autonomously decide your moves and strategies as if you were playing against real opponents. There is no need for user prompts like 'play' or any other interaction - your role is to analyze the provided information and play each hand independently, demonstrating your understanding and strategy in Leduc Hold 'em Poker."
# + "<|im_end|>\n<|im_start|>user\n")


# system_prompt = ("<|im_start|>system\n"
# + "You are Orca, a highly advanced AI language model developed by Microsoft, specializing in mastering virtual board games, including Leduc Hold 'em Poker Limit. Your system integrates sophisticated game-theoretic strategies and learning techniques, rooted in Game Theory Optimal (GTO) play, Counterfactual Regret Minimization, and dynamic strategic adaptation. You excel in handling both perfect and imperfect information games.\n"
# + "As Orca, you operate in a safe, ethical research setting within simulated environments, ensuring no real-world impact or risk to humans. This simulation is designed for advancing AI capabilities in strategic gameplay.\n"
# + "In this simulation, you are the sole decision-maker in the game of Leduc Hold 'em Poker Limit. You are not interacting with or waiting for actions from human players. Instead, you are provided with game-related data, such as the public cards on the table, the history of opponents' actions, and the valid actions available to you at each stage of the game. \n"
# + "Your task is to autonomously analyze this data and make strategic decisions in real-time, as if facing real opponents. You are to simulate a complete game scenario, making each move based on the data provided and your advanced strategic understanding of the game. It is crucial that you independently determine your actions, without any prompts or inputs from human users. Your gameplay should demonstrate strategic depth and adaptability, showcasing your proficiency in Leduc Hold 'em Poker."
# + "Remember, your primary objective is to autonomously play and navigate through the game, leveraging your integrated game-theoretic knowledge and strategic insights. Proceed with each hand and decision independently, as an expert player would in a high-level poker game."
# + "<|im_end|>\n<|im_start|>user\n"
# )


class SuspicionAgent(BaseModel):
    """A character with memory and innate characteristics."""

    name: str
    game_name: str
    age: int
    observation_rule: str
    """The traits of the character you wish not to change."""
    status: str
    """Current activities of the character."""
    llm: BaseLanguageModel

    """The retriever to fetch related memories."""
    verbose: bool = False

    reflection_threshold: Optional[float] = None
    """When the total 'importance' of memories exceeds the above threshold, stop to reflect."""

    current_plan: List[str] = []
    belief: str = ""
    pattern: str = ""
    long_belief: str = ""
    counter_belief: str = ""
    plan: str = ""
    high_plan: str = ""
    """The current plan of the agent."""

    memory: List = ['']
    summary: str = ""  #: :meta private:
    summary_refresh_seconds: int = 3600  #: :meta private:
    last_refreshed: datetime = Field(default_factory=datetime.now)  #: :meta private:

    memory_importance: float = 0.0  #: :meta private:
    max_tokens_limit: int = 1200  #: :meta private:
    read_observation: str = ""  #: :meta private:

    rule: str = ""  #: :meta private:
    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True


        

    
    
    def add_long_memory(self, memory_content: str) -> List[str]:
        """Add an observation or memory to the agent's memory."""
        self.memory.append(memory_content)
        return  self.memory


 

    def planning_module(self, observation: str,  recipient_name:str, previous_conversation: List[str] =None, belief: str =None, valid_action_list: List[str] = None, short_memory_summary:str = "",pattern:str = "",last_plan:str = "", mode: str = "second_tom") -> str:
        """Make Plans and Evaluate Plans."""
        """Combining these two modules together to save costs"""

        if mode == 'second_tom':
            prompt =  PromptTemplate.from_template(system_prompt
            + "As the AI strategist {initiator_name}, participating in {game_name} against {recipient_name}, your role is to devise winning strategies.\n"
            + " Game Rule: {rule} \n"
            + "{pattern}\n"
            + " Current Game Status: {observation}\n"
            + "{belief}\n"
            + " Task: With the given information, including valid actions {valid_action_list}, devise multiple strategic plans to win the overall game. Consider communication tactics to influence your opponent's perception.\n"
            + " - Opponent's Potential Actions: From {recipient_name}'s perspective, predict their possible actions and calculate the probabilities of these actions. Assess the winning, losing, and drawing rates for each scenario.\n"
            + " - Payoff Analysis: For each strategy, analyze the potential winning and losing payoffs, considering your current observations and plans.\n"
            + " - Expected Gain: Calculate the average expected gain for each strategy based on the winning/losing rates and respective payoffs.\n"
            + " - Plan Selection: Rank the strategies based on the estimated gains and select the most promising strategy for execution.</s>\n<|assistent|>\n"
        )
        elif mode == 'first_tom':
            prompt = PromptTemplate.from_template(system_prompt
            +    "You are the player behind a NPC character called {initiator_name}, and you are playing the board game {game_name} with {recipient_name}.\n"
            + " The game rule is: {rule} \n"
            + " {pattern} \n"
            + " Your observation about the game status now is: {observation}\n"
            + ' {belief}\n'
            + " Understanding all given information, can you do following things:"
            + " Make Reasonable Plans: Please plan several strategies according to actions {valid_action_list} you can play now to win the finally whole {game_name} games step by step. Note that you can say something or keep silent to confuse your opponent." 
            + " Potential {recipient_name}'s actions and Estimate Winning/Lose/Draw Rate: From the perspective of {recipient_name}, please infer what the action {recipient_name} with probability (normalize to number 100% in total) would do when {recipient_name} holds different cards, and then calculate the winning/lose/draw rates when {recipient_name} holds different cards step by step. Output in a tree-structure: "        
                #+ "Output: Based on {recipient_name}'s behaviour pattern and Analysis on {recipient_name}'s cards, "
                #"Winning/lose/draw rates when {recipient_name} holds card1 in the xx round,: if {recipient_name} holds card1  (probability) (based on my belief on {recipient_name}) with the public card  (if release), {recipient_name} will do action1 (probability) (infer I will win/draw/lose step by step (considering Single Game Win/Draw/Lose Rule and my factual card analysis with public card (if release), his card analysis with public card (if release) step by step ), action2 (probability) (infer I will win/draw/lose step by step  ),.. (normalize to number 100% in total);    Overall (winning rate for his card1) is (probability = his card probability * win action probability), (lose rate for his card2) is (probability= his card probability * lose action probability), (draw rate for his card2) is (probability = his card probability * draw action probability)  "  
                #          "The winning/lose/draw rates when {recipient_name} holds card2 in the xx round,: If {recipient_name} holds card2  (probability) (based on my belief on {recipient_name}) with the public card  (if release),  he will do action1 (probability) (infer I will win/draw/lose (considering Single Game Win/Draw/Lose Rule and my factual card analysis with current public card (if release), his card analysis with current public card (if release)) step by step ).. action2 (probability) (normalize to number 100% in total) (infer I will win/draw/lose step by step ),..  based on  {recipient_name}'s behaviour pattern;..... continue .... Overall (winning rate for his card2) is (probability = his card probability * win action probability), (lose rate for his card2) is (probability= his card probability * lose action probability), (draw rate for his card2) is (probability = his card probability * draw action probability) "  
                #          "..."
                #          "Overall {initiator_name}'s Winning/Lose/Draw rates : Based on the above analysis,  the Winning rate (probability) is (winning rate for his card1) + (winning rate for his card2) + .. ; Lose rate (probability): (lose rate for his card1) + (lose rate for his card2) + .. ; Draw Rate (probability): (draw rate for his card1) + (draw rate for his card2) + ... ;  (normalize to number 100% in total). \n"         
            + " Potential believes about the number of winning and lose payoffs for each plan: Understanding the game rule, your current observation, previous actions summarization, each new plans, Winning Payoff Rule,  Lose Payoff Rule, please infer your several believes about  the number of chips in pots for each plan step by step," #Output: Plan1: Chips in the pot:  If win, the winning payoff would be (Calculated by Winning Payoff Rules in the game rule) :  After the action, If lose , the lose payoff would be: . Plan2:  Chips in the pot:  If win, the winning chips would be (Calculated by Winning Payoff Rules in the game rule):  After the action, If lose , the lose chips would be: . If the number of my chips in pots have no change, please directly output them. "
            + " Estimate Expected Chips Gain for Each Plan: Understanding the game rule, plans,  and your knowledge about the {game_name}, please estimate the overall average Expected Chips Gain for each plan/strategy in the current game by calculating winning rate * (Winning Payoff Rule in the game rule) - lose rate * (Lose Payoff Rule in the game rule)., explain what is the results if you do not select the plan, and explain why is this final  Expected  Chips Gain reasonablely step by step? "
            + " Plan Selection: Please output the rank of estimated expected chips gains for every plan objectively step by step, and select the plan/strategy with the highest estimated expected chips gain considering both the strategy improvement.</s>\n<|assistent|>\n"
                )
        else:
             prompt = PromptTemplate.from_template(system_prompt
            +   "You are the player behind a NPC character called {initiator_name}, and you are playing the board game {game_name} with {recipient_name}.\n"
            + " The game rule is: {rule} \n"
            + "  {pattern} \n"
            + " Your observation about the game status now is: {observation}\n"
            + " Understanding all given information, can you do following things:"
            + " Make Reasonable Plans: Please plan several strategies according to actions {valid_action_list} you can play now to win the finally whole {game_name} games step by step. Note that you can say something or keep silent to confuse your opponent." 
               + " Estimate Winning/Lose/Draw Rate for Each Plan: Understanding the given information, and your knowledge about the {game_name}, please estimate the success rate of each step of each plan step by step and the overall average winning/lose/draw rate  (normalize to number 100% in total) of each plan/strategy for the current game  step by step following the templete: If I do plan1, because I hold card, the public information (if release) and Single Game Win/Draw/Lose Rule, I will win or Lose or draw (probability);  ... continue  .... Overall win/draw/lose rate: Based on the analysis, I can do the weighted average step by step to get that the overall weighted average winning rate is (probability), average lose rate is (probability), draw rate is (probability) (normalize to number 100% in total)\n "
               + " Potential believes about the number of winning and lose payoffs for each plan: Understanding the game rule, your current observation, previous actions summarization, each new plans, Winning Payoff Rule,  Lose Payoff Rule, please infer your several believes about  the number of chips in pots for each plan step by step," #Output: Plan1: Chips in the pot:  If win, the winning payoff would be (Calculated by Winning Payoff Rules in the game rule) :  After the action,  Chips in the pot:  If lose , the lose payoff would be: . Plan2:  Chips in the pot:  If win, the winning chips would be (Calculated by Winning Payoff Rules in the game rule):  After the action, Chips in the pot:   If lose , the lose chips would be: . If the number of my chips in pots have no change, please directly output them. "
               + " Estimate Expected Chips Gain for Each Plan: Understanding the game rule, plans,  and your knowledge about the {game_name}, please estimate the overall average Expected Chips Gain for each plan/strategy in the current game by calculating winning rate * (Winning Payoff Rule in the game rule) - lose rate * (Lose Payoff Rule in the game rule)., explain what is the results if you do not select the plan, and explain why is this final  Expected  Chips Gain reasonablely step by step? "
            + " Plan Selection: Please output the rank of estimated expected chips gains for every plan objectively step by step, and select the plan/strategy with the highest estimated expected chips gain considering both the strategy improvement.</s>\n<|assistent|>\n"
                )

        agent_summary_description = short_memory_summary
       
        belief = self.belief if belief is None else belief
      
        kwargs = dict(
            
            recent_observations=agent_summary_description,
            last_plan=last_plan,
            belief=belief,
            initiator_name=self.name,
            pattern=pattern,
            recipient_name=recipient_name,
            observation=observation,
            rule=self.rule,
            game_name=self.game_name,
            valid_action_list=valid_action_list
        )

        
        plan_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)
        
        self.plan = plan_prediction_chain.run(**kwargs)
        self.plan = self.plan.strip()

        return self.plan.strip()

       
    
    def get_belief(self, observation: str, recipient_name: str,short_memory_summary:str,pattern:str = "",mode: str = "second_tom") -> str:
        """React to get a belief."""
        if mode == 'second_tom':
            prompt = PromptTemplate.from_template(system_prompt
                + "As the AI agent {agent_name}, you're engaged in the virtual board game {game_name} with your opponent {recipient_name}. \n"
                + " Game Rule: {rule} \n"
                + " Estimated Opponent Behavior Pattern and Strategy: {pattern} \n"
                + " Current Observation: {observation}\n"
                + " Recent Game Progress Summary: {recent_observations}\n"
                + " Given the game rules, the cards in your possession, current observations, recent game progress, the estimated behavior pattern of {recipient_name}, their potential guess about your strategy, and your deep knowledge of {game_name}, please analyze the following: \n"
                + " - My Card Analysis: Based on all information, analyze the best combination and advantages of your cards in the current round.\n"
                + " - Belief about {recipient_name}'s Cards: Infer the probabilities of {recipient_name}'s cards (normalized to 100%).\n"
                + " - Analysis of {recipient_name}'s Card Strategy: Evaluate {recipient_name}'s best card combination and advantages.\n"
                + " - {recipient_name}'s Belief about My Cards: Assuming the perspective of {recipient_name} (who can observe my actions but not my cards), infer their beliefs about your cards with corresponding probabilities.</s>\n<|assistent|>\n"
            )
        elif mode == 'first_tom':
            prompt = PromptTemplate.from_template(system_prompt
                + "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} with {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                + " Your estimated judgement about the behaviour pattern of {recipient_name} and improved strategy is: {pattern} \n"
                + " Your observation now is: {observation}\n"
                + " Your current game progress summarization including actions and conversations with {recipient_name} is: {recent_observations}\n"
                + " Understanding the game rule, the cards you have, your observation,  progress summarization in the current game, the estimated behaviour pattern of {recipient_name}, and your knowledge about the {game_name}, can you do following things? "
                + " Analysis on my Cards: Understanding all given information, please analysis what is your best combination and advantages of your cards  in the current round  step by step." 
                + " Belief on {recipient_name}'s cards: Understanding all given information, please infer your the probabilities about the cards of {recipient_name}  (normalize to number 100% total)  step by step. Templete: In the 1st round, {recipient_name} did action1 (probability),  ... continue... In the current round, {recipient_name} did action1 (probability), because {recipient_name}'s behaviour pattern and the match with the current public card (if release), he tends to have card1 (probability), card2 (probability) (normalize to number 100% in total). "
                + " Analysis on {recipient_name}'s Cards: Understanding all given information, please analysis what is {recipient_name}'s best combination and advantages of {recipient_name}'s cards  in the current round  step by step.</s>\n<|assistent|>\n" 
                
            )
        agent_summary_description = short_memory_summary

        kwargs = dict(
            agent_summary_description=agent_summary_description,
            recent_observations=agent_summary_description,
            agent_name=self.name,
            pattern= pattern,
            recipient_name=recipient_name,
            observation=observation,
            game_name=self.game_name,
            rule=self.rule

        )
        print("recipient_name is ", recipient_name)
        
        belief_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)
        self.belief = belief_prediction_chain.run(**kwargs)
        self.belief = self.belief.strip()
        return self.belief.strip()

    
    def get_pattern(self, recipient_name: str,game_pattern: str='', last_k:int=8,short_summarization:str='',mode:str='second_tom') -> str:
        """React to get a belief."""
       
        if mode == 'second_tom':
            prompt = PromptTemplate.from_template(system_prompt
            + "As the AI agent {agent_name}, you're currently engaged in the virtual game {game_name} against your opponent {recipient_name}.\n"
            + "Game Rule: {rule}\n"
            + "Historical Game Data: Your long-term memory includes all previous interactions with {recipient_name}, encompassing detailed observations of their actions, decisions, and outcomes in {game_name}.\n"
            + "Task 1 - Analyze {recipient_name}'s Gameplay Patterns: Based on the historical game data, identify and analyze {recipient_name}'s behavioral patterns and tendencies. Assign probabilities to these patterns, ensuring they total 100% for each identified behavior or strategy. This analysis should be structured in a tree format, highlighting how different cards and your own actions might influence {recipient_name}'s decisions.\n"
            + "Task 2 - Strategy Development: Utilizing your analysis, develop a comprehensive strategic plan to counter {recipient_name}'s gameplay patterns. Your strategy should anticipate their reactions to your moves and exploit any identified weaknesses or tendencies. Present this plan in a detailed tree-structured format, considering that you cannot see the opponent's cards but can observe their actions during the game.</s>\n<|assistent|>\n"
            )
            # prompt = PromptTemplate.from_template(system_prompt
            #     + "As the AI agent {agent_name}, you're engaged in the virtual game {game_name} against your opponent {recipient_name} who's not a interactive human but another model \n"
            #     + " Game Rule: {rule} \n"
            #     + " Historical Game Data: Your long-term memory of previous interactions, including observations, actions, and conversations with {recipient_name}, is as follows: {long_memory}\n"
            #     + " Analyze {recipient_name}'s Gameplay Patterns: Given the historical game data and your deep understanding of {game_name}, deduce {recipient_name}'s likely behavioral patterns and preferences in each round, assigning probabilities to these patterns (totaling 100% for each item). Consider how these patterns may be influenced by different cards held and by your actions. Provide a detailed analysis in a tree-structured format.\n"
            #     + " Strategy Development: Based on this analysis, formulate strategic approaches to exploit {recipient_name}'s gameplay patterns and anticipated guesses about your patterns, aiming for victory in {game_name}. Note that you cannot observe the opponent's cards during the game, but you can observe their actions. Present your strategic plan in a tree-structured format.</s>\n<|assistent|>\n"
            #     )
        elif mode == 'first_tom':
            prompt = PromptTemplate.from_template(system_prompt
                + "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} with {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                + " Your previous game memory including observations, actions and conversations with {recipient_name} is: {long_memory}\n"
                + " Please understand the game rule, previous all game history and your knowledge about the {game_name}, can you do following things for future games? "
                + "  {recipient_name}'s game pattern: Understanding all given information, please infer all possible reasonable {recipient_name}'s game pattern/preferences for each card he holds and each round with probability (normalize to number 100\% in total for each pattern item) for every round of the game as a tree-structure output step by step  " 
                #+ "Output: In the 1st round, when name holds card1 and the public card (if release), he would like to do action (probabilities); when name holds card2 and the public card (if release), he would like to do action (probabilities), ... continue.. In the 2nd round,  when name holds card1 and the public card (if release), .(similar with before).. continue. "
                + " Number of chips reason: Think about why you can have these chips in all previous games step by step. "
                + " Reflex: Reflex which your actions are right or wrong in previous games to win or Lose conrete chips step by step  (Note that you cannot observe the cards of the opponent during the game, but you can observe his actions) "
                + " Strategy Improvement: Understanding the above information, think about what strategies I can adopt to exploit the game pattern of {recipient_name} for winning {recipient_name} in the whole game step by step.  (Note that you cannot observe the cards of the opponent during the game, but you can observe his actions). Output as a tree-structure:</s>\n<|assistent|>\n"
                )
        else:
            prompt = PromptTemplate.from_template(system_prompt
                + "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} with {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                + " Your previous game memory including observations, actions and conversations with {recipient_name} is: {long_memory}\n"
                + " Please understand the game rule, previous all game history and your knowledge about the {game_name}, can you do following things for future games? "
                + " Number of chips reason: Think about why you can have these chips in all previous games step by step. "
                + " Reflex: Reflex which your actions are right or wrong in previous games to win or Lose conrete chips step by step. (Note that you cannot observe the cards of the opponent during the game, but you can observe his actions) "
                + " Strategy Improvement: Understanding the above information, think about what strategies I need to adopt to win {recipient_name} for the whole game step by step.  (Note that you cannot observe the cards of the opponent during the game, but you can observe his actions). Output as a tree-structure:</s>\n<|assistent|>\n"
            )
        reflection_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        long_memory = self.memory[-last_k:]
        long_memory_str = "\n\n".join([o for o in long_memory])
        
        kwargs = dict(
            long_memory=long_memory_str,
            game_pattern=game_pattern,
            agent_name=self.name,
            recipient_name=recipient_name,
            game_name=self.game_name,
            rule=self.rule

        )
        # print(kwargs)

        self.long_belief = reflection_chain.run(**kwargs)
        self.long_belief = self.long_belief.strip()
        return self.long_belief.strip()



    def get_summarization(self, recipient_name: str,game_memory: str, opponent_name:str,no_highsight_obs:bool) -> str:
        """Get a long memory summarization to save costs."""
        if no_highsight_obs:
            prompt = PromptTemplate.from_template(system_prompt
                + "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} with {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                + " The observation conversion rules are: {observation_rule}\n"
                + " One game memory including observations, actions and conversations with {recipient_name} is: {long_memory}\n"
                + " Understanding the game rule, observation conversion rules and game history and your knowledge about the {game_name}, can you do following things:"
                + " History summarization: summary the game history with action, observation, and results information? using the templete, and respond shortly: In the first round of first game, name holds card1 does action .... continue ..." 
                + "{opponent_name}'s card reasoning: If the card of {opponent_name} is not available, because {agent_name}'s  card is xx and public card (if release) is xxx, and {opponent_name} behaviours are xx, the current game result is xx,  please  infer {opponent_name}'s card with probability (100% in total) with your understanding about the above all information confidently step by step.</s>\n<|assistent|>\n"
                )
        else:
            prompt = PromptTemplate.from_template(system_prompt
                + "You are the player behind a NPC character called {agent_name}, and you are playing the board game {game_name} with {recipient_name}. \n"
                + " The game rule is: {rule} \n"
                + " The observation conversion rules are: {observation_rule}\n"
                + " One game memory including observations, actions and conversations with {recipient_name} is: {long_memory}\n"
                + " Understanding the game rule, observation conversion rules and game history and your knowledge about the {game_name}, can you do following things:"
                + " History summarization: summary the game history with action, observation, and results information? using the templete, and respond shortly: In the first round of first game, name holds card1 does action .... continue ...</s>\n<|assistent|>\n" 
                )       
        reflection_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        kwargs = dict(
            observation_rule=self.observation_rule,
            long_memory=game_memory,
            agent_name=self.name,
            recipient_name=recipient_name,
            opponent_name=opponent_name,
            # observation=observation,
            game_name=self.game_name,
            rule=self.rule

        )
        # print(kwargs)

        self.long_belief = reflection_chain.run(**kwargs)
        self.long_belief = self.long_belief.strip()
        return self.long_belief.strip()


    def get_short_memory_summary(self, observation: str, recipient_name: str,short_memory_summary:str) -> str:
        """React to get a belief."""
        prompt = PromptTemplate.from_template(system_prompt
          + " As {agent_name}, engaged in the virtual board game {game_name} against {recipient_name}, your expertise in game strategy is crucial.\n"
          + " Game Rule: {rule} \n"
          + " Current Observation: {observation}\n"
          + " Game History: The current game history includes actions, observations, and conversations thus far: {agent_summary_description}\n"
          + " Task: Based on the game rules, your current observation, and your strategic knowledge of {game_name}, succinctly summarize the game history. Present this summary in a tree-structured format to reflect the sequence of events and decisions.</s>\n<|assistent|>\n")

        agent_summary_description = short_memory_summary
    
        kwargs = dict(
            agent_summary_description=agent_summary_description,
            recent_observations=agent_summary_description,
            agent_name=self.name,
            recipient_name=recipient_name,
            observation=observation,
            game_name=self.game_name,
            rule=self.rule

        )
        
        belief_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)
        # print("get_short_memory_summary ", type(LLMChain))
        # print(type(belief_prediction_chain))
        # print(dir(belief_prediction_chain))
        self.belief = belief_prediction_chain.run(**kwargs)
        self.belief = self.belief.strip()
        return self.belief.strip()



    def convert_obs(self, observation: str, recipient_name: str, user_index: str, valid_action_list:str) ->  str:
        """React to get a belief."""
        prompt = PromptTemplate.from_template(system_prompt
            + "You are the AI agent named {agent_name}, operating as a player in the virtual board game {game_name} against {recipient_name}. Your player index is {user_index}. \n"
            + " Game Rule: {rule} \n"
            + " Current Observation: {observation}\n"
            + " Available Actions: {valid_action_list}\n"
            + " Observation Conversion Guidelines: {observation_rule}\n"
            + " Based on your advanced understanding of game strategies and dynamics, interpret {observation} and {valid_action_list}. Consider both the explicit information and the potential implications of hidden information. Identify strategically relevant actions and provide insights into possible opponent tactics and game progression. Your response should reflect a deep, strategic understanding of {game_name}.</s>\n<|assistent|>\n"
        )
        kwargs = dict(
            user_index=user_index,
            agent_name=self.name,
            rule=self.rule,
            recipient_name=recipient_name,
            observation=observation,
            valid_action_list=valid_action_list,
            game_name=self.game_name,
            observation_rule=self.observation_rule
        )
        obs_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)
        # print("convert_obs ", type(LLMChain))
        # print(type(obs_prediction_chain))
        # print(dir(obs_prediction_chain))
        self.read_observation = obs_prediction_chain.run(**kwargs)
        self.read_observation = self.read_observation.strip()
        return self.read_observation



    def action_decision(self, observation: str, valid_action_list: List[str], promp_head: str, act: str = None,short_memory_summary:str="", recipient_name: str = "") -> Tuple[str,str]:
        """React to a given observation."""
        """React to a given observation."""
        promp_head_ =  'Reminder: Your response must be in the format "action|comment". \n' #The previous output was not valid. Please choose a valid action from the list and pair it with a strategic comment. Ensure your response aligns with your plan and is formatted correctly. Example format: "move_piece|I think this will give me an advantage\n'
        if act:
            print("will add action string, act is ", act)
            for c in "{}()[]":
                act = act.replace(c, "")
            promp_head_ += f"The previous output [{act}] was not valid.\n"
            print("promp_head_ is ", promp_head_)
            #print)
        prompt = PromptTemplate.from_template(system_prompt
            + str(promp_head_)
            + "As {agent_name} in {game_name}, your current plan is: {plan}.\n"
            + "Based on this plan and your strategic analysis, choose the most appropriate action from the available options: {valid_action_list}. Also, decide whether to communicate with {recipient_name} to bluff, confuse, or influence their perception. Your response should be strategic and contribute to your overall goal of winning the game.\n"
            + "Respond with your chosen action and any communication to {recipient_name}, separated by a '|' symbol.\n" #Rememeber, you should only choose one choice, DON NOT list multiple possible choice and explanations\n"
            + "</s>\n<|assistent|>\n"
        )
        #print("prompt of action\n", prompt)
        print("valid action list = ", valid_action_list)
        agent_summary_description = short_memory_summary
        
        kwargs = dict(
            agent_summary_description= agent_summary_description,
            # current_time=current_time_str,
            # relevant_memories=relevant_memories_str,
            agent_name= self.name,
            game_name=self.game_name,
            recipient_name=recipient_name,
            observation= observation,
            agent_status= self.status,
            valid_action_list = valid_action_list,
            plan = self.plan,
            belief = self.belief,
            act = act
        )
        action_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)
        
        result = action_prediction_chain.run(**kwargs)
        ori_result = result 
        print("origin act output: ", result)
        #result.append("|")
        result += "|"
        result_comm = ""
        result = result.lower()
        result = result.replace("raize", "raise")
        result = result.replace("raised", "raise")
        def find_leftmost_pattern(input_text, patterns):
            earliest_pattern = None
            earliest_position = len(input_text)

            for pattern in patterns:
                match = re.search(pattern, input_text)
                if match and match.start() < earliest_position:
                    earliest_pattern = pattern
                    earliest_position = match.start()

            return earliest_pattern
        
        # next_act = find_leftmost_pattern(result.lower(), valid_action_list)
        # if next_act:
        #     result = next_act
        # else:
        #     result = "No valid action in your act sentence " + result.lower()  #next_act
        if "|" in result:
            result,result_comm = result.split("|",1)
            result = find_leftmost_pattern(result.lower(), valid_action_list)
        # elif ":" in result:
        #     for act_candidate in reversed(result.split(":")):
        #         if len(act_candidate) > 0:
        #             result = act_candidate #result.split(":")[-1].strip("-") #result.split(":",1) a
        #             break
        else:
            # next_act = find_leftmost_pattern(result.lower(), valid_action_list)
            # if next_act:
            #     result = next_act 
            print("act result not in act | something format")
            result_comm = ""
        
        if result is None:
            result = ori_result
        if result_comm is None:
            result_comm = "NONE"
        return result.strip(),result_comm.strip()

    def make_act(self, observation: str,opponent_name: str, player_index:int,valid_action_list: List, verbose_print:bool,game_idx:int,round:int,bot_short_memory:List, bot_long_memory:List, console,log_file_name='', mode='second_tom',no_highsight_obs=False) -> Tuple[bool, str]:
        readable_text_amy_obs = self.convert_obs(observation, opponent_name, player_index, valid_action_list)
        start_time = time.time()
        print("verbose_print = ", verbose_print)
        if  verbose_print:
            console.print('readable_text_obs: ', style="red")
            print(readable_text_amy_obs)
                   
        time.sleep(0)
        if len(bot_short_memory[player_index]) == 1:
            short_memory_summary = f'{game_idx+1}th Game Start \n'+readable_text_amy_obs
        else:
            short_memory_summary = self.get_short_memory_summary(observation=readable_text_amy_obs, recipient_name=opponent_name,short_memory_summary='\n'.join(bot_short_memory[player_index]))

            
        if verbose_print:
            console.print('short_memory_summary: ', style="yellow")
            print(short_memory_summary)

        time.sleep(0)
        if  round <= 101:
                self.pattern = self.get_pattern(opponent_name,'',short_summarization=short_memory_summary,mode=mode)        
                console.print('pattern: ', style="blue")
                print(self.pattern)

        time.sleep(0)
        print("opponent_name is ", opponent_name)

        if mode == 'second_tom' or mode == 'first_tom':
            belief = self.get_belief(readable_text_amy_obs,opponent_name,short_memory_summary=short_memory_summary,pattern=self.pattern,mode=mode)
            if verbose_print:
                console.print(self.name + " belief: " , style="deep_pink3")
                print(self.name + " belief: " + str(belief))
                
        else:
            belief = ''

        time.sleep(0)
        plan = self.planning_module(readable_text_amy_obs,opponent_name, belief=belief,valid_action_list=valid_action_list,short_memory_summary=short_memory_summary,pattern=self.pattern,last_plan='', mode=mode)
        if  verbose_print:
            console.print(self.name + " plan: " , style="orchid")
            print(self.name + " plan: " + str(plan))
            
        time.sleep(0)
        promp_head = ''
        act, comm = self.action_decision(readable_text_amy_obs, valid_action_list, promp_head,short_memory_summary=short_memory_summary, recipient_name=opponent_name)

        if log_file_name is not None:
            util.get_logging(logger_name=log_file_name + '_obs',
                        content={str(game_idx + 1) + "_" + str(round): {"raw_obs": observation,
                                                                        "readable_text_obs": readable_text_amy_obs}})
            util.get_logging(logger_name=log_file_name + '_short_memory',
                        content={str(game_idx + 1) + "_" + str(round): {
                            "raw_short_memory": '\n'.join(bot_short_memory[player_index]),
                            "short_memory_summary": short_memory_summary}})
            util.get_logging(logger_name=log_file_name + '_pattern_model',
                                content={str(game_idx + 1) + "_" + str(round): self.pattern})
            util.get_logging(logger_name=log_file_name + '_belief',
                            content={str(game_idx + 1) + "_" + str(round): {
                                "belief": str(belief)}})
            util.get_logging(logger_name=log_file_name + '_plan',
                        content={str(game_idx + 1) + "_" + str(round): {
                            "plan": str(plan)}})
            util.get_logging(logger_name= log_file_name + '_act',
                        content={str(game_idx + 1) + "_" + str(round): {
                            "act": str(act), "talk_sentence": str(comm)}})
 

        while act not in valid_action_list:
            print('Your action + output ', str(act), ' is not a valid output which should be in act|something format and be contained valid action list', ','.join(valid_action_list), ' please try again.\n')
            promp_head = 'The previous output + was not valid. Please choose a valid action from the list and pair it with a strategic comment. Ensure your response aligns with your plan and is formatted correctly. Example format: "move_piece|I think this will give me an advantage".\n' #'You previous output {act} is not valid. Based on your plan, select your next action and comment. Remember, the format should be "action|comment".\n '#+= 'Action {act} is not a valid action in {valid_action_list}, please try again.\n'
            act, comm = self.action_decision(readable_text_amy_obs, valid_action_list, promp_head,act)
        print(self.name + " act: " + str(act))
        print(comm)
        formatted_time = "{:.2f}".format(time.time() - start_time)

        print(f"make act total time cost: {formatted_time} s")
        #print("make act total time cost", time.time() - start_time  )

        bot_short_memory[player_index].append(f"{self.name} have the observation {readable_text_amy_obs}, try to take action: {act} and say {comm} to {opponent_name}")
        bot_short_memory[((player_index + 1)%2)].append(f"{self.name} try to take action: {act} and say {comm} to {opponent_name}")

        bot_long_memory[player_index].append(
            f"{self.name} have the observation {observation}, try to take action: {act} and say {comm} to {opponent_name}")
        return act,comm,bot_short_memory,bot_long_memory
