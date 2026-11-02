import os
import pandas as pd
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LLM_Interaction.llm_models_api import generate_response

CASES = ["trait", "try", "neutral"]
from LLM_Interaction.sys_simulator import InteractiveChatSession
from datetime import datetime
import hydra
from dotmap import DotMap
from pathlib import Path
from omegaconf import DictConfig
from LLM_Interaction.prompts.consts import (
    p2_descriptions,
    p2_descriptions_reversed,
    p2_descriptions_user,
    p2_descriptions_reversed_user,
)

class ChatManager:
    def __init__(self, cfg):
        self.cfg = DotMap(cfg)
        self.model_name=cfg.model_name
        self.project_root = cfg.project_root
        self.data_path = cfg.dataset_path
        self.cfg.project_root = Path(cfg.project_root)
        self.sevice_path = cfg.dataset_path
        self.output_dir = cfg.output_path
        self.add_example = cfg.add_example
        self.max_turns = cfg.max_turns
        self.temperature = cfg.gpt_temperature
        self.gpt_max_tokens = cfg.gpt_max_tokens 
        self.instruction_system_path =cfg.sys_prompt_path
        self.instruction_user_path =cfg.user_prompt_path
        self.dialog_id = cfg.dialog_id
        self.max_iterations = cfg.max_iteration
        self.prompt_name = cfg.prompt_name
        self.api_source = cfg.api_source
        self.domain = cfg.domain
        self.service_name = cfg.service_name
        self.model = cfg.model
    def get_unique_values(self, df, column_name):
        """
        Returns a list of unique non-null values from a specified column in the DataFrame.
        """
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")
        return df[column_name].dropna().unique().tolist()
      
                            
    def run(self, trait, description, user_description,case,instruction_system_path):
        
        df = pd.read_csv(self.data_path)
        
        max_api = 5
        # Numberize each turn
        
        i=0

         #initialize_prompts
        with open(instruction_system_path, "r") as f:
            self.instruction_prompt_system = f.read() 

        with open(self.instruction_user_path, "r") as f:
            self.instruction_prompt_user = f.read() 
        if self.domain == 'Restaurant':    
            sys_GPT_input = self.instruction_prompt_system.format(personality_definition = user_description, domain = self.domain,
                                                                    trait = trait, cuisine = self.get_unique_values(df,"cuisine"), price_range = self.get_unique_values(df,"price_range"))
            user_GPT_input = self.instruction_prompt_user.format(unique_cities=", ".join(self.get_unique_values(df,"city")), domain = self.domain,
                                                                          personality_definition= description, trait=trait)
        if self.domain == 'Travel':
            sys_GPT_input = self.instruction_prompt_system.format(personality_definition = user_description, domain = self.domain,
                                                                    trait = trait, category = self.get_unique_values(df,"category"))
            user_GPT_input = self.instruction_prompt_user.format(unique_location=", ".join(self.get_unique_values(df,"location")), domain = self.domain,
                                                                          personality_definition= description, trait=trait)    
            
        if self.domain == 'Media':
            sys_GPT_input = self.instruction_prompt_system.format(personality_definition = user_description, domain = self.domain,
                                                                    trait = trait, genre = self.get_unique_values(df,"genre"), price_range = self.get_unique_values(df,"price"))
            user_GPT_input = self.instruction_prompt_user.format(genre = self.get_unique_values(df,"genre"), domain = self.domain,
                                                                          subtitle = self.get_unique_values(df,"subtitle_language"), personality_definition= description, trait=trait)    
        
            
        if self.domain == 'Hotel':
            sys_GPT_input = self.instruction_prompt_system.format(personality_definition = user_description, domain = self.domain,
                                                                    trait = trait, star_rating = self.get_unique_values(df,"star_rating")) #, number_rooms = self.get_unique_values(df,"price"))
            user_GPT_input = self.instruction_prompt_user.format(location = self.get_unique_values(df,"location"), domain = self.domain,
                                                                          personality_definition= description, trait=trait)    
      
        
        
        chat_session = InteractiveChatSession(
                                    domain = self.service_name,
                                    api_source = self.api_source,
                                    model_name = self.model_name,
                                    instruction_file = sys_GPT_input,
                                    activate_retry = True,
                                    service_path = self.sevice_path,
                                    add_example = self.add_example,
                                    max_tokens = self.gpt_max_tokens,
                                    temperature = self.temperature,
                                )
        #initialize dialog info
        dialoge_info = {
            'id':[],
            'trait':[trait],
            'user_log':[],
            'sys_log':[],
            'turn_tracker':[],
            'api_calls': [],
            'user_req_slots': [],
            'dialog': [],
            'search_results': [],
            'api_tracker' : [],
            'turn_id' : [],
            }

        dialoge_info["turn_tracker"] = "user"
        dialoge_info["id"] = f"{self.dialog_id}_{i+1}" 
        i += 1    
        dialoge_info['turn_id'] = -1
        dialoge_info['api_tracker'] = 0
        
        while(True):
                progress_bar = tqdm(total=self.max_iterations)
                if 'turn_id' in dialoge_info:
                        dialoge_info['turn_id'] += 1       
                # Initialize with user's first message
                if dialoge_info["turn_tracker"] == "user":
              
                    dialoge_info["turn_tracker"] = "system"
                                    #initialize some of dialoge_info
                    # User-agent generation call
                    if dialoge_info['turn_id'] == 0:
                        user_prompt = user_GPT_input
                    else:
                        user_prompt = user_GPT_input + "\n " + "\n".join(dialoge_info["user_log"])
                    response = generate_response(user_prompt, self.model_name, self.api_source,
                                                 max_tokens=self.gpt_max_tokens, temperature=self.temperature)
                            
                        
                    
                    #user_init_message = "I'm looking for somewhere I can grab some food in Fremont. Something intermediate priced would be perfect."
                    if response[:5].lower() == "user:":
                        dialoge_info["user_log"].append(f"{response[:5].capitalize() + response[5:]}")
                        # if dialoge_info['turn_id'] == 0 and trait in response[5:]:
                        #     continue
                        # else:
                        dialoge_info["sys_log"].append(f"{response[:5].capitalize() + response[5:]}")
                    
                    else:
                        dialoge_info["user_log"].append(f"User: {response}")
                        # if dialoge_info['turn_id'] == 0 and trait in response:
                        #     continue
                        # else:
                        dialoge_info["sys_log"].append(f"User: {response}")
                    print(f"User: {response}")

                    end_phrases = [
                        "thanks a lot! that's all i need.",
                        "that's all i need"
                    ]
                    normalized_response = response.strip().lower()
                    if any(phrase in normalized_response for phrase in end_phrases):
                        progress_bar.update(1)

                        break  # Finish the current dialog
                else:
                    chat_session.interactive_chat(dialoge_info['sys_log'])
                    last_system_turn = dialoge_info['sys_log'][-1]
                    
                    if last_system_turn == "<Failed Dialog>":
                        progress_bar.update(1)
                        break
                    
                    if last_system_turn == "<Failed Dialog Due to Wrong Result>":
                        progress_bar.update(1)
                        break
                    
                    if last_system_turn.startswith("Search Results\n"):
                        # end the dialog when the number of generated api calls gets larger than the actual numbers of the generated api calls
                        if dialoge_info['api_tracker'] >= max_api:
                                progress_bar.update(1)
                                # Finish the current dialog
                                break
                        dialoge_info['api_tracker'] += 1
                    
                    elif last_system_turn.endswith("Take care and have a lovely evening!"):
                        progress_bar.update(1)
                        # Finish the current dialog
                        break
                    else:
                        # change the turn_tracker to user
                        dialoge_info["turn_tracker"] = "user"
                        # add the generated text to the user context if the generated text is not an api call
                        # it already has "System:" at the beginning
                        dialoge_info['user_log'].append(last_system_turn)
                    
                # specifying the maximum number of turns
                if dialoge_info['turn_id'] >= self.max_turns:
                    progress_bar.update(1)
                    break
                
        progress_bar.close()       
            
        
        output_dir = os.path.join(self.output_dir, case)
        output_dir = os.path.join(output_dir, self.model)
        prompt_name = f'{self.domain.lower()}_{case}_{self.model}'
        output_dir = os.path.join(output_dir, prompt_name) 
        suffix = "addExample" if self.add_example else "no_example"
        
        example_folder = os.path.join(output_dir, suffix)
        # Trait
        trait_folder = os.path.join(example_folder, trait)
        # Create directory if it doesn't exist
        if not os.path.exists(trait_folder):
            os.makedirs(trait_folder, exist_ok=True)



        # Timestamp
        timestamp = datetime.now().strftime("%m-%d_%H-%M")
        final_output_dir = os.path.join(trait_folder, timestamp)
        os.makedirs(final_output_dir, exist_ok=True)
        # Example: Write sys_log to a file
        with open(os.path.join(final_output_dir, "sys_log.txt"), "w", encoding="utf-8") as f:
            for line in dialoge_info["sys_log"]:
                f.write(line + "\n")
                
    def prompt_write(self,instruction_system_path):
        
        prompt_folder = os.path.join(self.output_dir, self.prompt_name)

        example_folder = prompt_folder 
        # Example setting
        #suffix = "addExample" if self.add_example else "no_example"
        #example_folder = os.path.join(prompt_folder, suffix)
        #os.makedirs(example_folder, exist_ok=True)

        # Timestamp
        
        os.makedirs(example_folder, exist_ok=True)
        # Example: Write sys_log to a file
        with open(instruction_system_path, 'r') as yml_file:
            yml_content = yml_file.read()  # Read as string
            
        with open(os.path.join(example_folder, 'prompt_system.txt'), 'w') as txt_file:
            txt_file.write(yml_content)
        
        with open(self.instruction_user_path, 'r') as yml_file:
            yml_content = yml_file.read()  # Read as string
        with open(os.path.join(example_folder, 'prompt_user.txt'), 'w') as txt_file:
            txt_file.write(yml_content)
        
      

@hydra.main(config_path="config_files", config_name="gpt-interactive-trait") #"./config_files"
def hydra_start(cfg: DictConfig) -> None:
    chat_manager = ChatManager(cfg)
    for case in CASES:
        if case == "neutral":
            #instruction_system = f'{cfg.domain.lower()}_prompt_{case}_SYS.yml'
            #instruction_system_path = os.path.join(chat_manager.instruction_system_path, instruction_system)
            #chat_manager.prompt_write(instruction_system_path)
        
                
            for traits, description in p2_descriptions_reversed.items():
                if traits == 'introversive':
                        #if traits == 'unconscientious':
                            for _ in range(9):
                                print(f" ***********************trait*********************: {traits}")
                                chat_manager.run(traits,description,p2_descriptions_reversed_user[traits],case,chat_manager.instruction_system_path)
            #             else:
        #                 for _ in range(50):
        #                         if traits !='introversive': #and traits !='stable' and traits !="closed": #and traits != 'unconscientious': #and traits !='stable': #or traits !='closed'  or traits !='stable' or traits != 'unconscientious':
        #                             print(f" ***********************trait*********************: {traits}")
        #                             chat_manager.run(traits,description,p2_descriptions_reversed_user[traits],case,instruction_system_path)
                    
        # for traits, description in p2_descriptions.items():
        #             #if traits != 'Neuroticism': #or traits == 'Openness': 
        #             #if traits != 'Extraversion'and traits != 'Agreeableness': #and traits != 'Conscientiousness': #and traits !='Neuroticism' : 
        #             for _ in range(50):    
        #                 print(f" ***********************trait*********************: {traits}")
        #                 chat_manager.run(traits,description,p2_descriptions_user[traits],case,instruction_system_path)   
    
if __name__ == "__main__":
    hydra_start()
    # wandb.finish()
    print(os.getcwd())
