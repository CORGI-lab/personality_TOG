import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import hydra
import re
from omegaconf import DictConfig
from pathlib import Path
from dotmap import DotMap
from datetime import datetime
from LLM_Interaction.prompts.consts import (
    p2_descriptions,
    p2_descriptions_reversed,
)
LLMS = ["gpt4", "qwen3", "gemeni"]
CASES = ["trait", "try", "neutral"]


TRAITS = [
    "introversive", "closed", "stable", "Extraversion", "Agreeableness",
    "Conscientiousness", "Neuroticism", "Openness", "disagreeable", "unconscientious"
]
class Task_Metric:
    
    def __init__(self, cfg):
        self.cfg = DotMap(cfg)
        self.project_root = cfg.project_root
        self.cfg.project_root = Path(cfg.project_root)
        self.output_path = cfg.output_path
        self.domain_schema = cfg.domain_schema
        self.domain = cfg.domain
        self.schema_file = cfg.schema_path
        self.dialog_path = cfg.dialog_path
        self.prompt_name = cfg.prompt_name
        
        self.total_intent = 0
        self.success_task = 0
        self.llm_total_dialog = 0
        self.covered_intents =0
        self.llm_success_task = 0
        self.missing_intents = 0
        self.text = ""
        self.trait = ""
        self.text_trait =""

    # def get_dated_folder(self, base_path, date_prefix):
    #     # List all folders/files in base_path
    #     for name in os.listdir(base_path):
    #         full_path = os.path.join(base_path, name)
    #         # Check if it's a directory and starts with the given date prefix
    #         if os.path.isdir(full_path) and name.startswith(date_prefix):
    #             return full_path
    #     return None
    
    def extract_intents(self):
        # Read the schema file
        with open(self.schema_file, 'r', encoding='utf-8') as f:
            schema_content = f.read()
        #intent_list =[]
        # Split the schema into domains
        domains = schema_content.split('service_name:')[1:]
        for dom in domains:
            dom = dom.strip()
            if dom.startswith(self.domain_schema):
                # Found the correct domain
                # Now find the method within this domain
                intent_names = re.findall(r'\tname:\s*(\w+)', dom)
                return intent_names
 
    def has_successful_last_reserve(self, dialogue_text: str) -> bool:
        """
        Return True if the LAST reservation API call for the current domain
        is followed by a non-empty Search Results block.
        """

        if self.domain.lower() == "restaurant":
            reserve_method = "ReserveRestaurant"
        elif self.domain.lower() == "hotel":
            reserve_method = "ReserveHotel"
        else:
            raise ValueError(f"Unsupported domain for reservation task success: {self.domain}")

        pattern = re.compile(
            rf"APICall\s*\(\s*method\s*=\s*['\"]?{reserve_method}['\"]?"
            r".*?\)\s*"
            r"Search Results\s*"
            r"(.*?)"
            r"\s*End Search Results",
            re.DOTALL | re.IGNORECASE
        )

        matches = pattern.findall(dialogue_text)

        if not matches:
            return False

        last_result_block = matches[-1].strip()

        if not last_result_block:
            return True

        if re.fullmatch(r"\[\s*\]", last_result_block):
            return False

        return last_result_block.startswith("[")
        
         
    def run(self):
        dialog_path = os.path.join(self.dialog_path, self.domain)
        for case in CASES:
            dialog_path_case = os.path.join(dialog_path, case) 
            for llm in LLMS:
                self.text_trait =""
                dialog_path_llm = os.path.join(dialog_path_case, llm) 
                if llm != "gpt4":
                    self.prompt_name = f'{self.domain.lower()}_{case}_{llm}'
                    dialog_path_name = os.path.join(dialog_path_llm, self.prompt_name) 
                    
                else:
                    self.prompt_name = f'{self.domain.lower()}_{case}_gpt'
                    dialog_path_name = os.path.join(dialog_path_llm, self.prompt_name) 
                    
                dialog_path_name = os.path.join(dialog_path_name, "addExample") 
                self.llm_success_task = 0
                self.llm_total_dialog = 0
                # if case == "try" and llm == "gpt4":
                #     TRAITS3 = TRAITS2
                # else:
                #     TRAITS3 = TRAITS
                for trait in TRAITS:
                    print(f" ***********************trait*********************: {trait}")
                    dialog_path_trait = os.path.join(dialog_path_name, trait) 
                    total_dialog = 0
                    successful_tasks = 0

                    for name in os.listdir(dialog_path_trait):
                        dialog_path_date = os.path.join(dialog_path_trait, name) 


                        dialog_file_sys = os.path.join(dialog_path_date, "sys_log.txt")

                        
                        total_dialog += 1
                        self.llm_total_dialog += 1

                        with open(dialog_file_sys, "r", encoding="utf-8") as f:
                            dialogue_text = f.read()

                        task_success = self.has_successful_last_reserve(dialogue_text)

                        if task_success:
                            successful_tasks += 1
                            self.llm_success_task += 1
                        

                    task_success_rate = (
                        successful_tasks / total_dialog
                        if total_dialog > 0 else 0.0
                    )
                    self.text_trait +=  f"Accuracy for {trait}: {task_success_rate}\n"

                    text = f"Total dialogue num: {total_dialog}\n"
                    text += f"Successful task dialogues: {successful_tasks}\n"
                    text += f"Failed task dialogues: {total_dialog - successful_tasks}\n"
                    text += f"Task Success for trait {trait}: {100 * task_success_rate:.2f}%\n"

                    print(text)
                    print("\n")

                    self.trait = trait
                    self.print_output(text, case,llm)
                    
                llm_success_rate = (
                        self.llm_success_task / self.llm_total_dialog
                        if self.llm_total_dialog > 0 else 0.0
                    ) 
                
                llm_text =  self.text_trait
                llm_text += "**********************************************************************************************"
                llm_text += f"Total Dialogue Num: {self.llm_total_dialog}\n"
                llm_text += f"Total success task: {self.llm_success_task}\n"

                llm_text += f"Total Success Task :  {100 * llm_success_rate:.2f}%\n"
                print("\n")
                print("**********************************************************************************************")


                self.print_output(llm_text, case, llm, trait=False)
                
    def print_output(self, text, case, llm, trait = True):
        output_dir = os.path.join(self.output_path, self.domain)
        output_dir = os.path.join(output_dir, case)
                        
        output_dir = os.path.join(output_dir, llm)
        if trait:
            output_dir = os.path.join(output_dir, self.trait)
    
             

                            # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

      
        # Example: Write response_log to a file
        with open(os.path.join(output_dir, "task_completion.txt"), "w", encoding="utf-8") as f:
            f.write(text)

@hydra.main(config_path="../../config_files", config_name="task_completion")
def hydra_start(cfg: DictConfig) -> None:
    Task_Eval = Task_Metric(cfg)
    

    Task_Eval.run()

    
if __name__ == "__main__":
    hydra_start()
    # wandb.finish()
    print(os.getcwd())        
        
     
                            
        
