import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import hydra
from omegaconf import DictConfig
from pathlib import Path
from dotmap import DotMap
from datetime import datetime
from LLM_Interaction.prompts.consts import (
    p2_descriptions,
    p2_descriptions_reversed,
)


class Dialog_Metric:
    
    def __init__(self, cfg):
        self.cfg = DotMap(cfg)
        self.project_root = cfg.project_root
        self.cfg.project_root = Path(cfg.project_root)
        self.output_path = cfg.output_path
        self.domain = cfg.domain
        self.correct_set=0
        self.total_sample=0
        self.trait_print=""
        self.text=""
        #self.date = cfg.date
        #gpt_setting
        # self.model_name = cfg.model_name
        # self.api_source = cfg.api_source
        # self.temperature = cfg.gpt_temperature
        # self.gpt_max_tokens = cfg.gpt_max_tokens 

        self.dialog_path = cfg.dialog_path
        self.prompt_name = cfg.prompt_name
        
    def get_dated_folder(self, base_path, date_prefix):
        # List all folders/files in base_path
        for name in os.listdir(base_path):
            full_path = os.path.join(base_path, name)
            # Check if it's a directory and starts with the given date prefix
            if os.path.isdir(full_path) and name.startswith(date_prefix):
                return full_path
        return None
    def print_output(self, text, trait=True):
        output_dir = os.path.join(self.output_path, self.prompt_name)
        
        # Trait
        if trait == True:
            output_dir = os.path.join(output_dir, self.trait_print)
        
        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)

            # Timestamp
        
        # timestamp = datetime.now().strftime("%m-%d_%H-%M")
        # final_output_dir = os.path.join(output_dir, timestamp)
        # os.makedirs(final_output_dir, exist_ok=True)
        # Example: Write response_log to a file
        with open(os.path.join(output_dir, "correct.txt"), "w", encoding="utf-8") as f:
                f.write(text)
                
    def run(self,trait):
        
        dialog_path = os.path.join(self.dialog_path, trait)  
        correct = 0
        total_sample = 0
        for name in os.listdir(dialog_path):
            dialog_path_date = self.get_dated_folder(dialog_path, name)
            dialog_file_sys = os.path.join(dialog_path_date, "sys_log.txt")
            with open(dialog_file_sys, "r", encoding="utf-8") as f:
                lines = f.readlines()
            

            if lines:
                for ln in reversed(lines):
                    if ln.strip() == "":
                        continue
                    if ln.lstrip().startswith("("):
                        continue
                    
                    last_line = ln.strip()
                    print("Last line:", last_line)
                    break
                #                 last_line = lines[-1] 
                # if last_line.startswith("("):
                #     last_line = lines[-2]
                # if last_line == '\n':
                #     last_line = lines[-3]
                # print("Last line:", last_line)
            else:
                print("The file is empty.")
                break
                
            end_phrases = [
                            "thanks a lot! that's all i need.",
                            "that's all i need",
                            "take care and have a lovely evening!"
                        ]
            normalized_response = last_line.strip().lower()
            if any(phrase in normalized_response for phrase in end_phrases):
                            correct += 1
                            self.correct_set += 1
                            print("correct")
            total_sample += 1
            self.total_sample += 1

        text = f"Total Correct:{correct}\n"   
        text += f"Total Samples:{total_sample}\n"   
        text += f"Total Accuracy:{100*(correct/total_sample)}"  
        accuracy = 100*(correct/total_sample)
        self.text +=  f"Accuracy for {trait}: {accuracy}\n"

        print(f"Accuracy for {trait}: {accuracy}")
        self.trait_print = trait       
        self.print_output(text)       
        # output_dir = os.path.join(self.output_path, self.prompt_name)
        
        # # Trait
        # trait_folder = os.path.join(output_dir, trait)
        # # Create directory if it doesn't exist
        # if not os.path.exists(trait_folder):
        #     os.makedirs(trait_folder, exist_ok=True)

        # Timestamp
        # timestamp = datetime.now().strftime("%m-%d_%H-%M")
        # final_output_dir = os.path.join(trait_folder, timestamp)
        # os.makedirs(final_output_dir, exist_ok=True)
        # # Example: Write response_log to a file
        # with open(os.path.join(final_output_dir, "correct.txt"), "w", encoding="utf-8") as f:
        #     f.write(text)

@hydra.main(config_path="../../config_files", config_name="dialog_completion")
def hydra_start(cfg: DictConfig) -> None:
    Dialog_Eval = Dialog_Metric(cfg)
    

    for _ in range(1):
        for traits, _ in p2_descriptions_reversed.items():
            print(f" ***********************trait*********************: {traits}")
            Dialog_Eval.run(traits)
            
        for traits, _ in p2_descriptions.items():
           print(f" ***********************trait*********************: {traits}")
           Dialog_Eval.run(traits)
    Dialog_Eval.text += "\n\n"
    Dialog_Eval.text += f"Total Correct:{Dialog_Eval.correct_set}\n"   
    Dialog_Eval.text += f"Total Samples:{Dialog_Eval.total_sample}\n"    
    total_accuracy = 100*(Dialog_Eval.correct_set/Dialog_Eval.total_sample)
    Dialog_Eval.text += f"Total Accuracy:{total_accuracy}"   
    print("Total Accuracy: ", total_accuracy)
    Dialog_Eval.print_output(Dialog_Eval.text,trait=False)
if __name__ == "__main__":
    hydra_start()
    # wandb.finish()
    print(os.getcwd())        
        
     