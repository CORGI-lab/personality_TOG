import os
import pandas as pd
import re

from rapidfuzz import fuzz
from LLM_Interaction.utils import (
    domain_name_schema, 
    extract_key_value_pairs, 
    extract_cuisine,
    extract_star_rating,
    api_params_check, 
    check_is_transactional, 
    fetch_service_results_from_csv, 
    format_service_results, 
    fetch_service_results_from_csv_no_params, 
    extract_method_name, 
    get_result_slots
    )
from LLM_Interaction.llm_models_api import generate_response


class InteractiveChatSession:
    def __init__(
        self,
        domain,
        api_source = 'openai',
        model_name = 'gpt-4o',
        schema_file = './LLM_Interaction/schemas/schema.txt',
        instruction_file = None,
        activate_retry = True,
        add_example = True,
        service_path=None,
        system_response_handler=None,
        max_tokens=250,
        temperature=0.7,
    ):
        self.instruction_file = instruction_file
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.activate_retry = activate_retry
        self.last_api_call =''
        self.last_query_params=''
        self.api_source = api_source
        self.model_name = model_name
        self.conversation_log = []
        self.domain = domain
        self.schema_file=schema_file
        self.system_response_handler = system_response_handler
        self.service_path = service_path
        self.domain_name = domain_name_schema(schema_file, domain)
        self.storage = ""
        

    def generate(self, prompt):
        """System-agent generation call (normal turn or API-call retry)."""
        return generate_response(prompt, self.model_name, self.api_source,
                                 max_tokens=self.max_tokens, temperature=self.temperature)

    def __call__(self):
        # Return the dialog_id when the object is called
        return self.dialog_id
    def process_api_call(self,
        llm_response,
        api_source,
        model_name,
        chat_GPT_input,
        conversation_log,
        service_results_path,
        schema_file,
        domain,
        last_api_call,
        activate_retry = True,
        system_response_handler=None
    ):
    # Preprocess the api call to convet \\_ ti _
        llm_response = llm_response.replace("\\_", "_")

        #print("self.last-apicall: ", self.last_api_call)
        query_params = extract_key_value_pairs(llm_response)
        to_remove = [k for k, v in query_params.items() if str(v).lower() == "dontcare"]
        for k in to_remove:
            query_params.pop(k, None)
        method_name = extract_method_name(llm_response)
        # if method_name == 'FindAttractions':
        #     if query_params.get("category") == "dontcare":
        #         query_params.pop("category", None)
        #     if query_params.get("free_entry") == "dontcare":
        #         query_params.pop("category", None)
        if method_name == 'FindRestaurants':
            self.last_query_params = query_params
        
        if method_name == 'ReserveRestaurant':
            print("ReserveRestaurant///////////////////////////////////////////")
            if self.last_query_params and type(self.last_query_params) != str :
                if self.last_query_params.get("cuisine") != None:
                    cuisine_value = extract_cuisine(self.last_api_call, self.last_query_params)
                    query_params = {'cuisine': cuisine_value, **query_params}
            if self.last_query_params and type(self.last_query_params) == str :
                #cuisine_value = re.search(r"'cuisine'\s*:\s*'([^']*)'", self.last_query_params)
                cuisine_value = extract_cuisine(self.last_api_call, self.last_query_params)
                query_params = {'cuisine': cuisine_value, **query_params}
        if method_name == 'SearchHotel':
            self.storage = extract_star_rating(query_params)
            if query_params.get("star_rating") == "dontcare":
                query_params.pop("star_rating", None)
            if query_params.get("smoking_allowed") == "dontcare":
                query_params.pop("smoking_allowed", None)
            if "number_of_rooms" in query_params:
                query_params.pop("number_of_rooms", None)
            
        if self.storage != None and method_name == 'ReserveHotel' :
            query_params = {'star_rating': self.storage, **query_params}
        if method_name =='FindMovies':
            print("FindMovies///////////////////////////////////////////")
            if query_params.get("actors") == "dontcare":
                query_params.pop("actors", None)
            if query_params.get("director") == "dontcare":
                query_params.pop("director", None)
        api_check = api_params_check(method_name, query_params, schema_file, domain)

        # Check if the required parameters are missing or method name is not extracted
        if api_check['missing_required_params']:
            if activate_retry:
                llm_conversation_log = conversation_log.copy()

                num_tries = 1
                while num_tries <= 3:
                    if api_check['missing_required_params']:
                        if api_check['missing_required_params'] == ['method name wrong pattern']:
                            msg = f"You are generating the method name in the wrong pattern"
                        
                        elif api_check['missing_required_params'] == ['method name not found']:
                            msg = f"Method name is wrong because it is not found in the schema"
                        
                        else:
                            msg = f"You are missing the required parameters in the API call"
                
                        ################# msg += f""
                            if len(api_check["provided_query_params"]) == 0:
                                msg += f", also you might be generating the API call with the wrong pattern"
                        print(f"msg: {msg}")

                        llm_conversation_log.append(f"\n Inform: {msg}, please generate the API call again. If user is open to any cuisine, select a random cuisine from list based on user's trait.")
                        dialog_history_prompt = "\n ".join(llm_conversation_log)
                
                        # Call the LLM
                        llm_response = self.generate(chat_GPT_input + "\n " + dialog_history_prompt)
                        
                        
                        if llm_response[:7].lower() == "system:":
                            llm_conversation_log.append(llm_response[:7].capitalize() + llm_response[7:])
                        else:
                            llm_conversation_log.append(f"System: {llm_response}")
                        if "APICall" not in llm_response:
                            if llm_response[:7].lower() == "system:":
                                conversation_log[-1] = llm_response[:7].capitalize() + llm_response[7:]
                            else:
                                conversation_log[-1] = f"System: {llm_response}"
                            return
                        query_params = extract_key_value_pairs(llm_response)
                        method_name = extract_method_name(llm_response)
                        api_check = api_params_check(method_name, query_params, schema_file, domain)
                        num_tries += 1
                    else:
                        if llm_response[:7].lower() == "system:":
                            conversation_log[-1] = llm_response[:7].capitalize() + llm_response[7:]
                        else:
                            conversation_log[-1] = f"System: {llm_response}"
                        break
                
                if num_tries == 4:
                    return conversation_log.append(f"<Failed Dialog>")
            else:
                
                print("Failed Dialog")
                return conversation_log.append(f"<Failed Dialog>")
            
        if api_check['provided_query_params']:
            print(f"Provided query parameters: {query_params}")
            if method_name == 'FindRestaurants':
                self.last_query_params = f"Provided query parameters: {query_params}"

            # Extract the method name from the APICall
            method_name = extract_method_name(llm_response)

            print(f"Fetching service results for the API Call...")

            # Fetch the service results after the API call
            service_results = fetch_service_results_from_csv(service_results_path, query_params, limit=3)

            if service_results is not None and not service_results.empty:
                # Format the service results for display
                formatted_service_results = format_service_results(service_results)
                for service_result in formatted_service_results:
                    # Send the service result to the handler if provided
                    if system_response_handler:
                        system_response_handler(service_result)
                    else:
                        print(f"Tool: {service_result}")
                conversation_log.append(f"Search Results\n{formatted_service_results}\nEnd Search Results")
            else:
                no_results_message = "No service results found."
                if system_response_handler:
                    system_response_handler(no_results_message)
                else:
                    print(f"Tool: {no_results_message}")
                # conversation_log.append(f"Search Results\n\nEnd Search Results")
                
                # Check if the method is transactional
                is_transactional = check_is_transactional(schema_file, domain, method_name)
                print(f"Method '{method_name}' is transactional: {is_transactional}")

                if is_transactional:
                    # Make a service entry to the CSV file using values from APICall parameters
                    print("Preparing to insert a new record into the database.")

                    # Construct the new service entry from the query_params
                    new_entry = query_params.copy()

                    # Ensure all columns are present in the new entry
                    if os.path.exists(service_results_path):
                        # Read CSV and set index_col=0 to avoid 'Unnamed: 0'
                        nullable_int_columns = {
                            'party_size': 'Int64',
                            'number_of_seats': 'Int64',
                            'number_of_tickets': 'Int64',
                            'stay_length': 'Int64',
                            'amount': 'Int64',
                            'transfer_amount': 'Int64'
                        }
                        df_existing = pd.read_csv(service_results_path, index_col=0, dtype=nullable_int_columns)
                        all_columns = df_existing.columns.tolist()

                        # **Updated Code Starts Here**
                        # Identify missing columns in new_entry
                        missing_columns = [col for col in all_columns if col not in new_entry]

                        if missing_columns:
                            print(f"Missing columns in the new entry: {missing_columns}")
                            # Attempt to fill missing columns from matching previous entries
                            # Look for records where at least 2 columns match
                            matching_records = df_existing.copy()

                            # Calculate match count for each existing record
                            def calculate_match_count(row):
                                count = 0
                                for key in query_params.keys():
                                    if key in row and pd.notna(row[key]):
                                        
                                        row_value = row[key]
                                        
                                        if key in nullable_int_columns:
                                            row_value = int(row_value)
                                        
                                        if isinstance(row_value, str):
                                            if fuzz.partial_ratio(str(query_params[key]).lower(), row_value.lower()) >= 90:  # Set threshold
                                                count += 1
                                        else: # Non-string comparison (assume exact match for numbers)
                                            if str(query_params[key]).lower() == str(row_value).lower():
                                                count += 1
                                return count

                            matching_records['match_count'] = matching_records.apply(calculate_match_count, axis=1)
                            # Filter records where at least 2 columns match
                            matching_records = matching_records[matching_records['match_count'] >= 2]

                            if not matching_records.empty:
                                # Sort by highest match count
                                matching_records = matching_records.sort_values(by='match_count', ascending=False)
                                # Use the first matching record to fill missing columns
                                best_match = matching_records.iloc[0]
                                for col in missing_columns:
                                    if col in best_match and pd.notna(best_match[col]):
                                        new_entry[col] = best_match[col]
                            else:
                                print("No matching records found with at least 2 matching columns to fill missing columns.")

                            # Drop the 'match_count' column from existing DataFrame
                            matching_records = matching_records.drop(columns=['match_count'], errors='ignore')
                        else:
                            print("No missing columns to fill from previous entries.")
                        # **Updated Code Ends Here**
                    else:
                        # If CSV doesn't exist, create columns based on result_slots
                        # Add missing result slots with empty values
                        result_slots = get_result_slots(schema_file, domain, method_name)
                        all_columns = result_slots
                        missing_columns = []  # No previous data to fill from

                    # Remove 'Unnamed: 0' from all_columns if present
                    if 'Unnamed: 0' in all_columns:
                        all_columns.remove('Unnamed: 0')

                    # Reorder new_entry to match the columns
                    ordered_entry = {col: new_entry.get(col, '') for col in all_columns}

                    # Print the service entry before inserting
                    print("Service Entry to be inserted:")
                    print(ordered_entry)

                    # Append the new entry to the CSV file
                    # append_to_csv(service_results_path, ordered_entry, all_columns)

                    # Inform the user that the record has been inserted
                    insertion_message = "The new record has been inserted into the database."
                    if system_response_handler:
                        system_response_handler(insertion_message)
                    else:
                        print(f"SYSTEM: {insertion_message}")
                    conversation_log.append(f"Search Results\n[{ordered_entry}]\nEnd Search Results")
                else:
                    conversation_log.append(f"Search Results\n\nEnd Search Results")
        else:
            print("No provided query parameters to process.")
            service_results = fetch_service_results_from_csv_no_params(service_results_path, limit=3)
            if service_results is not None and not service_results.empty:
                # Format the service results for display
                formatted_service_results = format_service_results(service_results)
                for service_result in formatted_service_results:
                    print(f"Tool: {service_result}")
                conversation_log.append(f"Search Results\n{formatted_service_results}\nEnd Search Results")
            else:
                conversation_log.append(f"Search Results\n\nEnd Search Results")
            #return conversation_log
    def interactive_chat(self, conversation_log):
        # Add user input to conversation log

        #self.conversation_log.append(f"User: {user_input}")
        self.conversation_log = conversation_log
        dialog_history_prompt = "\n ".join(self.conversation_log)
        #dialog_id = self.dialog_id
        #print(dialog_id)

        # Call the LLM
        llm_response = self.generate(self.instruction_file + "\n " + dialog_history_prompt)
        
        if llm_response:
            # Handle the system response
            if self.system_response_handler:
                self.system_response_handler(llm_response)
            else:
                print(f"{llm_response}")

            if llm_response[:7].lower() == "system:":
                self.conversation_log.append(llm_response[:7].capitalize() + llm_response[7:])
            else:
                self.conversation_log.append(f"System: {llm_response}")

            # Process API call if present in the response
            
                
            if "APICall" in llm_response:
                if "service_result" in llm_response:
                    conversation_log.append(f"<Failed Dialog Due to Wrong Result>")
                    self.last_api_call = llm_response
                    return 
                else:
                    self.process_api_call(
                        llm_response,
                        self.api_source,
                        self.model_name,
                        self.instruction_file,
                        self.conversation_log,
                        self.service_path,
                        self.schema_file,
                        self.domain,
                        self.last_api_call,
                        self.activate_retry,
                        self.system_response_handler
                    )
                    self.last_api_call = llm_response
                    
            return 
        

