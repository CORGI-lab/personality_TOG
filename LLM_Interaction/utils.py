import os
import re
import csv
import difflib
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz import process



def get_best_match(query, reference_list, threshold=80):
    match = process.extractOne(query, reference_list)
    if match:
        matched_element, score, _ = match
        return matched_element if score >= threshold else None
    return None
    


def check_is_transactional(schema_file, domain, method_name):
    # Read the schema file
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema_content = f.read()
    
    # Split the schema into domains
    domains = schema_content.split('service_name:')[1:]
    for dom in domains:
        dom = dom.strip()
        if dom.startswith(domain):
            # Found the correct domain
            # Now find the method within this domain
            intents = dom.split('intents_')
            for intent in intents:
                if f"name: {method_name}" in intent:
                    # Found the method
                    # Check if it's transactional
                    lines = intent.strip().split('\n')
                    for line in lines:
                        if line.strip().startswith('is_transactional:'):
                            type_line = line.strip()
                            # Extract the type value
                            type_value = type_line.replace('is_transactional: ', '').strip()
                            is_transactional = type_value.lower() == 'true'
                            return is_transactional
                    # If 'is_transactional:' line not found, default to False
                    return False
            # Method not found in this domain
            break  # Exit after searching the correct domain
    # Domain or method not found
    print(f"Domain '{domain}' or method '{method_name}' not found in schema.")
    return False

def extract_method_name(llm_response):
    # Extract the method name from the llm_response string
    match = re.search(r"APICall\s*\(\s*method\s*=\s*'([^']+)'\s*,", llm_response)
    if match:
        return match.group(1)
    else:
        return None


def domain_name_schema(schema_file, domain):
    with open(schema_file, 'r') as file:
        schema_data = file.read()
        domains = [line.split(':')[1].strip() for line in schema_data.splitlines() if line.startswith('service_name')]
        closest_match = difflib.get_close_matches(domain, domains, n=1, cutoff=0.7)

        if not closest_match:
            raise ValueError(f"Domain '{domain}' not found in schema. Did you mean: {', '.join(domains)}?")

        domain_corrected = closest_match[0]
        domain_start = schema_data.find(f"service_name: {domain_corrected}")
        domain_schema = schema_data[domain_start:]
        next_domain = domain_schema.find("service_name:", 1)
        if next_domain != -1:
            domain_schema = domain_schema[:next_domain]
        return domain_corrected
def extract_star_rating(query_params):
    match = re.search(r"star_rating\s*:\s*['\"]?([A-Za-z\s]+)['\"]?", str(query_params))
    if match:
        return match.group(1).strip()
    else:
        match = re.search(r"'star_rating'\s*:\s*'([^']+)'", str(query_params))
        if match:
            return match.group(1).strip()
        else: 
            None
def extract_cuisine(last_api_call,last_query_params):
    # This regex handles:
    # - cuisine: Japanese
    # - cuisine: 'Japanese'
    # - cuisine: "Japanese"
    match = re.search(r"cuisine\s*:\s*['\"]?([A-Za-z\s]+)['\"]?", last_api_call)
    if match:
        return match.group(1).strip()
    else:
        match = re.search(r"'cuisine'\s*:\s*'([^']+)'", last_query_params)
        if match:
            return match.group(1).strip()
        else: 
            None
import re, ast

import re, ast

import re, ast

def extract_key_value_pairs(api_call_text: str) -> dict:
    m = re.search(r"parameters\s*=\s*(\{.*?\})\s*\)", api_call_text, flags=re.S)
    if not m:
        return {}

    s = m.group(1).strip()

    # 1) Quote bare keys: { place_name: ... } -> { 'place_name': ... }
    s = re.sub(r"(?<=\{|,)\s*([A-Za-z_]\w*)\s*:", r" '\1':", s)

    # 2) Quote bare values safely (use double quotes to handle apostrophes)
    def repl_value(match):
        val = match.group(1).strip()

        # Already quoted
        if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
            return f": {val}"

        # Dates like 2024-11-15 should be strings
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", val):
            return f': "{val}"'

        # Numbers / booleans / None
        if re.fullmatch(r"[-+]?\d+(\.\d+)?", val) or val in {"True", "False", "None"}:
            return f": {val}"

        # Otherwise: treat as a string, wrap in double quotes
        val = val.replace('"', r'\"')
        return f': "{val}"'

    s = re.sub(r":\s*([^,}\n]+)", repl_value, s)

    try:
        d = ast.literal_eval(s)
        if not isinstance(d, dict):
            return {}
        # Return strings (compat with your old pipeline)
        out = {}
        for k, v in d.items():
            out[str(k).strip("'\"")] = str(v).strip("'\"")
        return out
    except Exception:
        return {}

    
def extract_key_value_pairs2(api_call_text):
    # Extract key-value pairs from the parameters in the APICall
    match = re.search(r"parameters\s*=\s*\{([^}]+)\}", api_call_text)
    if match:
        params_str = match.group(1)
        pairs = re.findall(r"(\w+):\s*([^,}]+)", params_str)
        params = {}
        for key, value in pairs:
            # Remove quotes from values if present
            value = value.strip().strip("'\"")
            params[key.strip()] = value
        if not params:
            pairs = re.findall(r"(['\"])(\w+)\1\s*:\s*(['\"])(.*?)\3", params_str)
            params = {}
            for match in pairs:
                key = match[1]
                value = match[3]
                params[key] = value
        return params
    else:
        return {}

def format_service_results(service_results):
    formatted_results = []
    
    columns_to_drop = ['unnamed: 0', 'Unnamed: 0']
        
    formatted_results = [
        row.dropna().drop(columns_to_drop, errors='ignore').to_dict() 
        for _, row in service_results.iterrows()
    ]
    
    return formatted_results

def fetch_service_results_from_csv_no_params(csv_path, limit=3):
    # Fetch random service results from the CSV file
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} does not exist.")
        return None

    df = pd.read_csv(csv_path)
    
    # If the dataframe has fewer rows than the limit, return all rows
    if len(df) <= limit:
        return df
    
    # Randomly sample 'limit' number of rows
    return df.sample(n=limit)

def fetch_service_results_from_csv(csv_path, query_params, limit=3):
    # Fetch service results from the CSV file based on query_params, ignoring 'date' parameter
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} does not exist.")
        return None
    
    nullable_int_columns = {
        'party_size': 'Int64',
        'number_of_seats': 'Int64',
        'number_of_tickets': 'Int64',
        'stay_length': 'Int64',
        'amount': 'Int64',
        'transfer_amount': 'Int64'
    }

    df = pd.read_csv(csv_path, dtype=nullable_int_columns)

    # Convert column names and query_params keys to lowercase for consistent comparison
    df.columns = df.columns.str.lower()
    query_params = {k.lower(): v for k, v in query_params.items()}
    
    # get the date keys
    query_date_keys = [key for key in query_params.keys() if 'date' in key.lower()]
    query_time_keys = [key for key in query_params.keys() if 'time' in key.lower()]
    query_party_keys = [key for key in query_params.keys() if 'party_size' in key.lower()]

    db_date_keys = [col for col in df.columns if 'date' in col.lower()]
    db_time_keys = [col for col in df.columns if 'time' in col.lower()]
    db_party_keys = [col for col in df.columns if 'party_size' in col.lower()]
    # Save the values first
    time = query_params.get('time')
    date = query_params.get('date')
    party_size = query_params.get('party_size')
    # Remove the keys
    query_params.pop('time', None)
    query_params.pop('date', None)
    query_params.pop('party_size', None)
    query_params.pop('number_of_rooms', None)
    query_params.pop('stay_length', None)
    if query_params.get("star_rating") == "dontcare":
                query_params.pop("star_rating", None)
    
    # Filter the DataFrame based on query_params, ignoring 'date' parameter
    for key, value in query_params.items():
        if key in query_date_keys:
            continue  # Ignore the 'date' parameter during filtering
        # if key in query_time_keys:
        #     continue
        # Remove quotes from value if present
        value = value.strip("'\"")
        if key in df.columns:
            
            # Use fuzzy matching for comparison
            def fuzzy_match(row_value):
                if pd.isna(row_value):
                    return False
                
                if key in nullable_int_columns:
                    row_value = int(row_value)
                    
                if isinstance(row_value, str):
                    return fuzz.partial_ratio(row_value.lower(), str(value).lower()) >= 90  # Set a threshold, e.g., 90
                else: # Non-string comparison (assume exact match for numbers)
                    return str(row_value).lower() == str(value).lower() 
            
            df = df[df[key].apply(fuzzy_match)]
            
            # df = df[df[key].astype(str).str.lower() == value.lower()]
            
            if df.empty:
                break  # No need to continue filtering if no rows are left
        else:
            # If the key is not in the columns, we can't filter on it
            continue
    if df.empty:
        return None
    else:
        
        # Add them back
        if party_size is not None:
            query_params['party_size'] = party_size
        if time is not None:
            query_params['time'] = time
        if date is not None:
            query_params['date'] = date
         
        if len(query_time_keys)>0 and len(db_time_keys)>0:
            for time_key in query_time_keys:
                if time_key in db_time_keys:
                    new_time = query_params[time_key].strip(":")
                    df[time_key] = new_time  # Replace the 'time' values with the new date
                    
        if len(query_party_keys)>0 and len(db_party_keys)>0:
            for party_key in query_party_keys:
                if party_key in db_party_keys:
                    new_party = query_params[party_key]
                    df[party_key] = new_party  # Replace the 'time' values with the new date  
                              
        if len(query_date_keys)>0 and len(db_date_keys)>0:
            for date_key in query_date_keys:
                if date_key in db_date_keys:
                    new_date = query_params[date_key].strip("'\"")
                    df[date_key] = new_date  # Replace the 'date' values with the new date 
            
        # Replace the 'date' column in the fetched results with the date from query_params

        return df.head(limit)



def api_params_check(method_name, query_params, schema_file, domain):
    # Check for missing required parameters
    missing_required_params = []
    provided_query_params = list(query_params.keys())

    if method_name is None:
        return {
            'missing_required_params': ['method name wrong pattern'],
            'provided_query_params': provided_query_params
        }
    
    required_slots = get_required_slots(schema_file, domain, method_name)
    
    if required_slots == ['None']:
        missing_required_params = []
    elif required_slots == ["Not found"]:
        missing_required_params = ['method name not found']
    else:
        for param in required_slots:
            if param not in query_params:
                missing_required_params.append(param)

    return {
        'missing_required_params': missing_required_params,
        'provided_query_params': provided_query_params
    }



def append_to_csv(csv_path, entry_dict, fieldnames):

    file_exists = os.path.exists(csv_path)

    # Open the CSV file in append mode
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()

        # Write the entry
        writer.writerow(entry_dict)

    print(f"Appended new entry to {csv_path}")

def get_result_slots(schema_file, domain, method_name):
    # Read the schema file and return the result slots for the given method
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema_content = f.read()

    # Split the schema into domains
    domains = schema_content.split('service_name:')[1:]
    for dom in domains:
        dom = dom.strip()
        if dom.startswith(domain):
            # Now find the intents within this domain
            # Split the schema into intents
            intents = dom.split('intents_')
            for intent in intents:
                if f"name: {method_name}" in intent:
                    # Find the line with result_slots
                    lines = intent.strip().split('\n')
                    for line in lines:
                        if line.strip().startswith('result_slots:'):
                            result_slots_line = line.strip()
                            # Remove 'result_slots:' and split the slots
                            slots = result_slots_line.replace('result_slots:', '').strip().split(',')
                            slots = [slot.strip() for slot in slots]
                            return slots
                    break

    return []

def get_required_slots(schema_file, domain, method_name):
    # Read the schema file and return the required slots for the given method
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema_content = f.read()

    # Split the schema into domains
    domains = schema_content.split('service_name:')[1:]
    for dom in domains:
        dom = dom.strip()
        if dom.startswith(domain):
            # Now find the intents within this domain
            # Split the schema into intents
            intents = dom.split('intents_')
            for intent in intents:
                if f"name: {method_name}" in intent:
                    # Find the line with required_slots
                    lines = intent.strip().split('\n')
                    for line in lines:
                        if line.strip().startswith('required_slots:'):
                            required_slots_line = line.strip()
                            # Remove 'required_slots:' and split the slots
                            slots = required_slots_line.replace('required_slots:', '').strip().split(',')
                            slots = [slot.strip() for slot in slots]
                            return slots
                    break

    return ["Not found"]

# Save the conversation log to a CSV file
def save_conversation_to_csv(conversation_log, output_csv_file):
    fieldnames = ['Speaker', 'Utterance']
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        # Process each conversation log entry before saving
        for entry in conversation_log:
            if ("service_result_" in entry['Utterance']):
                writer.writerow({'Speaker': 'Tool', 'Utterance': entry['Utterance']})
            else:
                writer.writerow(entry)
    
    print(f"Conversation saved to {output_csv_file}")
