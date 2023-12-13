import json
import openai
from tqdm import tqdm

api_key = "" # Enter Your API Key!!!
dataset_name = "" # Enter Your Dataset Name, e.g. EuroSAT

def get_completion(prompt, api_key, model="gpt-3.5-turbo", temperature=1):
    openai.api_key = api_key # Enter Your API Key!!!
    instruction = "Please reconstruct the following sentence into 4 parts and output a JSON object. \
              1. All entities, \
              2. All attributes, \
              3. Relationships between entity and entity, \
              4. Relationships between attributes and entities. \
              The target JSON object contains the following keys: \
              Entities, \
              Attributes, \
              Entity-to-Entity Relationships, \
              Entity-to-Attribute Relationships. \
              For the key Entity-to-Entity Relationships, the value is a list of JSON object that contains the following keys: entity1, relationship, entity2. \
              For the key Entity-to-Attribute Relationships, the value is a list of JSON object that contains the following keys: entity, relationship, attribute. \
              Do not output anything other than the JSON object. \
              \
              Sentence: '''{}'''".format(prompt)
    messages = [{"role": "system", "content": "You are good at image classification."}, {"role": "user", "content": instruction}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response['choices'][0]['message']['content']

with open('./data/gpt_data/description/'+dataset_name+'.json', 'r') as f:
    prompt_templates = json.load(f)

result = {}
for classname in tqdm(prompt_templates.keys()):
    prompts = prompt_templates[classname]
    responses = [json.loads(get_completion(prompt, api_key)) for prompt in prompts]
    result[classname] = responses

    with open('./data/gpt_data/structure/'+dataset_name+'.json','w') as f:
        json.dump(result, f, indent=4)
