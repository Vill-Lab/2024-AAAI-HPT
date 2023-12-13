import json
import openai
from tqdm import tqdm

api_key = "" # Enter Your API Key!!!
dataset_name = "" # Enter Your Dataset Name, e.g. EuroSAT

templates = ["What does {} look like among all {}? ",
             "What are the distinct features of {} for recognition among all {}? ",
             "How can you identify {} in appearance among all {}? ",
             "What are the differences between {} and other {} in appearance? ",
             "What visual cue is unique to {} among all {}? "]

infos = {
    'ImageNet':             ["{}",                "objects"],
    'OxfordPets':           ["a pet {}",          "types of pets"], 
    'Caltech101':           ["{}",                "objects"],
    'DescribableTextures':  ["a {} texture",      "types of texture"],
    'EuroSAT':              ["{}",                "types of land in a centered satellite photo"],
    'FGVCAircraft':         ["a {} aircraft",     "types of aircraft"],
    'Food101':              ["{}",                "types of food"],
    'OxfordFlowers':        ["a flower {}",       "types of flowers"],
    'StanfordCars':         ["a {} car",          "types of car"],
    'SUN397':               ["a {} scene",        "types of scenes"],
    'UCF101':               ["a person doing {}", "types of action"],
}

def get_completion(prompt, api_key, model="gpt-3.5-turbo", temperature=1):
    openai.api_key = api_key
    messages = [{"role": "system", "content": "You are good at image classification."}, {"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response['choices'][0]['message']['content']

with open('./data/gpt_data/classname/'+dataset_name+'.txt', 'r') as f:
    classnames = f.read().split("\n")[:-1]

result = {}
for classname in tqdm(classnames):
    info = infos[dataset_name]
    prompts = [template.format(info[0], info[1]).format(classname) + "Describe it in 20 words." for template in templates]
    responses = [get_completion(prompt, api_key) for prompt in prompts]
    result[classname] = responses

    with open('./data/gpt_data/description/'+dataset_name+'.json','w') as f:
        json.dump(result, f, indent=4)
