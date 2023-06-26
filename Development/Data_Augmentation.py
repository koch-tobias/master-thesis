# %%
import openai
import pandas as pd
import numpy as np
import random
from loguru import logger
from boundingbox_calculations import find_valid_space, random_centerpoint_in_valid_space, calculate_corners
from config import gpt_settings

# %%
def random_order(description: str) -> str:
    '''
    This function takes a sentence as input and changes the word order randomly. 
    '''
    # Split the description into individual words
    words = description.split()
    new_description = description

    # Shuffle the words in the description
    while new_description == description:
        random.shuffle(words)

        # Join the shuffled words back into a string
        new_description = ' '.join(words)

    return new_description

# %%
def swap_chars(description: str) -> str:
    '''
    This function takes a text as input and randomly swap two chars which are next to each other
    '''
    # Convert description to a list of chars
    chars = list(description)

    # Choose the chars which should be swaped 
    index1 = random.randrange(len(chars) - 1)
    index2 = index1 + random.choice([-1,1])

    # Change their position
    chars[index1], chars[index2] = chars[index2], chars[index1]
    
    return ''.join(chars)


# %%
def add_char(description: str) -> str:
    # German alphabet
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    # Randomly choose a char of the alphabet
    char = random.choice(alphabet)

    # Randomly select the position where to add the char
    index = random.randrange(len(description)+1)

    # Add the selected char
    new_description = description[:index] + char + description[index:]

    return new_description 

# %%
def delete_char(description: str) -> str:
    index = random.randrange(len(description))
    new_description = description[:index] + description[index+1:]
    
    return new_description   

# %%
def random_mistakes(description: str) -> str:
    '''
    This function takes a text as input and inserts a random spelling error. 
    This is done either by swapping chars, adding a chars, or deleting a chars.
    '''
    # Generate three random probabilities between 0 and 1 
    probs = [random.uniform(0,1) for _ in range(3)]
    probs = [p/sum(probs) for p in probs]

    if probs[0] > (probs[1] and probs[2]):
        # Randomly swap two chars
        new_description = swap_chars(description)
        
        return new_description
    
    elif probs[1] > (probs[0] and probs[2]):
        # Randomly add a char
        new_description = add_char(description)

        return new_description

    else:
        # Randomly delete one char
        new_description = delete_char(description)
        
        return new_description

# %%
def remove_prefix(response: str) -> str:
    ''' 
    The output of GPT has always a prefix like "Answer: " or "Modified Designation: ". This prefix will be deleted that we keep only the generated text.
    '''
    index = response.find(":")

    if index != -1:
        new_response = response[index+1:].lstrip()
    else:
        new_response = response
    
    return new_response

# %%
def init_openai():
    openai.api_type = "azure"
    openai.api_base = "https://convaip-sbx-openai.openai.azure.com/"
    # openai.api_version = "2022-12-01" # For GPT3.0
    openai.api_version = "2023-03-15-preview" # For GPT 3.5
    openai.api_key = '9e6fa24631f54cf58866766bd31a2bff' #os.getenv("OPENAI_API_KEY")

# %%
def create_prompt(text: str) -> str:
    prompt = f'''
                Task:
                Write a modified car component designation based on the given German language input. Your output must be slightly different to the input while retaining the same meaning.

                Context:
                The original designation is written in German and may contain abbreviations and technical terms. You are not allowed to add or remove words. Abbreviations that often appear in the data set are 'HI' for 'HINTEN', 'VO' for 'VORN', 'HKL' for 'HECKKLAPPE','GPR' for GEPAECKRAUM', 'VKL' for 'VERKLEIDUNG', 'ISP' for 'INNENSPIEGEL', 'TV' for 'TUER VORNE', 'TH' for 'TUER HINTEN', 'HBL' for 'HOCHGESETZTE BREMSLEUCHTE', 'STF' for 'STOSSFAENGER', 'KST' for 'KOPFSTUETZE', 'ND' for 'NORMALDACH', 'DAHAUBE' for DACHANTENNENHAUBE. This abbreviation help you to create designations.

                Instructions:
                1. Carefully examine the input and understand its meaning.
                2. Modify the designation to create a new version which is very close to the original.
                3. Make sure that the meaning of the designation remains the same.
                4. Make sure that the modified designation is not equal to the input data

                Examples: 
                - Input Data: ABDECKLEISTE EINSTIEG HI
                - Modified Designation: ABDECKLEISTE EINSTIEG HINTEN

                - Original: KANTENSCHUTZ TUER VORN
                - Modified Designation: KANTENSCHUTZ TV

                - Original: MD KST VERSTELLBAR AUSSEN MAT
                - Modified Designation: MD KST AUßENMATTE VERSTELLBAR

                - Original: SEITENVERKLEIDUNG
                - Modified Designation: SEITENVERKL

                - Orignal: ABDECKUNG FENSTERRAHMEN TUER VORNE
                - Modified Designation: ABDECKUNG FENSTERRAHMEN TUERE VORN

                """
                {text}
                """
            '''
    return prompt

# %%
def gpt30_designation(text: str) -> str:
    prompt= create_prompt(text)
    new_response = text
    while new_response == text:
        response = openai.Completion.create(
            engine="davinci-003-deployment",
            prompt=prompt,
            temperature=0.7,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            n=1
        )
        response = [choice.text.strip() for choice in response.choices]
        new_response = remove_prefix(response[0])
    return new_response

# %%
def gpt35_designation(text:str) -> str:
    prompt= create_prompt(text)
    new_response = text
    while new_response == text:
        response = openai.ChatCompletion.create(
            engine="chat-gpt-0301",
            messages=[{"role":"system","content":"You are an AI assistant that helps to create a car component designation."},{"role":"user","content": prompt}],
            temperature = gpt_settings["temperature"],
            max_tokens = gpt_settings["max_tokens"],
            top_p = gpt_settings["top_p"],
            n = gpt_settings["n"]
        )
        response = [choice.message.content for choice in response.choices]
        new_response = remove_prefix(response[0])
    return new_response

# %%
def augmented_boundingbox(df_original, df_temp):
    corners, valid_length, valid_width, valid_height = find_valid_space(df_original)
    list_corners = []
    list_corners.append(corners)

    # Generate random values for length, width, and height until the volume is within the desired range
    min_volume = df_original['volume'].min()
    max_volume = df_original['volume'].max()
    if min_volume == max_volume:
        min_volume = min_volume*0.9
        max_volume = max_volume*1.1
    
    min_length = df_original['length'].min() * 0.90
    min_width = df_original['width'].min() * 0.90
    min_height = df_original['height'].min() * 0.90
    max_length = df_original['length'].max() * 1.1
    max_width = df_original['width'].max() * 1.1
    max_height = df_original['height'].max() * 1.1
       
    volume = 0
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    z_min = 0
    z_max = 0
    length = 0
    width = 0
    height = 0
    
    while(volume < min_volume or volume > max_volume):
        length = np.random.uniform(min_length, max_length)
        width = np.random.uniform(min_width, max_width)
        height = np.random.uniform(min_height, max_height)
        volume = length * width * height
        

    center_point = random_centerpoint_in_valid_space(corners, length, width, height)
    x_min, x_max, y_min, y_max, z_min, z_max = calculate_corners(center_point, length, width, height, df_temp["theta_x"], df_temp["theta_y"], df_temp["theta_z"])

    # Modify the bounding box information
    df_temp.loc[0, "X-Min_transf"] = x_min
    df_temp.loc[0, "X-Max_transf"] = x_max
    df_temp.loc[0, "Y-Min_transf"] = y_min
    df_temp.loc[0, "Y-Max_transf"] = y_max
    df_temp.loc[0, "Z-Min_transf"] = z_min
    df_temp.loc[0, "Z-Max_transf"] = z_max
    df_temp.loc[0, "center_x"] = center_point[0]
    df_temp.loc[0, "center_y"] = center_point[1]
    df_temp.loc[0, "center_z"] = center_point[2]
    df_temp.loc[0, "length"] = length
    df_temp.loc[0, "width"] = width
    df_temp.loc[0, "height"] = height
    df_temp.loc[0, "volume"] = volume
    df_temp.loc[0, "density"] = volume/df_temp.loc[0, "Wert"]

    return df_temp

# %%
def data_augmentation(df: pd.DataFrame) -> pd.DataFrame:
    ''' 
    This function generates synthetic data to extend the data set. 
    Only the data points that are relevant for a measurement are expanded, since this class is very underrepresented. 
    A component name is expanded if it contains more than 2 words and at least one of the three data augmentation techniques is activated.
    Return: New dataset 
    '''
    init_openai()

    logger.info("Start adding artificial designations...")

    df_relevant_parts = df[df["Relevant fuer Messung"] == "Ja"]    

    unique_names = df_relevant_parts["Einheitsname"].unique().tolist()
    unique_names.sort()
    for name in unique_names:
        df_new = df_relevant_parts[(df_relevant_parts["Einheitsname"] == name)].reset_index(drop=True)
        df_temp = df_new.iloc[[0]]

        if df_new.shape[0] < 3:
            if len(df_temp.loc[0,"Benennung (bereinigt)"].split()) > 1:
                df_temp.loc[0,"Benennung (bereinigt)"] = random_order(df_new.loc[0,"Benennung (bereinigt)"])
                if df_temp.loc[0,"volume"] > 0:
                    df_temp = augmented_boundingbox(df_new, df_temp)  
                    df = pd.concat([df, df_temp]).reset_index(drop=True)   

                df_temp.loc[0,"Benennung (bereinigt)"] = gpt35_designation(df_new.loc[0,"Benennung (bereinigt)"])
                if len(df_temp["Benennung (bereinigt)"][0]) < 40 and df_temp.loc[0,"volume"] > 0:
                    df_temp = augmented_boundingbox(df_new, df_temp)
                    df = pd.concat([df, df_temp], ignore_index=True).reset_index(drop=True)

    return df
