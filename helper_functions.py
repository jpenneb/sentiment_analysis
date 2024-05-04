import argparse
import csv
from datasets import Dataset
from datetime import datetime
from dotenv import load_dotenv
import json
from nltk.tokenize import sent_tokenize
import openai
import os
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

load_dotenv()
with open("config.json", "r") as config_file:
    config = json.load(config_file)

TRANSFORMER_PIPELINE = config["TRANSFORMER_PIPELINE"]
TRANSFORMER_MODEL = config["TRANSFORMER_MODEL"]
MODEL_KEYWORD = config["MODEL_KEYWORD"]
EARNINGS_CALL_TRANSCRIPTS_DIRECTORY = config["EARNINGS_CALL_TRANSCRIPTS_DIRECTORY"]
OVERVIEW_FILE = config["OVERVIEW_FILE"]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AIERA_API_KEY = os.getenv("AIERA_API_KEY")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some financial data.')
    parser.add_argument('--start_date', type=str, default='2023-01-01', help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, default=datetime.today().strftime('%Y-%m-%d'), help='End date in YYYY-MM-DD format')
    parser.add_argument('--tickers_list', nargs='+', default=['GPS'], help='List of ticker symbols separated by space')
    parser.add_argument('--topic', type=str, default='margin', help='Key drivers topic, such as "margin" or "inflation"')
    parser.add_argument('--training_document_type', type=str, default='earnings call transcripts', help='Type of document for creating synthetic data for training')
    parser.add_argument('--base_model_for_training', type=str, default='roberta-base', help='Pretrained model for fine tuning on synthetic data, such as roberta-base')
    args = parser.parse_args()
    return args

def initialize_overview_file(filepath):
    if not os.path.isfile(filepath):
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Company Ticker', 'Date', 'Sentiment Score'])

def tokenize_text_into_sentences_and_filter_by_keyword(text):
    sentences = sent_tokenize(text)
    keyword_sentences = [sentence for sentence in sentences if MODEL_KEYWORD in sentence.lower()]
    return keyword_sentences

def extract_date_and_text_from_api(ticker, quarter, year):
    url = f"https://discountingcashflows.com/api/transcript/{ticker}/{quarter}/{year}/"
    response = requests.get(url).json()
    date = response[0]['date']
    text = response[0]['content']
    return date, text

def analyze_results_from_pipeline_on_text_segments(pipeline, text_segments):
    analysis_results = []
    for text_segment in text_segments:
        sentiment_result = pipeline(text_segment)
        analysis_results.append({
        "text": text_segment,
        "sentiment": sentiment_result[0]['label'],
        "confidence": sentiment_result[0]['score']
    })
    return analysis_results

def generate_response_via_llm(OPENAI_API_KEY, analysis_results):
    openai.api_key = OPENAI_API_KEY
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"Summarize the following sentiments related to 'margin' in the document: {analysis_results}",
            }
        ]
    )
    return response

def create_csv_of_results(file, results_json):
    df = pd.DataFrame(results_json['extracts'])
    for header in results_json:
        df[header] = results_json[header]
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    df.to_csv(f'outputs/{file.split(".pdf")[0]}_output.csv', index=False)

def correct_analysis_results(results, topic):
    i = 0
    for sentence in results:
        if sentence['sentiment'] != 'LABEL_0':
            if topic not in sentence['text']:
                prompt_text = f'''
                    You are a financial analyst specializing in analyzing {topic} sentiment as it pertains to key drivers of financial performance mentioned in financial documents. You are tasked with evaluating each "text" and corresponding "sentiment" label in the following JSON and identify and fix any incorrect sentiment labeling. The sentiment should be neutral (LABEL_0) unless the text is clearly related to {topic} commentary driving key financial performance.
                    For example:
                    {{"text": "Good afternoon", "sentiment": "LABEL_1", "confidence: "0.99987445"}}
                    Should be:
                     {{"text": "Good afternoon", "sentiment": "LABEL_0", "confidence: "0.99987445"}}
                     Since this sentence has nothing to do with sentiment related to financial key drivers for {topic}. Your fix will be based on relevance to key financial performance drivers based on the text and the assigned sentiment label. You understand the following mapping of sentiment labels: LABEL_0 is neutral, LABEL_1 is positive, LABEL_2 is negative.
                    JSON for fixing:
                    {sentence}
                    Fixed JSON:
                    '''
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    response_format={"type": "json_object"},
                    messages=[
                                {
                                    "role": "system",
                                    "content": prompt_text
                                }
                            ],
                    max_tokens=4096,
                    n=1
                )
                try:
                    results[i] = json.loads(response.choices[0].message.content)
                except:
                    print(response.choices[0].finish_reason)
        i += 1
    return results

def calculate_document_sentiment_score(results_json):
    num_positive = sum(result['sentiment'].lower() in ('positive', 'label_1') for result in results_json)
    num_negative = sum(result['sentiment'].lower() in ('negative', 'label_2') for result in results_json)
    return (num_positive - num_negative) / (num_positive + num_negative + 1)

def create_csv_for_overview_analysis(OVERVIEW_FILE, json_of_results):
    with open(OVERVIEW_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([json_of_results["company_ticker"], json_of_results["date"], json_of_results["sentiment_score"]])

def generate_synthetic_data(topic, document_type, sentiment, type, n_entries):
    sentiment_descriptions = {
        "pos": f"clearly positive, relevant as a 'key driver' of financial growth or success, such as 'We noted solid {topic}' or 'The initiative led to a 15% increase in annual {topic}s.'",
        "neg": "explicitly negative, in the context of a 'key driver' of financial decline or challenges, such as 'This quarter was not so great for {topic}' or 'The project resulted in a 20% decrease in annual {topic} compared to last year.'",
        "neutral": "either completely unrelated to any financial topics, like 'The meeting started at 3 PM.', or indirectly related but without any sentiment, such as 'The CEO discussed new office locations.'"
    }

    types = {
        "quantitative": f"i.e. reporting on numeric values and how they compare to previous reports, such as '{topic} was $3.5M, representing a Y/Y increase of 5%.'",
        "qualitative": f"i.e. without numeric metrics, such as '{topic} is doing well.'"
    }

    prompt_text =  f'''
    You are tasked with creating realistic and believable synthetic data intended for fine tuning a pretrained roBERTa model, which will be used to label sentiment for setences in {document_type}.

    The synthetic data must be as diverse and as varied as possible as well as focused on being {type}, {types[type]}. And it's important that for neutral examples, you include some completely irrelant sentences that are not related to {topic} at all.

    Generate exactly {n_entries} entries of only {sentiment_descriptions[sentiment]} sentences about {topic} as would be found in {document_type}.
    
    Each entry should be structured as a JSON with "sentence" and "sentiment" keys, where the value of "sentiment" is {sentiment}. Do not include the entry numbers, only the dictionary structure.
    Example:
    {{
        "data":
            [
                {{
                    "sentence": "This is an example of a {sentiment_descriptions[sentiment]} sentence about {topic} found in {document_type}.",
                    "sentiment": "{sentiment}"
                }},
                ...
                {{
                    "sentence": "This is an example of the nth {sentiment_descriptions[sentiment]} sentence about {topic} found in {document_type}.",
                    "sentiment": "{sentiment}"
                }}
            ]
    }}

    Your response, structured as a dictionary:
    '''
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={"type": "json_object"},
        messages=[
                    {
                        "role": "system",
                        "content": prompt_text
                    }
                ],
        max_tokens=4096,
        n=1
    )
    print(response.choices[0].message.content)
    return json.loads(response.choices[0].message.content)

def generate_synthetic_data_for_topic_on_document_type(topic, document_type):
    data = {"data": []}
    iterations = 15
    for sentiment in ['pos', 'neg', 'neutral']:
        for type in ['quantitative', 'qualitative']:
            for i in list(range(iterations)):
                print(f"Processing {sentiment} {type} batch {i+1} of {iterations}.")
                data['data'].extend(generate_synthetic_data(topic, document_type, sentiment, type, 50)['data'])
    return data

def train_base_model_on_topic_for_document_type_with_synthetic_data(topic, document_type, base_model):
    data = generate_synthetic_data_for_topic_on_document_type(topic, document_type)
    polarity_mapping = {
        'pos': 1,
        'neg': 2,
        'neutral': 0
    }

    data_df = pd.DataFrame(data['data'])

    data_df['sentiment'] = data_df['sentiment'].map(polarity_mapping)

    data_df.to_csv("synthetic_training_data.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        data_df['sentence'],
        data_df['sentiment'],
        test_size=0.2,
        random_state=42
    )

    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=3) # 3 corresponds to 3 sentiment classes: pos, neg, neutral
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    train_df = pd.concat([X_train, y_train.rename('labels')], axis=1)
    test_df = pd.concat([X_test, y_test.rename('labels')], axis=1)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    def tokenize(examples):
        return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)

    tokenized_train_dataset = train_dataset.map(tokenize, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=f"./training/{topic}_on_{document_type.replace(' ', '_')}_model/fine_tuned/{base_model}_on_synthetic_data/results",          # The output directory for the model predictions and checkpoints
        num_train_epochs=3,              # Total number of training epochs
        per_device_train_batch_size=8,   # Batch size per device during training
        per_device_eval_batch_size=8,    # Batch size for evaluation
        warmup_steps=500,                # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # Strength of weight decay
        logging_dir="./logs",            # Directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",     # Evaluation is done at the end of each epoch
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
    )
    trainer.train()
    trainer.evaluate()
    model.save_pretrained(f"./training/{topic}_on_{document_type.replace(' ', '_')}_model/fine_tuned/{base_model}_on_synthetic_data/")
    tokenizer.save_pretrained(f"./training/{topic}_on_{document_type.replace(' ', '_')}_model/fine_tuned/{base_model}_on_synthetic_data/")

def analyze_text_and_create_output(ticker, date, text_for_pipeline_analysis, model_name, model, topic):
    analysis_results = analyze_results_from_pipeline_on_text_segments(pipeline(TRANSFORMER_PIPELINE, model=model), text_for_pipeline_analysis)
    corrected_analysis_results = correct_analysis_results(analysis_results, topic)
    sentiment_score = calculate_document_sentiment_score(corrected_analysis_results)
    json_of_results = {
            "company_ticker": f"{model_name}_{ticker}",
            "date": date,
            "sentiment_score": sentiment_score,
            "extracts": analysis_results
            # "summary": "response_via_llm(OPENAI_API_KEY, analysis_results).choices[0].message.content"
        }
    create_csv_of_results(f"{model_name}_{ticker}_{date}", json_of_results)
    create_csv_for_overview_analysis(OVERVIEW_FILE, json_of_results)

def analyze_text_and_create_output_for_on_the_fly_model(ticker, date, text, topic, document_type, base_model):
    text_for_pipeline_analysis = tokenize_text_into_sentences_and_filter_by_keyword(text)
    model_name = f"{base_model}_fine_tuned_on_synthetic_{document_type.replace(' ', '_')}_{topic}_data"
    model = f"./training/{topic}_on_{document_type.replace(' ', '_')}_model/fine_tuned/{base_model}_on_synthetic_data/"
    analyze_text_and_create_output(ticker, date, text_for_pipeline_analysis, model_name, model, topic)

def analyze_text_and_create_output_for_list_of_fine_tuned_models(ticker, date, text):
    text_for_pipeline_analysis = tokenize_text_into_sentences_and_filter_by_keyword(text)
    for model in TRANSFORMER_MODEL:
        analyze_text_and_create_output(ticker, date, text_for_pipeline_analysis, model, TRANSFORMER_MODEL[model])
