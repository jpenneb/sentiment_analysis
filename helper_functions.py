import argparse
import csv
from datasets import load_dataset, DatasetDict, Dataset
from datetime import datetime
from dotenv import load_dotenv
import fitz
import json
import nltk
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
    parser.add_argument('--document_source', type=str, default='discountingcashflows', help='Source of document, either Aiera (aiera), Discounting Cashflows (discountingcashflows), or PDFs (pdfs)')
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

def extract_text_from_pdf_and_tokenize_as_sentences_and_filter_by_keyword(path):
    with fitz.open(path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return tokenize_text_into_sentences_and_filter_by_keyword(text)

def extract_date_and_text_from_discounting_cashflow_api_call(ticker, quarter, year):
    url = f"https://discountingcashflows.com/api/transcript/{ticker}/{quarter}/{year}/"
    response = requests.get(url).json()
    date = response[0]['date']
    text = response[0]['content']
    return date, text

def extract_date_and_text_from_airea_api_call(ticker, start_date, end_date): #TODO: This returns only the last transcript in the time range...
    url = "https://premium.aiera.com/api/events/"
    params = {"start_date": start_date, "end_date": end_date, "ticker": ticker}
    headers = {"X-API-Key": f"{AIERA_API_KEY}"}

    events = requests.get(url, params=params, headers=headers).json()
    for event in events:
        if event["event_type"] == "earnings":
            date = event["event_date"]
            text = ""
            for item in requests.get(f"{url}{event['event_id']}/transcript", headers=headers).json():
                text += item["transcript"] + "\n"
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

def create_csv_of_results(file, json):
    df = pd.DataFrame(json['extracts'])
    for header in json:
        df[header] = json[header]
    df.to_csv(f'outputs/{file.split(".pdf")[0]}_output.csv', index=False)

def calculate_document_sentiment_score(results_json):
    num_positive = sum(result['sentiment'].lower() in ('positive', 'label_1') for result in results_json)
    num_negative = sum(result['sentiment'].lower() in ('negative', 'label_2') for result in results_json)
    return (num_positive - num_negative) / (num_positive + num_negative + 1)

def create_csv_for_overview_analysis(OVERVIEW_FILE, json_of_results):
    with open(OVERVIEW_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([json_of_results["company_ticker"], json_of_results["date"], json_of_results["sentiment_score"]])

def generate_synthetic_data(topic, document_type, sentiment, n_entries):
    sentiment_prompt = {
        "pos": "positive",
        "neg": "negative",
        "neutral": "a mix of neutral and irrelevant (such as 'Operator: Good afternoon, ladies and gentlemen')"
    }

    prompt_text = f'''
    You are tasked with creating synthetic data intended for training an encoder model, which will be used to label sentiment for setences in {document_type}.
    The synthetic data must be diverse, including an equal mix of qualititive commentary (i.e. lacking numeric metrics, such as "{topic} is doing well.") and quanititative commentary (i.e. reporting on numeric values and how they compare to previous reports, such as "{topic} was $3.5M, representing a Y/Y increase of 5%."). And it's important that for neutral examples, you include some completely irrelant sentences that are not related to {topic} at all.

    Generate exactly {n_entries} entries of only {sentiment_prompt[sentiment]} financial commentary sentences about {topic} as seen in {document_type}.
    Each entry should be structured as a JSON with "sentence" and "sentiment" keys, where the value of "sentiment" is {sentiment}. Do not include the entry numbers, only the dictionary structure.
    Example:
    {{
        "data":
            [
                {{
                    "sentence": "This is an example of a {sentiment_prompt[sentiment]} sentence about {topic} found in {document_type}.",
                    "sentiment": "{sentiment}"
                }},
                ...
                {{
                    "sentence": "This is an example of the nth {sentiment_prompt[sentiment]} sentence about {topic} found in {document_type}.",
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
    iterations = 20
    for sentiment in ['pos', 'neg', 'neutral']:
        for i in list(range(iterations)):
            print(f"Processing {sentiment} batch {i+1} of {iterations}.")
            data['data'].extend(generate_synthetic_data(topic, document_type, sentiment, 25)['data']) # TODO: Possibly "seed" with more generic neutral data
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

def analyze_text_and_create_output(ticker, date, text_for_pipeline_analysis, model_name, model):
    analysis_results = analyze_results_from_pipeline_on_text_segments(pipeline(TRANSFORMER_PIPELINE, model=model), text_for_pipeline_analysis)
    sentiment_score = calculate_document_sentiment_score(analysis_results)
    json_of_results = {
            "company_ticker": f"{model_name}_{ticker}",
            "date": date,
            "sentiment_score": sentiment_score,
            "extracts": analysis_results,
            "summary": "response_via_llm(OPENAI_API_KEY, analysis_results).choices[0].message.content"
        }
    create_csv_of_results(f"{model_name}_{ticker}_{date}", json_of_results)
    create_csv_for_overview_analysis(OVERVIEW_FILE, json_of_results)

def analyze_text_and_create_output_for_on_the_fly_model(ticker, date, text, topic, document_type, base_model):
    text_for_pipeline_analysis = tokenize_text_into_sentences_and_filter_by_keyword(text)
    model_name = f"{base_model}_fine_tuned_on_synthetic_{document_type.replace(' ', '_')}_{topic}_data"
    model = f"./training/{topic}_on_{document_type.replace(' ', '_')}_model/fine_tuned/{base_model}_on_synthetic_data/"
    analyze_text_and_create_output(ticker, date, text_for_pipeline_analysis, model_name, model)

def analyze_text_and_create_output_for_list_of_fine_tuned_models(ticker, date, text):
    text_for_pipeline_analysis = tokenize_text_into_sentences_and_filter_by_keyword(text)
    for model in TRANSFORMER_MODEL:
        analyze_text_and_create_output(ticker, date, text_for_pipeline_analysis, model, TRANSFORMER_MODEL[model])
