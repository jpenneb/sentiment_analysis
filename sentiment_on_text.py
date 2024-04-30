from datetime import datetime
from helper_functions import OVERVIEW_FILE, analyze_text_and_create_output_for_on_the_fly_model, extract_date_and_text_from_api, initialize_overview_file, parse_arguments, train_base_model_on_topic_for_document_type_with_synthetic_data
import os
import time


def process_documents():
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

    start_year = start_date_obj.year

    if not os.path.isdir(f"training/{topic}_on_{training_document_type.replace(' ', '_')}_model/fine_tuned/{base_model_for_training}_on_synthetic_data/"):
        train_base_model_on_topic_for_document_type_with_synthetic_data(topic, training_document_type, base_model_for_training)
    
    for ticker in tickers_list:
        end_year = end_date_obj.year
        while end_year >= start_year:
            for quarter in ["Q1", "Q2", "Q3", "Q4"]:
                date, text = extract_date_and_text_from_api(ticker, f"{quarter}", f"{end_year}")
                analyze_text_and_create_output_for_on_the_fly_model(ticker, date, text, topic, training_document_type, base_model_for_training)
            end_year -= 1

def main():
    initialize_overview_file(OVERVIEW_FILE)
    process_documents()

if __name__ == "__main__":
    start_time = time.time()

    args = parse_arguments()
    tickers_list = args.tickers_list
    start_date = args.start_date
    end_date = args.end_date
    topic = args.topic
    training_document_type = args.training_document_type
    base_model_for_training = args.base_model_for_training

    main()
    print(f"{time.time() - start_time} seconds")
