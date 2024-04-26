from datetime import datetime
from helper_functions import EARNINGS_CALL_TRANSCRIPTS_DIRECTORY, OVERVIEW_FILE, TRANSFORMER_MODEL, TRANSFORMER_PIPELINE, analyze_results_from_pipeline_on_text_segments, analyze_text_and_create_output_for_list_of_fine_tuned_models, analyze_text_and_create_output_for_on_the_fly_model, calculate_document_sentiment_score, create_csv_for_overview_analysis, create_csv_of_results, extract_date_and_text_from_airea_api_call, extract_date_and_text_from_discounting_cashflow_api_call, extract_text_from_pdf_and_tokenize_as_sentences_and_filter_by_keyword, initialize_overview_file, parse_arguments, train_base_model_on_topic_for_document_type_with_synthetic_data
import os
import re
import time
from transformers import pipeline


def process_discountingcashflows():
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

    start_year = start_date_obj.year

    #train_base_model_on_topic_for_document_type_with_synthetic_data(topic, training_document_type, base_model_for_training)
    print(tickers_list)
    for ticker in tickers_list:
        end_year = end_date_obj.year
        while end_year >= start_year:
            for quarter in ["Q1", "Q2", "Q3", "Q4"]:
                date, text = extract_date_and_text_from_discounting_cashflow_api_call(ticker, f"{quarter}", f"{end_year}")
                analyze_text_and_create_output_for_on_the_fly_model(ticker, date, text, topic, training_document_type, base_model_for_training)
            end_year -= 1

def process_aiera():
    for ticker in tickers_list:
        date, text = extract_date_and_text_from_airea_api_call(ticker, start_date, end_date)
        analyze_text_and_create_output_for_list_of_fine_tuned_models(ticker, date, text)

def process_pdfs():
    for file in os.listdir(EARNINGS_CALL_TRANSCRIPTS_DIRECTORY):
        file_path = os.path.join(EARNINGS_CALL_TRANSCRIPTS_DIRECTORY, file)
        if not os.path.isfile(file_path):
            continue
        text_for_pipeline_analysis = extract_text_from_pdf_and_tokenize_as_sentences_and_filter_by_keyword(file_path)
        analysis_results = analyze_results_from_pipeline_on_text_segments(pipeline(TRANSFORMER_PIPELINE, model=TRANSFORMER_MODEL["FINE_TUNED_BERT"]), text_for_pipeline_analysis)
        sentiment_score = calculate_document_sentiment_score(analysis_results)
        json_of_results = {
            "company_name": f'{file.split("(")[0]}',
            "company_ticker": re.search(r'\((.*?)\)', file).group(1),
            "date": re.search(r'(Q\d 20\d\d)', file).group(1),
            "sentiment_score": sentiment_score,
            "extracts": analysis_results,
            "summary": "response_via_llm(OPENAI_API_KEY, analysis_results).choices[0].message.content"
        }
        create_csv_of_results(file, json_of_results)
        create_csv_for_overview_analysis(OVERVIEW_FILE, json_of_results)

def main():
    initialize_overview_file(OVERVIEW_FILE)

    source_to_function_map = {
        "discountingcashflows": process_discountingcashflows,
        "aiera": process_aiera,
        "pdfs": process_pdfs
    }

    processing_function = source_to_function_map.get(document_source)

    if processing_function:
        processing_function()
    else:
        print(f"Unsupported document source: {document_source}")

if __name__ == "__main__":
    start_time = time.time()
    args = parse_arguments()
    tickers_list = args.tickers_list
    start_date = args.start_date
    end_date = args.end_date
    document_source = args.document_source
    topic = args.topic
    training_document_type = args.training_document_type
    base_model_for_training = args.base_model_for_training
    main()
    print(f"{time.time() - start_time} seconds")
