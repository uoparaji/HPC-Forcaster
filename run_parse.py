def runParseRawData():
	from parse import parse_raw_data
	parse_raw_data("CFMS-2016jul-2018jun_2160.csv")

def runParseRawData():
	from main_code import split_original_parsed_dataset
	split_original_parsed_dataset(number_weeks_training=80, number_test_sets=9, "parsed_data_size=211120_weeks=104.csv")