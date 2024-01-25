import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-id', '--input-data',required=True)
parser.add_argument('-ig', '--input-gold',required=True)
parser.add_argument('-od', '--output-data',required=True)
parser.add_argument('-og', '--output-gold',required=True)

args = parser.parse_args()

if __name__=="__main__":
    input_data_path=args.input_data
    input_gold_path=args.input_gold
    output_data_path=args.output_data
    output_gold_path=args.output_gold

    # Read the data files and split the hypernyms such that we make one hyponym - hypernym pair each.
    input_data=pd.read_csv(input_data_path, header=None)
    input_gold = pd.read_csv(input_gold_path,
                  header=None)
    new_input_data=[]
    new_input_gold=[]
    for j in range(input_gold.shape[0]):
      split_hypernyms=input_gold[0][j].split('\t')
      for h in split_hypernyms:
        new_input_data.append(input_data[0][j].split('\t')[0])
        new_input_gold.append(h)
    new_input_data_df=pd.DataFrame(new_input_data)
    new_input_gold_df=pd.DataFrame(new_input_gold)
    new_input_data_df.to_csv(output_data_path,index=False,header=None)
    new_input_gold_df.to_csv(output_gold_path,index=False,header=None)