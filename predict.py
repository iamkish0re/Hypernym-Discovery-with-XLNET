import argparse
import pandas as pd
from transformers import XLNetTokenizer, XLNetLMHeadModel,XLNetTokenizerFast
import torch



# Puts the given hyponym through the model to get the logits
def predict_for_hyponym(model,tokenizer,hyponym,prefix_word):
    PADDING_TEXT=""""The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the remainder of the story.A pamphlet with photos and comments from the journals kept by the students added to the display. <eod> <eos>"""
    test_input_ids = torch.tensor(tokenizer.encode(PADDING_TEXT + " "+hyponym+" is a "+prefix_word+" <mask>", add_special_tokens=True,)).unsqueeze(0)
    targets = [-3]
    perm_mask = torch.zeros((1, test_input_ids.shape[1], test_input_ids.shape[1]), dtype=torch.float)
    perm_mask[0, :, targets] = 1.0  # Previous tokens don't see last token
    target_mapping = torch.zeros((1, len(targets), test_input_ids.shape[1]), dtype=torch.float)
    target_mapping[0, 0, targets[0]] = 1.0  # Our first  prediction
    
    test_input_ids_tensor = test_input_ids.to("cuda")
    target_mapping_tensor = target_mapping.to("cuda")
    perm_mask_tensor = perm_mask.to("cuda")
    
    model.eval()
    if torch.cuda.is_available(): model.to('cuda')
    with torch.no_grad():
      outputs = model(test_input_ids_tensor, perm_mask=perm_mask_tensor, target_mapping=target_mapping_tensor)
    return outputs


# function that get the hypernyms for the hyponym based on the vocab list. This generates multiple woord hypernyms by querying the model.
def get_output_preditions(model,tokenizer,hyponym,vocab_list,k):
    word_prob_dict={}
    n=2
    next_words_to_consider=[""]
    words_seen_so_far=set()
    words_seen_so_far.add(hyponym)
    while(True):
        if len(next_words_to_consider)==0:
            break
        extra_word=next_words_to_consider.pop(0)
        if(len(extra_word.strip().split(" "))>n):
            break
        outputs=predict_for_hyponym(model,tokenizer,hyponym,extra_word)
        m=torch.nn.Softmax(dim=0)
        predicted_k_indexes = torch.topk(outputs[0][0][0],k=k)
        probability_list=m(predicted_k_indexes[0])
        predicted_indexes_list = predicted_k_indexes[1]
        
        for i,item  in enumerate(predicted_indexes_list):
            the_index = predicted_indexes_list[i].item()
            next_word=tokenizer.decode(the_index)
            # print("word and logits",next_word,predicted_logits_list[i].item(),probability_list[i].item())
            if(next_word not in extra_word.strip().split(" ") and (len((extra_word+" "+next_word).strip().split(" "))<=n)) and ((extra_word+" "+next_word).strip() in vocab_list):
                next_words_to_consider.append((extra_word+" "+next_word).strip())
                if(len(extra_word)>0):
                    word_prob_dict[(extra_word+" "+next_word).strip()]= word_prob_dict[(extra_word).strip()] * probability_list[i].item()
                else:
                    word_prob_dict[(extra_word+" "+next_word).strip()]= probability_list[i].item()
                    
                words_seen_so_far.add(next_word)
    return sorted(word_prob_dict.items(), key=lambda x:x[1],reverse=True)


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',required=True)
parser.add_argument('-i', '--input-file',required=True)
parser.add_argument('-o', '--output-file',required=True)  
parser.add_argument('-v', '--vocab-file',required=True)  

args = parser.parse_args()

if __name__=="__main__":

    model_path=args.model
    input_path=args.input_file
    output_path=args.output_file
    vocab_path=args.vocab_file

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=torch.load(model_path)

    vocab=pd.read_table(vocab_path,header=None,index_col=False)
    vocab_list=vocab.values
    vocab_list={x[0] for x in vocab_list.tolist()}

    if(input_path is not None and len(input_path)>0):
        data_file_path=input_path
        outfile=open(output_path,"w")
        print(data_file_path)
        input_data=pd.read_csv(data_file_path, header=None)
        new_input_data=[]
        for i in range(input_data.shape[0]):
            if(i%50==0):
                print(str(i)+"/"+str(input_data.shape[0])+" is done.")
            input_hyponym=input_data[0][i].split('\t')[0]
            preds=get_output_preditions(model,tokenizer,input_hyponym,vocab_list,10)
            printstr=""
            for hypernym_prob in preds:
                printstr=printstr+hypernym_prob[0]+"\t"
            outfile.writelines(printstr+"\n")
        
