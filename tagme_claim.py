import tagme
import json
from tqdm import tqdm
import re
import sys
import os
# Set the authorization token for subsequent calls.
tagme.GCUBE_TOKEN = "6b362171-ff34-4097-a1c2-df426a5b7452-843339462"

def match(input_string):
    pattern = r'(.+?) -> (.+?) \(score: (.+?)\)'
    matches = re.match(pattern, input_string)

    if matches:
        original_name = matches.group(1)
        normalized_name = matches.group(2)
        score = matches.group(3)

        # print("Original Name:", original_name)
        # print("Normalized Name:", normalized_name)
        # print("Score:", score)
        return (original_name, normalized_name, score)
    else:
        # print("No match found.")
        return None


def tageme_process(input_file, output_file):

    with open(input_file, 'r') as rf, open(output_file, 'w') as wf:
        for line in tqdm(rf.readlines()):
            data= json.loads(line)
            # Print annotations with a score higher than 0.1
            id2splits = dict()
            item_id, claim = data['id'], data['claim']

            splits = []
            try:
                lunch_annotations = tagme.annotate(claim)
                for ann in lunch_annotations.get_annotations(0.1):
                    mention, entity, score = match(str(ann))
                    # print(mention, " >> ", entity, score)
                    splits.append([mention, entity, score])
            except:
                    print('error annotation about: ', item_id, claim)
            
            data['claim_tagme'] = splits
            wf.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    # input_file = '/root/autodl-tmp/KernelGAT/data/bert_dev.json'
    input_file = sys.argv[1]
    folder_name = os.path.dirname(input_file)
    file_name = "tagme_" + os.path.basename(input_file).replace(".json", ".jsonl")
    output_file = os.path.join(folder_name, file_name)
    print("output_file: ", output_file)
    tageme_process(input_file, output_file)
    print("finish !!")