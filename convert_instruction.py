import json
import os

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data


if __name__ == '__main__':
    dataset = 'ICEWS14'
    data_path = './data/{0}'.format(dataset)
    train_json = read_json(os.path.join(data_path, '{0}_train.json'.format(dataset)))
    test_json = read_json(os.path.join(data_path, '{0}_test.json'.format(dataset)))
    train = []
    for i,item in enumerate(train_json):
        if 'index' not in item:
            item['index'] = i
        temp_json = {
            'id': item['index'],
            'conversations': [
                {
                    "from": "human",
                    "value": item['instruction'] + item['input']
                },
                {
                    "from": "assistant",
                    "value": item['output']
                }
            ],
            'graph_id': item['index']
        }
        train.append(temp_json)

    test = []
    for i, item in enumerate(test_json):
        if 'index' not in item:
            item['index'] = i
        temp_json = {
            'id': item['index'],
            'conversations': [
                {
                    "from": "human",
                    "value": item['instruction'] + item['input']
                },
                {
                    "from": "assistant",
                    "value": item['output']
                }
            ],
            'graph_id': item['index']
        }
        test.append(temp_json)

    with open('./data/{0}/train.json'.format(dataset), 'w') as f:
        json.dump(train, f)

    with open('./data/{0}/test.json'.format(dataset), 'w') as f:
        json.dump(test, f)


