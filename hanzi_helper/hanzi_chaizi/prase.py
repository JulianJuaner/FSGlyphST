import pickle

data = {}

# load data from chaizi dataset.
with open('hanzi_helper/hanzi_chaizi/chaizi-jt.txt', 'rt') as fd:
    for line in fd:
        item_list = line.strip().split('〖')
        # print(item_list)
        for item in item_list:
            if '〗' in item:
                chaizi_res = item.split(' ')
                key = chaizi_res[-1][0]
                value = chaizi_res[:-1]
                if key in data.keys():
                    data[key] = value

with open('hanzi_helper/hanzi_chaizi/chaizi_origin-jt.txt', 'rt') as fd:    
    for line in fd:
        item_list = line.strip().split('\t')
        key = item_list[0]
        value = [i.strip().split() for i in item_list[1:]]
        data[key] = value

output_file = 'hanzi_helper/hanzi_chaizi/hanzi_chaizi/data/data.pkl'

value_count = {}
# for key in data.keys():
#     print(data[key])
#     if len(data[key])

with open(output_file, 'wb') as fd:
    pickle.dump(data, fd)

