
from hanzi_helper.hanzi_feat.hanzi_char_featurizer import Featurizor, featurize_as_tensor
from hanzi_helper.hanzi_chaizi.hanzi_chaizi import HanziChaizi
from sourcecode.utils.font_set import CHINESE_CHAR

# 横折（𠃍）、横撇（㇇）、横钩（乛）、横折钩（𠃌）、横折提（㇊）、横折弯（㇍）、
# 横折折（㇅）、横斜钩（⺄）、横折弯钩（㇈）、横撇弯钩（㇌）、横折折撇（㇋）、
# 横折折折钩（𠄎）、横折折折（㇎）、竖提（𠄌）、竖折（𠃊）、竖钩（亅）、竖弯（㇄）、
# 竖弯钩（乚）、竖折撇（ㄣ）、竖折折（𠃑）、竖折折钩（㇉）、撇点（𡿨）、撇折（𠃋）、
# 斜钩（㇂）、弯钩（㇁）

basic_components = [
    '丶', '乁', '辶', '丩', '乚', '乀', '龴', '㇆', 
    '𠄌', '𠃊', '亅', '丿', '㔾', '乛', '𠃋', '㇄',
    '㇉', '匚', '匸', '丨', '冂', '一', '㇏', '㇀', 
    '𠄎', 'ㄣ', '㇍', '𡿨', '㇌', '㇎', '⺄',
    ]

comp_of_basic_components = set()
featurizor = Featurizor()
chaizi = HanziChaizi()
sub_component_set = set()
component_set = set()
flag = 0
for i in range(len(CHINESE_CHAR)):
    # print(CHINESE_CHAR[i], i)
    # result = featurizor.featurize(CHINESE_CHAR[i])
    chaizi_result = chaizi.query(CHINESE_CHAR[i])
    
    # print(result)
    if chaizi_result is not None:
        form_basic = 0

        for component in chaizi_result[-1]:
            if len(chaizi_result[-1]) == 1:
                comp_of_basic_components.add(component)
            elif component != CHINESE_CHAR[i]:
                if component in basic_components:
                    form_basic += 1
                sub_component_set.add(component)
        if form_basic == len(chaizi_result[-1]):
            comp_of_basic_components.add(CHINESE_CHAR[i])

count = 0
while True:
    count += 1
    for sub_comp in sub_component_set:
        chaizi_result = chaizi.query(sub_comp)
        if chaizi_result is not None and len(chaizi_result)!= 0:
            form_basic = 0
            for component in chaizi_result[-1]:
                if len(chaizi_result[-1]) == 1:
                    comp_of_basic_components.add(component)
                else:
                    if component != sub_comp:
                        component_set.add(component)
                    if component in basic_components:
                        form_basic += 1
    
            if form_basic == len(chaizi_result[-1]):
                comp_of_basic_components.add(sub_comp)
        else:
            component_set.add(sub_comp)
    if len(component_set) == len(sub_component_set):
        break
    sub_component_set = component_set
    component_set = set()

component_set |= comp_of_basic_components
print(component_set)
print(len(component_set), len(sub_component_set), count)
component_set = list(component_set)
import json
from tqdm import tqdm
out_file = 'data/component_feat.json'
with open('data/key_value.json') as f:
    key_value_pair = json.load(f)

zi_feat = dict()
for i in tqdm(range(len(CHINESE_CHAR))):
    # print(CHINESE_CHAR[i])
    four_corner_result = featurizor.featurize(CHINESE_CHAR[i])[3:]
    
    four_corner = [str(int(_[0])) for _ in four_corner_result]
    char_num = str(key_value_pair[str(ord(CHINESE_CHAR[i]))])
    zi_feat[char_num] = dict()
    zi_feat[char_num]['four_corner'] = four_corner
    zi_feat[char_num]['char'] = CHINESE_CHAR[i] + ' ' + str(ord(CHINESE_CHAR[i]))
    zi_feat[char_num]['component'] = [0 for _ in range(len(component_set))]
    query_in = CHINESE_CHAR[i]
    if query_in in component_set:
        zi_feat[char_num]['component'][component_set.index(query_in)] += 1

    # implement a stack.
    else:
        remaining_components = [query_in]
        while len(remaining_components) > 0:
            query_in = remaining_components[0]
            remaining_components = remaining_components[1:]
            # print(remaining_components, query_in)
            chaizi_result = chaizi.query(query_in)
            
            if chaizi_result is not None and len(chaizi_result)!= 0:
                for component in chaizi_result[-1]:
                    if component in component_set:
                        zi_feat[char_num]['component'][component_set.index(component)] += 1
                    else:
                        remaining_components.append(component)
            else:
                if query_in in component_set:
                        zi_feat[char_num]['component'][component_set.index(query_in)] += 1
                else:
                    raise RunTimeError

with open(out_file, 'w') as out_:
    json.dump(zi_feat, out_)

sub_components = ['氏', '凹', '乙', '予', '心', '飠', '頁', '扌', '离', '亡', '才', '尤', '厂', '丶', '並', '爭', '㇆', '廴', '巾', '乃',
                '爿', '凵', '儿', '长', '匚', '冊', '卜', '𠄌', '乡', '不', '豸', '十', '几', '刁', '乑', '力', '卩', '亻', '角', 
                '刂', '𠃋', '马', '丌', '了', '冂', '\uf7ee', '乂', '丿', '匸', '辶', '乚', '巸', '刀', '巨', '艮', '入', '冫', '书', 
                '舟', '𠔉', '勻', '丹', '丅', '丫', '瓦', '已', '女', '㠯', '上', '㡀', '骨', '弓', '又', '亠', '彡', '亅', '己', '曰',
                '而', '冖', '尢', '廾', '耳', '丄', '氵', '夃', '丁', '㇉', '𠂢', '乀', '㐆', '㔾', '爫', '灬', '见', '民', '巛', '厶',
                '戉', '川', '月', '与', '夕', '匕', '茲', '彑', '八', '门', '戈', '龴', '犭', '丩', '三', '丷', '人', '丨', '二', 
                '癶', '一', '尸', '七', '丑', '旡', '忄', '阝', '口', '龹', '工', '弋', '下', '㓁', '阜', '纟', '龷', '囗', '𠃊', '水',
                '勹', '乛', '夂', '⺆', '艹', '㐬', '寸', '竹', '彐', '𢦏', '丽', '巳', '雨', '之', '凸', '广', '乁', '豕', '为']

out_file = 'data/component_vector.txt'
with open(out_file, 'w') as fd:
    for key in zi_feat.keys():
        four_corner = ' '
        four_corner = four_corner.join(zi_feat[key]['four_corner'])
        string_list = [str(i) for i in zi_feat[key]['component']]
        component = ' '.join(string_list)
        fd.write(key + " " + four_corner + " "+ component + '\n')

# write into glyph generation.

# original glyph vector representation.
# import tensorflow as tf
# feature = featurize_as_tensor('data/data_text.txt')
# out_file = 'data/glyph_onehot.txt'

# import json
# with open('data/key_value.json') as f:
#   key_value_pair = json.load(f)

# with open(out_file, 'w') as fd:
#     with tf.Session() as sess:
#         sess.run(tf.initializers.tables_initializer())
#         for i in range(3500):
#             char_num = key_value_pair[str(ord(CHINESE_CHAR[i]))]
#             print(char_num)
#             data = sess.run(feature)
#             if i == 1:
#                 print(len(data[0]))
#             data = str(data[0])[1:-1]
#             data = data.replace('\n', '').replace('.', '')
#             fd.write(str(char_num) + " " + data + '\n')