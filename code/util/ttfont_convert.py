'''
    References: 
    https://blog.csdn.net/JineD/article/details/106622398,
    https://blog.csdn.net/mouday/article/details/81512870
'''

from PIL import Image, ImageDraw, ImageFont
import sys
# sys.path[0] = "."
print(sys.path)
from font_set import WESTERN_CHAR, CHINESE_CHAR
import os

DIR = './data/TTFfont/'
OUT_DIR = './data/PNGfont/'
OUT_CHINESE_DIR = './data/PNGfont_cn/'
PNG_SIZE = 256

def is_contain_chinese(check_str):
    """
    判断字符串中是否包含中文
    :param check_str: {str} 需要检测的字符串
    :return: {bool} 包含返回True， 不包含返回False
    """
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def draw_png(name, font_size = 128, extension='.ttf'):
    font = ImageFont.truetype(DIR + name + extension, font_size)
    max_width, max_height = (PNG_SIZE, PNG_SIZE)
    print('Start processing font {} to PNG file.'.format(name))
    os.makedirs(os.path.join(OUT_DIR, name), exist_ok=True)


    for char in WESTERN_CHAR:
        text_width, text_height = font.getsize(char)
        image = Image.new(mode='RGBA', size=(max_width, max_height))
        padding = ((max_width-text_width)//2, (max_height-text_height)//2)
        draw_table = ImageDraw.Draw(im=image)
        draw_table.text(xy=padding, text=char, fill='#000000', font=font)
        image.save(os.path.join(OUT_DIR, name) + '/' + str(ord(char)) + '.png', 'PNG')  
        image.close()
    print(is_contain_chinese(name), name)
    if is_contain_chinese(name):
        os.makedirs(os.path.join(OUT_CHINESE_DIR, name), exist_ok=True)
        for char in CHINESE_CHAR:
            text_width, text_height = font.getsize(char)
            image = Image.new(mode='RGBA', size=(max_width, max_height))
            padding = ((max_width-text_width)//2, (max_height-text_height)//2)
            draw_table = ImageDraw.Draw(im=image)
            draw_table.text(xy=padding, text=char, fill='#000000', font=font)
            image.save(os.path.join(OUT_CHINESE_DIR, name) + '/' + str(ord(char)) + '.png', 'PNG')  
            image.close()

 
if __name__ == "__main__":
    for name in os.listdir(DIR):
        try:
            name = name.split('.')[0]
            draw_png(name, extension='.ttf')
        except Exception as e:
            print(name, ' ERR: ', e)
            continue