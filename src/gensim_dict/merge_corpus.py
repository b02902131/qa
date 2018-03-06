# -*- coding: utf-8 -*-

import jieba
import logging

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # jieba custom setting.
    jieba.set_dictionary('jieba_dict/dict.txt.big')

    # load stopwords set
    # stopwordset = set()
    # with open('jieba_dict/stopwords.txt','r',encoding='utf-8') as sw:
    #     for line in sw:
    #         stopwordset.add(line.strip('\n'))

    texts_num = 0

    output = open('wiki_qa_seg.txt','w')
    copy_txt_list = ['training_seg.txt', 'wiki_seg.txt']    
    for filename in copy_txt_list:
        with open(filename,'r') as content :
            logging.info("複製 %s 中的內容" % filename)
            for line in content:
                output.write(line)
                texts_num += 1
                if texts_num % 10000 == 0:
                    logging.info("已完成前 %d 行的複製" % texts_num)


    output.close()

if __name__ == '__main__':
    main()
