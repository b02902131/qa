#! /usr/bin/env python   
# -*- coding: utf-8 -*- 

# import modules & set up logging
import gensim, logging
from gensim.models import word2vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import re, os
import numpy as np
import json
import jieba

from argparse import ArgumentParser

def loadTrainingSentences(args):
	with open(args.training_data_path) as data_file:    
		data = json.load(data_file)["data"]

	sentences = []

	for d in data:
		paragraphs = d['paragraphs']
		for paragraph in paragraphs:
			context = paragraph['context']
			sentences.append(context)

	return sentences

def DoJieba(args):
	sentences = loadTrainingSentences(args)

	# jieba custom setting.
	jieba.set_dictionary('jieba_dict/dict.txt.big')

	# load stopwords set
	# stopwordset = set()
	# with open('jieba_dict/stopwords.txt','r',encoding='utf-8') as sw:
	# 	for line in sw:
	# 		stopwordset.add(line.strip('\n'))

	output = open('training_seg.txt','w')

	texts_num = 0
    
	for line in sentences:
	    words = jieba.cut(line, cut_all=False)
	    for word in words:
	    	output.write(word +' ')
	    texts_num += 1
	    if texts_num % 10000 == 0:
	    	logging.info("已完成前 %d 行的斷詞" % texts_num)
	output.close()

def DoGensim(args):
	sentences = word2vec.Text8Corpus("training_seg.txt")
	model = word2vec.Word2Vec(sentences, size=args.embedding_dim, min_count=args.min_count)

	# Save our model.
	# model.save("med250.model.bin")

def UpdateGensim(args):
	model = gensim.models.Word2Vec.load(args.gensim_path)
	

if __name__ == "__main__":

	parser = ArgumentParser(description='gensim word2vec final qa')
	parser.add_argument('action', choices=['jieba', 'gensim'])

	# training arguments
	parser.add_argument('--training_data_path', default='data/train-v1.1.json')
	parser.add_argument('--gensim_path', default='med250.model.bin')

	# word2vec arguments
	parser.add_argument('-emb_dim', '--embedding_dim', default=128, type=int)
	parser.add_argument('-min_count', '--min_count', default=1,type=int)

	parser.add_argument('--onDeepQ', default=False, type=bool)
	parser.add_argument('--fastMode', default=False, type=bool)
	args = parser.parse_args()

	if args.action == 'jieba':
		DoJieba(args)
	elif args.action == 'gensim':
		DoGensim(args)
	


