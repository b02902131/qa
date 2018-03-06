from gensim.models import word2vec
import logging

from datetime import datetime, date
from argparse import ArgumentParser

import numpy as np

from itertools import chain

def main(args):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(args.train_txt)
    # sentences_training_data = word2vec.Text8Corpus("training_seg.txt")

    model = word2vec.Word2Vec(sentences, size=args.embedding_dim, min_count=args.min_count)

    # model.train(sentences)

    #保存模型，供日後使用
    if args.only_wv:
    	dt_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    	np.save("wv_"+dt_now+"_.npy", model.wv)
    	print("model saved at: "+"wv_"+dt_now+"_.npy")
    else:
	    dt_now = datetime.now().strftime("%Y%m%d_%H%M%S")
	    model.save("med250_"+dt_now+"_.model.bin")


    #模型讀取方式
    # model = word2vec.Word2Vec.load("your_model.bin")

if __name__ == "__main__":
	parser = ArgumentParser(description='gensim word2vec final qa')

	# word2vec arguments
	parser.add_argument('--train_txt',default="wiki_qa_seg.txt")
	parser.add_argument('-emb_dim', '--embedding_dim', default=256, type=int)
	parser.add_argument('-min_count', '--min_count', default=1,type=int)
	parser.add_argument('--only_wv', default=True, type=bool)

	args = parser.parse_args()

	main(args)
