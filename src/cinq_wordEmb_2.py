import numpy as np
import pandas as pd

import sys, os, re, json

from argparse import ArgumentParser

from utils.load import loadTrainingData, loadTestingData, cleanSentence, answer_convert, one_hot_encoding, one_hot_decoding, output_predicts, randomize_dataset, split_by_valid_ratio
from utils.compare_CQ import context_compare_vector, context_compare_vector_level, context_compare_punctuation_vector, context_punctuation_vector_all_level, context_cinq_vector_level
from utils.evaluate import mean_f1, my_f1_score, mean_f1_2, mean_f1_3
from utils.word_embed import load_char2idx, sentence_encode_charLevel, load_gensim, sentence_encode_wordLevel

from models.cinq_wordEmb_model import cinq_wordEmb_start_model, cinq_wordEmb_end_model, cinq_wordEmb_model_2

from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback


post_fix = "_1228_punctuation2.h5"
start_model_path = "model/start_model"+post_fix
end_model_path =  "model/end_model"+post_fix

import json
import jieba, gensim
# jieba custom setting.
jieba.set_dictionary('jieba_dict/dict.txt.big')

# load stopwords set
stopwordset = set()
with open('jieba_dict/stopwords.txt','r',encoding='utf-8') as sw:
	for line in sw:
		stopwordset.add(line.strip('\n'))

def DoTrain(args):
	print ('Loading data from %s' %args.training_data_path)
	contexts, questions, answers, ids = loadTrainingData(args)
	answers_start, answers_end = answer_convert(contexts, answers)

	print ('Load word2idx, char2idx')
	word2idx, embeddings_matrix = load_gensim(wv_path=args.wv_path, wv_only=True)
	char2idx, idx2char = load_char2idx()
	args.char_size = len(char2idx)+1
	args.vocab_size = len(embeddings_matrix)
	args.embedding_dim = len(embeddings_matrix[0])

	print ('jieba cut ...')
	jieba_questions = [list(jieba.cut(question, cut_all=False)) for question in questions]
	jieba_contexts = [list(jieba.cut(context, cut_all=False)) for context in contexts]

	print ('Sentence encode --wordLevel...')
	questions_word = sentence_encode_wordLevel(questions, jieba_questions, word2idx, args.q_max_len)
	contexts_word = sentence_encode_wordLevel(contexts, jieba_contexts, word2idx, args.c_max_len)
	
	# print ('contexts_word.shape', contexts_word.shape)
	# print ('questions_word.shape', questions_word.shape)
	print ('questions[0]', questions[0])
	print ('jieba_questions[0]',jieba_questions[0])
	print ('questions_word[0]', questions_word[0])

	print ('Sentence encode --charLevel...')
	contexts_char = sentence_encode_charLevel(contexts, char2idx=char2idx, max_len=args.c_max_len)
	questions_char = sentence_encode_charLevel(questions, char2idx=char2idx, max_len=args.q_max_len)
	print ('contexts_char.shape', contexts_char.shape)
	print ('questions_char.shape', questions_char.shape)
	
	print ('Encoding the answers --one_hot...')
	answers_start = np.array([one_hot_encoding(x, dim=args.c_max_len) for x in answers_start])
	answers_end   = np.array([one_hot_encoding(x, dim=args.c_max_len) for x in answers_end])
	print(answers_start.shape)

	print ('Heuristic --comparing_CQ...')
	cinq_vector = context_cinq_vector_level(contexts, questions, args.c_max_len, args.context_vector_level)
	punctuation_vector = context_punctuation_vector_all_level(contexts, args.c_max_len)

	print ('Split validation data...')
	valid_cinq_vector, train_cinq_vector = split_by_valid_ratio(cinq_vector, args.val_ratio)
	valid_punctuation_vector, train_punctuation_vector = split_by_valid_ratio(punctuation_vector, args.val_ratio)
	valid_contexts_char	, train_contexts_char 	= split_by_valid_ratio(contexts_char, args.val_ratio)
	valid_questions_char, train_questions_char 	= split_by_valid_ratio(questions_char, args.val_ratio)
	valid_contexts_word, train_contexts_word	= split_by_valid_ratio(contexts_word, args.val_ratio)
	valid_questions_word, train_questions_word	= split_by_valid_ratio(questions_word, args.val_ratio)
	valid_answers_start	, train_answers_start 	= split_by_valid_ratio(answers_start, args.val_ratio)
	valid_answers_end	, train_answers_end 	= split_by_valid_ratio(answers_end, args.val_ratio)
	
	print ('Build up model for start point...')
	model = cinq_wordEmb_model_2(args, embeddings_matrix, isTeacherForcing=args.isTeacherForcing)
	print(model.summary())

	# load model
	if args.load_model is not None:
	    path = args.load_model
	    if os.path.exists(path):
	        print('load model from %s' % path)
	        model.load_weights(path, by_name=True)

    # call back functions
	batch_print_callback = LambdaCallback(
		on_epoch_end=lambda batch, logs: print(
			'\nINFO:root:Epoch[%d] Train-accuracy=%f\nINFO:root:Epoch[%d] Validation-accuracy=%f' %
			(batch, logs['acc'], batch, logs['val_acc'])))

	earlystopping = EarlyStopping(
	    monitor='val_loss', patience=5, verbose=1, mode='auto')

	outdir = args.save_dir
	if not os.path.exists(outdir):
	    os.makedirs(outdir)
	filepath = outdir+"/"+args.model_name+"-{epoch:02d}.hdf5"
	checkpoint = ModelCheckpoint(filepath=filepath,
	                             verbose=1,
	                             save_best_only=True,
	                             save_weights_only=True,
	                             monitor='val_loss',
	                             mode='min')

	callbacks_list = [checkpoint, earlystopping]

	# model fit
	history = model.fit([
		train_cinq_vector, train_punctuation_vector, 
		train_contexts_word, train_questions_word, 
		train_contexts_char, train_questions_char,
		train_answers_start], 
		[train_answers_start, train_answers_end],	
		epochs=args.nb_epoch,
		batch_size=args.batch_size,
		validation_data=([
			valid_cinq_vector, valid_punctuation_vector, 
			valid_contexts_word, valid_questions_word, 
			valid_contexts_char, valid_questions_char,
			valid_answers_start],
			[valid_answers_start, valid_answers_end]),
		callbacks=callbacks_list)

	# model save
	filepath = "save_model/model.h5"
	print("filepath: ", filepath)
	model.save_weights(filepath)
	print("Save model at: %s" %filepath)

def fromCQI2Ypred(args, contexts, questions, ids):
	print ('Load word2idx, char2idx')
	word2idx, embeddings_matrix = load_gensim(wv_path=args.wv_path, wv_only=True)
	char2idx, idx2char = load_char2idx()
	args.char_size = len(char2idx)+1
	args.vocab_size = len(embeddings_matrix)
	args.embedding_dim = len(embeddings_matrix[0])

	print ('jieba cut ...')
	jieba_questions = [list(jieba.cut(question, cut_all=False)) for question in questions]
	jieba_contexts = [list(jieba.cut(context, cut_all=False)) for context in contexts]

	print ('Sentence encode --wordLevel...')
	questions_word = sentence_encode_wordLevel(questions, jieba_questions, word2idx, args.q_max_len)
	contexts_word = sentence_encode_wordLevel(contexts, jieba_contexts, word2idx, args.c_max_len)
	
	print ('Sentence encode --charLevel...')
	contexts_char = sentence_encode_charLevel(contexts, char2idx=char2idx, max_len=args.c_max_len)
	questions_char = sentence_encode_charLevel(questions, char2idx=char2idx, max_len=args.q_max_len)

	print ('Heuristic --comparing_CQ...')
	cinq_vector = context_cinq_vector_level(contexts, questions, args.c_max_len, args.context_vector_level)
	punctuation_vector = context_punctuation_vector_all_level(contexts, args.c_max_len)

	print ('Build up model for start point...')
	model = cinq_wordEmb_model_2(args, embeddings_matrix, isTeacherForcing=False)
	print(model.summary())

	path = start_model_path
	if os.path.exists(path):
	    print('load model from %s' % path)
	    model.load_weights(path, by_name=True)
	else:
	    raise ValueError("Can't find the file %s" % path)

	answers_start_place_holder = np.zeros((cinq_vector.shape[0], args.c_max_len), dtype=np.float64)
	print ('Predict start point...')
	Y_pred = model.predict([
		cinq_vector, punctuation_vector, 
		contexts_word, questions_word, 
		contexts_char, questions_char,
		answers_start_place_holder], 
		batch_size=args.batch_size, 
		verbose=1)
	Y_str = [np.argmax(y) for y in Y_pred[0]]
	Y_end = [np.argmax(y) for y in Y_pred[1]]

	return Y_str, Y_end

def DoValid(args):

	print ('Loading data from %s' %args.training_data_path)
	contexts, questions, answers, ids = loadTrainingData(args)
	answers_start, answers_end = answer_convert(contexts, answers)

	Y_str, Y_end = fromCQI2Ypred(args, contexts, questions, ids)

	ans_start = answers_start
	ans_end = answers_end

	
	print ('Output prediction...')
	output_predicts(args, Y_str, Y_end, ids, contexts)

	### print F1 score
	print ('Calculating F1 score...')
	score = mean_f1_3(Y_str, Y_end, ans_start, ans_end)
	print("score: ", score.mean())
	for r in [100, 1000, 2000, 3000, 5000, 8000, 10000]:
		print("score[:", r, "]: ", score[:r].mean())

	valid_num = len(Y_str)
	print("valid set size: ", valid_num)



def DoTest(args):
	print ('Loading data from %s' %args.testing_data_path)
	contexts, questions, ids = loadTestingData(args)	
	
	Y_str, Y_end = fromCQI2Ypred(args, contexts, questions, ids)

	print ('Output result to %s' %args.result_path)
	output_predicts(args, Y_str, Y_end, ids, contexts)


if __name__ == "__main__":

	parser = ArgumentParser(description='Final chinese QA')

	parser.add_argument('action', choices=['train', 'valid', 'test'])

	# training arguments
	parser.add_argument('--isTeacherForcing', default=False, type=bool)
	parser.add_argument('--batch_size', default=64, type=int)
	parser.add_argument('--nb_epoch', default=200, type=int)
	parser.add_argument('--val_ratio', default=0.2, type=float)
	parser.add_argument('--data_folder', default='data/')
	parser.add_argument('--training_data_path', default='data/train-v1.1.json')
	parser.add_argument('--wv_path', default="word_vec/wv_20180118_234210_mincount300_dim100.npy")
	parser.add_argument('--vocab_size', default=20000, type=int)
	parser.add_argument('--char_size', default=5493, type=int)
	parser.add_argument('--char_emb_dim', default=8, type=int)
	parser.add_argument('--embedding_trainable', default=False, type=bool)
	parser.add_argument('-emb_dim', '--embedding_dim', default=64, type=int)
	parser.add_argument('-hid_siz', '--hidden_size', default=64, type=int)
	parser.add_argument('--optimizer', default='adam', help='Optimizer', type=str)
	parser.add_argument('--loss', default='categorical_crossentropy', help='Loss', type=str)
	parser.add_argument('--char_level_embeddings', default=True)
	parser.add_argument('--context_vector_level', default=4, type=int)
	parser.add_argument('--punctuation_level', default=9, type=int)
	parser.add_argument('--randomize', default=True)
	parser.add_argument('--latent_dim', default=6)
	parser.add_argument('--RNN_Type', choices=['gru', 'lstm'], default='gru')
	parser.add_argument('--word_trainable', default=False)

	# testing arguments
	parser.add_argument('--testing_data_path', default='data/test-v1.1.json')

	# put model in the same directory
	parser.add_argument('--load_model', default=None)
	parser.add_argument('--load_model2', default=None)
	parser.add_argument('--save_dir', default='save_model', type=str)
	parser.add_argument('--model_name', default='model', type=str)
	parser.add_argument('--result_path', default='result/result.csv')
	parser.add_argument('--result_text_path', default='result/result_text.csv')
	parser.add_argument('--learningRate', default=0.001, type=float)

	parser.add_argument('--onDeepQ', default=False, type=bool)
	parser.add_argument('--fastMode', default=False, type=bool)

	parser.add_argument('--seg_len', default=20, type=int)
	parser.add_argument('--shiftting', default=0, type=int)
	parser.add_argument('--enlarge', default=0, type=int)
	parser.add_argument('--useJieba', default=False, type=bool)
	parser.add_argument('--density', default=2, type=int)
	parser.add_argument('--fastConst', default=10, type=int)
	parser.add_argument('--c_max_len', default=1160, type=int)
	parser.add_argument('--q_max_len', default=90, type=int)

	args = parser.parse_args()

	fastConst = args.fastConst

	if args.onDeepQ:
		data_path = os.environ.get("GRAPE_DATASET_DIR")
		args.training_data_path = os.path.join(data_path, "data/train-v1.1.json")
		args.testing_data_path = os.path.join(data_path, "data/test-v1.1.json")
		args.result_path = "model/res.csv"
		args.batch_size = 64


	if args.action == "train":
		DoTrain(args)
	elif args.action == "valid":
		args.randomize =False
		start_model_path = args.load_model
		end_model_path = args.load_model2
		DoValid(args)
	elif args.action == "test":
		start_model_path = args.load_model
		end_model_path = args.load_model2
		DoTest(args)
	else:
		print ('wrong action arguments!!!')