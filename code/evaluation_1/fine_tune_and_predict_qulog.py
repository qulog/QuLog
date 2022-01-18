import copy

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
from tqdm import trange
import pickle
import json
import sys
import time
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append("classes")
from classes.loss_functions import NuLogsyLossCompute
from classes.model import *
from classes.networks import *
from classes.tokenizer import *
from classes.data_loader import *

from collections import defaultdict

# dataset_name = sys.argv[1]

name_ = "cassandra_original.csv"
dataset_name = "/home/matilda/PycharmProjects/huawei_qulog/log_level_estimation/3_preprocessed_data/" + name_

def run_test(dataloader, model, optimizer, f_loss, polars=None):
    model.eval()
    total_loss = 0
    preds = []
    tmps = []
    scores_head1 = []
    scores_head2 = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            load, y = batch

            out = model.forward(load.cuda().long())
            # out = model.forward(load.long())
            if isinstance(f_loss, nn.CosineSimilarity):
                x = F.normalize(out, p=2, dim=1)

                x = torch.mm(x, polars.t().cuda())
                pred = x.max(1, keepdim=True)[1].reshape(1, -1)[0]
                preds += list(pred.detach().cpu().numpy())
            else:
                #                 tmp = out.detach().cpu().numpy()
                tmp = out.cpu().numpy()
                preds += list(np.argmax(tmp, axis=1))
                tmps += list(tmp)
                #                 scores +=
                #                 scores_head1 += model.encoder.layers[0].self_attn.attn[:, 0, :, :].detach().cpu()
                #                 scores_head2 += model.encoder.layers[0].self_attn.attn[:, 1, :, :].detach().cpu()
                scores_head1 += model.encoder.layers[0].self_attn.attn[:, 0, :, :]
                scores_head2 += model.encoder.layers[0].self_attn.attn[:, 1, :, :]

    #             if i%5==0:
    #                 print("Epoch %d Test Step: %d / %d Loss: %f" %
    #                       (epoch,i, len(dataloader), loss), end='\r')

    #     print("Epoch %d Test Step: %d / %d Loss: %f" %
    #                   (epoch,i, len(dataloader), loss), end='\r')
    return preds, scores_head1, scores_head2


# def get_relevant_indecies(class1, class2, labels_, predictions, test_data_tokenized):
#     label1 = label_mapper[class1]
#     label2 = np.int(label_mapper[class2])
#     idx_true_class = set(np.arange(len(labels_))[np.where(labels_== label1)])
#     idx_predicted_class = set(np.arange(len(predictions))[np.where(predictions==label2)])
#     print(type(label2))
#     print(type(predictions[0]))
#     print(label_mapper)
#     print(idx_true_class)
#     print(idx_predicted_class)

#     idx_valid_class = list(idx_true_class.intersection(idx_predicted_class))
# #     print(idx_true_class.intersection(idx_predicted_class))
# #     print("The type is ", type(test_data_tokenized))
#     target_data_ = test_data_tokenized[idx_valid_class]

#     return target_data_, idx_valid_class

# def assign_scores_to(target_data_, scores_mean):
#     words_per_class_importances = defaultdict(list)
#     for ms_id, log_message in enumerate(target_data_):
#         dict_ = {}
#         for idx, word in enumerate(log_message):
#             for idy, word2 in enumerate(log_message):
#                 dict_[tokenizer.index2word[word]] = scores_mean[ms_id, idx, idy]
#             dict_['log message'] = [tokenizer.index2word[x] for x in log_message]
#             dict_['msg_id'] = ms_id
#             words_per_class_importances[tokenizer.index2word[word]].append(dict_)
#     return words_per_class_importances

# def get_attention_scores_intersection(test_data_tokenized, labels_, predictions, scores_mean, class1, class2):
#     target_data_, idx_valid_class = get_relevant_indecies(class1, class2, labels_, predictions, test_data_tokenized)
#     scores_mean = np.array(scores_mean[idx_valid_class])
#     words_per_class_importances = assign_scores_to(target_data_, scores_mean)
#     return words_per_class_importances


def get_relevant_indecies(class1, class2, labels_, predictions, test_data_tokenized):
    label1 = label_mapper[class1]
    label2 = label_mapper[class2]

    idx_true_class = set(np.arange(len(labels_))[np.where(labels_ == label1)])
    idx_predicted_class = set(np.arange(len(predictions))[np.where(predictions == label2)])

    idx_valid_class = list(idx_true_class.intersection(idx_predicted_class))
    print(idx_valid_class)
    target_data_ = test_data_tokenized[idx_valid_class]
    return target_data_, idx_valid_class


def get_all_indecies(labels_, predictions, test_data_tokenized):
    idx_valid_class = np.arange(len(test_data_tokenized)).tolist()
    target_data_ = test_data_tokenized
    return target_data_, idx_valid_class


def assign_scores_to(target_data_, scores_mean):
    words_per_class_importances = defaultdict(list)
    for ms_id, log_message in enumerate(target_data_):
        dict_ = {}
        for idx, word in enumerate(log_message):
            for idy, word2 in enumerate(log_message):
                dict_[tokenizer.index2word[word2]] = scores_mean[ms_id, idx, idy]
            dict_['log message'] = [tokenizer.index2word[x] for x in log_message]
            dict_['msg_id'] = ms_id
            words_per_class_importances[tokenizer.index2word[word]].append(dict_)
    return words_per_class_importances


def assign_scores_words_log_message(target_data_, scores_mean):
    words_per_class_importances = defaultdict(list)
    list_msgs = []
    for ms_id, log_message in enumerate(target_data_):
        dict_ = {}
        for idx, word in enumerate(log_message):
            dict_[tokenizer.index2word[word]] = scores_mean[ms_id, idx, :][:len(log_message)]
            dict_['log message'] = [tokenizer.index2word[x] for x in log_message]
            dict_['msg_id'] = ms_id
        list_msgs.append(dict_)

    return list_msgs


def get_attention_scores_intersection(test_data_tokenized, labels_, predictions, scores_mean, class1, class2):
    target_data_, idx_valid_class = get_relevant_indecies(class1, class2, labels_, predictions, test_data_tokenized)
    scores_mean = np.array(scores_mean[idx_valid_class])
    words_per_class_importances = assign_scores_to(target_data_, scores_mean)
    scores_per_log_message = assign_scores_words_log_message(target_data_, scores_mean)
    return words_per_class_importances, scores_per_log_message


def get_attention_scores_all_test(test_data_tokenized, labels_, predictions, scores_mean):
    target_data_, idx_valid_class = get_all_indecies(labels_, predictions, test_data_tokenized)
    scores_mean = np.array(scores_mean[idx_valid_class])
    words_per_class_importances = assign_scores_to(target_data_, scores_mean)
    scores_per_log_message = assign_scores_words_log_message(target_data_, scores_mean)
    return words_per_class_importances, scores_per_log_message



def run_train(dataloader, model, optimizer, f_loss, epoch, polars=None, device="cpu"):
    model.train()
    total_loss = 0
    start = time.time()
    for i, batch in enumerate(dataloader):
        load, y = batch
        if polars is not None:
            y = polars[y.numpy()]
            y = torch.autograd.Variable(y).cuda()

        if device == "gpu":
            out = model.forward(load.cuda().long())
        else:
            out = model.forward(load.long())

        if isinstance(f_loss, nn.CosineSimilarity):
            loss = (1 - f_loss(out, y)).pow(2).sum()
        else:
            if device=="gpu":
                loss = f_loss(out, y.cuda().long())
            else:
                loss = f_loss(out, y.long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss
        elapsed = time.time() - start
        if i % 5 == 0:
            print("Epoch %d Train Step: %d / %d Loss: %f" %
                  (epoch, i, len(dataloader), loss), end='\r')

    print("Epoch %d Train Step: %d / %d Loss: %f" %
          (epoch, i, len(dataloader), loss), end='\r')
    return total_loss / len(dataloader)


def run_optimizer(model, train_dataloader, test_dataloader, labels_test, optimizer, n_epochs, f_loss, polars, class_weights, device):
    conf_matrix_good = []
    conf_matrix_bad = []
    best_f1_good = 0
    best_f1_bad = 0
    idx_good = 0
    idx_bad = 0
    best_model = 0
    for epoch in range(1, 1 + n_epochs):
        print("Epoch", epoch)
        loss = run_train(train_dataloader, model, optimizer, f_loss, epoch, polars, device)
        print("Epoch %d Train Loss: %f" % (epoch, loss), " " * 30)
        start_time = time.time()

        print("----------GOOD REPOS----------")
        preds1, scores11, scores12 = run_test(test_dataloader, model, optimizer, f_loss, device)
        print(f"Accuracy:{round(accuracy_score(preds1, labels_test), 2)}")
        print(f"f1_score:{round(f1_score(preds1, labels_test, average='micro'), 2)}")
        print(f"recall_score:{round(recall_score(preds1, labels_test, average='micro'), 2)}")
        print(f"precision_score:{round(precision_score(preds1, labels_test, average='micro'), 2)}")

        conf_matrix_good.append(confusion_matrix(preds1, labels_test))

        pp = confusion_matrix(preds1, labels_test)

        if pp.shape[0]<3:
            if best_f1_good < f1_score(preds1, labels_test, average='micro') and pp[0][0] >0 and pp[1][1]  > 0:
                best_f1_good = f1_score(preds1, labels_test, average='micro')
                idx_good = epoch-1
                best_model = copy.deepcopy(model)
                best_preds = preds1
        else:
            if best_f1_good < f1_score(preds1, labels_test, average='micro') and pp[0][0]  >0 and pp[1][1]  > 0 and pp[2][2]:
                best_f1_good = f1_score(preds1, labels_test, average='micro')
                idx_good = epoch-1
                best_model = copy.deepcopy(model)
                best_preds = preds1



    return best_model, best_preds, conf_matrix_good


batch_size = 32
pad_len = 32
tgt_vocab = 16
n_layers = 2
in_features = 16
out_features = 16
num_heads = 4
dropout = 0.05
max_len = 32
learning_rate = 0.0001
decay = 0.001
betas = (0.9, 0.999)
momentum = 0.9
n_epochs = 10

label_mapper = {'error': 0, 'info': 1, 'warning': 2}
inverse_log_mapper = {0: 'error', 1: 'info', 2: 'warning'}

# label_mapper = {'debug': 0, 'error': 1, 'info': 2, 'log': 3, 'warning': 4}
# inverse_log_mapper = {0: 'debug', 1: 'error', 2:'info', 3:'log', 4:'warning'}

# df = pd.read_csv("./filtered_log_df.csv")
print("Step 1) Loading the prediction dataset.")
df = pd.read_csv(dataset_name)
print(df.shape)
# df = df.iloc[:10000, :]

# df.loc[:, 'log_message'] = df.loc[:, 'log_message'].apply(lambda x: re.sub(, " ", x))

print(pd.value_counts(df.log_level))

df.loc[df['log_level'] == 'debug'] = 'info'
# df.loc[df['log_level']=='critical'] = 'error'
# df.loc[df['log_level']=='exception'] = 'error'
# df.loc[df['log_level']=='fatal'] = 'error'
# df.loc[df['log_level']=='warn'] = 'warning'

print("Step 2) Loading the tokenizer.")
with open("../../5_results/models/learning_scenario1/info_error_warning/tokenizer_True_.pickle", 'rb') as file:
    tokenizer = pickle.load(file)

# with open("tokenizer/tokenizer_INFODEBUG.pickle",'rb') as file:
#     tokenizer = pickle.load(file)


src_vocab = tokenizer.n_words
# print("Step 3) Loading the weights for the loss fcn.")
# with open("./weights_INFO_ERROR_WARNING.pickle", "rb") as file:
#    class_weights = pickle.load(file)

# with open("./weights_class3_INFODEBUG.pickle", "rb") as file:
#     class_weights = pickle.load(file)

# print("The weights are {}".format(class_weights))
# cross_entropoy_loss = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights)).cuda()
# cross_entropoy_loss = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights))
cross_entropoy_loss = nn.CrossEntropyLoss()

# print(class_weights)

load = df['log_message'].values
labels = df['log_level'].values

start_time = time.time()

# replace all special characters
regex = re.compile('[^a-zA-Z ]')
df['log_message'] = df['log_message'].apply(lambda x: ' '.join(regex.sub('', x).strip().split()))

labels_tokenized = [label_mapper[label] for label in labels]

print("Step 4) Updating the new vocabulary.")
tokenized = []
for i in trange(0, len(df)):
    tokenized.append(np.array(tokenizer.tokenize_test(df['log_message'][i])))

print("There are {} new tokens".format(np.abs(tokenizer.n_words - src_vocab)))

print("Step 5) Create the dataloader.")
load_test, labels_test = np.array(tokenized), np.array(labels_tokenized)


print("Step 6) Load the model.")
model = torch.load('../../5_results/models/learning_scenario1/100_info_error_warning_model.pth')
# model = torch.load('entire_model_INFO_WARNING_ERROR_DEBUGREPLACED_class.pth')
model.cuda()
# model.cpu()
sgd_opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
adam_opt = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=decay)
optimizers = {"adam": adam_opt, "sgd": sgd_opt}

optimizer = optimizers['adam']
loss_f = cross_entropoy_loss

device = "gpu"
if device == "gpu":
    torch.cuda.set_device(0)
    model.cuda()

class_count = df.groupby("log_level").count()['log_message']
label_mapper = {class_count.index[i]: i for i in range(len(class_count))}

calculate_weights = lambda x, i: x.sum() / (len(x) * x[i])
weights = [calculate_weights(class_count, i) for i in range(len(class_count))]
weights /= max(weights)

weights = [1, 0.8, 1]

if device == "gpu":
    class_weights = torch.FloatTensor(weights).cuda()
    cross_entropoy_loss = nn.CrossEntropyLoss(weight=class_weights).cuda()
else:
    class_weights = torch.FloatTensor(weights)
    cross_entropoy_loss = nn.CrossEntropyLoss(weight=class_weights)


train_indencies = np.random.randint(0, df.shape[0], 50)
test_indecies = list(set(np.arange(df.shape[0])).difference(train_indencies))

load_train = load_test[train_indencies]
load_train_labels = labels_test[train_indencies]

df = df.loc[test_indecies]


load_test = load_test[test_indecies]
load_test_good_repos_labels = labels_test[test_indecies]

# test_dataloader = create_test_data_loaders(load_test, labels_test, pad_len, batch_size)

train_dataloader, test_dataloader_good_repos = create_data_loaders(load_train, load_train_labels, load_test, load_test_good_repos_labels, pad_len, batch_size)

best_model, preds, conf_matrix_good = run_optimizer(model, train_dataloader, test_dataloader_good_repos, labels_test=load_test_good_repos_labels, optimizer=optimizer, n_epochs=n_epochs, f_loss=cross_entropoy_loss, polars=None, class_weights=weights, device=device)

print("Step 7) Generate the predictions.")
predictions, scores1, scores2 = run_test(test_dataloader_good_repos, best_model, optimizer, cross_entropoy_loss, polars=None)

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

labels_tokenized = [label_mapper[label] for label in labels]
labels_tokenized = labels_test[test_indecies]

# print(labels)
# print(predictions)
print("Step 8) CALUCLATE SCORES")
print("The F1 score is {}".format(f1_score(y_true=labels_tokenized, y_pred=predictions, average="micro")))
print("The Precision score is {}".format(
    precision_score(y_true=labels_tokenized, y_pred=predictions, average="micro")))
print("The Recall score is {}".format(recall_score(y_true=labels_tokenized, y_pred=predictions, average="micro")))
print("The Accuracy score is {}".format(accuracy_score(y_true=labels_tokenized, y_pred=predictions)))
print("Confusion matrix {}".format(confusion_matrix(y_true=labels_tokenized, y_pred=predictions)))

df["log_message_pred"] = predictions
pom = df.loc[:, ["log_message", "log_level", "log_message_pred"]]
pom.log_message_pred = pom.log_message_pred.apply(lambda x: inverse_log_mapper[x])
#
# print("Step 9) PROCESS ATTENTION SCORES")
#
# class1 = "error"  ## class 1 to compare
# class2 = "info"  ## class 2 to compare
#
# test_data_tokenized = np.array(tokenized)
# labels_ = test_dataloader.dataset.tensors[1].numpy()
# predictions = np.array(predictions)
#
# scores_head_1 = [x.numpy() for x in scores1]
# scores_head_2 = [x.numpy() for x in scores2]
# scores_mean = (np.array(scores_head_1) + np.array(scores_head_2)) / 2
#
# words_per_class_importances, scores_per_log_message = get_attention_scores_intersection(np.array(load_test), labels_,
#                                                                                         predictions, scores_mean,
#                                                                                         class1, class2)
#
# top_list_dictionary = {}
# for top_word in words_per_class_importances.keys():
#     list_dictionary = defaultdict(list)
#     for possible_words in words_per_class_importances[top_word]:
#         #     x.pop('log message')
#         for word in possible_words.keys():
#             list_dictionary[word].append(possible_words[word])
#     top_list_dictionary[top_word] = list_dictionary
#
# w_all, s_all = get_attention_scores_all_test(test_data_tokenized, labels_, predictions, scores_mean)
# full_data = pd.DataFrame(s_all, index=np.arange(len(s_all))).fillna(0).round(5)

def diff_input(load, tokenized):
    difference = []
    for x in range(load.shape[0]):
        parsed_seq = tokenizer.convert_ids_to_tokens(tokenized[x])
        original_seq = load[x].lower().rsplit()
        difference.append(" ".join(list(set(original_seq).difference(set(parsed_seq)))))
    return pd.DataFrame(difference)


a = diff_input(load, tokenized)

print("Step 10) Store the predictions in predictions.csv file. The format is (log message, real log level, predicted log level).")
print("Total time {}".format(time.time() - start_time))
pd.concat([pom, a], axis=1).to_csv(name_ +'_predictions_model2.csv')


