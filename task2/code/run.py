import json
import numpy as np
from xgboost import XGBClassifier


def read_json(path):
    train_list_ = []
    label_list_ = []
    with open(path, "r", encoding="utf-8") as file_read:
        data_line = json.loads(file_read.read())

        for idx, line_i in enumerate(data_line):

            for idx_i, speaker_i, utterances_i, emotion_i, trigger_i in zip(range(len(line_i['speakers'])),
                                                                            line_i['speakers'],
                                                                            line_i['utterances'],
                                                                            line_i['emotions'], line_i['triggers']):
                if idx_i == len(line_i['speakers']) - 1:
                    line_emo = [label_name.index(emotion_i), label_name.index(first_emotion),
                                label_name.index(first_emotion), label_name.index(first_emotion)]
                elif idx_i == len(line_i['speakers']) - 2:
                    line_emo = [label_name.index(emotion_i), label_name.index(line_i['emotions'][idx_i + 1]),
                                label_name.index(first_emotion), label_name.index(first_emotion)]
                elif idx_i == len(line_i['speakers']) - 3:
                    line_emo = [label_name.index(emotion_i), label_name.index(line_i['emotions'][idx_i + 1]),
                                label_name.index(line_i['emotions'][idx_i + 2]), label_name.index(first_emotion)]
                else:
                    line_emo = [label_name.index(emotion_i), label_name.index(line_i['emotions'][idx_i + 1]),
                                label_name.index(line_i['emotions'][idx_i + 2]),
                                label_name.index(line_i['emotions'][idx_i + 3])]

                try:
                    label_list_.append(int(trigger_i))
                    train_list_.append(line_emo)


                except Exception as e:
                    pass

    return np.array(train_list_), np.array(label_list_)


def read_json_test(path):
    train_list_ = []
    with open(path, "r", encoding="utf-8") as file_read:
        data_line = json.loads(file_read.read())

        for idx, line_i in enumerate(data_line):

            for idx_i, speaker_i, utterances_i, emotion_i in zip(range(len(line_i['speakers'])),
                                                                 line_i['speakers'],
                                                                 line_i['utterances'],
                                                                 line_i['emotions']):
                if idx_i == len(line_i['speakers']) - 1:
                    line_emo = [label_name.index(emotion_i), label_name.index(first_emotion),
                                label_name.index(first_emotion), label_name.index(first_emotion)]
                elif idx_i == len(line_i['speakers']) - 2:
                    line_emo = [label_name.index(emotion_i), label_name.index(line_i['emotions'][idx_i + 1]),
                                label_name.index(first_emotion), label_name.index(first_emotion)]
                elif idx_i == len(line_i['speakers']) - 3:
                    line_emo = [label_name.index(emotion_i), label_name.index(line_i['emotions'][idx_i + 1]),
                                label_name.index(line_i['emotions'][idx_i + 2]), label_name.index(first_emotion)]
                else:
                    line_emo = [label_name.index(emotion_i), label_name.index(line_i['emotions'][idx_i + 1]),
                                label_name.index(line_i['emotions'][idx_i + 2]),
                                label_name.index(line_i['emotions'][idx_i + 3])]

                train_list_.append(line_emo)

    return np.array(train_list_)


if __name__ == '__main__':
    label_name = ['-1', 'disgust', 'contempt', 'anger', 'neutral', 'joy', 'sadness', 'fear', 'surprise']

    first_emotion = "-1"

    train_path = "../data/MaSaC_train_efr.json"
    test_path = "../data/MaSaC_test_efr.json"

    X_train, y_train = read_json(train_path)

    X_test = read_json_test(test_path)

    model = XGBClassifier(scale_pos_weight=1.08, random_state=42)

    # 训练模型
    model.fit(X_train, y_train)
    # 预测测试集
    y_pred = model.predict(X_test)

    # 评估模型性能
    #
    with open("answer2.txt", "w", encoding="utf-8") as file_write:
        for pred_i in y_pred:
            pred_i = "{:.1f}".format(pred_i)
            file_write.write(pred_i + "\n")

    file_write.close()
