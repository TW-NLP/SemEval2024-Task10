from tqdm import tqdm

from others.car_normal_text.code.utils import FGM


def train(train_loader, model, loss, optimizer, schedule, device):
    """

    :param train_loader:
    :param model:
    :param loss:
    :param optimizer:
    :param schedule:
    :param device:
    :return:
    """
    model.train()

    fgm = FGM(model, emb_name="word_embeddings", epsilon=1.0)
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        targets = batch['labels'].to(device)
        outputs1 = model(input_ids, attention_mask, token_type_ids)
        #
        # # 来计算loss
        loss_value1 = loss(outputs1, targets)

        loss_value1.backward()

        # 对抗训练
        fgm.attack()  # attack在embedding上添加对抗扰动
        outputs_ = model(input_ids, attention_mask, token_type_ids)
        loss_ = loss(outputs_, targets)
        loss_.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore()  # 恢复embedding参数

        optimizer.step()
        schedule.step()


