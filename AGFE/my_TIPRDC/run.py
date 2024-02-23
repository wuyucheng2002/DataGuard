"""
接收用户参数，并执行整个任务
the way to run "run.py" is to type:
python run.py --target_label='label' --u='sex' --lbd=0.66
in the terminal
"""
import argparse
from data_prepare import Data
from component import Feature_Extractor, Classifier, MutlInfo
from pretrain import train_FE_CF
from main import get_FE, test_downstream_task

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="获取任务相关参数")
    parse.add_argument("--target_label", type=str, default='label', help="the label for downstream task")
    parse.add_argument("--u", type=str, default='sex', help="the private attribute")
    parse.add_argument("--epochs", type=int, default=50, help="training epochs of components")
    parse.add_argument("--batch_size", type=int, default=256, help="training batch_size of components")
    parse.add_argument("--lr", type=float, default=0.1, help="training learning_rate of components")
    parse.add_argument("--lbd", type=float, default=0.5, help="for controlling the tradeoff between utility and privacy, higher lbd leads to higher privacy")
    parse.add_argument("--pretrain_epochs", type=int, default=20, help="training epochs of FE and CF for pretraining")
    parse.add_argument("--pretrain_batch_size", type=int, default=256, help="training batch size of FE and CF for pretraining")
    parse.add_argument("--pretrain_lr", type=float, default=0.1, help="training learning rate of FE and CF for pretraining")
    parse.add_argument("--embedding_dim", type=int, default=100, help="the dimension of the features extracted (embedding)")
    parse.add_argument("--hidden_dim", type=int, default=100, help="the dimension of classifier's hidden layer")
    args = parse.parse_args()

    data = Data(target=args.target_label, u=args.u)
    FE = Feature_Extractor(args.embedding_dim)
    CF = Classifier(target_size=data.num_of_label_u, hidden_dim=args.hidden_dim)

    train_FE_CF(FE, CF, data, args.pretrain_epochs, args.pretrain_batch_size, args.pretrain_lr)

    FE, attack_acc_before, attack_acc_after = get_FE(data, args.epochs, args.batch_size, args.lr, args.lbd)
    Embedding, acc_before, acc_after = test_downstream_task(FE, data)




