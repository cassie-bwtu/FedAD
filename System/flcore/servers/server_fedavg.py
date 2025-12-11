import copy
import time
from flcore.clients.client_fedavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread

class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients (两类)
        self.set_slow_clients()

        self.set_clients(clientAVG)  #生成clients

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []  #统计每一轮的时间消耗

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            '''
            if i == 0:
                print("===============globalmodel================")
                for name, parameters in self.global_model.blocks.block5.named_parameters():
                    print("name: ", name)
                    print("parameters: ", parameters)
            '''

            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                #print("Selected clients: {}".format([c.id for c in self.selected_clients]))
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

                '''
                if i == 0:
                    print("===============submodel================")
                    for name, parameters in client.model.blocks.block5.named_parameters():
                        print("name: ", name)
                        print("grad: ", parameters.grad)
                        print("parameters: ", parameters)
                '''

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))  #最高acc
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))  #平均（每一轮）时间消耗

        # self.save_results()
        # self.save_global_model()


    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        # self.global_model = copy.deepcopy(self.global_model)
        for name, param in self.global_model.state_dict().items():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
        # 全局模型更新

    def add_parameters(self, w, client_model):
        for (server_name, server_param), (client_name, client_param) in zip(self.global_model.state_dict().items(), client_model.state_dict().items()):
            if 'num_batches_tracked' in server_name:
                server_param.data += int(client_param.data.clone() * w)
            else:
                server_param.data += client_param.data.clone() * w


