import torch


class SaveActive(object):
    def __init__(self, model):
        self.model = model
        self.fw_output = {}
        self.fw_input = {}
        self.bw_output = {}
        self.bw_input = {}
        self.fw_hook_lst = []
        self.bw_hook_lst = []
        self.clear_buffer()
        self.__registor_model(model)

    def __enter__(self):
        return self

    def __call__(self, model):
        self.__init__(model)

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove_hook()
        self.clear_buffer()

    def clear_buffer(self):
        for name, layer in self.model.named_modules():
            if len(list(layer.named_children())) == 0:
                self.fw_input[name] = []
                self.fw_output[name] = []
                self.bw_input[name] = []
                self.bw_output[name] = []

    def __registor_model(self, model):
        for name, layer in model.named_modules():
            if len(list(layer.named_children())) == 0:
                # print(f'hook in {name}')
                fw_handle = layer.register_forward_hook(self.fw_save(name))
                self.fw_hook_lst.append(fw_handle)
                # except inplace for https://github.com/pytorch/pytorch/issues/61519
                # layer.register_full_backward_hook(self.bw_save(name))
                bw_handle = layer.register_backward_hook(self.bw_save(name))
                self.bw_hook_lst.append(bw_handle)

    def remove_hook(self):
        for fw_handle in self.fw_hook_lst:
            fw_handle.remove()
        for bw_handle in self.bw_hook_lst:
            bw_handle.remove()

    def fw_save(self, name):
        def forward_hook(model, inputs, output):
            tmp1 = inputs[0].detach().clone().cpu().to(torch.float32)
            if tmp1.dim() == 0:
                tmp1 = tmp1.unsqueeze(0)
            self.fw_input[name].append(tmp1)
            if output is not None:
                tmp2 = output.detach().clone().cpu().to(torch.float32)
                if tmp2.dim() == 0:
                    tmp2 = tmp2.unsqueeze(0)  # dim!=0 for torch.concat
                self.fw_output[name].append(tmp2)

        return forward_hook

    def bw_save(self, name):
        def backward_hook(module, grad_input, grad_output):
            if len(grad_input) >= 2:
                if grad_input[1] is not None:
                    tmp1 = grad_input[1].detach().clone().cpu().to(torch.float32)
                    if tmp1.dim() == 0:
                        tmp1 = tmp1.unsqueeze(0)
                    self.bw_input[name].append(tmp1)
            tmp2 = grad_output[0].detach().clone().cpu().to(torch.float32)
            if tmp2.dim() == 0:
                tmp2 = tmp2.unsqueeze(0)
            self.bw_output[name].append(tmp2)

        return backward_hook

    def get_fw_input_mean_norm(self):
        if self.__is_null(self.fw_input):
            print('Error: Please try foward prop')
            return {}
        means = {}
        for key in self.fw_input:
            if len(self.fw_input[key]) != 0:
                means[key] = torch.cat(self.fw_input[key], dim=0).mean(0).norm()
                n_data = len(torch.cat(self.fw_input[key], dim=0))
        # rint(f'mean norm by n_sameples: {n_data}')
        return means

    def get_fw_output_mean_norm(self):
        if self.__is_null(self.fw_output):
            print('Error: Please try foward prop')
            return {}
        means = {}
        for key in self.fw_output:
            if len(self.fw_output[key]) != 0:
                means[key] = torch.cat(self.fw_output[key], dim=0).mean(0).norm()
                n_data = len(torch.cat(self.fw_output[key], dim=0))
        # print(f'mean norm by n_sameples: {n_data}')
        return means

    def get_bw_input_mean_norm(self):
        if self.__is_null(self.bw_input):
            print('Error: Please try backward prop')
            return {}
        means = {}
        for key in self.bw_input:
            if len(self.bw_input[key]) != 0:
                means[key] = torch.cat(self.bw_input[key], dim=0).mean(0).norm()
                n_data = len(torch.cat(self.bw_input[key], dim=0))
        # print(f'mean norm by n_sameples: {n_data}')
        return means

    def get_bw_output_mean_norm(self):
        if self.__is_null(self.bw_output):
            print('Error: Please try backward prop')
            return {}
        means = {}
        for key in self.bw_output:
            if len(self.bw_output[key]) != 0:
                means[key] = torch.cat(self.bw_output[key], dim=0).mean(0).norm()
                n_data = len(torch.cat(self.bw_output[key], dim=0))
        # print(f'mean norm by n_sameples: {n_data}')
        return means

    def get_fw_input_mean_mean(self):
        if self.__is_null(self.fw_input):
            print('Error: Please try foward prop')
            return {}
        means = {}
        for key in self.fw_input:
            if len(self.fw_input[key]) != 0:
                means[key] = torch.cat(self.fw_input[key], dim=0).mean(0).norm()
                n_data = len(torch.cat(self.fw_input[key], dim=0))
        # rint(f'mean norm by n_sameples: {n_data}')
        return means

    def get_fw_output_mean_mean(self):
        if self.__is_null(self.fw_output):
            print('Error: Please try foward prop')
            return {}
        means = {}
        for key in self.fw_output:
            if len(self.fw_output[key]) != 0:
                means[key] = torch.cat(self.fw_output[key], dim=0).mean(0).mean()
                n_data = len(torch.cat(self.fw_output[key], dim=0))
        # print(f'mean norm by n_sameples: {n_data}')
        return means

    def get_bw_input_mean_mean(self):
        if self.__is_null(self.bw_input):
            print('Error: Please try backward prop')
            return {}
        means = {}
        for key in self.bw_input:
            if len(self.bw_input[key]) != 0:
                means[key] = torch.cat(self.bw_input[key], dim=0).mean(0).mean()
                n_data = len(torch.cat(self.bw_input[key], dim=0))
        # print(f'mean norm by n_sameples: {n_data}')
        return means

    def get_bw_output_mean_mean(self):
        if self.__is_null(self.bw_output):
            print('Error: Please try backward prop')
            return {}
        means = {}
        for key in self.bw_output:
            if len(self.bw_output[key]) != 0:
                means[key] = torch.cat(self.bw_output[key], dim=0).mean(0).mean()
                n_data = len(torch.cat(self.bw_output[key], dim=0))
        # print(f'mean norm by n_sameples: {n_data}')
        return means

    def __is_null(self, dat):
        n_data = 0
        for key in dat:
            n_data += len(dat[key])
        return n_data == 0
