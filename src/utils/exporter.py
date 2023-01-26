import torch
import os
from .model import load_checkpoint
from .train_utils import instantiate_architecture

class Exporter:
    def __init__(self, config):
        self.config = config
        self.checkpoint = config.get('checkpoint')
        self.gnn = config.get('gnn')
        if self.gnn:
            self.model1, self.model2, self.model3, self.model4 = instantiate_architecture(ftr_dim=512, model_name=config.get('backbone'), gnn=True)
        else:
            self.model1, self.model2, self.model3 = instantiate_architecture(ftr_dim=512, model_name=config.get('backbone'), gnn=False)

        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        if self.gnn:
            self.model4.eval()
            checkpoint=torch.load(self.checkpoint)
            glbl_cnv_wt=checkpoint['cnv_lyr_state_dict']
            glbl_backbone_wt=checkpoint['backbone_model_state_dict']
            glbl_fc_wt=checkpoint['fc_layers_state_dict']
            sit0_gnn_wt=checkpoint['sit0_gnn_model']
            # sit1_gnn_wt=checkpoint['sit1_gnn_model']
            # sit2_gnn_wt=checkpoint['sit2_gnn_model']
            # sit3_gnn_wt=checkpoint['sit3_gnn_model']
            # sit4_gnn_wt=checkpoint['sit4_gnn_model']
            self.model1.load_state_dict(glbl_cnv_wt)
            self.model2.load_state_dict(glbl_backbone_wt)
            self.model3.load_state_dict(glbl_fc_wt)
            self.model4.load_state_dict(sit0_gnn_wt)
        else:
            

    def export_model_ir(self):
        input_model = os.path.join(
            os.path.split(self.checkpoint)[0], self.config.get('model_name_onnx'))
        input_shape = self.config.get('input_shape')
        output_dir = os.path.split(self.checkpoint)[0]
        export_command = f"""mo \
        --framework onnx \
        --input_model {input_model} \
        --input_shape "{input_shape}" \
        --output_dir {output_dir}"""

        if self.config.get('verbose_export'):
            print(export_command)
        os.system(export_command)

    def export_model_onnx(self):

        print(f"Saving model to {self.config.get('model_name_onnx')}")
        res_path = os.path.join(os.path.split(self.checkpoint)[0], self.config.get('model_name_onnx'))

        if self.phase == 2:
            dummy_input = torch.randn(1, 16, 160, 324)
        else:
            dummy_input = torch.randn(1, 1, 1024, 1024)

        torch.onnx.export(self.model, dummy_input, res_path,
                          opset_version=11, do_constant_folding=True,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}},
                          verbose=False)