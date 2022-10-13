from SETR.transformer_seg import SETRModel, Vit
import torch 

if __name__ == "__main__":
    # 重点是SETR模型的实现
    net = SETRModel(patch_size=(16, 16),
                    in_channels=3, 
                    out_channels=1, 
                    hidden_size=1024,
                    num_hidden_layers=1,
                    num_attention_heads=1,
                    decode_features=[512, 1256, 128, 64])
    t1 = torch.rand(1, 3, 256, 256)
    print("input: " + str(t1.shape))

    print("output: " + str(net(t1).shape))

'''
    # ViT模型完成分类任务
    model = Vit(patch_size=(32, 32), 
                    in_channels=1, 
                    out_class=10, 
                    sample_rate=4,
                    hidden_size=1024, 
                    num_hidden_layers=1, 
                    num_attention_heads=16)
    
    t1 = torch.rand(1, 1, 512, 512)
    print("input: " + str(t1.shape))

    print("output: " + str(model(t1).shape))
'''


