import torch 
import torch.distributed as dist

@torch.no_grad()
def concat_all_gather(tensor):
    
    # gather all tensor shape 
    shape_tensor = torch.tensor(tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(shape_list, shape_tensor)
    
    # padding tensor to the max length
    if shape_list[0].numel() > 1:
        max_shape = torch.tensor([_[0] for _ in shape_list]).max()
        padding_tensor = torch.zeros((max_shape, shape_tensor[1]), device='cuda').type_as(tensor)
    else:
        max_shape = torch.tensor(shape_list).max()
        padding_tensor = torch.zeros(max_shape, device='cuda').type_as(tensor)
       
    padding_tensor[:shape_tensor[0]] = tensor
    
    tensor_list = [torch.zeros_like(padding_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, padding_tensor)

    sub_tensor_list = []
    for sub_tensor, sub_shape in zip(tensor_list, shape_list):
        sub_tensor_list.append(sub_tensor[:sub_shape[0]])
    output = torch.cat(sub_tensor_list, dim=0)

    return output