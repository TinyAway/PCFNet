import torch
import torch.nn.functional as F
import faiss


res = faiss.StandardGpuResources()  # use a single GPU

def swig_ptr_from_FloatTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)

def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    # return faiss.cast_integer_to_long_ptr(
    #     x.storage().data_ptr() + x.storage_offset() * 8)
    return faiss.cast_integer_to_idx_t_ptr(x.storage().data_ptr() + x.storage_offset() * 8)

def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)

    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize()
    xptr = swig_ptr_from_FloatTensor(x)
    Iptr = swig_ptr_from_LongTensor(I)
    Dptr = swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr,
                   k, Dptr, Iptr)
    torch.cuda.synchronize()
    return D, I


def knn_faiss(feature_B, feature_A, mask, k):
    b, ch, nA = feature_A.shape
    feature_A = feature_A.view(b, ch, -1).permute(0, 2, 1).contiguous()
    feature_B = feature_B.view(b, ch, -1).permute(0, 2, 1).contiguous()
    similarities = []
    for i in range(b):
        index_cpu = faiss.IndexFlatL2(ch)
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        torch.cuda.synchronize()
        feature_B_ptr = swig_ptr_from_FloatTensor(feature_B[i])
        index.add_c(feature_B[i].shape[0], feature_B_ptr)
        _, indx_i = search_index_pytorch(index, feature_A[i], k=k)
        indx_i = indx_i.t().contiguous()
        indx_i = indx_i.flatten()
        similarity = mask[i][indx_i]
        similarity = similarity.view(k, -1).unsqueeze(0)
        similarities.append(similarity)
    similarities = torch.cat(similarities, dim=0)
    return similarities


def cosine_similarity(query_feat, support_feat, mask):
    eps = 1e-5
    support_feat = support_feat * mask
    bsz, ch, hb, wb = support_feat.size()
    support_feat = support_feat.view(bsz, ch, -1)
    support_feat_norm = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

    bsz, ch, ha, wa = query_feat.size()
    query_feat = query_feat.view(bsz, ch, -1)
    query_feat_norm = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

    corr = torch.bmm(query_feat_norm.transpose(1, 2), support_feat_norm)
    corr = corr.clamp(min=0)
    area = torch.sum(mask.view(bsz, -1), dim=1).view(bsz, 1) + eps
    corr = corr.sum(dim=-1)
    corr = corr / area

    corr = corr.view(bsz, ha, wa)
    return corr

def euclid_distance(query_feat, support_feat, mask, k=10):
    bsz, ch, ha, wa = query_feat.size()
    query_feat = query_feat.view(bsz, ch, -1)
    support_feat = support_feat.view(bsz, ch, -1)
    mask = mask.view(bsz, -1)

    with torch.no_grad():
        similarities = knn_faiss(support_feat, query_feat, mask, k)
    similarities = similarities.mean(dim=1).view(bsz, ha, wa)
    return similarities


class Correlation:

    @classmethod
    def correlation(cls, query_feats, support_feats, stack_ids, support_mask):
        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
                                 align_corners=True)
            corr = cosine_similarity(query_feat, support_feat, mask)
            corrs.append(corr)
            similarity = euclid_distance(query_feat, support_feat, mask)
            corrs.append(similarity)

        corr_l4 = torch.stack(corrs[-stack_ids[0] * 2:]).transpose(0, 1).contiguous()
        corr_l3 = torch.stack(corrs[-stack_ids[1] * 2:-stack_ids[0] * 2]).transpose(0, 1).contiguous()
        corr_l2 = torch.stack(corrs[-stack_ids[2] * 2:-stack_ids[1] * 2]).transpose(0, 1).contiguous()

        return [corr_l4, corr_l3, corr_l2]


