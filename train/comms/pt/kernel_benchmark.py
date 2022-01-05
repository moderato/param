import numpy as np
import torch, functools
import table_batched_embeddings, table_batched_embeddings_ops
np.random.seed(42)

def div_round_up(a, b):
    return int((a + b - 1) // b) * b


def get_table_batched_offsets_from_dense(merged_indices):
    (T, B, L) = merged_indices.size()
    lengths = np.ones((T, B)) * L
    flat_lengths = lengths.flatten()
    return (
        merged_indices.int().contiguous().view(-1).cuda(),
        torch.tensor(([0] + np.cumsum(flat_lengths).tolist())).int().cuda(),
    )


def benchmark_torch_function(iters, warmup_iters, f, *args, **kwargs):
    for _ in range(warmup_iters): # Warmup
        f(*args, **kwargs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_time = 0.0
    for _ in range(iters):
        torch.cuda.synchronize()
        start_event.record()
        f(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        total_time += start_event.elapsed_time(end_event) * 1.0e-3
    return total_time / iters


def benchmark_conv(batch_size, H, W, IC, OC, stride, dilation, FH, FW, is_dw, iters, warmup_iters, backward=False):
    input_feature = torch.randn(batch_size, IC, H, W, requires_grad=True).cuda() # NCHW
    padding = []
    for f in [FH, FW]:
        padding.append((f - 1) // 2) # Only consider SAME with dilation = 1 for now
    conv = torch.nn.Conv2d(IC, OC, (FH, FW), stride=stride, dilation=dilation, padding=padding, groups=(IC if is_dw else 1)).cuda()

    if not backward:
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            conv,
            input_feature
        )
    else:
        out = conv(input_feature)
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            out.mean().backward,
            retain_graph=True
        )
    return time_per_iter


def benchmark_linear(M, N, K, iters, warmup_iters, backward=False):
    A = torch.randn(M, K, requires_grad=True).cuda()
    linear = torch.nn.Linear(K, N).cuda()
    if not backward:
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            linear,
            A
        )
    else:
        out = linear(A)
        out_mean = out.mean()
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            out_mean.backward,
            retain_graph=True,
        )
    return time_per_iter


def benchmark_fc(batch_size, M, N, K, iters, warmup_iters, backward=False):
    if batch_size == 1:
        A = torch.randn(M, K, requires_grad=True).cuda()
        B = torch.randn(N, K, requires_grad=True).cuda()
        C = torch.randn(M, N, requires_grad=True).cuda()
        if not backward:
            time_per_iter = benchmark_torch_function(
                iters,
                warmup_iters,
                torch.addmm,
                C, A, B.T,
            )
        else:
            torch.addmm(C, A, B.T)
            time_per_iter = benchmark_torch_function(
                iters,
                warmup_iters,
                C.mean().backward,
                retain_graph=True,
            )
        return time_per_iter

    else:
        A = torch.randn(batch_size, M, K, requires_grad=True).cuda()
        B = torch.randn(batch_size, N, K, requires_grad=True).cuda()
        if not backward:
            time_per_iter = benchmark_torch_function(
                iters,
                warmup_iters,
                torch.bmm,
                A, torch.transpose(B, 1, 2),
            )
        else:
            C = torch.bmm(A, torch.transpose(B, 1, 2))
            time_per_iter = benchmark_torch_function(
                iters,
                warmup_iters,
                C.mean().backward,
                retain_graph=True,
            )
        return time_per_iter

def benchmark_tril(batch_size, M, N, diag, iters, warmup_iters, backward=False):
    assert M == N, "Input tensor should be square!"
    Z = torch.randn(batch_size, M, N, requires_grad=True).cuda()
    li = torch.tensor([i for i in range(M) for j in range(i + diag)])
    lj = torch.tensor([j for i in range(N) for j in range(i + diag)])
    def zflat_wrapper(Z, i, j):
        return Z[:, i, j]

    if not backward:
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            zflat_wrapper,
            Z,
            li,
            lj
        )
    else:
        out = zflat_wrapper(Z, li, lj)
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            out.mean().backward,
            retain_graph=True,
        )
    return time_per_iter


def benchmark_bn(batch_size, H, W, OC, iters, warmup_iters, backward=False):
    out_feature = torch.randn(batch_size, OC, H, W, requires_grad=True).cuda()
    bn = torch.nn.BatchNorm2d(OC).cuda()

    if not backward:
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            bn,
            out_feature
        )
    else:
        output = bn(out_feature)
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            output.mean().backward,
            retain_graph=True,
        )
    return time_per_iter


def benchmark_concat(sizes, dim, iters, warmup_iters):
    tensors = [torch.randn(size) for size in sizes]

    time_per_iter = benchmark_torch_function(
        iters,
        warmup_iters,
        torch.cat,
        tensors,
        dim=dim
    )
    return time_per_iter


def benchmark_memcpy(size, iters, warmup_iters):
    A = torch.randn(size)

    time_per_iter = benchmark_torch_function(
        iters,
        warmup_iters,
        A.to,
        device="cuda"
    )
    return time_per_iter


def benchmark_transpose(batch_size, M, N, trans_type, iters, warmup_iters):
    A = torch.randn(batch_size, M, N).cuda()
    if trans_type == 0:
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            A.permute(0, 2, 1).contiguous
        )
    elif trans_type == 1:
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            A.permute(2, 1, 0).contiguous
        )
    else: # 2
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            A.permute(1, 0, 2).contiguous
    )
    return time_per_iter


def benchmark_pool(batch_size, H, W, OC, stride, dilation, FH, FW, pool_type, iters, warmup_iters, backward=False):
    A = torch.randn(batch_size, OC, H, W, requires_grad=True).cuda()
    padding = []
    for f in [FH, FW]:
        padding.append((f - 1) // 2) # Only consider SAME with dilation = 1 for now

    if pool_type == "max": # Max
        pool = torch.nn.MaxPool2d((FH, FW), stride=stride, dilation=dilation, padding=padding)
    else: # Avg
        pool = torch.nn.AvgPool2d((FH, FW), stride=stride, padding=padding)

    if not backward:
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            pool,
            A
        )
    else:
        output = torch.pool(A)
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            output.mean().backward,
            retain_graph=True,
        )
    return time_per_iter


def benchmark_relu(size, iters, warmup_iters, backward=False):
    A = torch.randn(size, requires_grad=True).cuda()
    
    if not backward:
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            torch.relu,
            A
        )
    else:
        output = torch.relu(A)
        time_per_iter = benchmark_torch_function(
            iters,
            warmup_iters,
            output.mean().backward,
            retain_graph=True,
        )
    return time_per_iter


def benchmark_embedding_lookup(B, E, T, L, D, BT_block_size, iters, warmup_iters, backward, shmem=False, sgd=False, fp16=False, managed=False, mixed=False):
    Es = [int(x) for x in E.split('-')] if isinstance(E, list) else [E]
    if len(Es) == 1:
        Es = Es * T
    assert len(Es) == T

    if mixed:
        mixed_D = [
            div_round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
            for _ in range(T)
        ]
        D = np.average(mixed_D)
    cc = (
        table_batched_embeddings_ops.TableBatchedEmbeddingBags(
            T,
            Es,
            D,
            optimizer=table_batched_embeddings_ops.Optimizer.APPROX_ROWWISE_ADAGRAD,
            learning_rate=0.1,
            managed=table_batched_embeddings_ops.EmbeddingLocation.DEVICE
            if not managed
            else table_batched_embeddings_ops.EmbeddingLocation.HOST_MAPPED,
            eps=0.1,
            stochastic_rounding=False,
            fp16=fp16,
        ).cuda()
        if not mixed
        else table_batched_embeddings_ops.MixedDimTableBatchedEmbeddingBags(
            [(Es, d) for d in mixed_D],
            optimizer=table_batched_embeddings_ops.Optimizer.APPROX_ROWWISE_ADAGRAD,
            learning_rate=0.1,
            managed=table_batched_embeddings_ops.EmbeddingLocation.DEVICE
            if not managed
            else table_batched_embeddings_ops.EmbeddingLocation.HOST_MAPPED,
            eps=0.1,
            stochastic_rounding=False,
            fp16=fp16,
        ).cuda()
    )

    R = False

    def w2(c):
        if not R:
            return c

        @functools.wraps(c)
        def z(w, o, x, *args):
            c(w, o, x.random_(0, E - 1), *args)

        return z

    def w3(c):
        if not R:
            return c

        @functools.wraps(c)
        def z(g, w, o, x, *args):
            c(g, w, o, x.random_(0, E - 1), *args)

        return z

    def w4(c):
        if not R:
            return c

        @functools.wraps(c)
        def z(g, w, o, a, x, *args):
            c(g, w, o, a, x.random_(0, E - 1), *args)

        return z

    def w6(c):
        if not R:
            return c

        @functools.wraps(c)
        def z(g, w, o, a, b, d, x, *args):
            c(g, w, o, a, b, d, x.random_(0, E - 1), *args)

        return z

    idxs = []
    for x in range(T):
        idxs.append(torch.randint(low=0, high=Es[x] - 1, size=(B, L)).int().cuda())
    merged_indices = torch.stack(idxs, dim=0)

    (indices, offsets) = get_table_batched_offsets_from_dense(merged_indices)

    assert indices.shape[0] == B * T * L
    assert all(
        l == L for l in (offsets[1:] - offsets[:-1]).detach().cpu().numpy().tolist()
    )
    per_sample_weights = None
    stochastic = False # TODO: Fix this
    exact = 1
    y0 = (
        table_batched_embeddings.forward(
            cc.embedding_weights,
            cc.table_offsets,
            indices,
            offsets,
            per_sample_weights,
            L,
            1,
            shmem,
        )
        if not mixed
        else table_batched_embeddings.forward_mixed_D(
            cc.embedding_weights,
            cc.table_offsets,
            cc.dim_offsets,
            cc.total_D,
            indices,
            offsets,
            per_sample_weights,
            L,
            1,
            shmem,
        )
    )

    y = (
        table_batched_embeddings.forward(
            cc.embedding_weights,
            cc.table_offsets,
            indices,
            offsets,
            per_sample_weights,
            L,
            BT_block_size,
            shmem,
        )
        if not mixed
        else table_batched_embeddings.forward_mixed_D(
            cc.embedding_weights,
            cc.table_offsets,
            cc.dim_offsets,
            cc.total_D,
            indices,
            offsets,
            per_sample_weights,
            L,
            BT_block_size,
            False,
        )
    )
    torch.testing.assert_allclose(y, y0)

    if not backward:
        time_per_iter = (
            benchmark_torch_function(
                iters,
                warmup_iters,
                w2(table_batched_embeddings.forward),
                cc.embedding_weights,
                cc.table_offsets,
                indices,
                offsets,
                per_sample_weights,
                L,
                BT_block_size,
                shmem,
            )
            if not mixed
            else benchmark_torch_function(
                iters,
                warmup_iters,
                w4(table_batched_embeddings.forward_mixed_D),
                cc.embedding_weights,
                cc.table_offsets,
                cc.dim_offsets,
                cc.total_D,
                indices,
                offsets,
                per_sample_weights,
                L,
                BT_block_size,
                shmem,
            )
        )

    else: # backward
        go = torch.randn_like(y0)

        learning_rate = 0.05
        eps = 0.01

        if sgd:
            time_per_iter = benchmark_torch_function(
                iters,
                warmup_iters,
                w3(table_batched_embeddings.backward_sgd),
                go,
                cc.embedding_weights,
                cc.table_offsets,
                indices,
                offsets,
                learning_rate,
                L,
                BT_block_size,
                shmem,
            )

        else: # adagrad
            if not exact:
                time_per_iter = (
                    benchmark_torch_function(
                        iters,
                        warmup_iters,
                        w3(table_batched_embeddings.backward_approx_adagrad),
                        go,
                        cc.embedding_weights,
                        cc.table_offsets,
                        indices,
                        offsets,
                        per_sample_weights,
                        cc.optimizer_state,
                        learning_rate,
                        eps,
                        L,
                        stochastic,
                        BT_block_size,
                    )
                    if not mixed
                    else benchmark_torch_function(
                        iters,
                        warmup_iters,
                        w6(
                            table_batched_embeddings.backward_approx_adagrad_mixed_D
                        ),
                        go,
                        cc.embedding_weights,
                        cc.table_offsets,
                        cc.table_dim_offsets,
                        cc.dim_offsets,
                        cc.total_D,
                        indices,
                        offsets,
                        per_sample_weights,
                        cc.optimizer_state,
                        learning_rate,
                        eps,
                        L,
                        stochastic,
                        BT_block_size,
                    )
                )
            else:
                time_per_iter = (
                    benchmark_torch_function(
                        iters,
                        warmup_iters,
                        w3(table_batched_embeddings.backward_exact_adagrad),
                        go,
                        cc.embedding_weights,
                        cc.table_offsets,
                        indices,
                        offsets,
                        per_sample_weights,
                        cc.optimizer_state,
                        learning_rate,
                        eps,
                        stochastic,
                        BT_block_size,
                    )
                    if not mixed
                    else benchmark_torch_function(
                        iters,
                        warmup_iters,
                        w6(table_batched_embeddings.backward_exact_adagrad_mixed_D),
                        go,
                        cc.embedding_weights,
                        cc.table_offsets,
                        cc.table_dim_offsets,
                        cc.dim_offsets,
                        cc.total_D,
                        indices,
                        offsets,
                        per_sample_weights,
                        cc.optimizer_state,
                        learning_rate,
                        eps,
                        stochastic,
                        BT_block_size,
                    )
                )
    return time_per_iter
