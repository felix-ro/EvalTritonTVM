
triton_red_fused_native_batch_norm_backward_threshold_backward_3 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @reduction(
     size_hints=[512, 4096],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_3', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
     xnumel = 512
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rbase = tl.arange(0, RBLOCK)[None, :]
     x0 = xindex
     _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
         rindex = roffset + rbase
         rmask = rindex < rnumel
         r1 = rindex % 49
         r2 = (rindex // 49)
         tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (25088*r2)), rmask & xmask, other=0)
         tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (25088*r2)), rmask & xmask, other=0)
         tmp8 = tl.load(in_ptr2 + (r1 + (49*x0) + (25088*r2)), rmask & xmask, other=0)
         tmp1 = 0.0
         tmp2 = tmp0 <= tmp1
         tmp4 = tl.where(tmp2, tmp1, tmp3)
         tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
         tmp7 = _tmp6 + tmp5
         _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
         tmp10 = tmp8 - tmp9
         tmp11 = tmp4 * tmp10
         tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
         tmp14 = _tmp13 + tmp12
         _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
     tmp6 = tl.sum(_tmp6, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp6, xmask)
     tmp13 = tl.sum(_tmp13, 1)[:, None]
     tl.store(out_ptr1 + (x0), tmp13, xmask)
     tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     tmp16 = tmp13 * tmp15
     tl.store(out_ptr2 + (x0), tmp16, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/se/cseid5ukasiwriptnd3wyppdcdd53fwnedzigiujn3nirg4yi6ju.py
 # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
 
triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_4', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]})
 @triton.jit
 def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ks0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x3 = xindex
     x1 = (xindex // 49) % 512
     tmp0 = tl.load(in_ptr0 + (x3), xmask)
     tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
     tmp5 = tl.load(in_ptr1 + (x3), xmask)
     tmp6 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
     tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
     tmp17 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
     tmp20 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
     tmp1 = 0.0
     tmp2 = tmp0 <= tmp1
     tmp4 = tl.where(tmp2, tmp1, tmp3)
     tmp7 = tmp5 - tmp6
     tmp9 = 0.0204081632653061*(1/ks0)
     tmp10 = tmp9.to(tl.float32)
     tmp11 = tmp8 * tmp10
     tmp13 = tmp12 * tmp12
     tmp14 = tmp11 * tmp13
     tmp15 = tmp7 * tmp14
     tmp16 = tmp4 - tmp15
     tmp18 = tmp17 * tmp10
     tmp19 = tmp16 - tmp18
     tmp21 = tmp12 * tmp20
     tmp22 = tmp19 * tmp21
     tl.store(in_out_ptr0 + (x3), tmp22, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/zb/czblqdkz33qyui6vkezvtomwrmppkztz6hmgwm2tedslprwe4idy.py
 # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
 
triton_red_fused_add_div_native_batch_norm_backward_threshold_backward_5 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @reduction(
     size_hints=[2048, 4096],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_native_batch_norm_backward_threshold_backward_5', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
     xnumel = 2048
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rbase = tl.arange(0, RBLOCK)[None, :]
     x0 = xindex
     _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
     _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
         rindex = roffset + rbase
         rmask = rindex < rnumel
         r1 = rindex % 49
         r2 = (rindex // 49)
         tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (100352*r2)), rmask, other=0)
         tmp3 = tl.load(in_ptr1 + (r1 + (49*x0) + (100352*r2)), rmask)
         tmp4 = tl.load(in_ptr2 + (x0 + (2048*r2)), rmask, eviction_policy='evict_last', other=0)
         tmp8 = tl.load(in_ptr3 + (r1 + (49*x0) + (100352*r2)), rmask, other=0)
         tmp14 = tl.load(in_ptr4 + (r1 + (49*x0) + (100352*r2)), rmask, other=0)
         tmp1 = 0.0
         tmp2 = tmp0 <= tmp1
         tmp5 = 49.0
         tmp6 = tmp4 / tmp5
         tmp7 = tl.where(tmp3, tmp1, tmp6)
         tmp9 = tmp7 + tmp8
         tmp10 = tl.where(tmp2, tmp1, tmp9)
         tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
         tmp13 = _tmp12 + tmp11
         _tmp12 = tl.where(rmask, tmp13, _tmp12)
         tmp16 = tmp14 - tmp15
         tmp17 = tmp10 * tmp16
         tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
         tmp20 = _tmp19 + tmp18
         _tmp19 = tl.where(rmask, tmp20, _tmp19)
     tmp12 = tl.sum(_tmp12, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp12, None)
     tmp19 = tl.sum(_tmp19, 1)[:, None]
     tl.store(out_ptr1 + (x0), tmp19, None)
     tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
     tmp22 = tmp19 * tmp21
     tl.store(out_ptr2 + (x0), tmp22, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/jd/cjdqomwphik75uixpx24zxafs4iqtf2o5c3r2qhswkdd3ivqdxtj.py
 # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
 
triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_6 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_6', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]})
 @triton.jit
 def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ks0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x3 = xindex
     x4 = (xindex // 49)
     x1 = (xindex // 49) % 2048
     tmp0 = tl.load(in_ptr0 + (x3), None)
     tmp3 = tl.load(in_ptr1 + (x3), None)
     tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
     tmp8 = tl.load(in_ptr3 + (x3), None)
     tmp11 = tl.load(in_ptr4 + (x3), None)
     tmp12 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
     tmp14 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
     tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
     tmp23 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
     tmp26 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
     tmp1 = 0.0
     tmp2 = tmp0 <= tmp1
     tmp5 = 49.0
     tmp6 = tmp4 / tmp5
     tmp7 = tl.where(tmp3, tmp1, tmp6)
     tmp9 = tmp7 + tmp8
     tmp10 = tl.where(tmp2, tmp1, tmp9)
     tmp13 = tmp11 - tmp12
     tmp15 = 0.0204081632653061*(1/ks0)
     tmp16 = tmp15.to(tl.float32)
     tmp17 = tmp14 * tmp16
     tmp19 = tmp18 * tmp18
     tmp20 = tmp17 * tmp19
     tmp21 = tmp13 * tmp20
     tmp22 = tmp10 - tmp21
     tmp24 = tmp23 * tmp16
     tmp25 = tmp22 - tmp24
     tmp27 = tmp18 * tmp26
     tmp28 = tmp25 * tmp27
     tl.store(in_out_ptr0 + (x3), tmp28, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/6q/c6qzvn2trotwbqbn2hla5rcprxuu5mutnszn4fso2rbfzpbzfyfw.py
 # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.threshold_backward]
 
triton_poi_fused_add_div_threshold_backward_7 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_threshold_backward_7', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]})
 @triton.jit
 def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x2 = xindex
     x1 = (xindex // 49)
     tmp0 = tl.load(in_ptr0 + (x2), None)
     tmp3 = tl.load(in_ptr1 + (x2), None)
     tmp5 = tl.load(in_ptr2 + (x2), None)
     tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
     tmp10 = tl.load(in_out_ptr0 + (x2), None)
     tmp13 = tl.load(in_ptr4 + (x2), None)
     tmp1 = 0.0
     tmp2 = tmp0 <= tmp1
     tmp4 = tmp3 <= tmp1
     tmp7 = 49.0
     tmp8 = tmp6 / tmp7
     tmp9 = tl.where(tmp5, tmp1, tmp8)
     tmp11 = tmp9 + tmp10
     tmp12 = tl.where(tmp4, tmp1, tmp11)
     tmp14 = tmp12 + tmp13
     tmp15 = tl.where(tmp2, tmp1, tmp14)
     tl.store(in_out_ptr0 + (x2), tmp15, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/7u/c7up7xfgf74nmxl7bubteriqym5ylfkkt65peupisi6ifrzkcgbc.py
 # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
 
triton_red_fused_native_batch_norm_backward_8 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @reduction(
     size_hints=[2048, 4096],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_8', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
     xnumel = 2048
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rbase = tl.arange(0, RBLOCK)[None, :]
     x0 = xindex
     _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
     _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
     _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
         rindex = roffset + rbase
         rmask = rindex < rnumel
         r1 = rindex % 49
         r2 = (rindex // 49)
         tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (100352*r2)), rmask, other=0)
         tmp4 = tl.load(in_ptr1 + (r1 + (49*x0) + (100352*r2)), rmask, other=0)
         tmp11 = tl.load(in_ptr3 + (r1 + (49*x0) + (100352*r2)), rmask, other=0)
         tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
         tmp3 = _tmp2 + tmp1
         _tmp2 = tl.where(rmask, tmp3, _tmp2)
         tmp6 = tmp4 - tmp5
         tmp7 = tmp0 * tmp6
         tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
         tmp10 = _tmp9 + tmp8
         _tmp9 = tl.where(rmask, tmp10, _tmp9)
         tmp13 = tmp11 - tmp12
         tmp14 = tmp0 * tmp13
         tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
         tmp17 = _tmp16 + tmp15
         _tmp16 = tl.where(rmask, tmp17, _tmp16)
     tmp2 = tl.sum(_tmp2, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp2, None)
     tmp9 = tl.sum(_tmp9, 1)[:, None]
     tl.store(out_ptr1 + (x0), tmp9, None)
     tmp16 = tl.sum(_tmp16, 1)[:, None]
     tl.store(out_ptr2 + (x0), tmp16, None)
     tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
     tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
     tmp19 = tmp9 * tmp18
     tmp21 = tmp16 * tmp20
     tl.store(out_ptr3 + (x0), tmp19, None)
     tl.store(out_ptr4 + (x0), tmp21, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/b6/cb6jkxngtjlysxosnaecvfm4oyzgeyrgt3ww6c6pyyomb7w5xfcc.py
 # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
 
 triton_poi_fused_convolution_backward_native_batch_norm_backward_9 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_9', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15,))]})
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x3 = xindex
     x1 = (xindex // 49) % 2048
     tmp0 = tl.load(in_ptr0 + (x3), None)
     tmp1 = tl.load(in_ptr1 + (x3), None)
     tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
     tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
     tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
     tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
     tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
     tmp19 = tl.load(in_ptr7 + (x3), None)
     tmp20 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
     tmp22 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
     tmp24 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
     tmp30 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
     tmp3 = tmp1 - tmp2
     tmp5 = 0.0204081632653061*(1/ks0)
     tmp6 = tmp5.to(tl.float32)
     tmp7 = tmp4 * tmp6
     tmp9 = tmp8 * tmp8
     tmp10 = tmp7 * tmp9
     tmp11 = tmp3 * tmp10
     tmp12 = tmp0 - tmp11
     tmp14 = tmp13 * tmp6
     tmp15 = tmp12 - tmp14
     tmp17 = tmp8 * tmp16
     tmp18 = tmp15 * tmp17
     tmp21 = tmp19 - tmp20
     tmp23 = tmp22 * tmp6
     tmp25 = tmp24 * tmp24
     tmp26 = tmp23 * tmp25
     tmp27 = tmp21 * tmp26
     tmp28 = tmp0 - tmp27
     tmp29 = tmp28 - tmp14
     tmp31 = tmp24 * tmp30
     tmp32 = tmp29 * tmp31
     tl.store(out_ptr0 + (x3), tmp18, None)
     tl.store(out_ptr1 + (x3), tmp32, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/24/c24vih7vxqlevvxqlliehwjwau6kdcug2ld75xmk5h6qmxfqgefz.py
 # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_red_fused_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @reduction(
     size_hints=[512, 16384],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_10', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
     xnumel = 512
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rbase = tl.arange(0, RBLOCK)[None, :]
     x0 = xindex
     _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
         rindex = roffset + rbase
         rmask = rindex < rnumel
         r1 = rindex % 196
         r2 = (rindex // 196)
         tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, other=0)
         tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, other=0)
         tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, other=0)
         tmp1 = 0.0
         tmp2 = tmp0 <= tmp1
         tmp4 = tl.where(tmp2, tmp1, tmp3)
         tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
         tmp7 = _tmp6 + tmp5
         _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
         tmp10 = tmp8 - tmp9
         tmp11 = tmp4 * tmp10
         tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
         tmp14 = _tmp13 + tmp12
         _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
     tmp6 = tl.sum(_tmp6, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp6, xmask)
     tmp13 = tl.sum(_tmp13, 1)[:, None]
     tl.store(out_ptr1 + (x0), tmp13, xmask)
     tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     tmp16 = tmp13 * tmp15
     tl.store(out_ptr2 + (x0), tmp16, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/e6/ce6tpxqubc2to42ednfgz53ny43guxofwpmdckaios5ngf5zfbmf.py
 # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]})
 @triton.jit
 def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ks0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x3 = xindex
     x1 = (xindex // 196) % 512
     tmp0 = tl.load(in_ptr0 + (x3), None)
     tmp3 = tl.load(in_out_ptr0 + (x3), None)
     tmp5 = tl.load(in_ptr1 + (x3), None)
     tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
     tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
     tmp17 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
     tmp20 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
     tmp1 = 0.0
     tmp2 = tmp0 <= tmp1
     tmp4 = tl.where(tmp2, tmp1, tmp3)
     tmp7 = tmp5 - tmp6
     tmp9 = 0.00510204081632653*(1/ks0)
     tmp10 = tmp9.to(tl.float32)
     tmp11 = tmp8 * tmp10
     tmp13 = tmp12 * tmp12
     tmp14 = tmp11 * tmp13
     tmp15 = tmp7 * tmp14
     tmp16 = tmp4 - tmp15
     tmp18 = tmp17 * tmp10
     tmp19 = tmp16 - tmp18
     tmp21 = tmp12 * tmp20
     tmp22 = tmp19 * tmp21
     tl.store(in_out_ptr0 + (x3), tmp22, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/23/c23rxrvmveb4qirlp643dhqxy7lr7fj4nshedoksbrr32sv4xu5f.py
 # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_red_fused_add_native_batch_norm_backward_threshold_backward_12 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @reduction(
     size_hints=[1024, 16384],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_12', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
     xnumel = 1024
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rbase = tl.arange(0, RBLOCK)[None, :]
     x0 = xindex
     _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
         rindex = roffset + rbase
         rmask = rindex < rnumel
         r1 = rindex % 196
         r2 = (rindex // 196)
         tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0)
         tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0)
         tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0)
         tmp10 = tl.load(in_ptr3 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0)
         tmp1 = 0.0
         tmp2 = tmp0 <= tmp1
         tmp5 = tmp3 + tmp4
         tmp6 = tl.where(tmp2, tmp1, tmp5)
         tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
         tmp9 = _tmp8 + tmp7
         _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
         tmp12 = tmp10 - tmp11
         tmp13 = tmp6 * tmp12
         tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
         tmp16 = _tmp15 + tmp14
         _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
     tmp8 = tl.sum(_tmp8, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp8, xmask)
     tmp15 = tl.sum(_tmp15, 1)[:, None]
     tl.store(out_ptr1 + (x0), tmp15, xmask)
     tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
     tmp18 = tmp15 * tmp17
     tl.store(out_ptr2 + (x0), tmp18, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/h4/ch42hpxbyv3eazoenftpjdt7ffgdy2yyscwun3qmn2vexvofjc7w.py
 # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_13', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]})
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x3 = xindex
     x1 = (xindex // 196) % 1024
     tmp0 = tl.load(in_ptr0 + (x3), None)
     tmp3 = tl.load(in_ptr1 + (x3), None)
     tmp4 = tl.load(in_ptr2 + (x3), None)
     tmp7 = tl.load(in_ptr3 + (x3), None)
     tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
     tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
     tmp14 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
     tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
     tmp22 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
     tmp1 = 0.0
     tmp2 = tmp0 <= tmp1
     tmp5 = tmp3 + tmp4
     tmp6 = tl.where(tmp2, tmp1, tmp5)
     tmp9 = tmp7 - tmp8
     tmp11 = 0.00510204081632653*(1/ks0)
     tmp12 = tmp11.to(tl.float32)
     tmp13 = tmp10 * tmp12
     tmp15 = tmp14 * tmp14
     tmp16 = tmp13 * tmp15
     tmp17 = tmp9 * tmp16
     tmp18 = tmp6 - tmp17
     tmp20 = tmp19 * tmp12
     tmp21 = tmp18 - tmp20
     tmp23 = tmp14 * tmp22
     tmp24 = tmp21 * tmp23
     tl.store(out_ptr0 + (x3), tmp24, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/4j/c4jsn7iionwndpvx3z5twl4hrfibgnsorqm7chqomnmboxbzny3z.py
 # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_red_fused_native_batch_norm_backward_threshold_backward_14 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @reduction(
     size_hints=[256, 16384],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_14', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
     xnumel = 256
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rbase = tl.arange(0, RBLOCK)[None, :]
     x0 = xindex
     _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
         rindex = roffset + rbase
         rmask = rindex < rnumel
         r1 = rindex % 196
         r2 = (rindex // 196)
         tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (50176*r2)), rmask & xmask, other=0)
         tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (50176*r2)), rmask & xmask, other=0)
         tmp8 = tl.load(in_ptr2 + (r1 + (196*x0) + (50176*r2)), rmask & xmask, other=0)
         tmp1 = 0.0
         tmp2 = tmp0 <= tmp1
         tmp4 = tl.where(tmp2, tmp1, tmp3)
         tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
         tmp7 = _tmp6 + tmp5
         _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
         tmp10 = tmp8 - tmp9
         tmp11 = tmp4 * tmp10
         tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
         tmp14 = _tmp13 + tmp12
         _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
     tmp6 = tl.sum(_tmp6, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp6, xmask)
     tmp13 = tl.sum(_tmp13, 1)[:, None]
     tl.store(out_ptr1 + (x0), tmp13, xmask)
     tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     tmp16 = tmp13 * tmp15
     tl.store(out_ptr2 + (x0), tmp16, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/4n/c4nplnvol55ybtebhkthtntaqg6wfyiaia4fs5zbioecu6lna4qx.py
 # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]})
 @triton.jit
 def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ks0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x3 = xindex
     x1 = (xindex // 196) % 256
     tmp0 = tl.load(in_ptr0 + (x3), xmask)
     tmp3 = tl.load(in_out_ptr0 + (x3), xmask)
     tmp5 = tl.load(in_ptr1 + (x3), xmask)
     tmp6 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
     tmp8 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
     tmp17 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
     tmp20 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
     tmp1 = 0.0
     tmp2 = tmp0 <= tmp1
     tmp4 = tl.where(tmp2, tmp1, tmp3)
     tmp7 = tmp5 - tmp6
     tmp9 = 0.00510204081632653*(1/ks0)
     tmp10 = tmp9.to(tl.float32)
     tmp11 = tmp8 * tmp10
     tmp13 = tmp12 * tmp12
     tmp14 = tmp11 * tmp13
     tmp15 = tmp7 * tmp14
     tmp16 = tmp4 - tmp15
     tmp18 = tmp17 * tmp10
     tmp19 = tmp16 - tmp18
     tmp21 = tmp12 * tmp20
     tmp22 = tmp19 * tmp21
     tl.store(in_out_ptr0 + (x3), tmp22, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/md/cmdm6lij2wlsgyarvjz3ndz6me7rsexpmvbovvdcjwgmi3rjfeaa.py
 # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
 
 triton_poi_fused_add_threshold_backward_16 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_16', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]})
 @triton.jit
 def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x0 = xindex
     tmp0 = tl.load(in_ptr0 + (x0), None)
     tmp3 = tl.load(in_ptr1 + (x0), None)
     tmp5 = tl.load(in_out_ptr0 + (x0), None)
     tmp6 = tl.load(in_ptr2 + (x0), None)
     tmp9 = tl.load(in_ptr3 + (x0), None)
     tmp1 = 0.0
     tmp2 = tmp0 <= tmp1
     tmp4 = tmp3 <= tmp1
     tmp7 = tmp5 + tmp6
     tmp8 = tl.where(tmp4, tmp1, tmp7)
     tmp10 = tmp8 + tmp9
     tmp11 = tl.where(tmp2, tmp1, tmp10)
     tl.store(in_out_ptr0 + (x0), tmp11, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/qj/cqjthocknb5gkvyp3ajjuie6byzm7fbdb3zrifiiuze4vks7vqgb.py
 # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
 
 triton_red_fused_native_batch_norm_backward_17 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @reduction(
     size_hints=[1024, 16384],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_17', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
     xnumel = 1024
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rbase = tl.arange(0, RBLOCK)[None, :]
     x0 = xindex
     _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
     _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
         rindex = roffset + rbase
         rmask = rindex < rnumel
         r1 = rindex % 196
         r2 = (rindex // 196)
         tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0)
         tmp4 = tl.load(in_ptr1 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0)
         tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
         tmp3 = _tmp2 + tmp1
         _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
         tmp6 = tmp4 - tmp5
         tmp7 = tmp0 * tmp6
         tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
         tmp10 = _tmp9 + tmp8
         _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
     tmp2 = tl.sum(_tmp2, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp2, xmask)
     tmp9 = tl.sum(_tmp9, 1)[:, None]
     tl.store(out_ptr1 + (x0), tmp9, xmask)
     tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     tmp12 = tmp9 * tmp11
     tl.store(out_ptr2 + (x0), tmp12, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/ep/ceplp5zb5co7duvxznyupkhfpqxc2nqo575fpxopin7nkg7r4sjg.py
 # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
 
 triton_poi_fused_convolution_backward_native_batch_norm_backward_18 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_18', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]})
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x3 = xindex
     x1 = (xindex // 196) % 1024
     tmp0 = tl.load(in_ptr0 + (x3), None)
     tmp1 = tl.load(in_ptr1 + (x3), None)
     tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
     tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
     tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
     tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
     tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
     tmp3 = tmp1 - tmp2
     tmp5 = 0.00510204081632653*(1/ks0)
     tmp6 = tmp5.to(tl.float32)
     tmp7 = tmp4 * tmp6
     tmp9 = tmp8 * tmp8
     tmp10 = tmp7 * tmp9
     tmp11 = tmp3 * tmp10
     tmp12 = tmp0 - tmp11
     tmp14 = tmp13 * tmp6
     tmp15 = tmp12 - tmp14
     tmp17 = tmp8 * tmp16
     tmp18 = tmp15 * tmp17
     tl.store(out_ptr0 + (x3), tmp18, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/2q/c2qmbwfrvomydgki2qvqcefck2horxo4c4rhbkpxwcb6fyqptk77.py
 # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
 
 triton_poi_fused_add_threshold_backward_19 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_19', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]})
 @triton.jit
 def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x0 = xindex
     tmp0 = tl.load(in_ptr0 + (x0), None)
     tmp3 = tl.load(in_ptr1 + (x0), None)
     tmp5 = tl.load(in_ptr2 + (x0), None)
     tmp6 = tl.load(in_out_ptr0 + (x0), None)
     tmp9 = tl.load(in_ptr3 + (x0), None)
     tmp1 = 0.0
     tmp2 = tmp0 <= tmp1
     tmp4 = tmp3 <= tmp1
     tmp7 = tmp5 + tmp6
     tmp8 = tl.where(tmp4, tmp1, tmp7)
     tmp10 = tmp8 + tmp9
     tmp11 = tl.where(tmp2, tmp1, tmp10)
     tl.store(in_out_ptr0 + (x0), tmp11, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/pk/cpk5ofsab3ypzazhi2dgkadefpxh7xxfmpqoeevzc4leybxpajko.py
 # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
 
 triton_red_fused_native_batch_norm_backward_20 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @reduction(
     size_hints=[1024, 16384],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_20', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
     xnumel = 1024
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rbase = tl.arange(0, RBLOCK)[None, :]
     x0 = xindex
     _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
     _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
         rindex = roffset + rbase
         rmask = rindex < rnumel
         r1 = rindex % 196
         r2 = (rindex // 196)
         tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0)
         tmp4 = tl.load(in_ptr1 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0)
         tmp11 = tl.load(in_ptr3 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, other=0)
         tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
         tmp3 = _tmp2 + tmp1
         _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
         tmp6 = tmp4 - tmp5
         tmp7 = tmp0 * tmp6
         tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
         tmp10 = _tmp9 + tmp8
         _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
         tmp13 = tmp11 - tmp12
         tmp14 = tmp0 * tmp13
         tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
         tmp17 = _tmp16 + tmp15
         _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
     tmp2 = tl.sum(_tmp2, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp2, xmask)
     tmp9 = tl.sum(_tmp9, 1)[:, None]
     tl.store(out_ptr1 + (x0), tmp9, xmask)
     tmp16 = tl.sum(_tmp16, 1)[:, None]
     tl.store(out_ptr2 + (x0), tmp16, xmask)
     tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
     tmp20 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
     tmp19 = tmp9 * tmp18
     tmp21 = tmp16 * tmp20
     tl.store(out_ptr3 + (x0), tmp19, xmask)
     tl.store(out_ptr4 + (x0), tmp21, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/pg/cpgynpabosa6ymdf5w2yej5aewg43ef23hsljkvc7t4bylmwjkag.py
 # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
 
 triton_poi_fused_convolution_backward_native_batch_norm_backward_21 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_21', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15,))]})
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x3 = xindex
     x1 = (xindex // 196) % 1024
     tmp0 = tl.load(in_ptr0 + (x3), None)
     tmp1 = tl.load(in_ptr1 + (x3), None)
     tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
     tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
     tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
     tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
     tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
     tmp19 = tl.load(in_ptr7 + (x3), None)
     tmp20 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
     tmp22 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
     tmp24 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
     tmp30 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
     tmp3 = tmp1 - tmp2
     tmp5 = 0.00510204081632653*(1/ks0)
     tmp6 = tmp5.to(tl.float32)
     tmp7 = tmp4 * tmp6
     tmp9 = tmp8 * tmp8
     tmp10 = tmp7 * tmp9
     tmp11 = tmp3 * tmp10
     tmp12 = tmp0 - tmp11
     tmp14 = tmp13 * tmp6
     tmp15 = tmp12 - tmp14
     tmp17 = tmp8 * tmp16
     tmp18 = tmp15 * tmp17
     tmp21 = tmp19 - tmp20
     tmp23 = tmp22 * tmp6
     tmp25 = tmp24 * tmp24
     tmp26 = tmp23 * tmp25
     tmp27 = tmp21 * tmp26
     tmp28 = tmp0 - tmp27
     tmp29 = tmp28 - tmp14
     tmp31 = tmp24 * tmp30
     tmp32 = tmp29 * tmp31
     tl.store(out_ptr0 + (x3), tmp18, None)
     tl.store(out_ptr1 + (x3), tmp32, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/o2/co2kiyiqsbf4d4qe6pllznazm5tqmiibcg7fai4doozqx4fsp6h5.py
 # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_red_fused_native_batch_norm_backward_threshold_backward_22 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @reduction(
     size_hints=[256, 65536],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_22', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
     xnumel = 256
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rbase = tl.arange(0, RBLOCK)[None, :]
     x0 = xindex
     _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
         rindex = roffset + rbase
         rmask = rindex < rnumel
         r1 = rindex % 784
         r2 = (rindex // 784)
         tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, other=0)
         tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, other=0)
         tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, other=0)
         tmp1 = 0.0
         tmp2 = tmp0 <= tmp1
         tmp4 = tl.where(tmp2, tmp1, tmp3)
         tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
         tmp7 = _tmp6 + tmp5
         _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
         tmp10 = tmp8 - tmp9
         tmp11 = tmp4 * tmp10
         tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
         tmp14 = _tmp13 + tmp12
         _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
     tmp6 = tl.sum(_tmp6, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp6, xmask)
     tmp13 = tl.sum(_tmp13, 1)[:, None]
     tl.store(out_ptr1 + (x0), tmp13, xmask)
     tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     tmp16 = tmp13 * tmp15
     tl.store(out_ptr2 + (x0), tmp16, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/ld/cldtb3mh6a6har3bb757hysirrqr3lkqc7hg6gvhisaiy3iskxul.py
 # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_23', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]})
 @triton.jit
 def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ks0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x3 = xindex
     x1 = (xindex // 784) % 256
     tmp0 = tl.load(in_ptr0 + (x3), None)
     tmp3 = tl.load(in_out_ptr0 + (x3), None)
     tmp5 = tl.load(in_ptr1 + (x3), None)
     tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
     tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
     tmp17 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
     tmp20 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
     tmp1 = 0.0
     tmp2 = tmp0 <= tmp1
     tmp4 = tl.where(tmp2, tmp1, tmp3)
     tmp7 = tmp5 - tmp6
     tmp9 = 0.00127551020408163*(1/ks0)
     tmp10 = tmp9.to(tl.float32)
     tmp11 = tmp8 * tmp10
     tmp13 = tmp12 * tmp12
     tmp14 = tmp11 * tmp13
     tmp15 = tmp7 * tmp14
     tmp16 = tmp4 - tmp15
     tmp18 = tmp17 * tmp10
     tmp19 = tmp16 - tmp18
     tmp21 = tmp12 * tmp20
     tmp22 = tmp19 * tmp21
     tl.store(in_out_ptr0 + (x3), tmp22, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/lw/clwcfhdpg7u5g75i3wtvoqwlvolqxecvwf2345sj27efrvsfuhtt.py
 # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_red_fused_add_native_batch_norm_backward_threshold_backward_24 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @reduction(
     size_hints=[512, 65536],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_24', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
     xnumel = 512
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rbase = tl.arange(0, RBLOCK)[None, :]
     x0 = xindex
     _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
         rindex = roffset + rbase
         rmask = rindex < rnumel
         r1 = rindex % 784
         r2 = (rindex // 784)
         tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, other=0)
         tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, other=0)
         tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, other=0)
         tmp10 = tl.load(in_ptr3 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, other=0)
         tmp1 = 0.0
         tmp2 = tmp0 <= tmp1
         tmp5 = tmp3 + tmp4
         tmp6 = tl.where(tmp2, tmp1, tmp5)
         tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
         tmp9 = _tmp8 + tmp7
         _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
         tmp12 = tmp10 - tmp11
         tmp13 = tmp6 * tmp12
         tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
         tmp16 = _tmp15 + tmp14
         _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
     tmp8 = tl.sum(_tmp8, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp8, xmask)
     tmp15 = tl.sum(_tmp15, 1)[:, None]
     tl.store(out_ptr1 + (x0), tmp15, xmask)
     tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
     tmp18 = tmp15 * tmp17
     tl.store(out_ptr2 + (x0), tmp18, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/tv/ctvilmuvwov64m67kg356xnce5i2xuu6tmgjypzuns7233mhgv2x.py
 # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_poi_fused_add_native_batch_norm_backward_threshold_backward_25 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_25', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]})
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x3 = xindex
     x1 = (xindex // 784) % 512
     tmp0 = tl.load(in_ptr0 + (x3), None)
     tmp3 = tl.load(in_ptr1 + (x3), None)
     tmp4 = tl.load(in_ptr2 + (x3), None)
     tmp7 = tl.load(in_ptr3 + (x3), None)
     tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
     tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
     tmp14 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
     tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
     tmp22 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
     tmp1 = 0.0
     tmp2 = tmp0 <= tmp1
     tmp5 = tmp3 + tmp4
     tmp6 = tl.where(tmp2, tmp1, tmp5)
     tmp9 = tmp7 - tmp8
     tmp11 = 0.00127551020408163*(1/ks0)
     tmp12 = tmp11.to(tl.float32)
     tmp13 = tmp10 * tmp12
     tmp15 = tmp14 * tmp14
     tmp16 = tmp13 * tmp15
     tmp17 = tmp9 * tmp16
     tmp18 = tmp6 - tmp17
     tmp20 = tmp19 * tmp12
     tmp21 = tmp18 - tmp20
     tmp23 = tmp14 * tmp22
     tmp24 = tmp21 * tmp23
     tl.store(out_ptr0 + (x3), tmp24, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/wz/cwz6bcsbu5gl2gzkaic263jrcxyztxolqohwgyvp4xa5cstshqig.py
 # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_red_fused_native_batch_norm_backward_threshold_backward_26 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @reduction(
     size_hints=[128, 65536],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_26', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
     xnumel = 128
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rbase = tl.arange(0, RBLOCK)[None, :]
     x0 = xindex
     _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
         rindex = roffset + rbase
         rmask = rindex < rnumel
         r1 = rindex % 784
         r2 = (rindex // 784)
         tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (100352*r2)), rmask & xmask, other=0)
         tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (100352*r2)), rmask & xmask, other=0)
         tmp8 = tl.load(in_ptr2 + (r1 + (784*x0) + (100352*r2)), rmask & xmask, other=0)
         tmp1 = 0.0
         tmp2 = tmp0 <= tmp1
         tmp4 = tl.where(tmp2, tmp1, tmp3)
         tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
         tmp7 = _tmp6 + tmp5
         _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
         tmp10 = tmp8 - tmp9
         tmp11 = tmp4 * tmp10
         tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
         tmp14 = _tmp13 + tmp12
         _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
     tmp6 = tl.sum(_tmp6, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp6, xmask)
     tmp13 = tl.sum(_tmp13, 1)[:, None]
     tl.store(out_ptr1 + (x0), tmp13, xmask)
     tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     tmp16 = tmp13 * tmp15
     tl.store(out_ptr2 + (x0), tmp16, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/ul/culfu5wtwnwl42rzqzommyruyd4gcoksiclufsirho2fz6d6dzty.py
 # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]})
 @triton.jit
 def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ks0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x3 = xindex
     x1 = (xindex // 784) % 128
     tmp0 = tl.load(in_ptr0 + (x3), None)
     tmp3 = tl.load(in_out_ptr0 + (x3), None)
     tmp5 = tl.load(in_ptr1 + (x3), None)
     tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
     tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
     tmp17 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
     tmp20 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
     tmp1 = 0.0
     tmp2 = tmp0 <= tmp1
     tmp4 = tl.where(tmp2, tmp1, tmp3)
     tmp7 = tmp5 - tmp6
     tmp9 = 0.00127551020408163*(1/ks0)
     tmp10 = tmp9.to(tl.float32)
     tmp11 = tmp8 * tmp10
     tmp13 = tmp12 * tmp12
     tmp14 = tmp11 * tmp13
     tmp15 = tmp7 * tmp14
     tmp16 = tmp4 - tmp15
     tmp18 = tmp17 * tmp10
     tmp19 = tmp16 - tmp18
     tmp21 = tmp12 * tmp20
     tmp22 = tmp19 * tmp21
     tl.store(in_out_ptr0 + (x3), tmp22, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/o6/co6gg22utt7tbrfoa6fmlebsgib5zr3ro7cyvahyiexfswknyya4.py
 # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
 
 triton_poi_fused_add_threshold_backward_28 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_28', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]})
 @triton.jit
 def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x0 = xindex
     tmp0 = tl.load(in_ptr0 + (x0), None)
     tmp3 = tl.load(in_ptr1 + (x0), None)
     tmp5 = tl.load(in_out_ptr0 + (x0), None)
     tmp6 = tl.load(in_ptr2 + (x0), None)
     tmp9 = tl.load(in_ptr3 + (x0), None)
     tmp1 = 0.0
     tmp2 = tmp0 <= tmp1
     tmp4 = tmp3 <= tmp1
     tmp7 = tmp5 + tmp6
     tmp8 = tl.where(tmp4, tmp1, tmp7)
     tmp10 = tmp8 + tmp9
     tmp11 = tl.where(tmp2, tmp1, tmp10)
     tl.store(in_out_ptr0 + (x0), tmp11, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/l5/cl5tvvoswz4iggh6ya2cucb54dr73ci5pjzvzao7sgxuxvrrxa35.py
 # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
 
 triton_red_fused_native_batch_norm_backward_29 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @reduction(
     size_hints=[512, 65536],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_29', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
     xnumel = 512
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rbase = tl.arange(0, RBLOCK)[None, :]
     x0 = xindex
     _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
     _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
         rindex = roffset + rbase
         rmask = rindex < rnumel
         r1 = rindex % 784
         r2 = (rindex // 784)
         tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, other=0)
         tmp4 = tl.load(in_ptr1 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, other=0)
         tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
         tmp3 = _tmp2 + tmp1
         _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
         tmp6 = tmp4 - tmp5
         tmp7 = tmp0 * tmp6
         tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
         tmp10 = _tmp9 + tmp8
         _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
     tmp2 = tl.sum(_tmp2, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp2, xmask)
     tmp9 = tl.sum(_tmp9, 1)[:, None]
     tl.store(out_ptr1 + (x0), tmp9, xmask)
     tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     tmp12 = tmp9 * tmp11
     tl.store(out_ptr2 + (x0), tmp12, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/x3/cx3dui54ukee3i5uyliwhh3xuy46u3bl2eftb6haddq5x4bqta4a.py
 # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
 
 triton_poi_fused_convolution_backward_native_batch_norm_backward_30 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_30', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]})
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x3 = xindex
     x1 = (xindex // 784) % 512
     tmp0 = tl.load(in_ptr0 + (x3), None)
     tmp1 = tl.load(in_ptr1 + (x3), None)
     tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
     tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
     tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
     tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
     tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
     tmp3 = tmp1 - tmp2
     tmp5 = 0.00127551020408163*(1/ks0)
     tmp6 = tmp5.to(tl.float32)
     tmp7 = tmp4 * tmp6
     tmp9 = tmp8 * tmp8
     tmp10 = tmp7 * tmp9
     tmp11 = tmp3 * tmp10
     tmp12 = tmp0 - tmp11
     tmp14 = tmp13 * tmp6
     tmp15 = tmp12 - tmp14
     tmp17 = tmp8 * tmp16
     tmp18 = tmp15 * tmp17
     tl.store(out_ptr0 + (x3), tmp18, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/4p/c4prtcmloooaw3gjiqloc4upp5ewgovkrnwwq5tf54ec2jjregwj.py
 # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
 
 triton_red_fused_native_batch_norm_backward_31 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @reduction(
     size_hints=[512, 65536],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_31', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
     xnumel = 512
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rbase = tl.arange(0, RBLOCK)[None, :]
     x0 = xindex
     _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
     _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
         rindex = roffset + rbase
         rmask = rindex < rnumel
         r1 = rindex % 784
         r2 = (rindex // 784)
         tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, other=0)
         tmp4 = tl.load(in_ptr1 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, other=0)
         tmp11 = tl.load(in_ptr3 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, other=0)
         tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
         tmp3 = _tmp2 + tmp1
         _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
         tmp6 = tmp4 - tmp5
         tmp7 = tmp0 * tmp6
         tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
         tmp10 = _tmp9 + tmp8
         _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
         tmp13 = tmp11 - tmp12
         tmp14 = tmp0 * tmp13
         tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
         tmp17 = _tmp16 + tmp15
         _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
     tmp2 = tl.sum(_tmp2, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp2, xmask)
     tmp9 = tl.sum(_tmp9, 1)[:, None]
     tl.store(out_ptr1 + (x0), tmp9, xmask)
     tmp16 = tl.sum(_tmp16, 1)[:, None]
     tl.store(out_ptr2 + (x0), tmp16, xmask)
     tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
     tmp20 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
     tmp19 = tmp9 * tmp18
     tmp21 = tmp16 * tmp20
     tl.store(out_ptr3 + (x0), tmp19, xmask)
     tl.store(out_ptr4 + (x0), tmp21, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/hz/chz64zuwnvig2mya5ku37q7j7xjyejhzfdoqaey3arweepl7s6oe.py
 # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
 
 triton_poi_fused_convolution_backward_native_batch_norm_backward_32 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_32', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15,))]})
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x3 = xindex
     x1 = (xindex // 784) % 512
     tmp0 = tl.load(in_ptr0 + (x3), None)
     tmp1 = tl.load(in_ptr1 + (x3), None)
     tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
     tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
     tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
     tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
     tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
     tmp19 = tl.load(in_ptr7 + (x3), None)
     tmp20 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
     tmp22 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
     tmp24 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
     tmp30 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
     tmp3 = tmp1 - tmp2
     tmp5 = 0.00127551020408163*(1/ks0)
     tmp6 = tmp5.to(tl.float32)
     tmp7 = tmp4 * tmp6
     tmp9 = tmp8 * tmp8
     tmp10 = tmp7 * tmp9
     tmp11 = tmp3 * tmp10
     tmp12 = tmp0 - tmp11
     tmp14 = tmp13 * tmp6
     tmp15 = tmp12 - tmp14
     tmp17 = tmp8 * tmp16
     tmp18 = tmp15 * tmp17
     tmp21 = tmp19 - tmp20
     tmp23 = tmp22 * tmp6
     tmp25 = tmp24 * tmp24
     tmp26 = tmp23 * tmp25
     tmp27 = tmp21 * tmp26
     tmp28 = tmp0 - tmp27
     tmp29 = tmp28 - tmp14
     tmp31 = tmp24 * tmp30
     tmp32 = tmp29 * tmp31
     tl.store(out_ptr0 + (x3), tmp18, None)
     tl.store(out_ptr1 + (x3), tmp32, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/fj/cfjgxpyhzmas3ywfmpanc6wenfp7zn2yyp7tuprf6f2echewlpea.py
 # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_red_fused_native_batch_norm_backward_threshold_backward_33 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @reduction(
     size_hints=[128, 262144],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_33', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
     xnumel = 128
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rbase = tl.arange(0, RBLOCK)[None, :]
     x0 = xindex
     _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
         rindex = roffset + rbase
         rmask = rindex < rnumel
         r1 = rindex % 3136
         r2 = (rindex // 3136)
         tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (401408*r2)), rmask & xmask, other=0)
         tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (401408*r2)), rmask & xmask, other=0)
         tmp8 = tl.load(in_ptr2 + (r1 + (3136*x0) + (401408*r2)), rmask & xmask, other=0)
         tmp1 = 0.0
         tmp2 = tmp0 <= tmp1
         tmp4 = tl.where(tmp2, tmp1, tmp3)
         tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
         tmp7 = _tmp6 + tmp5
         _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
         tmp10 = tmp8 - tmp9
         tmp11 = tmp4 * tmp10
         tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
         tmp14 = _tmp13 + tmp12
         _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
     tmp6 = tl.sum(_tmp6, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp6, xmask)
     tmp13 = tl.sum(_tmp13, 1)[:, None]
     tl.store(out_ptr1 + (x0), tmp13, xmask)
     tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     tmp16 = tmp13 * tmp15
     tl.store(out_ptr2 + (x0), tmp16, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/ap/cap263e23grvlefwsdz7ij4wtjgc7rbpl27rnndcprizgikyqa2j.py
 # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_34 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_34', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]})
 @triton.jit
 def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ks0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x3 = xindex
     x1 = (xindex // 3136) % 128
     tmp0 = tl.load(in_ptr0 + (x3), None)
     tmp3 = tl.load(in_out_ptr0 + (x3), None)
     tmp5 = tl.load(in_ptr1 + (x3), None)
     tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
     tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
     tmp17 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
     tmp20 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
     tmp1 = 0.0
     tmp2 = tmp0 <= tmp1
     tmp4 = tl.where(tmp2, tmp1, tmp3)
     tmp7 = tmp5 - tmp6
     tmp9 = 0.000318877551020408*(1/ks0)
     tmp10 = tmp9.to(tl.float32)
     tmp11 = tmp8 * tmp10
     tmp13 = tmp12 * tmp12
     tmp14 = tmp11 * tmp13
     tmp15 = tmp7 * tmp14
     tmp16 = tmp4 - tmp15
     tmp18 = tmp17 * tmp10
     tmp19 = tmp16 - tmp18
     tmp21 = tmp12 * tmp20
     tmp22 = tmp19 * tmp21
     tl.store(in_out_ptr0 + (x3), tmp22, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/bq/cbqr7k63kkwcjfsuzgmq7ikuy77y7zsrmrfscsa6ztnygmnzt3ru.py
 # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_red_fused_add_native_batch_norm_backward_threshold_backward_35 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @reduction(
     size_hints=[256, 262144],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_35', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
     xnumel = 256
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rbase = tl.arange(0, RBLOCK)[None, :]
     x0 = xindex
     _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
         rindex = roffset + rbase
         rmask = rindex < rnumel
         r1 = rindex % 3136
         r2 = (rindex // 3136)
         tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, other=0)
         tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, other=0)
         tmp4 = tl.load(in_ptr2 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, other=0)
         tmp10 = tl.load(in_ptr3 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, other=0)
         tmp1 = 0.0
         tmp2 = tmp0 <= tmp1
         tmp5 = tmp3 + tmp4
         tmp6 = tl.where(tmp2, tmp1, tmp5)
         tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
         tmp9 = _tmp8 + tmp7
         _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
         tmp12 = tmp10 - tmp11
         tmp13 = tmp6 * tmp12
         tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
         tmp16 = _tmp15 + tmp14
         _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
     tmp8 = tl.sum(_tmp8, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp8, xmask)
     tmp15 = tl.sum(_tmp15, 1)[:, None]
     tl.store(out_ptr1 + (x0), tmp15, xmask)
     tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
     tmp18 = tmp15 * tmp17
     tl.store(out_ptr2 + (x0), tmp18, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/cn/ccn3oxfnvgixxzs2z6meklfn3ebkqalm6hyyw6mqkh4mldylurlm.py
 # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_poi_fused_add_native_batch_norm_backward_threshold_backward_36 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_36', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]})
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x3 = xindex
     x1 = (xindex // 3136) % 256
     tmp0 = tl.load(in_ptr0 + (x3), None)
     tmp3 = tl.load(in_ptr1 + (x3), None)
     tmp4 = tl.load(in_ptr2 + (x3), None)
     tmp7 = tl.load(in_ptr3 + (x3), None)
     tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
     tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
     tmp14 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
     tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
     tmp22 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
     tmp1 = 0.0
     tmp2 = tmp0 <= tmp1
     tmp5 = tmp3 + tmp4
     tmp6 = tl.where(tmp2, tmp1, tmp5)
     tmp9 = tmp7 - tmp8
     tmp11 = 0.000318877551020408*(1/ks0)
     tmp12 = tmp11.to(tl.float32)
     tmp13 = tmp10 * tmp12
     tmp15 = tmp14 * tmp14
     tmp16 = tmp13 * tmp15
     tmp17 = tmp9 * tmp16
     tmp18 = tmp6 - tmp17
     tmp20 = tmp19 * tmp12
     tmp21 = tmp18 - tmp20
     tmp23 = tmp14 * tmp22
     tmp24 = tmp21 * tmp23
     tl.store(out_ptr0 + (x3), tmp24, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/v5/cv5ahamyjoepiaaivfdt6ngwkd5roxs7lu2v46ih4ypsiavg4umn.py
 # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_red_fused_native_batch_norm_backward_threshold_backward_37 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @reduction(
     size_hints=[256, 65536],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_37', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
     xnumel = 256
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rbase = tl.arange(0, RBLOCK)[None, :]
     x0 = xindex % 64
     x1 = (xindex // 64)
     _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     x3 = xindex
     tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
         rindex = roffset + rbase
         rmask = rindex < rnumel
         r2 = rindex
         tmp0 = tl.load(in_ptr0 + ((56*(((r2 + (784*ks0*x1)) // 56) % 56)) + (3136*x0) + (200704*((r2 + (784*ks0*x1)) // 3136)) + (r2 % 56)), rmask & xmask, other=0)
         tmp3 = tl.load(in_ptr1 + ((56*(((r2 + (784*ks0*x1)) // 56) % 56)) + (3136*x0) + (200704*((r2 + (784*ks0*x1)) // 3136)) + (r2 % 56)), rmask & xmask, other=0)
         tmp8 = tl.load(in_ptr2 + ((56*(((r2 + (784*ks0*x1)) // 56) % 56)) + (3136*x0) + (200704*((r2 + (784*ks0*x1)) // 3136)) + (r2 % 56)), rmask & xmask, other=0)
         tmp1 = 0.0
         tmp2 = tmp0 <= tmp1
         tmp4 = tl.where(tmp2, tmp1, tmp3)
         tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
         tmp7 = _tmp6 + tmp5
         _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
         tmp10 = tmp8 - tmp9
         tmp11 = tmp4 * tmp10
         tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
         tmp14 = _tmp13 + tmp12
         _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
     tmp6 = tl.sum(_tmp6, 1)[:, None]
     tl.store(out_ptr0 + (x3), tmp6, xmask)
     tmp13 = tl.sum(_tmp13, 1)[:, None]
     tl.store(out_ptr1 + (x3), tmp13, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/gv/cgvkspuz5gcamic2i3mwhrdwnhumwqco6chaezqof5dt2sff6szd.py
 # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_per_fused_native_batch_norm_backward_threshold_backward_38 = async_compile.triton('triton_', '''
[2023-12-21 13:33:53,446] [0/1] torch._inductor.graph.__output_code: [DEBUG] import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @persistent_reduction(
     size_hints=[64, 4],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_38', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]}
 )
 @triton.jit
 def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
     xnumel = 64
     rnumel = 4
     RBLOCK: tl.constexpr = 4
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rindex = tl.arange(0, RBLOCK)[None, :]
     rmask = rindex < rnumel
     r1 = rindex
     x0 = xindex
     tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0)
     tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
     tmp3 = tl.where(rmask & xmask, tmp1, 0)
     tmp4 = tl.sum(tmp3, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp4, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/bs/cbsesfeqiw2xwqrsc6cr6slwi4366ijfxumpwjscfqxri2d7omj6.py
 # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_per_fused_native_batch_norm_backward_threshold_backward_39 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @persistent_reduction(
     size_hints=[64, 4],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_39', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
     xnumel = 64
     rnumel = 4
     RBLOCK: tl.constexpr = 4
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rindex = tl.arange(0, RBLOCK)[None, :]
     rmask = rindex < rnumel
     r1 = rindex
     x0 = xindex
     tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0)
     tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
     tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
     tmp3 = tl.where(rmask & xmask, tmp1, 0)
     tmp4 = tl.sum(tmp3, 1)[:, None]
     tmp6 = tmp4 * tmp5
     tl.store(out_ptr1 + (x0), tmp6, xmask)
     tl.store(out_ptr0 + (x0), tmp4, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/64/c64ls52gebq2c44eg2zvnn4bauqjbdfmimmjgckzgghm3krmmyjt.py
 # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]})
 @triton.jit
 def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ks0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x3 = xindex
     x1 = (xindex // 3136) % 64
     tmp0 = tl.load(in_ptr0 + (x3), None)
     tmp3 = tl.load(in_out_ptr0 + (x3), None)
     tmp5 = tl.load(in_ptr1 + (x3), None)
     tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
     tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
     tmp17 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
     tmp20 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
     tmp1 = 0.0
     tmp2 = tmp0 <= tmp1
     tmp4 = tl.where(tmp2, tmp1, tmp3)
     tmp7 = tmp5 - tmp6
     tmp9 = 0.000318877551020408*(1/ks0)
     tmp10 = tmp9.to(tl.float32)
     tmp11 = tmp8 * tmp10
     tmp13 = tmp12 * tmp12
     tmp14 = tmp11 * tmp13
     tmp15 = tmp7 * tmp14
     tmp16 = tmp4 - tmp15
     tmp18 = tmp17 * tmp10
     tmp19 = tmp16 - tmp18
     tmp21 = tmp12 * tmp20
     tmp22 = tmp19 * tmp21
     tl.store(in_out_ptr0 + (x3), tmp22, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/p2/cp2oomdeto57sstdgct7hxcp4zffimtat3mhpocqrrbhivip67go.py
 # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
 
 triton_poi_fused_add_threshold_backward_41 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_41', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]})
 @triton.jit
 def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x0 = xindex
     tmp0 = tl.load(in_ptr0 + (x0), None)
     tmp3 = tl.load(in_ptr1 + (x0), None)
     tmp5 = tl.load(in_out_ptr0 + (x0), None)
     tmp6 = tl.load(in_ptr2 + (x0), None)
     tmp9 = tl.load(in_ptr3 + (x0), None)
     tmp1 = 0.0
     tmp2 = tmp0 <= tmp1
     tmp4 = tmp3 <= tmp1
     tmp7 = tmp5 + tmp6
     tmp8 = tl.where(tmp4, tmp1, tmp7)
     tmp10 = tmp8 + tmp9
     tmp11 = tl.where(tmp2, tmp1, tmp10)
     tl.store(in_out_ptr0 + (x0), tmp11, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/or/cor5udrs4nraamoqd54i7t7bzsbk72xkt526do37fyffwzmdot3e.py
 # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
 
 triton_red_fused_native_batch_norm_backward_42 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @reduction(
     size_hints=[256, 262144],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_42', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
     xnumel = 256
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rbase = tl.arange(0, RBLOCK)[None, :]
     x0 = xindex
     _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
     _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
         rindex = roffset + rbase
         rmask = rindex < rnumel
         r1 = rindex % 3136
         r2 = (rindex // 3136)
         tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, other=0)
         tmp4 = tl.load(in_ptr1 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, other=0)
         tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
         tmp3 = _tmp2 + tmp1
         _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
         tmp6 = tmp4 - tmp5
         tmp7 = tmp0 * tmp6
         tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
         tmp10 = _tmp9 + tmp8
         _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
     tmp2 = tl.sum(_tmp2, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp2, xmask)
     tmp9 = tl.sum(_tmp9, 1)[:, None]
     tl.store(out_ptr1 + (x0), tmp9, xmask)
     tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     tmp12 = tmp9 * tmp11
     tl.store(out_ptr2 + (x0), tmp12, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/om/com7ll47yjb7uronw6tbnsbydpsijyjqlhjsu7fqiokfa3jzycll.py
 # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
 
 triton_poi_fused_convolution_backward_native_batch_norm_backward_43 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_43', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]})
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x3 = xindex
     x1 = (xindex // 3136) % 256
     tmp0 = tl.load(in_ptr0 + (x3), None)
     tmp1 = tl.load(in_ptr1 + (x3), None)
     tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
     tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
     tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
     tmp13 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
     tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
     tmp3 = tmp1 - tmp2
     tmp5 = 0.000318877551020408*(1/ks0)
     tmp6 = tmp5.to(tl.float32)
     tmp7 = tmp4 * tmp6
     tmp9 = tmp8 * tmp8
     tmp10 = tmp7 * tmp9
     tmp11 = tmp3 * tmp10
     tmp12 = tmp0 - tmp11
     tmp14 = tmp13 * tmp6
     tmp15 = tmp12 - tmp14
     tmp17 = tmp8 * tmp16
     tmp18 = tmp15 * tmp17
     tl.store(out_ptr0 + (x3), tmp18, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/fv/cfvnlbibzomacws7lym6k2dqzgjyuencw4g3ovnpt6d3rdsbaag6.py
 # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_red_fused_add_native_batch_norm_backward_threshold_backward_44 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @reduction(
     size_hints=[256, 262144],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_44', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
     xnumel = 256
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rbase = tl.arange(0, RBLOCK)[None, :]
     x0 = xindex
     _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
     _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     tmp18 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
     _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
         rindex = roffset + rbase
         rmask = rindex < rnumel
         r1 = rindex % 3136
         r2 = (rindex // 3136)
         tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, other=0)
         tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, other=0)
         tmp4 = tl.load(in_ptr2 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, other=0)
         tmp10 = tl.load(in_ptr3 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, other=0)
         tmp17 = tl.load(in_ptr5 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, other=0)
         tmp1 = 0.0
         tmp2 = tmp0 <= tmp1
         tmp5 = tmp3 + tmp4
         tmp6 = tl.where(tmp2, tmp1, tmp5)
         tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
         tmp9 = _tmp8 + tmp7
         _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
         tmp12 = tmp10 - tmp11
         tmp13 = tmp6 * tmp12
         tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
         tmp16 = _tmp15 + tmp14
         _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
         tmp19 = tmp17 - tmp18
         tmp20 = tmp6 * tmp19
         tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
         tmp23 = _tmp22 + tmp21
         _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
     tmp8 = tl.sum(_tmp8, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp8, xmask)
     tmp15 = tl.sum(_tmp15, 1)[:, None]
     tl.store(out_ptr1 + (x0), tmp15, xmask)
     tmp22 = tl.sum(_tmp22, 1)[:, None]
     tl.store(out_ptr2 + (x0), tmp22, xmask)
     tmp24 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
     tmp26 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
     tmp25 = tmp15 * tmp24
     tmp27 = tmp22 * tmp26
     tl.store(out_ptr3 + (x0), tmp25, xmask)
     tl.store(out_ptr4 + (x0), tmp27, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/4u/c4umiogtjmooiholzsqrvba54s763ibs6bgxdpeck4grmcieqeky.py
 # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_poi_fused_add_native_batch_norm_backward_threshold_backward_45 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32', 17: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_45', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(17,))]})
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x3 = xindex
     x1 = (xindex // 3136) % 256
     tmp0 = tl.load(in_ptr0 + (x3), None)
     tmp3 = tl.load(in_ptr1 + (x3), None)
     tmp4 = tl.load(in_ptr2 + (x3), None)
     tmp7 = tl.load(in_ptr3 + (x3), None)
     tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
     tmp10 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
     tmp14 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
     tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
     tmp22 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
     tmp25 = tl.load(in_ptr9 + (x3), None)
     tmp26 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
     tmp28 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
     tmp30 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
     tmp36 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
     tmp1 = 0.0
     tmp2 = tmp0 <= tmp1
     tmp5 = tmp3 + tmp4
     tmp6 = tl.where(tmp2, tmp1, tmp5)
     tmp9 = tmp7 - tmp8
     tmp11 = 0.000318877551020408*(1/ks0)
     tmp12 = tmp11.to(tl.float32)
     tmp13 = tmp10 * tmp12
     tmp15 = tmp14 * tmp14
     tmp16 = tmp13 * tmp15
     tmp17 = tmp9 * tmp16
     tmp18 = tmp6 - tmp17
     tmp20 = tmp19 * tmp12
     tmp21 = tmp18 - tmp20
     tmp23 = tmp14 * tmp22
     tmp24 = tmp21 * tmp23
     tmp27 = tmp25 - tmp26
     tmp29 = tmp28 * tmp12
     tmp31 = tmp30 * tmp30
     tmp32 = tmp29 * tmp31
     tmp33 = tmp27 * tmp32
     tmp34 = tmp6 - tmp33
     tmp35 = tmp34 - tmp20
     tmp37 = tmp30 * tmp36
     tmp38 = tmp35 * tmp37
     tl.store(out_ptr0 + (x3), tmp24, None)
     tl.store(out_ptr1 + (x3), tmp38, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/6y/c6yuhy2yyyxkyz53uiahwzp63l5rkr3r3g6ooznmezcyhkfs55hy.py
 # Source Nodes: [], Original ATen: [aten.add]
 
 triton_poi_fused_add_46 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_46', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
 @triton.jit
 def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x0 = xindex
     tmp0 = tl.load(in_out_ptr0 + (x0), None)
     tmp1 = tl.load(in_ptr0 + (x0), None)
     tmp2 = tmp0 + tmp1
     tl.store(in_out_ptr0 + (x0), tmp2, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/mi/cmi67v4v4o6dzqxrpaeo62qu3z3c3mftldnqnfkgkhlwvwnhn7am.py
 # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
 
 triton_poi_fused_add_max_pool2d_with_indices_backward_47 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_max_pool2d_with_indices_backward_47', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]})
 @triton.jit
 def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x0 = xindex % 112
     x1 = (xindex // 112) % 112
     x2 = (xindex // 12544)
     x3 = xindex % 12544
     x5 = xindex
     tmp0 = tl.load(in_ptr0 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
     tmp1 = tl.load(in_ptr1 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
     tmp6 = tl.load(in_ptr0 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
     tmp7 = tl.load(in_ptr1 + ((56*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
     tmp19 = tl.load(in_ptr0 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
     tmp20 = tl.load(in_ptr1 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
     tmp30 = tl.load(in_ptr0 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
     tmp31 = tl.load(in_ptr1 + ((56*(tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2)))))) + (56*(tl.where((tl.math.min(1 + (tl.math.max(0, (x1 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x1) // 2))))) >= 0, 0, 56))) + (3136*x2) + (tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) + (tl.where((tl.math.min(1 + (tl.math.max(0, (x0 // 2))), (-1) + (tl.math.min(56, 1 + ((1 + x0) // 2))))) >= 0, 0, 56))), None)
     tmp2 = x3
     tmp3 = tmp0 == tmp2
     tmp4 = 0.0
     tmp5 = tl.where(tmp3, tmp1, tmp4)
     tmp8 = tmp6 == tmp2
     tmp9 = tl.math.max(0, (x1 // 2))
     tmp10 = tl.math.min(56, 1 + ((1 + x1) // 2))
     tmp11 = tmp9 < tmp10
     tmp12 = 1 + (tl.math.max(0, (x0 // 2)))
     tmp13 = tl.math.min(56, 1 + ((1 + x0) // 2))
     tmp14 = tmp12 < tmp13
     tmp15 = tmp11 & tmp14
     tmp16 = tmp15 & tmp8
     tmp17 = tmp5 + tmp7
     tmp18 = tl.where(tmp16, tmp17, tmp5)
     tmp21 = tmp19 == tmp2
     tmp22 = 1 + (tl.math.max(0, (x1 // 2)))
     tmp23 = tmp22 < tmp10
     tmp24 = tl.math.max(0, (x0 // 2))
     tmp25 = tmp24 < tmp13
     tmp26 = tmp23 & tmp25
     tmp27 = tmp26 & tmp21
     tmp28 = tmp18 + tmp20
     tmp29 = tl.where(tmp27, tmp28, tmp18)
     tmp32 = tmp30 == tmp2
     tmp33 = tmp23 & tmp14
     tmp34 = tmp33 & tmp32
     tmp35 = tmp29 + tmp31
     tmp36 = tl.where(tmp34, tmp35, tmp29)
     tl.store(out_ptr0 + (x5), tmp36, None)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/dn/cdnoo74u7t64wqygd5ztbngvkk56x2tmislcwu72ofqvrt4amqyg.py
 # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_red_fused_native_batch_norm_backward_threshold_backward_48 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @reduction(
     size_hints=[512, 131072],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_48', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
     xnumel = 448
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rbase = tl.arange(0, RBLOCK)[None, :]
     x0 = xindex % 64
     x1 = (xindex // 64)
     _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     x3 = xindex
     tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
     for roffset in range(0, rnumel, RBLOCK):
         rindex = roffset + rbase
         rmask = rindex < rnumel
         r2 = rindex
         tmp0 = tl.load(in_ptr0 + ((112*(((r2 + (1792*ks0*x1)) // 112) % 112)) + (12544*x0) + (802816*((r2 + (1792*ks0*x1)) // 12544)) + (r2 % 112)), rmask & xmask, other=0)
         tmp3 = tl.load(in_ptr1 + ((112*(((r2 + (1792*ks0*x1)) // 112) % 112)) + (12544*x0) + (802816*((r2 + (1792*ks0*x1)) // 12544)) + (r2 % 112)), rmask & xmask, other=0)
         tmp8 = tl.load(in_ptr2 + ((112*(((r2 + (1792*ks0*x1)) // 112) % 112)) + (12544*x0) + (802816*((r2 + (1792*ks0*x1)) // 12544)) + (r2 % 112)), rmask & xmask, other=0)
         tmp1 = 0.0
         tmp2 = tmp0 <= tmp1
         tmp4 = tl.where(tmp2, tmp1, tmp3)
         tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
         tmp7 = _tmp6 + tmp5
         _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
         tmp10 = tmp8 - tmp9
         tmp11 = tmp4 * tmp10
         tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
         tmp14 = _tmp13 + tmp12
         _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
     tmp6 = tl.sum(_tmp6, 1)[:, None]
     tl.store(out_ptr0 + (x3), tmp6, xmask)
     tmp13 = tl.sum(_tmp13, 1)[:, None]
     tl.store(out_ptr1 + (x3), tmp13, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/6n/c6nf5ptfptqptmhflx77pvxdfwykdustv34zjapepld5bifl36cr.py
 # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_per_fused_native_batch_norm_backward_threshold_backward_49 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @persistent_reduction(
     size_hints=[64, 8],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_49', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]}
 )
 @triton.jit
 def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
     xnumel = 64
     rnumel = 7
     RBLOCK: tl.constexpr = 8
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rindex = tl.arange(0, RBLOCK)[None, :]
     rmask = rindex < rnumel
     r1 = rindex
     x0 = xindex
     tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0)
     tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
     tmp3 = tl.where(rmask & xmask, tmp1, 0)
     tmp4 = tl.sum(tmp3, 1)[:, None]
     tl.store(out_ptr0 + (x0), tmp4, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/kp/ckpjbtq26iclfse3y4lncwb2xc4adkkk4lsfu3v3ezvh7nyozyqv.py
 # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_per_fused_native_batch_norm_backward_threshold_backward_50 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @persistent_reduction(
     size_hints=[64, 8],
     reduction_hint=ReductionHint.INNER,
     filename=__file__,
     meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_50', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]}
 )
 @triton.jit
 def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
     xnumel = 64
     rnumel = 7
     RBLOCK: tl.constexpr = 8
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
     xmask = xindex < xnumel
     rindex = tl.arange(0, RBLOCK)[None, :]
     rmask = rindex < rnumel
     r1 = rindex
     x0 = xindex
     tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0)
     tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
     tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
     tmp3 = tl.where(rmask & xmask, tmp1, 0)
     tmp4 = tl.sum(tmp3, 1)[:, None]
     tmp6 = tmp4 * tmp5
     tl.store(out_ptr1 + (x0), tmp6, xmask)
     tl.store(out_ptr0 + (x0), tmp4, xmask)
 ''')
 
 
 # kernel path: /tmp/torchinductor_root/t6/ct6rzjsokg46b4zvfpwyvw3a7ai23qix5kh5kgbw654jg3mq4dmp.py
 # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
 
 triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_51 = async_compile.triton('triton_', '''
 import triton
 import triton.language as tl
 from torch._inductor.ir import ReductionHint
 from torch._inductor.ir import TileHint
 from torch._inductor.triton_heuristics import AutotuneHint, pointwise
 from torch._inductor.utils import instance_descriptor
 from torch._inductor import triton_helpers
 
 @pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_51', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]})
 @triton.jit
 def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ks0, xnumel, XBLOCK : tl.constexpr):
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x3 = xindex
     x1 = (xindex // 12544) % 64
     tmp0 = tl.load(in_ptr0 + (x3), None)
     tmp3 = tl.load(in_out_ptr0 + (x3), None)
     tmp5 = tl.load(in_ptr1 + (x3), None)
     tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
     tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
     tmp17 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
     tmp20 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
     tmp1 = 0.0
     tmp2 = tmp0 <= tmp1
     tmp4 = tl.where(tmp2, tmp1, tmp3)
     tmp7 = tmp5 - tmp6
     tmp9 = (7.97193877551020e-5)*(1/ks0)
     tmp10 = tmp9.to(tl.float32)
     tmp11 = tmp8 * tmp10
     tmp13 = tmp12 * tmp12
     tmp14 = tmp11 * tmp13
     tmp15 = tmp7 * tmp14
     tmp16 = tmp4 - tmp15
     tmp18 = tmp17 * tmp10
     tmp19 = tmp16 - tmp18
     tmp21 = tmp12 * tmp20
     tmp22 = tmp19 * tmp21
     tl.store(in_out_ptr0 + (x3), tmp22, None)
 ''')