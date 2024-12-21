# Version History

- flash-attention-1-v01.cu
  - All batch in one kernel launch
- flash-attention-1-v02.cu
  - Batch in multiple kernel launches in different streams
  - Padding the tensors
- flash-attention-1-v03.cu
  - No padding
- flash-attention-2-v01.cu
  - Flash attention 2
- flash-attention-2-v02.cu
  - v01 No row max
  - Fused qk dot and exp
