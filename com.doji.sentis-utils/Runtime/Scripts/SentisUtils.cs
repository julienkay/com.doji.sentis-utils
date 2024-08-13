using System;
using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine.Assertions;

namespace Doji.AI {

    /// <summary>
    /// Extends Ops class with not-yet implemented operators
    /// and some overloads for more convenience.
    /// </summary>
    public static class SentisUtils {

        private static Tensor[] _tmpTensorRefs = new Tensor[2];

        /// <summary>
        /// Computes the q-th quantiles of each row of the input tensor along the dimension dim.
        /// torch.quantile
        /// </summary>
        public static TensorFloat Quantile(this Ops ops, TensorFloat tensor, float q, int dim) {
            TensorFloat sorted = ops.Sort(tensor, dim);

            if (q < 0 || q > 1) {
                throw new ArgumentException("Quantile value must be between 0 and 1");
            }

            float index = (tensor.shape[dim] - 1) * q;

            using TensorInt lowerIndex = new TensorInt((int)MathF.Floor(index));
            using TensorInt upperIndex = new TensorInt((int)MathF.Ceiling(index));

            TensorFloat lowerValues = ops.Gather(sorted, lowerIndex, dim);
            TensorFloat upperValues = ops.Gather(sorted, upperIndex, dim);
            float weights = index - (int)MathF.Floor(index);

            TensorFloat sub = ops.Sub(upperValues, lowerValues);
            TensorFloat mul = ops.Mul(sub, weights);
            TensorFloat interpolated = ops.Add(mul, lowerValues);
            return interpolated;
        }

        public static TensorFloat Sort(this Ops ops, TensorFloat tensor, int dim) {
            int num = tensor.shape[dim];
            return ops.TopK(tensor, num, dim, largest: false /* sort lowest-to-highest */, true).values;
        }

        public static TensorFloat Clamp(this Ops ops, TensorFloat tensor, TensorFloat min, TensorFloat max) {
            return ops.Min(ops.Max(tensor, min), max);
        }

        /// <summary>
        /// Alias for <see cref="Ops.Concat(Tensor[], int)"/> to match numpy.concatenate()
        /// naming and for convenience of not needing to create a Tensor array.
        /// </summary>
        public static Tensor Concatenate(this Ops ops, Tensor tensor1, Tensor tensor2, int axis = 0) {
            _tmpTensorRefs[0] = tensor1;
            _tmpTensorRefs[1] = tensor2;
            return ops.Concat(_tmpTensorRefs, axis);
        }

        /// <summary>
        /// Alias for <see cref="Ops.Concat(Tensor[], int)"/> to match numpy.concatenate()
        /// naming and for convenience by adding a List<TensorFloat> overload.
        /// </summary>
        public static Tensor Concatenate(this Ops ops, List<Tensor> tensors, int axis = 0) {
            return ops.Concat(tensors, axis);
        }

        public static TensorFloat Concatenate(this Ops ops, TensorFloat tensor1, TensorFloat tensor2, int axis = 0) {
            return ops.Concatenate(tensor1 as Tensor, tensor2 as Tensor, axis) as TensorFloat;
        }

        public static TensorInt Concatenate(this Ops ops, TensorInt tensor1, TensorInt tensor2, int axis = 0) {
            return ops.Concatenate(tensor1 as Tensor, tensor2 as Tensor, axis) as TensorInt;
        }

        /// <summary>
        /// Similar to torch.repeat() or numpy.tile()
        /// </summary>
        public static TensorFloat Repeat(this Ops ops, TensorFloat tensor, int repeats, int axis) {
            if (repeats <= 0) {
                throw new ArgumentException($"Repeat count must be greater than zero, was {repeats}.", nameof(repeats));
            }

            if (repeats == 1) {
                return tensor;
            }

            int[] r = ArrayUtils.Full(tensor.shape.rank, 1);
            r[axis] = repeats;
            return ops.Tile(tensor, r);
        }

        /// <summary>
        /// Similar to torch.repeat() or numpy.tile()
        /// </summary>
        public static TensorInt Repeat(this Ops ops, TensorInt tensor, int repeats, int axis) {
            if (repeats <= 0) {
                throw new ArgumentException($"Repeat count must be greater than zero, was {repeats}.", nameof(repeats));
            }

            if (repeats == 1) {
                return tensor;
            }

            int[] r = ArrayUtils.Full(tensor.shape.rank, 1);
            r[axis] = repeats;
            return ops.Tile(tensor, r);
        }

        /// <summary>
        /// Similar to torch.repeat_interleave() or numpy.repeat()
        /// </summary>
        public static TensorFloat RepeatInterleave(this Ops ops, TensorFloat tensor, int repeats, int dim) {
            if (repeats <= 0) {
                throw new ArgumentException($"Repeat count must be greater than zero, was {repeats}.", nameof(repeats));
            }
            if (tensor.shape.rank > 1) {
                throw new ArgumentException($"RepeatInterleave not supported yet for tensors with rank > 1. Shape was {tensor.shape}");
            }

            // implement repeat_interleave using repeat, reshape & transpose ops
            var repeat = ops.Repeat(tensor, repeats, dim);
            var flatShape = new TensorShape(repeat.shape.length);
            repeat.Reshape(flatShape);
            repeat.Reshape(new TensorShape(repeats, flatShape.length / repeats));
            var transpose = ops.Transpose(repeat, new int[] { 1, 0 });
            transpose.Reshape(flatShape);
            return transpose;
        }

        /// <summary>
        /// Similar to torch.repeat_interleave() or numpy.repeat()
        /// </summary>
        public static TensorInt RepeatInterleave(this Ops ops, TensorInt tensor, int repeats, int dim) {
            if (repeats <= 0) {
                throw new ArgumentException($"Repeat count must be greater than zero, was {repeats}.", nameof(repeats));
            }
            if (tensor.shape.rank > 1) {
                throw new ArgumentException($"RepeatInterleave not supported yet for tensors with rank > 1. Shape was {tensor.shape}");
            }

            // implement repeat_interleave using repeat, reshape & transpose ops
            var repeat = ops.Repeat(tensor, repeats, dim);
            var flatShape = new TensorShape(repeat.shape.length);
            repeat.Reshape(flatShape);
            repeat.Reshape(new TensorShape(repeats, flatShape.length / repeats));
            var transpose = ops.Transpose(repeat, new int[] { 1, 0 });
            transpose.Reshape(flatShape);
            return transpose;
        }

        /// <summary>
        /// Alias for <see cref="Ops.Split{T}(T, int, int, int)"/> to match the
        /// arguments of numpy.split() or torch.chunk() i.e. providing <paramref name="sections"/>
        /// that the original tensor is split into.
        /// </summary>
        public static void Split(this Ops ops, Tensor tensor, int sections, int axis = 0, List<TensorFloat> splitTensors = null) {
            if (tensor.shape[axis] % sections != 0) {
                throw new ArgumentException($"Tensor dimension {axis} (length: {tensor.shape[axis]}) can not be divided into {sections} sections.");
            }
            splitTensors ??= new List<TensorFloat>();
            splitTensors.Clear();

            int step = tensor.shape[axis] / sections;
            int end = tensor.shape[axis] - step;
            for (int i = 0; i < end; i += step) {
                var section = ops.Split(tensor, axis: axis, i, i + step) as TensorFloat;
                splitTensors.Add(section);
            }
        }

        /// <summary>
        /// Splits a tensor into two sections.
        /// </summary>
        public static (TensorFloat a, TensorFloat b) SplitHalf(this Ops ops, Tensor tensor, int axis = 0) {
            if (tensor.shape[axis] % 2 != 0) {
                throw new ArgumentException($"Tensor dimension {axis} (length: {tensor.shape[axis]}) can not be divided into 2 sections.");
            }
            int half = tensor.shape[axis] / 2;
            int start = 0;
            int end = tensor.shape[axis];
            var a = ops.Split(tensor, axis: axis, start, half) as TensorFloat;
            var b = ops.Split(tensor, axis: axis, half, end) as TensorFloat;
            return (a, b);
        }

        public static T Slice<T>(this Ops ops, T tensor, Index i) where T : Tensor {
            return ops.Slice(tensor, i.ToRange());
        }
        public static T Slice<T>(this Ops ops, T tensor, Index i, Range r) where T : Tensor {
            return ops.Slice(tensor, i.ToRange(), r);
        }
        public static T Slice<T>(this Ops ops, T tensor, Range r, Index i) where T : Tensor {
            return ops.Slice(tensor, r, i.ToRange());
        }
        public static T Slice<T>(this Ops ops, T tensor, Index i, Range r1, Range r2) where T : Tensor {
            T O = ops.Slice(tensor, i.ToRange(), r1, r2);
            return O;
        }
        public static T Slice<T>(this Ops ops, T tensor, Range r0, Index i, Range r2) where T : Tensor {
            T O = ops.Slice(tensor, r0, i.ToRange(), r2);
            return O;
        }
        public static T Slice<T>(this Ops ops, T tensor, Range r0, Range r1, Index i) where T : Tensor {
            T O = ops.Slice(tensor, r0, r1, i.ToRange());
            return O;
        }
        public static T Slice<T>(this Ops ops, T tensor, Index i0, Index i1, Range r) where T : Tensor {
            T O = ops.Slice(tensor, i0.ToRange(), i1.ToRange(), r);
            return O;
        }
        public static T Slice<T>(this Ops ops, T tensor, Range r, Index i0, Index i1) where T : Tensor {
            T O = ops.Slice(tensor, r, i0.ToRange(), i1.ToRange());
            return O;
        }
        public static T Slice<T>(this Ops ops, T tensor, Index i0, Range r, Index i2) where T : Tensor {
            T O = ops.Slice(tensor, i0.ToRange(), r, i2.ToRange());
            return O;
        }
        public static T Slice<T>(this Ops ops, T tensor, Index i0, Index i1, Index i2) where T : Tensor {
            T O = ops.Slice(tensor, i0.ToRange(), i1.ToRange(), i2.ToRange());
            return O;
        }
        public static Range ToRange(this Index i) {
            return i.IsFromEnd ? i..new Index(i.Value - 1, true) : (i..new Index(i.Value + 1, false));
        }

        /// <summary>
        /// A Slice() method that uses C# indices and ranges rather than (start, end axes, steps) parameters.
        /// For example:
        /// Doing 'tensor[  :  , -1 ,  :  ]' in python directly translates to 
        /// .Slice(tensor,  .. , ^1 ,  .. )
        /// </summary>
        public static T Slice<T>(this Ops ops, T tensor, params Range[] ranges) where T : Tensor {
            var shape = tensor.shape;
            int rank = shape.rank;
            int numDimensionsToSlice = ranges.Length;

            Assert.IsTrue(rank >= numDimensionsToSlice, $"Too many indices ({numDimensionsToSlice}) for tensor of rank {rank}.");

            unsafe {
                Span<int> starts = stackalloc int[rank];
                Span<int> ends = stackalloc int[rank];
                Span<int> axes = stackalloc int[rank];
                Span<int> steps = stackalloc int[rank];

                for (int i = 0; i < rank; i++) {
                    Range range = i < numDimensionsToSlice ? ranges[i] : ..;
                    (int offset, int length) = range.GetOffsetAndLength(shape[i]);
                    starts[i] = offset;
                    ends[i] = offset + length;
                    axes[i] = i;
                    steps[i] = 1;
                }

                Tensor outputTensor = ops.Slice(tensor, starts, ends, axes, steps);
                return outputTensor as T;
            }
        }

        /// <summary>
        /// Fills elements of <paramref name="tensor"/> with <paramref name="value"/> where <paramref name="mask"/> is 1
        /// </summary>
        public static TensorFloat MaskedFill(this Ops ops, TensorFloat tensor, TensorInt mask, float value) {
            using TensorFloat A = new TensorFloat(value);
            return ops.Where(mask, A, tensor);
        }

        /// <summary>
        /// Fills elements of <paramref name="tensor"/> with <paramref name="value"/> where <paramref name="mask"/> is 1
        /// </summary>
        public static TensorInt MaskedFill(this Ops ops, TensorInt tensor, TensorInt mask, int value) {
            using TensorInt A = new TensorInt(value);
            return ops.Where(mask, A, tensor);
        }

        //TODO: needs test, https://discuss.pytorch.org/t/alternatives-to-torch-isin/190297/3
        /*public static TensorInt IsIn(this Ops ops, TensorInt tensor1, TensorInt tensor2) {
            tensor1.Reshape(tensor1.shape.Unsqueeze(1));
            var eq = ops.Equal(tensor1, tensor2);
            var sum = ops.ReduceSum(eq, new int[] { 0, 1 });
            e.unsqueeze(1) == t).sum();
            tensor1.Reshape(tensor1.shape.Squeeze(1));
            return sum;
        }*/
    }
}