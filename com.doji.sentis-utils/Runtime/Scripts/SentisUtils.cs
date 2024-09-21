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

        /// <summary>
        /// Computes the q-th quantiles of each row of the input tensor along the dimension dim.
        /// torch.quantile
        /// </summary>
        public static Tensor<float> Quantile(this Ops ops, Tensor<float> tensor, float q, int dim) {
            Tensor<float> sorted = ops.Sort(tensor, dim);

            if (q < 0 || q > 1) {
                throw new ArgumentException("Quantile value must be between 0 and 1");
            }

            float index = (tensor.shape[dim] - 1) * q;

            Tensor<int> lowerIndex = ops.NewTensor((int)MathF.Floor(index));
            Tensor<int> upperIndex = ops.NewTensor((int)MathF.Ceiling(index));

            Tensor<float> lowerValues = ops.Gather(sorted, lowerIndex, dim);
            Tensor<float> upperValues = ops.Gather(sorted, upperIndex, dim);
            float weights = index - (int)MathF.Floor(index);

            Tensor<float> sub = ops.Sub(upperValues, lowerValues);
            Tensor<float> mul = ops.Mul(sub, weights);
            Tensor<float> interpolated = ops.Add(mul, lowerValues);
            return interpolated;
        }

        public static Tensor<float> Sort(this Ops ops, Tensor<float> tensor, int dim) {
            int num = tensor.shape[dim];
            return ops.TopK(tensor, num, dim, largest: false /* sort lowest-to-highest */, true).values;
        }

        public static Tensor<float> Clamp(this Ops ops, Tensor<float> tensor, Tensor<float> min, Tensor<float> max) {
            return ops.Min(ops.Max(tensor, min), max);
        }

        /// <summary>
        /// Alias for <see cref="Ops.Concat(Tensor[], int)"/> to match torch.cat()
        /// naming and for convenience of not needing to create a Tensor array.
        /// </summary>
        public static T Cat<T>(this Ops ops, T tensor1, T tensor2, int axis = 0) where T : Tensor {
            return ops.Concat(tensor1, tensor2, axis);
        }

        /// <summary>
        /// Alias for <see cref="Ops.Concat(Tensor[], int)"/> to match numpy.concatenate()
        /// naming and for convenience of not needing to create a Tensor array.
        /// </summary>
        public static T Concatenate<T>(this Ops ops, T tensor1, T tensor2, int axis = 0) where T : Tensor {
            return ops.Concat(tensor1, tensor2, axis);
        }

        /// <summary>
        /// A List<Tensor> overload for Cat().
        /// </summary>
        public static T Cat<T>(this Ops ops, List<T> tensors, int axis = 0) where T : Tensor {
            return ops.Concat(tensors, axis) as T;
        }

        /// <summary>
        /// A List<TensorFloat> overload for Cat().
        /// </summary>
        public static Tensor<float> Cat(this Ops ops, List<Tensor<float>> tensors, int axis = 0) {
            return ops.Concat(tensors, axis) as Tensor<float>;
        }

        /// <summary>
        /// A List<TensorInt> overload for Cat().
        /// </summary>
        public static Tensor<int> Cat(this Ops ops, List<Tensor<int>> tensors, int axis = 0) {
            return ops.Concat(tensors, axis) as Tensor<int>;
        }

        /// <summary>
        /// A List<Tensor> overload for Concatenate().
        /// </summary>
        public static T Concatenate<T>(this Ops ops, List<Tensor> tensors, int axis = 0) where T : Tensor {
            return ops.Concat(tensors, axis) as T;
        }

        /// <summary>
        /// A List<TensorFloat> overload for Concatenate().
        /// </summary>
        public static Tensor<float> Concatenate(this Ops ops, List<Tensor<float>> tensors, int axis = 0) {
            return ops.Concat(tensors, axis) as Tensor<float>;
        }

        /// <summary>
        /// A List<TensorFloat> overload for Concatenate().
        /// </summary>
        public static Tensor<int> Concatenate(this Ops ops, List<Tensor<int>> tensors, int axis = 0) {
            return ops.Concat(tensors, axis) as Tensor<int>;
        }

        /// <summary>
        /// Similar to torch.repeat() or numpy.tile()
        /// </summary>
        public static Tensor<float> Repeat(this Ops ops, Tensor<float> tensor, int repeats, int axis) {
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
        public static Tensor<int> Repeat(this Ops ops, Tensor<int> tensor, int repeats, int axis) {
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
        public static Tensor<float> RepeatInterleave(this Ops ops, Tensor<float> tensor, int repeats, int dim) {
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
        public static Tensor<int> RepeatInterleave(this Ops ops, Tensor<int> tensor, int repeats, int dim) {
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
        public static void Split(this Ops ops, Tensor tensor, int sections, int axis = 0, List<Tensor<float>> splitTensors = null) {
            if (tensor.shape[axis] % sections != 0) {
                throw new ArgumentException($"Tensor dimension {axis} (length: {tensor.shape[axis]}) can not be divided into {sections} sections.");
            }
            splitTensors ??= new List<Tensor<float>>();
            splitTensors.Clear();

            int step = tensor.shape[axis] / sections;
            int end = tensor.shape[axis] - step;
            for (int i = 0; i < end; i += step) {
                var section = ops.Split(tensor, axis: axis, i, i + step) as Tensor<float>;
                splitTensors.Add(section);
            }
        }

        /// <summary>
        /// Splits a tensor into two sections.
        /// </summary>
        public static (Tensor<float> a, Tensor<float> b) SplitHalf(this Ops ops, Tensor tensor, int axis = 0) {
            if (tensor.shape[axis] % 2 != 0) {
                throw new ArgumentException($"Tensor dimension {axis} (length: {tensor.shape[axis]}) can not be divided into 2 sections.");
            }
            int half = tensor.shape[axis] / 2;
            int start = 0;
            int end = tensor.shape[axis];
            var a = ops.Split(tensor, axis: axis, start, half) as Tensor<float>;
            var b = ops.Split(tensor, axis: axis, half, end) as Tensor<float>;
            return (a, b);
        }

        public static T Slice<T>(this Ops ops, T tensor, Index i) where T : Tensor {
            T O = ops.Slice(tensor, i.ToRange());
            O.Reshape(O.shape.Squeeze(0));
            return O;
        }
        public static T Slice<T>(this Ops ops, T tensor, Index i, Range r) where T : Tensor {
            T O = ops.Slice(tensor, i.ToRange(), r);
            O.Reshape(O.shape.Squeeze(0));
            return O;
        }
        public static T Slice<T>(this Ops ops, T tensor, Range r, Index i) where T : Tensor {
            T O = ops.Slice(tensor, r, i.ToRange());
            O.Reshape(O.shape.Squeeze(1));
            return O;
        }
        public static T Slice<T>(this Ops ops, T tensor, Index i, Range r1, Range r2) where T : Tensor {
            T O = ops.Slice(tensor, i.ToRange(), r1, r2);
            O.Reshape(O.shape.Squeeze(0));
            return O;
        }
        public static T Slice<T>(this Ops ops, T tensor, Range r0, Index i, Range r2) where T : Tensor {
            T O = ops.Slice(tensor, r0, i.ToRange(), r2);
            O.Reshape(O.shape.Squeeze(1));
            return O;
        }
        public static T Slice<T>(this Ops ops, T tensor, Range r0, Range r1, Index i) where T : Tensor {
            T O = ops.Slice(tensor, r0, r1, i.ToRange());
            O.Reshape(O.shape.Squeeze(2));
            return O;
        }
        public static T Slice<T>(this Ops ops, T tensor, Index i0, Index i1, Range r) where T : Tensor {
            T O = ops.Slice(tensor, i0.ToRange(), i1.ToRange(), r);
            O.Reshape(O.shape.Squeeze(1));
            O.Reshape(O.shape.Squeeze(0));
            return O;
        }
        public static T Slice<T>(this Ops ops, T tensor, Range r, Index i0, Index i1) where T : Tensor {
            T O = ops.Slice(tensor, r, i0.ToRange(), i1.ToRange());
            O.Reshape(O.shape.Squeeze(2));
            O.Reshape(O.shape.Squeeze(1));
            return O;
        }
        public static T Slice<T>(this Ops ops, T tensor, Index i0, Range r, Index i2) where T : Tensor {
            T O = ops.Slice(tensor, i0.ToRange(), r, i2.ToRange());
            O.Reshape(O.shape.Squeeze(2));
            O.Reshape(O.shape.Squeeze(0));
            return O;
        }
        public static T Slice<T>(this Ops ops, T tensor, Index i0, Index i1, Index i2) where T : Tensor {
            T O = ops.Slice(tensor, i0.ToRange(), i1.ToRange(), i2.ToRange());
            O.Reshape(O.shape.Squeeze(2));
            O.Reshape(O.shape.Squeeze(1));
            O.Reshape(O.shape.Squeeze(0));
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
        public static Tensor<float> MaskedFill(this Ops ops, Tensor<float> tensor, Tensor<int> mask, float value) {
            using Tensor<float> A = new Tensor<float>(new TensorShape(), new float[] { value });
            return ops.Where(mask, A, tensor);
        }

        /// <summary>
        /// Fills elements of <paramref name="tensor"/> with <paramref name="value"/> where <paramref name="mask"/> is 1
        /// </summary>
        public static Tensor<int> MaskedFill(this Ops ops, Tensor<int> tensor, Tensor<int> mask, int value) {
            using Tensor<int> A = new Tensor<int>(new TensorShape(), new int[] { value });
            return ops.Where(mask, A, tensor);
        }

        /// <summary>
        /// Creates a tensor of the given <paramref name="shape"/> size filled with <paramref name="fillValue"/>.
        /// </summary>
        public static Tensor<int> Full(this Ops ops, TensorShape shape, int fillvalue) {
            return ops.NewTensor(shape, ArrayUtils.Full(shape.length, fillvalue));
        }

        /// <summary>
        /// Creates a tensor of the given <paramref name="shape"/> size filled with <paramref name="fillValue"/>.
        /// </summary>
        public static Tensor<int> Full(this Ops ops, TensorShape shape, bool fillvalue) {
            return ops.NewTensor(shape, ArrayUtils.Full(shape.length, fillvalue ? 1 : 0));
        }

        /// <summary>
        /// Creates a 1D tensor of the given <paramref name="size"/> size filled with <paramref name="fillValue"/>.
        /// </summary>
        public static Tensor<int> Full(this Ops ops, int size, bool fillvalue) {
            TensorShape shape = new TensorShape(size);
            return ops.NewTensor(shape, ArrayUtils.Full(shape.length, fillvalue ? 1 : 0));
        }

        /// <summary>
        /// Creates a tensor of the given <paramref name="shape"/> size filled with <paramref name="fillValue"/>.
        /// </summary>
        public static Tensor<float> Full(this Ops ops, TensorShape shape, float fillvalue) {
            return ops.NewTensor(shape, ArrayUtils.Full(shape.length, fillvalue));
        }

        //TODO: needs test, https://discuss.pytorch.org/t/alternatives-to-torch-isin/190297/3
        /*public static Tensor<int> IsIn(this Ops ops, Tensor<int> tensor1, Tensor<int> tensor2) {
            tensor1.Reshape(tensor1.shape.Unsqueeze(1));
            var eq = ops.Equal(tensor1, tensor2);
            var sum = ops.ReduceSum(eq, new int[] { 0, 1 });
            e.unsqueeze(1) == t).sum();
            tensor1.Reshape(tensor1.shape.Squeeze(1));
            return sum;
        }*/
    }
}