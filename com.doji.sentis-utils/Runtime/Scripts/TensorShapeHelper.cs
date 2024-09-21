using System;
using System.Collections.Generic;
using System.Reflection;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI {

    internal static class TensorShapeHelper {

        private delegate TensorShape SliceDelegate(ReadOnlySpan<int> starts, ReadOnlySpan<int> ends, ReadOnlySpan<int> axes, ReadOnlySpan<int> steps);
        private static readonly MethodInfo _sliceMethod;

        private delegate TensorShape ConcatDelegate(TensorShape other, int axis);
        private static readonly MethodInfo _concatMethod;

        private delegate TensorShape TileDelegate(ReadOnlySpan<int> repeats);
        private static readonly MethodInfo _tileMethod;

        private delegate TensorShape TransposeDelegate(ReadOnlySpan<int> permutations);
        private static readonly MethodInfo _transposeMethod;

        static TensorShapeHelper() {
            _sliceMethod = typeof(TensorShape).GetMethod("Slice", BindingFlags.NonPublic | BindingFlags.Instance);
            _concatMethod = typeof(TensorShape).GetMethod("Concat", BindingFlags.NonPublic | BindingFlags.Instance);
            _tileMethod = typeof(TensorShape).GetMethod("Tile", BindingFlags.NonPublic | BindingFlags.Instance);
            Debug.Log("HERE??");
            _transposeMethod = typeof(TensorShape).GetMethod("Transpose", BindingFlags.NonPublic | BindingFlags.Instance, null, new Type[] { typeof(int[]) }, null );
        }

        public static TensorShape ConcatShape<T>(T[] tensors, int axis) where T : Tensor {
            Debug.Log("HERE??");
            TensorShape output = tensors[0].shape;

            for (int i = 1; i < tensors.Length; ++i) {
                TensorShape shape = tensors[i].shape;
                output = output.Concat(shape, axis);
            }

            return output;
        }

        public static TensorShape ConcatShape<T>(List<T> tensors, int axis) where T : Tensor {
            TensorShape output = tensors[0].shape;

            for (int i = 1; i < tensors.Count; ++i) {
                TensorShape shape = tensors[i].shape;
                output = output.Concat(shape, axis);
            }

            return output;
        }

        public static TensorShape ConcatShape(Tensor tensor1, Tensor tensor2, int axis) {
            return tensor1.shape.Concat(tensor2.shape, axis);
        }

        public static TensorShape BroadcastShape(Tensor a, Tensor b) {
            return a.shape.Broadcast(b.shape);
        }

        /// <summary>
        /// Creates a `TensorShape` that results from slicing `this` along given axes with given starts, ends, and steps.
        /// </summary>
        /// <param name="starts">The start indices along each of the `axes`.</param>
        /// <param name="ends">The end indices along each of the `axes`.</param>
        /// <param name="axes">The axes along which to slice.</param>
        /// <param name="steps">The step sizes for each of the `axes`.</param>
        /// <returns>The sliced tensor shape.</returns>
        public static TensorShape Slice(this TensorShape shape, ReadOnlySpan<int> starts, ReadOnlySpan<int> ends, ReadOnlySpan<int> axes, ReadOnlySpan<int> steps) {
            var sliceMethod = (SliceDelegate)_sliceMethod.CreateDelegate(typeof(SliceDelegate), shape);
            return sliceMethod(starts, ends, axes, steps);
        }

        public static TensorShape Concat(this TensorShape shape, TensorShape other, int axis) {
            var concatMethod = (ConcatDelegate)_concatMethod.CreateDelegate(typeof(ConcatDelegate), shape);
            return concatMethod(other, axis);
        }

        public static TensorShape Tile(this TensorShape shape, ReadOnlySpan<int> repeats) {
            var tileMethod = (TileDelegate)_tileMethod.CreateDelegate(typeof(TileDelegate), shape);
            return tileMethod(repeats);
        }

        public static TensorShape Transpose(this TensorShape shape, ReadOnlySpan<int> permutations) {
            var transposeMethod = (TransposeDelegate)_transposeMethod.CreateDelegate(typeof(TransposeDelegate), shape);
            return transposeMethod(permutations);
        }
    }
}